"""Component-level profile of `extract` -- the stage that owns the latency tail.

Everything else in the fast path sums to ~10ms P50. `extract` runs 32ms P50 but
175ms P99 and 237ms P100, and it alone accounts for every query over the 200ms
budget. Averages hide this completely, so this reports P50/P70/P90/P95/P99/P100
per component and, crucially, correlates the tail against *batch shape*.

The hypothesis under test: ONNX pads a batch to its longest member, and Day 1
already measured embedding throughput as superlinear in sequence length. If one
300-token sentence shares a batch with nine 20-token ones, all ten cost the long
one. That would make the tail a function of `max_sentence_chars`, not of the
number of sentences -- a distinction that decides whether the fix is "embed
fewer" or "truncate longer".

    python -m bench.profile_extract --tag full --n 300
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from core.embedder import Embedder, EmbedderConfig
from core.extractive import extract_answer
from core.retriever import DEFAULT_ENSEMBLE, AdaptiveRetriever
from ingest.pipeline import DATA_ROOT, read_jsonl

PERCENTILES = {"p50": 50, "p70": 70, "p90": 90, "p95": 95, "p99": 99}


def pct(values: list[float]) -> dict[str, float]:
    a = np.array(values, dtype=float)
    out = {k: round(float(np.percentile(a, p)), 3) for k, p in PERCENTILES.items()}
    out["p100"] = round(float(a.max()), 3)
    out["mean"] = round(float(a.mean()), 3)
    return out


def run(tag: str, ensemble: list[str], langs: list[str], n: int, warmup: int, top_k: int) -> dict:
    queries: list[dict] = []
    for lang in langs:
        queries.extend(read_jsonl(DATA_ROOT / "raw" / f"{lang}_train_queries.jsonl"))

    rng = np.random.default_rng(7)  # same seed as bench.fastpath -> same query sample
    picked = [queries[i] for i in rng.choice(len(queries), min(n + warmup, len(queries)), False)]

    embedder = Embedder(EmbedderConfig(threads=int(os.getenv("ORT_THREADS", "0"))))
    retriever = AdaptiveRetriever.load(DATA_ROOT / "index" / tag, ensemble)

    stages: dict[str, list[float]] = defaultdict(list)
    totals: list[float] = []
    rows: list[dict] = []

    for i, q in enumerate(picked):
        text = q["query"]
        qv = embedder.encode_query(text)
        hits = retriever.search(qv, text, k=top_k).hits
        ans = extract_answer(text, qv, hits, embedder)

        if i < warmup:
            continue
        for k, v in ans.stage_ms.items():
            stages[k].append(v)
        totals.append(ans.took_ms)
        rows.append(
            {
                "took_ms": ans.took_ms,
                "embed_ms": ans.stage_ms.get("embed_sentences", 0.0),
                "n_sentences": ans.n_sentences_embedded,
                "max_chars": ans.max_sentence_chars,
            }
        )

    # Tail attribution: what distinguishes the slowest decile from the median?
    rows.sort(key=lambda r: r["took_ms"])
    n_rows = len(rows)
    tail = rows[int(n_rows * 0.9) :]
    body = rows[: int(n_rows * 0.5)]

    def summarise(group: list[dict]) -> dict:
        return {
            "n": len(group),
            "took_ms_mean": round(float(np.mean([r["took_ms"] for r in group])), 2),
            "embed_ms_mean": round(float(np.mean([r["embed_ms"] for r in group])), 2),
            "n_sentences_mean": round(float(np.mean([r["n_sentences"] for r in group])), 2),
            "max_chars_mean": round(float(np.mean([r["max_chars"] for r in group])), 1),
            "max_chars_max": max(r["max_chars"] for r in group),
        }

    mc = np.array([r["max_chars"] for r in rows], dtype=float)
    ns = np.array([r["n_sentences"] for r in rows], dtype=float)
    tm = np.array([r["took_ms"] for r in rows], dtype=float)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return round(float(np.corrcoef(a, b)[0, 1]), 4)

    return {
        "tag": tag,
        "ensemble": ensemble,
        "langs": langs,
        "n_queries": len(totals),
        "machine": os.uname().machine,
        "embedder_variant": embedder.variant,
        "threads": os.getenv("ORT_THREADS", "auto"),
        "seed": 7,
        "extract_total_ms": pct(totals),
        "stages_ms": {k: pct(v) for k, v in stages.items()},
        "tail_vs_body": {"slowest_10pct": summarise(tail), "fastest_50pct": summarise(body)},
        "correlation_with_took_ms": {
            "max_sentence_chars": corr(mc, tm),
            "n_sentences_embedded": corr(ns, tm),
        },
    }


def render(r: dict) -> str:
    lines = [
        "",
        f"## `extract` component profile ({r['n_queries']} queries, {r['machine']}, "
        f"{r['embedder_variant']}, {r['threads']} threads)",
        "",
        "| Component | P50 | P70 | P90 | P95 | P99 | P100 | mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in r["stages_ms"].items():
        lines.append(
            f"| {name} | {s['p50']} | {s['p70']} | {s['p90']} | {s['p95']} | "
            f"{s['p99']} | {s['p100']} | {s['mean']} |"
        )
    t = r["extract_total_ms"]
    lines.append(
        f"| **extract total** | **{t['p50']}** | **{t['p70']}** | {t['p90']} | {t['p95']} | "
        f"**{t['p99']}** | **{t['p100']}** | {t['mean']} |"
    )

    tb = r["tail_vs_body"]
    lines += [
        "",
        "### What distinguishes the tail",
        "",
        "| Group | n | extract mean | embed mean | sentences | max chars (mean) | max chars (worst) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, g in (("slowest 10%", tb["slowest_10pct"]), ("fastest 50%", tb["fastest_50pct"])):
        lines.append(
            f"| {label} | {g['n']} | {g['took_ms_mean']}ms | {g['embed_ms_mean']}ms | "
            f"{g['n_sentences_mean']} | {g['max_chars_mean']} | {g['max_chars_max']} |"
        )

    c = r["correlation_with_took_ms"]
    lines += [
        "",
        f"Correlation with extract latency: **max_sentence_chars {c['max_sentence_chars']}**, "
        f"n_sentences_embedded {c['n_sentences_embedded']}.",
        "",
        "_A batch is padded to its longest member, so the longest sentence sets the cost_",
        "_for the whole batch. Sentence count is capped at 10 and barely varies._",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--ensemble", nargs="+", default=DEFAULT_ENSEMBLE)
    ap.add_argument("--langs", nargs="+", default=["hin", "mar"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--label", default="extract_profile")
    args = ap.parse_args()

    r = run(args.tag, args.ensemble, args.langs, args.n, args.warmup, args.top_k)
    md = render(r)
    print(md)

    out = DATA_ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}_{args.label}.json").write_text(json.dumps(r, indent=2))
    (out / f"{args.tag}_{args.label}.md").write_text(md)


if __name__ == "__main__":
    main()

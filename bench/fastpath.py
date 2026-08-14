"""End-to-end fast-path latency: question -> grounded extractive answer.

`bench/latency.py` measures a *single* ChunkIndex. That is not the served path.
`AdaptiveRetriever` queries three indexes, and at pilot scale the fan-out was free
(3 x ~1ms), so the difference never showed. At full scale BM25 costs ~30ms per
index because `bm25s` scores by sparse matmul -- O(corpus) -- while HNSW is
logarithmic, and a serial fan-out therefore sums into the budget:

    3 x ~60ms retrieval + ~40ms extract  ~=  220ms   -> over

This measures the real harness path (guardrail -> embed -> retrieve -> extract ->
guardrail) so the reported P50/P70/P100 belongs to the system that actually serves,
not to one component of it. Generation is excluded by construction: the task scopes
the budget as "chunking + vector DB retrieval + everything through to final output",
and the extractive answer is complete and grounded on its own.

    python -m bench.fastpath --tag full --n 300
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from core.embedder import Embedder, EmbedderConfig
from core.harness import RAGHarness
from core.retriever import DEFAULT_ENSEMBLE
from ingest.pipeline import DATA_ROOT, read_jsonl

PERCENTILES = {"p50": 50, "p70": 70, "p90": 90, "p99": 99}


class _NoLLM:
    """Generation is out of scope here; the harness must never call this."""

    def generate(self, *a, **kw):  # pragma: no cover - defensive
        raise AssertionError("fast-path benchmark must not invoke generation")


def pct(values: list[float]) -> dict[str, float]:
    a = np.array(values)
    out = {k: round(float(np.percentile(a, p)), 3) for k, p in PERCENTILES.items()}
    out["p100"] = round(float(a.max()), 3)
    out["mean"] = round(float(a.mean()), 3)
    return out


def run(tag: str, ensemble: list[str], langs: list[str], n: int, warmup: int, top_k: int) -> dict:
    queries: list[dict] = []
    for lang in langs:
        queries.extend(read_jsonl(DATA_ROOT / "raw" / f"{lang}_train_queries.jsonl"))

    rng = np.random.default_rng(7)
    picked = [queries[i] for i in rng.choice(len(queries), min(n + warmup, len(queries)), False)]

    threads = int(os.getenv("ORT_THREADS", "0"))
    embedder = Embedder(EmbedderConfig(threads=threads))
    index_root = DATA_ROOT / "index" / tag
    h = RAGHarness(index_root, ensemble, llm=_NoLLM(), embedder=embedder, top_k=top_k)

    sizes = {name: len(ix) for name, ix in h.retriever.indexes.items()}
    print(f"ensemble={list(sizes)}  chunks={sum(sizes.values()):,}  queries={n}  warmup={warmup}")

    stages: dict[str, list[float]] = defaultdict(list)
    fast: list[float] = []
    per_index: dict[str, list[float]] = defaultdict(list)
    decisions: dict[str, int] = defaultdict(int)
    cold: dict[str, float] = {}

    for i, q in enumerate(picked):
        text = q["query"]

        r = h.answer(text, generate=False)

        # Per-index fan-out detail, measured separately so it cannot perturb the
        # headline number above.
        qv = embedder.encode_query(text)
        res = h.retriever.search(qv, text, k=top_k)

        if i == 0:
            cold = {"fast_path_ms": r.fast_path_ms, **r.timings_ms}
        if i >= warmup:
            for k, v in r.timings_ms.items():
                stages[k].append(v)
            fast.append(r.fast_path_ms)
            for name, ms in res.per_index_ms.items():
                per_index[name].append(ms)
            decisions[r.answer_source] += 1

    slowest = {name: pct(v)["p50"] for name, v in per_index.items()}
    serial_p50 = round(sum(slowest.values()), 3)

    return {
        "tag": tag,
        "ensemble": list(sizes),
        "n_chunks": sizes,
        "n_queries": len(fast),
        "langs": langs,
        "machine": os.uname().machine,
        "embedder_variant": embedder.variant,
        "threads": os.getenv("ORT_THREADS", "auto"),
        "cold_first_query_ms": cold,
        "fast_path_ms": pct(fast),
        "stages_ms": {k: pct(v) for k, v in stages.items()},
        "per_index_ms": {k: pct(v) for k, v in per_index.items()},
        "fan_out_serial_p50_ms": serial_p50,
        "fan_out_slowest_p50_ms": round(max(slowest.values()), 3),
        "answer_source": dict(decisions),
        "over_budget": sum(1 for v in fast if v > 200),
        "budget_ms": 200,
    }


def render(r: dict) -> str:
    total = sum(r["n_chunks"].values())
    lines = [
        "",
        f"## Fast-path latency — ensemble ({total:,} chunks across "
        f"{len(r['ensemble'])} indexes, {r['n_queries']} queries, {r['machine']})",
        "",
        f"**{r['fast_path_ms']['p50']}ms P50 · {r['fast_path_ms']['p70']}ms P70 · "
        f"{r['fast_path_ms']['p100']}ms P100** — "
        f"{r['n_queries'] - r['over_budget']}/{r['n_queries']} under {r['budget_ms']}ms",
        "",
        "| Stage | P50 | P70 | P90 | P99 | P100 | mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in r["stages_ms"].items():
        lines.append(
            f"| {name} | {s['p50']} | {s['p70']} | {s['p90']} | "
            f"{s['p99']} | {s['p100']} | {s['mean']} |"
        )
    f = r["fast_path_ms"]
    lines.append(
        f"| **fast_path_total** | **{f['p50']}** | **{f['p70']}** | {f['p90']} | "
        f"{f['p99']} | **{f['p100']}** | {f['mean']} |"
    )

    lines += [
        "",
        "Per-index fan-out (the ensemble cost):",
        "",
        "| Index | chunks | P50 | P100 |",
        "|---|---:|---:|---:|",
    ]
    for name, s in r["per_index_ms"].items():
        lines.append(f"| {name} | {r['n_chunks'][name]:,} | {s['p50']} | {s['p100']} |")
    lines += [
        "",
        f"Serial fan-out sums to {r['fan_out_serial_p50_ms']}ms P50; the slowest single "
        f"index is {r['fan_out_slowest_p50_ms']}ms — that gap is what parallelising the "
        "fan-out would recover.",
        "",
        f"Answer source: {r['answer_source']}",
        "",
        f"Cold first query: {r['cold_first_query_ms'].get('fast_path_ms', '?')}ms "
        "(excluded — it measures warmup, not steady state)",
        "",
        "_All times in ms. Speech-to-text and generation are excluded: the task scopes the_",
        '_budget as "chunking + vector DB retrieval + everything through to final output"._',
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
    ap.add_argument("--label", default="fastpath")
    args = ap.parse_args()

    r = run(args.tag, args.ensemble, args.langs, args.n, args.warmup, args.top_k)
    md = render(r)
    print(md)

    out = DATA_ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}_{args.label}_latency.json").write_text(json.dumps(r, indent=2))
    (out / f"{args.tag}_{args.label}_latency.md").write_text(md)


if __name__ == "__main__":
    main()

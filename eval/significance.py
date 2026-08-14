"""Paired significance testing and a leak test for the metadata hint.

Two questions the ablation table cannot answer on its own.

**1. Are the differences real?** The configurations sit within ~0.005 of each
other on recall. With 1,500 queries an unpaired standard error is roughly 0.010,
so the entire ranking could be noise. Because every configuration answers the
*same* queries, the correct test is paired: bootstrap the per-query differences
and report a confidence interval on the delta, not on each mean. A recommendation
that rests on a difference whose CI straddles zero is a coin flip with extra
steps.

**2. Is `metadata_128` winning for a reason that will survive deployment?**
`chunk_metadata` embeds `[{query_type}] {hint} | {body}`, and `query_type` is
taken from the *query that owns the passage* -- so every gold passage of a
NUMERIC query is tagged NUMERIC, while the rest of the corpus mostly is not. That
is a label a real corpus does not carry.

If the hint works the way its docstring claims (the encoder maps "how many..."
near numeric-flavoured text), the benefit should be roughly uniform across query
types. If it is the leak, the benefit must scale with how much of the corpus the
tag excludes -- large for rare types, ~zero for DESCRIPTION at 53% of the corpus.
Those predictions differ sharply, so the per-type breakdown decides it.

    python -m eval.significance --tag full --n-queries 1500
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

from core.embedder import Embedder, EmbedderConfig
from core.index import ChunkIndex
from core.retriever import AdaptiveRetriever
from eval.evaluate import _norm, rank_units, score_one
from ingest.pipeline import DATA_ROOT, read_jsonl

ALL_INDEXES = ["fixed_256", "semantic_128", "metadata_128"]
SIG_LEN = 120
N_BOOT = 10_000

CONFIGS: dict[str, list[str]] = {
    "metadata_128": ["metadata_128"],
    "fixed_256": ["fixed_256"],
    "fixed+metadata": ["fixed_256", "metadata_128"],
    "ENSEMBLE": ALL_INDEXES,
}
BASELINE = "metadata_128"


def paired_bootstrap(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    """CI on mean(a - b) by resampling queries, preserving the pairing."""
    d = a - b
    n = len(d)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "delta": round(float(d.mean()), 5),
        "ci95": [round(float(lo), 5), round(float(hi), 5)],
        "significant": bool(lo > 0 or hi < 0),
    }


def run(tag: str, langs: list[str], split: str, n_queries: int, top_k: int) -> dict:
    index_root = DATA_ROOT / "index" / tag

    queries: list[dict] = []
    signatures: dict[str, str] = {}
    for lang in langs:
        queries.extend(read_jsonl(DATA_ROOT / "raw" / f"{lang}_{split}_queries.jsonl"))
        for p in read_jsonl(DATA_ROOT / "raw" / f"{lang}_{split}_passages.jsonl"):
            signatures[p["passage_id"]] = _norm(p["text_translated"])[:SIG_LEN]

    rng = np.random.default_rng(7)
    picked = [queries[i] for i in rng.choice(len(queries), min(n_queries, len(queries)), False)]
    picked = [q for q in picked if q.get("gold_passage_ids")]

    embedder = Embedder(EmbedderConfig(threads=int(os.getenv("ORT_THREADS", "0"))))
    indexes = {n: ChunkIndex.load(index_root, n) for n in ALL_INDEXES}

    # Corpus composition by query_type -- the denominator for the leak test.
    type_counts = Counter(indexes["metadata_128"].query_types)
    total_chunks = sum(type_counts.values())
    corpus_share = {t: c / total_chunks for t, c in type_counts.items()}

    # Per-query metric vectors, per configuration.
    per_query: dict[str, dict[str, list[float]]] = {}
    meta: list[dict] = []

    for name, names in CONFIGS.items():
        retriever = AdaptiveRetriever({n: indexes[n] for n in names})
        store: dict[str, list[float]] = defaultdict(list)
        for qi, q in enumerate(picked):
            gold = set(q["gold_passage_ids"])
            qv = embedder.encode_query(q["query"])
            hits = retriever.search(qv, q["query"], k=top_k).hits
            m = score_one(rank_units(hits, gold, signatures, top_k), gold)
            for k, v in m.items():
                store[k].append(v)
            if name == BASELINE:
                meta.append({"lang": q["lang"], "query_type": q["query_type"]})
            if qi % 500 == 0:
                print(f"  {name}: {qi}/{len(picked)}")
        per_query[name] = {k: np.array(v, dtype=float) for k, v in store.items()}
        print(f"  {name}: done")

    boot = np.random.default_rng(11)
    comparisons = {}
    for name in CONFIGS:
        if name == BASELINE:
            continue
        comparisons[f"{name} - {BASELINE}"] = {
            metric: paired_bootstrap(
                per_query[name][metric], per_query[BASELINE][metric], boot
            )
            for metric in ("mrr@10", "recall@10", "recall@20")
        }

    # Leak test: per-type advantage of metadata_128 over fixed_256, against the
    # share of the corpus carrying that type.
    types = np.array([m["query_type"] for m in meta])
    langs_arr = np.array([m["lang"] for m in meta])
    leak = []
    for t in sorted(set(types)):
        mask = types == t
        if mask.sum() < 20:
            continue
        d = per_query["metadata_128"]["mrr@10"][mask] - per_query["fixed_256"]["mrr@10"][mask]
        leak.append(
            {
                "query_type": t,
                "n_queries": int(mask.sum()),
                "corpus_share": round(corpus_share.get(t, 0.0), 4),
                "metadata_mrr": round(float(per_query["metadata_128"]["mrr@10"][mask].mean()), 4),
                "fixed_mrr": round(float(per_query["fixed_256"]["mrr@10"][mask].mean()), 4),
                "advantage": round(float(d.mean()), 4),
                **paired_bootstrap(
                    per_query["metadata_128"]["mrr@10"][mask],
                    per_query["fixed_256"]["mrr@10"][mask],
                    boot,
                ),
            }
        )

    shares = np.array([r["corpus_share"] for r in leak])
    advs = np.array([r["advantage"] for r in leak])
    leak_corr = (
        round(float(np.corrcoef(shares, advs)[0, 1]), 4)
        if len(leak) > 2 and shares.std() > 0
        else None
    )

    # Per-language paired deltas.
    by_lang = {}
    for lg in sorted(set(langs_arr)):
        mask = langs_arr == lg
        by_lang[lg] = {
            "n_queries": int(mask.sum()),
            **{
                f"{name} - {BASELINE}": {
                    metric: paired_bootstrap(
                        per_query[name][metric][mask], per_query[BASELINE][metric][mask], boot
                    )
                    for metric in ("mrr@10", "recall@10")
                }
                for name in CONFIGS
                if name != BASELINE
            },
        }

    return {
        "tag": tag,
        "langs": langs,
        "n_queries": len(picked),
        "top_k": top_k,
        "machine": os.uname().machine,
        "seed": 7,
        "n_bootstrap": N_BOOT,
        "baseline": BASELINE,
        "means": {
            name: {k: round(float(v.mean()), 4) for k, v in m.items()}
            for name, m in per_query.items()
        },
        "paired_vs_baseline": comparisons,
        "leak_test": {"per_type": leak, "corr_share_vs_advantage": leak_corr},
        "per_language": by_lang,
    }


def render(r: dict) -> str:
    lines = [
        "",
        f"## Paired significance ({r['n_queries']} queries, {r['n_bootstrap']:,} bootstrap "
        f"resamples, baseline `{r['baseline']}`)",
        "",
        "| Comparison | Metric | Δ | 95% CI | significant |",
        "|---|---|---:|---|---|",
    ]
    for comp, metrics in r["paired_vs_baseline"].items():
        for metric, s in metrics.items():
            ci = f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"
            lines.append(
                f"| {comp} | {metric} | {s['delta']:+.4f} | {ci} | "
                f"{'**yes**' if s['significant'] else 'no'} |"
            )

    lines += [
        "",
        "### Leak test — does the metadata hint help most where it excludes most corpus?",
        "",
        "| query_type | queries | corpus share | metadata MRR | fixed MRR | advantage | 95% CI | sig |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for t in sorted(r["leak_test"]["per_type"], key=lambda x: -x["corpus_share"]):
        ci = f"[{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}]"
        lines.append(
            f"| {t['query_type']} | {t['n_queries']} | {t['corpus_share']:.1%} | "
            f"{t['metadata_mrr']:.4f} | {t['fixed_mrr']:.4f} | {t['advantage']:+.4f} | {ci} | "
            f"{'**yes**' if t['significant'] else 'no'} |"
        )
    lines += [
        "",
        f"Correlation between corpus share and metadata advantage: "
        f"**{r['leak_test']['corr_share_vs_advantage']}**.",
        "",
        "_A strong negative correlation is the leak signature: the tag helps precisely_",
        "_when it rules out most of the corpus, and stops helping for the majority type._",
        "",
        "### Per language",
        "",
        "| Lang | Comparison | Metric | Δ | 95% CI | significant |",
        "|---|---|---|---:|---|---|",
    ]
    for lg, block in r["per_language"].items():
        for comp, metrics in block.items():
            if comp == "n_queries":
                continue
            for metric, s in metrics.items():
                ci = f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"
                lines.append(
                    f"| {lg} | {comp} | {metric} | {s['delta']:+.4f} | {ci} | "
                    f"{'**yes**' if s['significant'] else 'no'} |"
                )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--langs", nargs="+", default=["hin", "mar"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-queries", type=int, default=1500)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    r = run(args.tag, args.langs, args.split, args.n_queries, args.top_k)
    md = render(r)
    print(md)

    out = DATA_ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}_significance.json").write_text(json.dumps(r, indent=2))
    (out / f"{args.tag}_significance.md").write_text(md)


if __name__ == "__main__":
    main()

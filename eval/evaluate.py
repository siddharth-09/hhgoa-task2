"""Score every chunking strategy on held-out queries, sliced by query_type.

This is the evidence for requirement #2. `is_selected` in MSMARCO-XI gives
ground-truth relevance labels for free, so the comparison is measured rather
than argued -- and the per-query_type slice is what justifies the router:
if one strategy won everywhere, routing would be pointless.

A retrieved chunk counts as a hit when its parent passage_id is in the query's
gold set. Chunks are compared at passage granularity because that is the unit
the labels are defined over, and because strategies emit different chunk counts
per passage -- scoring at chunk granularity would reward whichever strategy
fragments the most.

    python -m eval.evaluate --tag pilot --n-queries 2000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from core.embedder import Embedder, EmbedderConfig
from core.index import ChunkIndex
from core.retriever import DEFAULT_ENSEMBLE, AdaptiveRetriever
from ingest.pipeline import DATA_ROOT, read_jsonl

K_VALUES = (1, 5, 10, 20)


SIG_LEN = 80


def _norm(s: str) -> str:
    return " ".join(s.split())


def covered_gold(hit, gold: set[str], signatures: dict[str, str]) -> set[str]:
    """Which gold passages does this retrieved chunk actually contain?

    Passage-granularity chunks map to exactly one parent passage, so the parent
    id decides it. Document-granularity chunks concatenate ~10 passages, so
    matching on the parent id would count a hit whenever the document was
    retrieved at all -- trivially true, and it would inflate doc variants past
    any fair comparison.

    Instead a doc chunk counts only for the gold passages whose text it actually
    contains, checked against a normalised prefix signature. A chunk holding
    half the answer is correctly not credited with it.
    """
    # Ids are language-namespaced ("hin_Deva:1185869:0", "hin_Deva:doc1185869"),
    # so the granularity marker is the last segment, not the start of the string.
    if not hit.passage_id.rpartition(":")[2].startswith("doc"):
        return {hit.passage_id} & gold
    text = _norm(hit.text)
    return {g for g in gold if (sig := signatures.get(g)) and sig in text}


def rank_units(hits, gold: set[str], signatures: dict[str, str], limit: int) -> list[set[str]]:
    """Rank-ordered, deduplicated retrieval units -> the gold each one covers."""
    seen: set[str] = set()
    out: list[set[str]] = []
    for h in hits:
        if h.passage_id in seen:
            continue
        seen.add(h.passage_id)
        out.append(covered_gold(h, gold, signatures))
        if len(out) >= limit:
            break
    return out


def score_one(ranked: list[set[str]], gold: set[str]) -> dict[str, float]:
    """Metrics over gold *coverage*, so passage- and doc-granularity compare fairly."""
    out: dict[str, float] = {}
    for k in K_VALUES:
        covered: set[str] = set()
        for c in ranked[:k]:
            covered |= c
        out[f"recall@{k}"] = len(covered) / len(gold) if gold else 0.0

    rr = 0.0
    for i, c in enumerate(ranked[:10]):
        if c:
            rr = 1.0 / (i + 1)
            break
    out["mrr@10"] = rr

    dcg = sum(1.0 / math.log2(i + 2) for i, c in enumerate(ranked[:10]) if c)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out


def evaluate(
    tag: str,
    langs: list[str],
    split: str,
    n_queries: int,
    top_k: int,
    ensemble_names: list[str] | None = None,
) -> dict:
    queries: list[dict] = []
    signatures: dict[str, str] = {}
    for lang in langs:
        queries.extend(read_jsonl(DATA_ROOT / "raw" / f"{lang}_{split}_queries.jsonl"))
        for p in read_jsonl(DATA_ROOT / "raw" / f"{lang}_{split}_passages.jsonl"):
            signatures[p["passage_id"]] = _norm(p["text_translated"])[:SIG_LEN]

    rng = np.random.default_rng(42)
    if len(queries) > n_queries:
        queries = [queries[i] for i in rng.choice(len(queries), n_queries, replace=False)]
    print(f"evaluating on {len(queries):,} queries")

    embedder = Embedder(EmbedderConfig(threads=int(os.getenv("ORT_THREADS", "0"))))
    qvecs = embedder.encode_queries([q["query"] for q in queries])

    results: dict[str, dict] = {}
    index_root = DATA_ROOT / "index" / tag
    # Evaluate whatever indexes exist, so the variant grid can grow without
    # this file needing to know the names.
    built = sorted(d.name for d in index_root.iterdir() if (d / "hnsw.bin").exists())
    print(f"indexes: {', '.join(built)}")

    # The ensemble is scored as one more row, so it is directly comparable with
    # every single-index variant on identical queries.
    ensemble = AdaptiveRetriever.load(index_root, ensemble_names) if ensemble_names else None
    runners: list[tuple[str, object]] = [(n, ChunkIndex.load(index_root, n)) for n in built]
    if ensemble and ensemble.indexes:
        runners.append((f"ENSEMBLE[{'+'.join(ensemble.indexes)}]", ensemble))

    for strat, ix in runners:
        overall: dict[str, list[float]] = defaultdict(list)
        by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        latencies: list[float] = []

        for q, qv in zip(queries, qvecs, strict=True):
            gold = set(q["gold_passage_ids"])
            if not gold:
                continue
            t0 = time.perf_counter()
            if isinstance(ix, AdaptiveRetriever):
                hits = ix.search(qv, q["query"], k=top_k).hits
            else:
                hits = ix.search(qv, q["query"], k=top_k)
            latencies.append((time.perf_counter() - t0) * 1000)

            m = score_one(rank_units(hits, gold, signatures, max(K_VALUES)), gold)
            for name, val in m.items():
                overall[name].append(val)
                by_type[q["query_type"]][name].append(val)

        lat = np.array(latencies)
        results[strat] = {
            "n_chunks": len(ix),
            "n_queries": len(latencies),
            "overall": {k: round(float(np.mean(v)), 4) for k, v in overall.items()},
            "by_query_type": {
                qt: {k: round(float(np.mean(v)), 4) for k, v in metrics.items()}
                for qt, metrics in sorted(by_type.items())
            },
            "search_ms": {
                "p50": round(float(np.percentile(lat, 50)), 2),
                "p70": round(float(np.percentile(lat, 70)), 2),
                "p100": round(float(lat.max()), 2),
            },
        }
        o = results[strat]["overall"]
        print(
            f"  {strat:<10} chunks={len(ix):>8,}  "
            f"mrr@10={o['mrr@10']:.4f}  ndcg@10={o['ndcg@10']:.4f}  "
            f"r@10={o['recall@10']:.4f}  search_p50={results[strat]['search_ms']['p50']}ms"
        )

    return results


def render_markdown(results: dict) -> str:
    """The table that goes in the README."""
    lines = ["## Chunking strategy comparison", ""]
    lines.append("| Strategy | Chunks | MRR@10 | nDCG@10 | R@1 | R@5 | R@10 | R@20 | search p50 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for strat, r in results.items():
        o = r["overall"]
        lines.append(
            f"| {strat} | {r['n_chunks']:,} | {o['mrr@10']:.4f} | {o['ndcg@10']:.4f} | "
            f"{o['recall@1']:.4f} | {o['recall@5']:.4f} | {o['recall@10']:.4f} | "
            f"{o['recall@20']:.4f} | {r['search_ms']['p50']}ms |"
        )

    lines += ["", "## MRR@10 by query type — the routing signal", ""]
    types = sorted({t for r in results.values() for t in r["by_query_type"]})
    lines.append("| Strategy | " + " | ".join(types) + " |")
    lines.append("|---" * (len(types) + 1) + "|")
    for strat, r in results.items():
        row = [f"{r['by_query_type'].get(t, {}).get('mrr@10', 0):.4f}" for t in types]
        lines.append(f"| {strat} | " + " | ".join(row) + " |")

    lines += ["", "**Best strategy per query type:**", ""]
    for t in types:
        best = max(results.items(), key=lambda kv: kv[1]["by_query_type"].get(t, {}).get("mrr@10", 0))
        score = best[1]["by_query_type"].get(t, {}).get("mrr@10", 0)
        lines.append(f"- `{t}` → **{best[0]}** (MRR@10 {score:.4f})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--langs", nargs="+", default=["hin", "mar"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-queries", type=int, default=2000)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--ensemble", nargs="+", default=None, help="indexes to fuse")
    ap.add_argument("--no-ensemble", action="store_true")
    args = ap.parse_args()

    results = evaluate(
        args.tag, args.langs, args.split, args.n_queries, args.top_k,
        None if args.no_ensemble else (args.ensemble or DEFAULT_ENSEMBLE),
    )

    out_dir = DATA_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.tag}_eval.json").write_text(json.dumps(results, indent=2))
    md = render_markdown(results)
    (out_dir / f"{args.tag}_eval.md").write_text(md)
    print(f"\n{md}")


if __name__ == "__main__":
    main()

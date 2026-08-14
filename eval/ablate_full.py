"""Full index ablation: every subset, scored on one evaluation set.

The serving ensemble was chosen on 1,200 Hindi-only pilot queries over ~75k
chunks. At full scale (682k chunks, bilingual) the three-index ensemble no longer
dominates, so the choice has to be re-made here rather than inherited.

The repository has three passage-granularity indexes, so the subset lattice is
2^3-1 = 7 configurations -- small enough to enumerate exhaustively instead of
guessing which pairs are worth trying.

Reported per configuration: retrieval quality, search latency percentiles,
end-to-end fast-path latency, and resident footprint. A configuration that wins
on MRR while costing 2.8x the memory is not obviously the right ship, and the
table is built so that trade is visible rather than argued.

Every configuration sees the same queries, the same seed and the same process.
Indexes are loaded once and shared, so a subset costs no extra memory to score.

    python -m eval.ablate_full --tag full --n-queries 1500
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from core.embedder import Embedder, EmbedderConfig
from core.extractive import extract_answer
from core.index import ChunkIndex
from core.retriever import AdaptiveRetriever
from eval.evaluate import _norm, rank_units, score_one
from ingest.pipeline import DATA_ROOT, read_jsonl

ALL_INDEXES = ["fixed_256", "semantic_128", "metadata_128"]
SIG_LEN = 120
PERCENTILES = {"p50": 50, "p95": 95, "p99": 99}


def pct(values: list[float]) -> dict[str, float]:
    a = np.array(values, dtype=float)
    out = {k: round(float(np.percentile(a, p)), 3) for k, p in PERCENTILES.items()}
    out["p100"] = round(float(a.max()), 3)
    return out


def footprint(root: Path, names: list[str]) -> dict[str, float]:
    """On-disk bytes per configuration -- the closest honest proxy for resident set."""
    total = 0
    for n in names:
        for f in (root / n).rglob("*"):
            if f.is_file() and f.suffix != ".bak":
                total += f.stat().st_size
    return {"disk_mb": round(total / 1e6, 1)}


def evaluate_subset(
    names: list[str],
    indexes: dict[str, ChunkIndex],
    embedder: Embedder,
    queries: list[dict],
    signatures: dict[str, str],
    top_k: int,
    index_root: Path,
) -> dict:
    retriever = AdaptiveRetriever({n: indexes[n] for n in names})

    per_metric: dict[str, list[float]] = {}
    by_lang: dict[str, dict[str, list[float]]] = {}
    search_ms: list[float] = []
    e2e_ms: list[float] = []

    for q in queries:
        gold = set(q["gold_passage_ids"])
        if not gold:
            continue
        lang = q["lang"]

        t0 = time.perf_counter()
        qv = embedder.encode_query(q["query"])
        res = retriever.search(qv, q["query"], k=top_k)
        t1 = time.perf_counter()
        search_ms.append((t1 - t0) * 1000)

        ranked = rank_units(res.hits, gold, signatures, top_k)
        m = score_one(ranked, gold)
        for k, v in m.items():
            per_metric.setdefault(k, []).append(v)
            by_lang.setdefault(lang, {}).setdefault(k, []).append(v)

        # End-to-end fast path on a subsample -- extraction is config-independent
        # in cost but the total is what the budget is measured against.
        if len(e2e_ms) < 300:
            extract_answer(q["query"], qv, res.hits, embedder)
            e2e_ms.append((time.perf_counter() - t0) * 1000)

    return {
        "indexes": names,
        "n_chunks": sum(len(indexes[n]) for n in names),
        "n_scored": len(next(iter(per_metric.values()))),
        "metrics": {k: round(float(np.mean(v)), 4) for k, v in per_metric.items()},
        "by_lang": {
            lg: {k: round(float(np.mean(v)), 4) for k, v in mm.items()}
            for lg, mm in sorted(by_lang.items())
        },
        "search_ms": pct(search_ms),
        "e2e_fastpath_ms": pct(e2e_ms),
        **footprint(index_root, names),
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

    embedder = Embedder(EmbedderConfig(threads=int(os.getenv("ORT_THREADS", "0"))))
    print(f"loading {len(ALL_INDEXES)} indexes once, shared across all subsets...")
    indexes = {n: ChunkIndex.load(index_root, n) for n in ALL_INDEXES}

    subsets: list[list[str]] = []
    for r in range(1, len(ALL_INDEXES) + 1):
        subsets.extend([list(c) for c in combinations(ALL_INDEXES, r)])

    results = []
    for names in subsets:
        r = evaluate_subset(
            names, indexes, embedder, picked, signatures, top_k, index_root
        )
        results.append(r)
        print(
            f"  {'+'.join(names):<48} chunks={r['n_chunks']:>7,} "
            f"mrr={r['metrics'].get('mrr@10', 0):.4f} r@10={r['metrics'].get('recall@10', 0):.4f} "
            f"r@20={r['metrics'].get('recall@20', 0):.4f} search_p50={r['search_ms']['p50']:.2f}ms "
            f"disk={r['disk_mb']:.0f}MB"
        )

    return {
        "tag": tag,
        "langs": langs,
        "n_queries": len(picked),
        "top_k": top_k,
        "machine": os.uname().machine,
        "embedder_variant": embedder.variant,
        "threads": os.getenv("ORT_THREADS", "auto"),
        "seed": 7,
        "configs": results,
    }


def render(r: dict) -> str:
    lines = [
        "",
        f"## Index ablation — all {len(r['configs'])} subsets "
        f"({r['n_queries']} queries, {'+'.join(r['langs'])}, {r['machine']})",
        "",
        "| Configuration | Chunks | MRR@10 | R@10 | R@20 | search P50 | search P95 | "
        "search P99 | e2e P50 | e2e P99 | disk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in r["configs"]:
        m, s, e = c["metrics"], c["search_ms"], c["e2e_fastpath_ms"]
        lines.append(
            f"| {'+'.join(c['indexes'])} | {c['n_chunks']:,} | {m.get('mrr@10', 0):.4f} | "
            f"{m.get('recall@10', 0):.4f} | {m.get('recall@20', 0):.4f} | {s['p50']} | {s['p95']} | "
            f"{s['p99']} | {e['p50']} | {e['p99']} | {c['disk_mb']:.0f}MB |"
        )

    lines += ["", "### Per-language", "", "| Configuration | Lang | MRR@10 | R@10 | R@20 |", "|---|---|---:|---:|---:|"]
    for c in r["configs"]:
        for lg, m in c["by_lang"].items():
            lines.append(
                f"| {'+'.join(c['indexes'])} | {lg} | {m.get('mrr@10', 0):.4f} | "
                f"{m.get('recall@10', 0):.4f} | {m.get('recall@20', 0):.4f} |"
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
    (out / f"{args.tag}_ablation.json").write_text(json.dumps(r, indent=2))
    (out / f"{args.tag}_ablation.md").write_text(md)


if __name__ == "__main__":
    main()

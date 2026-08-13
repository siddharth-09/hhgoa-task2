"""Ensemble ablation: does each index in the fusion actually earn its place?

The full eval showed the 4-index ensemble beating every single variant, but a
spot check found `doc_fixed_256` contributing zero hits to any top-10. Those two
facts are compatible -- it could be helping only in the tail (ranks 10-50, which
recall@20 sees) -- but "it helps somewhere" is not a reason to keep an index
resident in RAM on the serving box.

So: score each combination on identical queries and let the numbers decide.

    python -m eval.ablate --tag pilot --n-queries 1500
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from core.embedder import Embedder, EmbedderConfig
from core.retriever import AdaptiveRetriever
from eval.evaluate import K_VALUES, SIG_LEN, _norm, rank_units, score_one
from ingest.pipeline import DATA_ROOT, read_jsonl

COMBOS: dict[str, list[str]] = {
    "single: fixed_256": ["fixed_256"],
    "single: metadata_128": ["metadata_128"],
    "single: doc_fixed_256": ["doc_fixed_256"],
    "P2: fixed+semantic": ["fixed_256", "semantic_128"],
    "P3: fixed+semantic+metadata": ["fixed_256", "semantic_128", "metadata_128"],
    "P3+doc (default)": ["fixed_256", "semantic_128", "metadata_128", "doc_fixed_256"],
    "P3+doc_sentence": ["fixed_256", "semantic_128", "metadata_128", "doc_sentence_256"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--langs", nargs="+", default=["hin"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-queries", type=int, default=1500)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    queries: list[dict] = []
    signatures: dict[str, str] = {}
    for lang in args.langs:
        queries.extend(read_jsonl(DATA_ROOT / "raw" / f"{lang}_{args.split}_queries.jsonl"))
        for p in read_jsonl(DATA_ROOT / "raw" / f"{lang}_{args.split}_passages.jsonl"):
            signatures[p["passage_id"]] = _norm(p["text_translated"])[:SIG_LEN]

    rng = np.random.default_rng(42)
    if len(queries) > args.n_queries:
        queries = [queries[i] for i in rng.choice(len(queries), args.n_queries, False)]
    queries = [q for q in queries if q["gold_passage_ids"]]
    print(f"{len(queries):,} labelled queries\n")

    emb = Embedder(EmbedderConfig(threads=int(os.getenv("ORT_THREADS", "0"))))
    qvecs = emb.encode_queries([q["query"] for q in queries])
    index_root = DATA_ROOT / "index" / args.tag

    rows: list[dict] = []
    for label, names in COMBOS.items():
        r = AdaptiveRetriever.load(index_root, names)
        if len(r.indexes) != len(names):
            print(f"  skip {label} (missing index)")
            continue

        acc: dict[str, list[float]] = {}
        lat: list[float] = []
        # How often each member index contributes to the returned top-k.
        contrib: dict[str, int] = dict.fromkeys(names, 0)

        for q, qv in zip(queries, qvecs, strict=True):
            res = r.search(qv, q["query"], k=args.top_k)
            lat.append(res.total_ms)
            for name, n in res.provenance().items():
                contrib[name] = contrib.get(name, 0) + n

            m = score_one(
                rank_units(res.hits, set(q["gold_passage_ids"]), signatures, max(K_VALUES)),
                set(q["gold_passage_ids"]),
            )
            for k, v in m.items():
                acc.setdefault(k, []).append(v)

        total_contrib = sum(contrib.values()) or 1
        row = {
            "label": label,
            "indexes": names,
            "chunks": len(r),
            **{k: round(float(np.mean(v)), 4) for k, v in acc.items()},
            "p50_ms": round(float(np.percentile(lat, 50)), 2),
            "share": {n: round(100 * c / total_contrib, 1) for n, c in contrib.items()},
        }
        rows.append(row)
        print(
            f"  {label:<30} mrr={row['mrr@10']:.4f}  ndcg={row['ndcg@10']:.4f}  "
            f"r@10={row['recall@10']:.4f}  r@20={row['recall@20']:.4f}  {row['p50_ms']}ms"
        )

    print("\n| Ensemble | Chunks | MRR@10 | nDCG@10 | R@10 | R@20 | p50 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['label']} | {r['chunks']:,} | {r['mrr@10']:.4f} | {r['ndcg@10']:.4f} | "
            f"{r['recall@10']:.4f} | {r['recall@20']:.4f} | {r['p50_ms']}ms |"
        )

    print("\nShare of returned hits contributed by each index:")
    for r in rows:
        if len(r["indexes"]) > 1:
            print(f"  {r['label']:<30} {r['share']}")

    out = DATA_ROOT / "reports" / f"{args.tag}_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

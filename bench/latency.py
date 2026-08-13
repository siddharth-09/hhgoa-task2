"""Per-stage latency benchmark -> P50 / P70 / P100.

Requirement #4 asks for percentiles "across a reasonable number of test queries
-- not a single best-case run", so this deliberately:

  * runs every query one at a time, with no batching. Batch throughput is a
    different and much flatter number; serving latency is what matters.
  * warms up first (ONNX session, HNSW graph, page cache) and reports the cold
    queries separately, because P100 on a cold index measures the warmup, not
    the system.
  * times each stage independently, so the budget is attributable rather than
    a single opaque total.

The task's wording -- "chunking + vector DB retrieval + everything through to
final output" -- starts at chunking, so speech-to-text sits outside the budget
and is reported separately.

    python -m bench.latency --tag pilot --variant fixed_256 --n 300
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from core.embedder import Embedder, EmbedderConfig
from core.index import ChunkIndex
from ingest.pipeline import DATA_ROOT, read_jsonl

PERCENTILES = {"p50": 50, "p70": 70, "p90": 90, "p99": 99}


def pct(values: list[float]) -> dict[str, float]:
    a = np.array(values)
    out = {k: round(float(np.percentile(a, p)), 3) for k, p in PERCENTILES.items()}
    out["p100"] = round(float(a.max()), 3)
    out["mean"] = round(float(a.mean()), 3)
    return out


def run(tag: str, variant: str, langs: list[str], n: int, warmup: int, top_k: int) -> dict:
    queries: list[dict] = []
    for lang in langs:
        queries.extend(read_jsonl(DATA_ROOT / "raw" / f"{lang}_train_queries.jsonl"))

    rng = np.random.default_rng(7)
    picked = [queries[i] for i in rng.choice(len(queries), min(n + warmup, len(queries)), False)]

    embedder = Embedder(EmbedderConfig(threads=int(os.getenv("ORT_THREADS", "0"))))
    ix = ChunkIndex.load(DATA_ROOT / "index" / tag, variant)
    print(f"variant={variant}  chunks={len(ix):,}  queries={n}  warmup={warmup}")

    stages: dict[str, list[float]] = defaultdict(list)
    cold: dict[str, float] = {}

    for i, q in enumerate(picked):
        text = q["query"]

        t0 = time.perf_counter()
        qv = embedder.encode_query(text)
        t1 = time.perf_counter()
        dense = ix.search_dense(qv, top_k)
        t2 = time.perf_counter()
        sparse = ix.search_sparse(text, top_k)
        t3 = time.perf_counter()
        ix.search(qv, text, k=top_k)  # full fused path
        t4 = time.perf_counter()

        row = {
            "embed_query": (t1 - t0) * 1000,
            "search_dense": (t2 - t1) * 1000,
            "search_sparse": (t3 - t2) * 1000,
            "search_fused_total": (t4 - t3) * 1000,
            "retrieval_total": (t4 - t0) * 1000 - (t4 - t3),  # embed + dense + sparse
        }
        if i == 0:
            cold = {k: round(v, 3) for k, v in row.items()}
        if i >= warmup:
            for k, v in row.items():
                stages[k].append(v)

        assert dense and sparse is not None

    return {
        "tag": tag,
        "variant": variant,
        "n_chunks": len(ix),
        "n_queries": len(stages["embed_query"]),
        "machine": os.uname().machine,
        "embedder_variant": embedder.variant,
        "threads": os.getenv("ORT_THREADS", "auto"),
        "cold_first_query_ms": cold,
        "stages_ms": {k: pct(v) for k, v in stages.items()},
    }


def render(r: dict) -> str:
    lines = [
        "",
        f"## Latency — {r['variant']} ({r['n_chunks']:,} chunks, "
        f"{r['n_queries']} queries, {r['machine']})",
        "",
        "| Stage | P50 | P70 | P90 | P99 | P100 | mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in r["stages_ms"].items():
        lines.append(
            f"| {name} | {s['p50']} | {s['p70']} | {s['p90']} | "
            f"{s['p99']} | {s['p100']} | {s['mean']} |"
        )
    lines += [
        "",
        f"Cold first query: {r['cold_first_query_ms'].get('retrieval_total', '?')}ms "
        "(excluded from percentiles — it measures warmup, not steady state)",
        "",
        "_All times in ms. Speech-to-text is excluded: the task scopes the budget as_",
        "_\"chunking + vector DB retrieval + everything through to final output\"._",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--variant", default="fixed_256")
    ap.add_argument("--langs", nargs="+", default=["hin"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    r = run(args.tag, args.variant, args.langs, args.n, args.warmup, args.top_k)
    md = render(r)
    print(md)

    out = DATA_ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}_{args.variant}_latency.json").write_text(json.dumps(r, indent=2))
    (out / f"{args.tag}_{args.variant}_latency.md").write_text(md)


if __name__ == "__main__":
    main()

"""Sweep the two `extract` latency levers against answer fidelity.

Retrieval is fixed and cached across configs, so this isolates extraction: every
config sees byte-identical hits for the same query, and differences are the
lever's alone.

Two levers, and they are not equivalent:

  embed_batch          how many sentences share a padded forward pass.
  max_sentence_chars   truncates the text handed to the encoder. Genuinely
                       changes the embedding, so it must be paid for in fidelity.

**The reference is `embed_batch=1`, not the largest batch.** Masked mean pooling
would make batching irrelevant in fp32 -- measured drift there is 4.7e-08. But
the served model is *dynamically* quantised int8: activation scales are computed
at runtime from a tensor that spans the batch, so one long padded member widens
the range and coarsens the scale for every short sentence beside it. Measured on
the same text: batched with a long sentence shifts the vector by 1.0e-02 (cos
0.9981), while batch_size=1 reproduces the standalone vector bitwise.

So a large batch is not the faithful configuration that small batches deviate
from -- it is the degraded one. Fidelity is therefore scored against batch=1,
which is the only config that embeds each sentence on its own activations.

    python -m bench.tune_extract --tag full --n 300
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from core.embedder import Embedder, EmbedderConfig
from core.extractive import extract_answer
from core.retriever import DEFAULT_ENSEMBLE, AdaptiveRetriever
from ingest.pipeline import DATA_ROOT, read_jsonl

PERCENTILES = {"p50": 50, "p70": 70, "p90": 90, "p95": 95, "p99": 99}

# (embed_batch, max_sentence_chars). REFERENCE first -- see the module docstring
# for why that is batch=1 and not the largest batch. batch=64 reproduces the
# pre-fix behaviour: 10 sentences in one batch, all padded to the longest.
REFERENCE: tuple[int, int] = (1, 0)
CONFIGS: list[tuple[int, int]] = [
    REFERENCE,
    (64, 0),
    (8, 0),
    (4, 0),
    (2, 0),
    (4, 512),
    (4, 256),
    (4, 192),
    (2, 256),
    # batch=1 removes the padding amplification but not the cost of a genuinely
    # long sentence embedded on its own -- that is what truncation caps.
    (1, 512),
    (1, 256),
    (1, 192),
]


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

    rng = np.random.default_rng(7)
    picked = [queries[i] for i in rng.choice(len(queries), min(n + warmup, len(queries)), False)]

    embedder = Embedder(EmbedderConfig(threads=int(os.getenv("ORT_THREADS", "0"))))
    retriever = AdaptiveRetriever.load(DATA_ROOT / "index" / tag, ensemble)

    # Retrieve once; every config then extracts over identical hits.
    print(f"retrieving {len(picked)} queries once, shared across {len(CONFIGS)} configs...")
    cached = []
    for q in picked:
        qv = embedder.encode_query(q["query"])
        cached.append((q["query"], qv, retriever.search(qv, q["query"], k=top_k).hits))

    results = []
    baseline_answers: list[str] | None = None

    for batch, trunc in CONFIGS:
        # Warm this config's shapes before measuring.
        for text, qv, hits in cached[:warmup]:
            extract_answer(text, qv, hits, embedder, embed_batch=batch, max_sentence_chars=trunc)

        times: list[float] = []
        answers: list[str] = []
        supports: list[float] = []
        for text, qv, hits in cached[warmup:]:
            a = extract_answer(
                text, qv, hits, embedder, embed_batch=batch, max_sentence_chars=trunc
            )
            times.append(a.took_ms)
            answers.append(a.text)
            supports.append(a.support)

        if baseline_answers is None:
            baseline_answers = answers  # CONFIGS[0] is REFERENCE
        identical = sum(1 for a, b in zip(answers, baseline_answers, strict=True) if a == b)

        r = {
            "embed_batch": batch,
            "max_sentence_chars": trunc,
            "extract_ms": pct(times),
            "identical_to_reference": round(identical / len(answers), 4),
            "mean_support": round(float(np.mean(supports)), 4),
        }
        results.append(r)
        print(
            f"  batch={batch:<3} trunc={trunc:<4} p50={r['extract_ms']['p50']:>7.2f} "
            f"p99={r['extract_ms']['p99']:>7.2f} p100={r['extract_ms']['p100']:>7.2f} "
            f"identical={r['identical_to_reference']:.3f} support={r['mean_support']:.4f}"
        )

    return {
        "tag": tag,
        "ensemble": ensemble,
        "langs": langs,
        "n_queries": len(cached) - warmup,
        "machine": os.uname().machine,
        "embedder_variant": embedder.variant,
        "threads": os.getenv("ORT_THREADS", "auto"),
        "seed": 7,
        "reference": "embed_batch=1, max_sentence_chars=0 (each sentence quantised alone)",
        "configs": results,
    }


def render(r: dict) -> str:
    lines = [
        "",
        f"## `extract` tuning sweep ({r['n_queries']} queries, {r['machine']}, "
        f"{r['embedder_variant']}, {r['threads']} threads)",
        "",
        f"Reference: {r['reference']}. Retrieval is cached and shared, so only extraction varies.",
        "",
        "| embed_batch | trunc chars | P50 | P90 | P95 | P99 | P100 | identical | mean support |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in r["configs"]:
        e = c["extract_ms"]
        trunc = c["max_sentence_chars"] or "—"
        lines.append(
            f"| {c['embed_batch']} | {trunc} | {e['p50']} | {e['p90']} | {e['p95']} | "
            f"{e['p99']} | {e['p100']} | {c['identical_to_reference']:.1%} | {c['mean_support']} |"
        )
    lines += [
        "",
        "_`identical` is the fraction of answers byte-identical to the **reference**_",
        "_(batch=1), which is the only config that embeds each sentence on its own_",
        "_activations. int8 activation scales are computed per batch, so a wide pad_",
        "_coarsens the quantisation of everything beside it -- large batches are the_",
        "_degraded end of this table, not the faithful one._",
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
    ap.add_argument("--label", default="extract_tuning")
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

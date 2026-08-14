"""End-to-end ingestion: download -> chunk -> embed -> index.

Every stage checkpoints to disk and is skipped when its output already exists.
The full run is 3-6 hours on the Oracle box, and a crash in the embedding stage
must not cost the download and chunking that came before it.

    python -m ingest.pipeline --langs hin --max-queries 5000 --tag pilot
    python -m ingest.pipeline --langs hin mar --max-queries 50000

Stage outputs, all under $DATA_ROOT (default /data locally ./data):

    raw/{lang}_{split}_passages.jsonl     download.py
    chunks/{tag}/{strategy}.jsonl         chunkers.py
    vectors/{tag}/{strategy}.npy          embedder.py
    index/{tag}/{strategy}/               index.py
    reports/{tag}_pipeline.json           stage timings
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from core.embedder import Embedder, EmbedderConfig
from core.index import ChunkIndex
from ingest.chunkers import Passage, TokenCounter, chunk_all, to_documents
from ingest.download import extract, open_shard, read_query_ids, write_jsonl

# A variant is (strategy x chunk size x granularity).
#
# The first pilot compared four strategies at one size and found them
# statistically identical (MRR@10 spread of 1.2%). Cause: Devanagari fragments
# into 2-3x more XLM-R subword tokens, so a "60-80 word" MSMARCO passage is
# ~250 tokens -- under a 256-token budget nothing ever split, and all four
# strategies emitted one chunk per passage.
#
# So size is the primary variable, and granularity the secondary one. Published
# benchmarks favour ~512 tokens with 10-20% overlap, but that is calibrated on
# English; the Devanagari token multiplier means smaller budgets here.
#
# Smaller chunks are also *faster* to embed: throughput is superlinear in
# sequence length (250 tok -> 74/s, 80 tok -> 267/s measured), so two 128-token
# chunks cost less than one 250-token chunk.
VARIANTS: list[dict] = [
    # passage granularity -- does size matter at all?
    {"name": "fixed_128", "strategy": "fixed", "max_tokens": 128, "gran": "passage"},
    {"name": "fixed_256", "strategy": "fixed", "max_tokens": 256, "gran": "passage"},
    {"name": "sentence_128", "strategy": "sentence", "max_tokens": 128, "gran": "passage"},
    {"name": "semantic_128", "strategy": "semantic", "max_tokens": 128, "gran": "passage"},
    {"name": "metadata_128", "strategy": "metadata", "max_tokens": 128, "gran": "passage"},
    # document granularity -- the setting real RAG chunking operates in
    {"name": "doc_fixed_256", "strategy": "fixed", "max_tokens": 256, "gran": "doc"},
    {"name": "doc_sentence_256", "strategy": "sentence", "max_tokens": 256, "gran": "doc"},
    {"name": "doc_semantic_256", "strategy": "semantic", "max_tokens": 256, "gran": "doc"},
]
VARIANTS_BY_NAME = {v["name"]: v for v in VARIANTS}

DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data" if Path("/data").is_dir() else "./data"))

_timings: dict[str, float] = {}


@contextmanager
def stage(name: str):
    print(f"\n▶ {name}")
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    _timings[name] = round(dt, 2)
    print(f"✓ {name} — {dt:,.1f}s")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --------------------------------------------------------------------------


def stage_download(
    langs: list[str], split: str, max_queries: int, force: bool, stream: bool
) -> None:
    raw = DATA_ROOT / "raw"
    for lang in langs:
        p_out = raw / f"{lang}_{split}_passages.jsonl"
        q_out = raw / f"{lang}_{split}_queries.jsonl"

        # Incremental by query_id: `max_queries` is the target *total*, so a
        # re-run with a bigger number extracts only the shortfall. Growing a
        # corpus costs the delta, not a rebuild.
        have = set() if force else read_query_ids(q_out)
        want = max_queries - len(have)
        if want <= 0:
            print(f"  skip {lang} ({len(have):,} queries already >= target {max_queries:,})")
            continue
        if have:
            print(f"  {lang}: have {len(have):,}, adding {want:,} more")

        with stage(f"{'stream' if stream else 'download'}+extract {lang}"):
            src = open_shard(lang, split, raw / "hf_cache", stream)
            passages, queries = extract(src, want, skip_query_ids=have)
            write_jsonl(passages, p_out, append=bool(have))
            write_jsonl(queries, q_out, append=bool(have))


def load_passages(langs: list[str], split: str) -> list[Passage]:
    out: list[Passage] = []
    for lang in langs:
        for r in read_jsonl(DATA_ROOT / "raw" / f"{lang}_{split}_passages.jsonl"):
            # Index the translated text -- that is what a spoken Hindi/Marathi
            # question will actually match against.
            out.append(
                Passage(
                    passage_id=r["passage_id"],
                    query_id=r["query_id"],
                    text=r["text_translated"],
                    lang=r["lang"],
                    query_type=r["query_type"],
                    translator=r.get("translator"),
                )
            )
    return out


def stage_chunk(
    passages: list[Passage],
    tag: str,
    embedder: Embedder,
    tc: TokenCounter,
    force: bool,
    variants: list[dict],
) -> dict[str, Path]:
    out_dir = DATA_ROOT / "chunks" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # Semantic chunking embeds every sentence before deciding where to cut --
    # the single most expensive part of ingestion.
    def embed_fn(sents: list[str]) -> np.ndarray:
        return embedder.encode_passages(sents)

    docs: list[Passage] | None = None

    for v in variants:
        name = v["name"]
        path = out_dir / f"{name}.jsonl"
        paths[name] = path

        # Which passages are already chunked for this variant? Chunk only the
        # rest and append, keeping chunk order aligned with the vector file.
        done_units: set[str] = set()
        if path.exists() and not force:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        done_units.add(json.loads(line)["passage_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue

        if v["gran"] == "doc":
            if docs is None:
                docs = list(to_documents(passages))
                print(f"  built {len(docs):,} documents from {len(passages):,} passages")
            source = docs
        else:
            source = passages

        if done_units:
            source = [p for p in source if p.passage_id not in done_units]
            if not source:
                print(f"  skip chunk:{name} (all {len(done_units):,} units already chunked)")
                continue
            print(f"  chunk:{name} extending by {len(source):,} new units")

        with stage(f"chunk:{name}"):
            n = 0
            seen_units = 0
            last_id = None
            t0 = time.perf_counter()
            total = len(source)
            # Semantic chunking embeds every sentence before choosing cut points,
            # so this stage runs for minutes with nothing to show. Report progress
            # or it reads as a hang.
            report_every = max(1, total // 20)
            with path.open("a" if done_units else "w", encoding="utf-8") as f:
                for c in chunk_all(
                    source,
                    tc,
                    v["strategy"],
                    embed_fn if v["strategy"] == "semantic" else None,
                    max_tokens=v["max_tokens"],
                ):
                    f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
                    n += 1
                    if c.passage_id != last_id:
                        last_id = c.passage_id
                        seen_units += 1
                        if seen_units % report_every == 0:
                            el = time.perf_counter() - t0
                            eta = (total - seen_units) / max(seen_units / el, 1e-9)
                            print(
                                f"    {seen_units:,}/{total:,} units  "
                                f"{n:,} chunks  eta {eta / 60:,.1f}m",
                                flush=True,
                            )
            per = n / max(total, 1)
            print(f"  {n:,} chunks  ({per:.2f} per unit)")
    return paths


def stage_embed(chunk_paths: dict[str, Path], tag: str, embedder: Embedder, force: bool) -> None:
    out_dir = DATA_ROOT / "vectors" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for strat, cpath in chunk_paths.items():
        vpath = out_dir / f"{strat}.npy"

        # Vectors are row-aligned to the chunk file, and chunks are only ever
        # appended -- so existing rows stay valid and only the tail needs work.
        # Embedding is ~90% of runtime, so this is where incremental pays off.
        existing = None
        if vpath.exists() and not force:
            existing = np.load(vpath)
            n_chunks = sum(1 for _ in cpath.open(encoding="utf-8"))
            if existing.shape[0] >= n_chunks:
                print(f"  skip embed:{strat} ({existing.shape[0]:,} vectors, up to date)")
                continue
            print(f"  embed:{strat} extending {existing.shape[0]:,} -> {n_chunks:,}")

        with stage(f"embed:{strat}"):
            chunks = read_jsonl(cpath)
            start = existing.shape[0] if existing is not None else 0
            texts = [c["text"] for c in chunks[start:]]
            vecs = np.empty((len(texts), 384), dtype=np.float32)
            # Progress must be newline-terminated and flushed: `docker logs` shows
            # nothing for a carriage-return progress bar, which makes a multi-hour
            # stage look like a hang.
            step = 512
            t0 = time.perf_counter()
            for i in range(0, len(texts), step):
                vecs[i : i + step] = embedder.encode_passages(texts[i : i + step])
                done = min(i + step, len(texts))
                rate = done / max(time.perf_counter() - t0, 1e-9)
                eta = (len(texts) - done) / max(rate, 1e-9)
                print(
                    f"    {done:,}/{len(texts):,}  {rate:,.1f}/s  eta {eta / 60:,.1f}m",
                    flush=True,
                )

            # Append, never replace. Rows are positionally aligned to the chunk
            # file, so dropping the existing block would silently misalign every
            # vector against its text -- the index would still build and still
            # return results, just wrong ones.
            if existing is not None:
                vecs = np.concatenate([existing, vecs], axis=0)

            n_lines = sum(1 for _ in cpath.open(encoding="utf-8"))
            if vecs.shape[0] != n_lines:
                raise RuntimeError(
                    f"{strat}: {vecs.shape[0]} vectors vs {n_lines} chunks -- refusing to "
                    "write a misaligned vector file"
                )
            np.save(vpath, vecs)


def stage_index(chunk_paths: dict[str, Path], tag: str, force: bool) -> None:
    root = DATA_ROOT / "index" / tag
    for strat, cpath in chunk_paths.items():
        info = root / strat / "info.json"
        if info.exists() and not force:
            built = json.loads(info.read_text()).get("n_chunks", -1)
            have = sum(1 for _ in cpath.open(encoding="utf-8"))
            if built == have:
                print(f"  skip index:{strat} ({built:,} chunks, up to date)")
                continue
            # HNSW is fixed-capacity and BM25 cannot append, so growth means a
            # rebuild -- but that costs ~13s per 20k chunks against hours of
            # embedding, so it is never the bottleneck.
            print(f"  rebuild index:{strat} ({built:,} -> {have:,} chunks)")
        with stage(f"index:{strat}"):
            chunks = read_jsonl(cpath)
            vecs = np.load(DATA_ROOT / "vectors" / tag / f"{strat}.npy")
            # For the metadata strategy the type hint helps the embedding but
            # would pollute lexical matching -- and must never reach the user, since
            # the extractive answer is drawn from the stored text. So BM25 and the
            # display copy both see the untagged body; only the vectors saw the hint.
            raw_texts = [c.get("extra", {}).get("raw_text") or c["text"] for c in chunks]
            ix = ChunkIndex(strat)
            ix.build(vecs, chunks, bm25_texts=raw_texts, display_texts=raw_texts)
            ix.save(root)
            print(f"  {len(ix):,} chunks indexed")


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["hin", "mar"])
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    ap.add_argument("--max-queries", type=int, default=50_000)
    ap.add_argument("--tag", default="full", help="namespaces outputs, e.g. 'pilot'")
    ap.add_argument("--force", action="store_true", help="ignore checkpoints")
    ap.add_argument("--stream", action="store_true", help="HTTP range reads; store no parquet")
    ap.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help=f"subset of: {' '.join(VARIANTS_BY_NAME)}",
    )
    ap.add_argument("--threads", type=int, default=int(os.getenv("ORT_THREADS", "0")))
    args = ap.parse_args()

    print(f"data root : {DATA_ROOT}")
    print(f"tag       : {args.tag}")
    print(f"langs     : {args.langs}  max_queries={args.max_queries:,}")

    t_start = time.perf_counter()

    stage_download(args.langs, args.split, args.max_queries, args.force, args.stream)

    with stage("load embedder"):
        embedder = Embedder(EmbedderConfig(threads=args.threads))
        print(f"  variant={embedder.variant}")
    tc = TokenCounter(embedder.tokenizer)

    with stage("load passages"):
        passages = load_passages(args.langs, args.split)
        print(f"  {len(passages):,} passages")

    variants = (
        [VARIANTS_BY_NAME[n] for n in args.variants] if args.variants else VARIANTS
    )
    print(f"variants  : {', '.join(v['name'] for v in variants)}")

    chunk_paths = stage_chunk(passages, args.tag, embedder, tc, args.force, variants)
    stage_embed(chunk_paths, args.tag, embedder, args.force)
    stage_index(chunk_paths, args.tag, args.force)

    total = time.perf_counter() - t_start
    _timings["TOTAL"] = round(total, 2)

    rep_dir = DATA_ROOT / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "tag": args.tag,
        "langs": args.langs,
        "max_queries": args.max_queries,
        "n_passages": len(passages),
        "embedder_variant": embedder.variant,
        "stage_seconds": _timings,
    }
    (rep_dir / f"{args.tag}_pipeline.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'=' * 52}\nTOTAL {total / 60:,.1f} min")
    for k, v in _timings.items():
        print(f"  {k:<28} {v:>10,.1f}s")


if __name__ == "__main__":
    main()

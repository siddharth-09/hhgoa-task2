"""Build the English index from `text_eng`, which is already on disk.

MSMARCO-XI ships every passage twice: the original English (`English_passages`)
and the Indic translation (`Translated_passages`). Only the translation was ever
chunked and embedded, so the retriever has never seen a word of English -- which
is why an English question abstains even when the answer is sitting in the corpus.
Measured: "What is the capital of India?" scores 0.359 against the Devanagari index
and abstains, while the English sentence "The capital of India is New Delhi" is
right there in `data/raw/`.

Nothing is downloaded. The text, the ids and the gold labels all already exist.

One detail that halves the work: the English source is *shared* across language
shards, so the 199,668 rows contain only 98,812 distinct English passages. That
sharing is the same property that caused the cross-language id collision fixed
earlier, so ids are namespaced `eng:<passage_id>` and deduped by content hash.

    docker compose run --rm bench python -m ingest.build_english
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
from dataclasses import asdict

from core.embedder import Embedder, EmbedderConfig
from core.index import ChunkIndex
from ingest.chunkers import Passage, TokenCounter, chunk_all
from ingest.pipeline import DATA_ROOT

VARIANT = "english_256"


def load_english() -> list[Passage]:
    """Deduped English passages, ordered stably so a rebuild is reproducible."""
    seen: set[bytes] = set()
    out: list[Passage] = []
    for lang in ("hin", "mar"):
        path = DATA_ROOT / "raw" / f"{lang}_train_passages.jsonl"
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                text = (r.get("text_eng") or "").strip()
                if not text:
                    continue
                h = hashlib.blake2b(text.encode(), digest_size=16).digest()
                if h in seen:
                    continue
                seen.add(h)
                # Keep the source passage_id so gold labels still resolve, but
                # namespace it: the same id exists in the Devanagari indexes.
                out.append(
                    Passage(
                        passage_id=f"eng:{r['passage_id']}",
                        query_id=r["query_id"],
                        text=text,
                        lang="eng_Latn",
                        query_type=r.get("query_type", ""),
                        translator="source",
                    )
                )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    t0 = time.perf_counter()
    passages = load_english()
    if args.limit:
        passages = passages[: args.limit]
    print(f"english passages (deduped) : {len(passages):,}")

    embedder = Embedder(EmbedderConfig(threads=int(os.getenv("ORT_THREADS", "0"))))
    tc = TokenCounter(embedder.tokenizer)

    # MSMARCO passages are already retrieval-sized, so at a 256-token budget this
    # is close to a no-op -- the same null result the Devanagari ablation found.
    # Run it anyway so an over-long passage is split rather than truncated.
    chunks = [asdict(c) for c in chunk_all(passages, tc, "fixed", max_tokens=256)]
    print(f"chunks                     : {len(chunks):,} "
          f"({len(chunks) / max(1, len(passages)):.2f} per passage)")

    texts = [c["text"] for c in chunks]
    print(f"embedding with {embedder.variant} ...")
    t1 = time.perf_counter()
    vecs = np.empty((len(texts), 384), dtype=np.float32)
    for i in range(0, len(texts), args.batch):
        vecs[i : i + args.batch] = embedder.encode_passages(texts[i : i + args.batch])
        if i and i % (args.batch * 40) == 0:
            done = i + args.batch
            rate = done / (time.perf_counter() - t1)
            print(f"  {done:>7,}/{len(texts):,}  {rate:>5.0f}/s  "
                  f"eta {(len(texts) - done) / rate / 60:>4.1f} min")
    embed_s = time.perf_counter() - t1
    print(f"embedded {len(texts):,} in {embed_s / 60:.1f} min ({len(texts) / embed_s:.0f}/s)")

    ix = ChunkIndex(VARIANT)
    ix.build(vecs, chunks)
    root = DATA_ROOT / "index" / args.tag
    root.mkdir(parents=True, exist_ok=True)
    ix.save(root)

    print(f"\nwrote {root / VARIANT}  ({len(ix):,} chunks)")
    print(f"total {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

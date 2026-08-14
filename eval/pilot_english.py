"""Pilot: would an English index actually fix the English queries?

Before spending ~30 minutes embedding 99k passages and adding a router to the
retrieval path eight days from a no-resubmission deadline, test the premise on a
20k subset: does an English query find its English passage, and does support clear
the 0.45 grounding gate?

The sample deliberately *includes* the passages known to contain the answers, plus
~20k distractors. That is a fair test of retrieval, not a guarantee of a hit: the
question is whether the right passage ranks first among realistic competition.

    docker compose run --rm bench python -m eval.pilot_english
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from core.embedder import Embedder, EmbedderConfig
from core.extractive import extract_answer
from core.index import ChunkIndex
from core.retriever import AdaptiveRetriever
from ingest.pipeline import DATA_ROOT

SAMPLE = 20_000
MIN_SUPPORT = 0.45  # the live grounding gate

# Queries that currently abstain against the Devanagari index.
PROBES = [
    ("What is the capital of India?",      "capital of india"),
    ("who built the Taj Mahal",            "taj mahal"),
    ("what is the meaning of lift",        "lift"),
    ("types of social security disability", "social security"),
    ("what was the impact of the Manhattan Project", "manhattan project"),
]


def load_english(limit: int) -> list[dict]:
    """Deduped English passages, with any passage matching a probe pulled in."""
    seen: set[bytes] = set()
    keep: list[dict] = []
    must: list[dict] = []
    needles = [n for _, n in PROBES]

    for lang in ("hin", "mar"):
        p = DATA_ROOT / "raw" / f"{lang}_train_passages.jsonl"
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                r = json.loads(line)
                txt = (r.get("text_eng") or "").strip()
                if not txt:
                    continue
                h = hashlib.blake2b(txt.encode(), digest_size=16).digest()
                if h in seen:
                    continue
                seen.add(h)
                row = {
                    "chunk_id": f"eng:{r['passage_id']}",
                    "passage_id": f"eng:{r['passage_id']}",
                    "query_id": r["query_id"],
                    "text": txt,
                    "strategy": "english",
                    "lang": "eng_Latn",
                    "query_type": r.get("query_type", ""),
                    "n_tokens": 0,
                    "translator": "",
                }
                low = txt.lower()
                (must if any(n in low for n in needles) else keep).append(row)

    # answers first, then distractors up to the sample budget
    return (must + keep)[:limit] if len(must) < limit else must[:limit]


def main() -> None:
    t0 = time.perf_counter()
    rows = load_english(SAMPLE)
    print(f"english passages sampled : {len(rows):,}")

    emb = Embedder(EmbedderConfig(threads=int(os.getenv("ORT_THREADS", "0"))))
    print(f"embedding with {emb.variant} ...")
    t1 = time.perf_counter()
    vecs = emb.encode_passages([r["text"] for r in rows])
    embed_s = time.perf_counter() - t1
    print(f"embedded {len(rows):,} in {embed_s:.0f}s ({len(rows)/embed_s:.0f}/s)")

    ix = ChunkIndex("english")
    ix.build(vecs, rows)
    root = DATA_ROOT / "index" / "pilot_eng"
    root.mkdir(parents=True, exist_ok=True)
    ix.save(root)
    r = AdaptiveRetriever({"english": ix})

    print(f"\n{'query':<48} {'support':>8}  {'gate':<8} top hit")
    print("-" * 108)
    passed = 0
    for q, _ in PROBES:
        qv = emb.encode_query(q)
        hits = r.search(qv, q, k=10).hits
        ans = extract_answer(q, qv, hits, emb)
        ok = ans.support >= MIN_SUPPORT
        passed += ok
        print(f"{q[:46]:<48} {ans.support:>8.3f}  {'ANSWER' if ok else 'abstain':<8} "
              f"{(hits[0].text[:52] if hits else '—')}")

    print("-" * 108)
    print(f"  {passed}/{len(PROBES)} would clear the {MIN_SUPPORT} grounding gate")
    print(f"  total {time.perf_counter() - t0:.0f}s")
    print(f"\n  extrapolated full build: {98812/(len(rows)/embed_s)/60:.0f} min for 98,812 passages")


if __name__ == "__main__":
    main()

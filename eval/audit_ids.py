"""Audit the corpus and indexes for id collisions and duplicate content.

The cross-language id collision was invisible for a full day because the pilot was
monolingual and every metric looked plausible. So this does not check whether the
numbers seem reasonable -- it checks the invariants directly, and it is written to
be re-run after any ingest.

Four separate questions, because they fail independently:

  1. structural   do meta / HNSW / BM25 agree on row count?
  2. identity     are chunk_ids and passage_ids unique within an index?
  3. contamination does one id ever map to two different texts, or one text to
                  two ids? The first is the collision; the second is duplicate
                  content, which inflates recall without being a bug per se.
  4. gold linkage do the eval's gold_passage_ids actually resolve into the index?
                  A silently unresolvable gold set scores zero and looks like a
                  quality problem rather than a plumbing one.

    python -m eval.audit_ids --tag full
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from core.index import ChunkIndex
from ingest.pipeline import DATA_ROOT, read_jsonl

ALL_INDEXES = ["fixed_256", "semantic_128", "metadata_128"]


def _sig(text: str) -> str:
    return hashlib.blake2b(" ".join(text.split()).encode(), digest_size=16).hexdigest()


def audit_index(root: Path, name: str) -> dict:
    ix = ChunkIndex.load(root, name)
    n = len(ix.chunk_ids)

    n_hnsw = ix.hnsw.get_current_count()
    n_bm25 = int(ix.bm25.scores["num_docs"])
    structural_ok = n == n_hnsw == n_bm25

    chunk_dupes = n - len(set(ix.chunk_ids))

    # A passage legitimately maps to several chunks, so duplicate passage_ids are
    # expected. The defect is one passage_id spanning more than one language.
    langs_per_pid: dict[str, set[str]] = {}
    for pid, lg in zip(ix.passage_ids, ix.langs, strict=True):
        langs_per_pid.setdefault(pid, set()).add(lg)
    cross_lang = sum(1 for v in langs_per_pid.values() if len(v) > 1)

    # One chunk_id -> two different texts is a true collision.
    text_per_cid: dict[str, str] = {}
    collisions = 0
    for cid, txt in zip(ix.chunk_ids, ix.texts, strict=True):
        s = _sig(txt)
        if cid in text_per_cid:
            if text_per_cid[cid] != s:
                collisions += 1
        else:
            text_per_cid[cid] = s

    # One text under several ids is duplicate content, not a collision.
    text_counts = Counter(_sig(t) for t in ix.texts)
    dup_text_rows = sum(c - 1 for c in text_counts.values() if c > 1)

    del ix
    return {
        "index": name,
        "n_rows": n,
        "n_hnsw": n_hnsw,
        "n_bm25": n_bm25,
        "structural_aligned": structural_ok,
        "duplicate_chunk_ids": chunk_dupes,
        "passage_ids_spanning_multiple_langs": cross_lang,
        "chunk_id_text_collisions": collisions,
        "duplicate_text_rows": dup_text_rows,
        "duplicate_text_pct": round(100 * dup_text_rows / n, 2),
        "clean": structural_ok and chunk_dupes == 0 and cross_lang == 0 and collisions == 0,
    }


def audit_gold(root: Path, langs: list[str], split: str) -> dict:
    known: set[str] = set()
    for name in ALL_INDEXES:
        ix = ChunkIndex.load(root, name)
        known |= set(ix.passage_ids)
        del ix

    total = resolved = 0
    empty = 0
    for lang in langs:
        for q in read_jsonl(DATA_ROOT / "raw" / f"{lang}_{split}_queries.jsonl"):
            gold = q.get("gold_passage_ids") or []
            if not gold:
                empty += 1
                continue
            total += len(gold)
            resolved += sum(1 for g in gold if g in known)

    return {
        "gold_ids_total": total,
        "gold_ids_resolvable": resolved,
        "gold_ids_missing": total - resolved,
        "resolvable_pct": round(100 * resolved / total, 2) if total else 0.0,
        "queries_with_no_gold": empty,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--langs", nargs="+", default=["hin", "mar"])
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    root = DATA_ROOT / "index" / args.tag
    per_index = [audit_index(root, n) for n in ALL_INDEXES]
    gold = audit_gold(root, args.langs, args.split)

    print("\n## Id audit\n")
    print("| Index | rows | hnsw | bm25 | aligned | dup chunk_ids | cross-lang pids | "
          "id->2 texts | dup text rows | clean |")
    print("|---|---:|---:|---:|---|---:|---:|---:|---:|---|")
    for r in per_index:
        print(
            f"| {r['index']} | {r['n_rows']:,} | {r['n_hnsw']:,} | {r['n_bm25']:,} | "
            f"{'yes' if r['structural_aligned'] else 'NO'} | {r['duplicate_chunk_ids']:,} | "
            f"{r['passage_ids_spanning_multiple_langs']:,} | {r['chunk_id_text_collisions']:,} | "
            f"{r['duplicate_text_rows']:,} ({r['duplicate_text_pct']}%) | "
            f"{'CLEAN' if r['clean'] else 'DIRTY'} |"
        )

    print(
        f"\nGold linkage: {gold['gold_ids_resolvable']:,}/{gold['gold_ids_total']:,} "
        f"({gold['resolvable_pct']}%) gold passage ids resolve into the index; "
        f"{gold['queries_with_no_gold']:,} queries carry no gold."
    )

    verdict = all(r["clean"] for r in per_index) and gold["resolvable_pct"] > 99.0
    print(f"\nVERDICT: {'CLEAN' if verdict else 'NOT CLEAN'}")

    out = DATA_ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}_id_audit.json").write_text(
        json.dumps({"per_index": per_index, "gold": gold, "clean": verdict}, indent=2)
    )


if __name__ == "__main__":
    main()

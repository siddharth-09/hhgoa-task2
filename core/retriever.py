"""Adaptive multi-index retriever: query several chunkings at once, fuse the ranks.

Motivation is empirical, not aesthetic. The variant sweep found no single
chunking significantly better than another at passage granularity -- but it did
find the granularities fail in *opposite directions*:

    doc_fixed_256   R@1 0.2688   R@10 0.3671   <- best precision, worst coverage
    fixed_256       R@1 0.1871   R@10 0.6763   <- worst precision, best coverage

Diverse retrievers surfacing different candidates is exactly the condition under
which rank fusion helps. So rather than choosing one chunking (and rather than
making the user choose), every query hits all of them and the ranks are fused.

Each index answers in ~1ms, so a four-index fan-out costs ~4ms against a 200ms
budget. The provenance of each result is preserved, so the UI can show which
chunking strategies actually contributed to a given answer -- the multi-strategy
work becomes visible in the product rather than only in the README.

Fusion happens at *unit* granularity (parent passage or document), not chunk id,
because different chunkings produce different chunk ids for the same underlying
text; fusing on chunk id would treat the same passage as several distinct
candidates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.index import ChunkIndex, Hit

# Chosen by ablation (eval/ablate.py, 1200 labelled queries), re-run after the
# Devanagari tokenisation fix in core/text.py:
#
#   fixed_256 + semantic_128                   MRR 0.4065  nDCG 0.4883  R@10 0.7678  2.13ms
#   + metadata_128                             MRR 0.4069  nDCG 0.4904  R@10 0.7757  3.16ms  <-
#   + metadata_128 + doc_fixed_256             MRR 0.4069  nDCG 0.4904  R@10 0.7757  3.96ms
#
# metadata_128 earns its place on recall (+1.0% R@10). It did *not* before the
# tokenizer fix -- it appeared to hurt, which was an artifact of BM25 indexing
# consonant fragments rather than a property of the strategy. Worth remembering
# that a broken component can invert an ablation's conclusion.
#
# doc_fixed_256 still contributes nothing: identical metrics to four decimals
# while filling 19.3% of returned slots with passages that are never relevant.
# It displaces better candidates and costs 6k chunks of RAM to do it.
#
# Latency is not the constraint here (3.16ms of a 200ms budget). Memory is: three
# passage indexes is ~1.5x the resident set of two, which matters at full scale.
# If the serving box is tight, fixed_256 + semantic_128 gives up ~1% R@10.
DEFAULT_ENSEMBLE = ["fixed_256", "semantic_128", "metadata_128"]


@dataclass(slots=True)
class FusedHit:
    unit_id: str  # parent passage_id, or doc id for document chunks
    text: str  # best-ranked chunk text for this unit
    score: float  # summed RRF score
    rank: int
    query_type: str
    lang: str
    # Which chunkings retrieved this unit, and at what rank in each.
    contributors: dict[str, int] = field(default_factory=dict)

    @property
    def n_votes(self) -> int:
        return len(self.contributors)

    @property
    def passage_id(self) -> str:
        """Alias so FusedHit and Hit score identically in eval/."""
        return self.unit_id


@dataclass(slots=True)
class RetrievalResult:
    hits: list[FusedHit]
    per_index_ms: dict[str, float]
    fuse_ms: float
    total_ms: float

    def provenance(self) -> dict[str, int]:
        """How many returned hits each chunking contributed to."""
        out: dict[str, int] = {}
        for h in self.hits:
            for name in h.contributors:
                out[name] = out.get(name, 0) + 1
        return out


class AdaptiveRetriever:
    def __init__(
        self,
        indexes: dict[str, ChunkIndex],
        weights: dict[str, float] | None = None,
        rrf_k: int = 60,
    ):
        if not indexes:
            raise ValueError("need at least one index")
        self.indexes = indexes
        self.weights = weights or dict.fromkeys(indexes, 1.0)
        self.rrf_k = rrf_k

    @classmethod
    def load(
        cls,
        index_root: Path,
        names: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ) -> AdaptiveRetriever:
        names = names or DEFAULT_ENSEMBLE
        indexes: dict[str, ChunkIndex] = {}
        for n in names:
            if (index_root / n / "hnsw.bin").exists():
                indexes[n] = ChunkIndex.load(index_root, n)
        missing = set(names) - set(indexes)
        if missing:
            print(f"  warning: missing indexes skipped: {sorted(missing)}")
        return cls(indexes, weights)

    def search(self, qvec: np.ndarray, query_text: str, k: int = 20, per_index_k: int = 30):
        t_start = time.perf_counter()
        per_index_ms: dict[str, float] = {}

        # unit_id -> accumulated RRF score, contributing ranks, best chunk seen
        scores: dict[str, float] = {}
        contributors: dict[str, dict[str, int]] = {}
        best: dict[str, Hit] = {}

        for name, ix in self.indexes.items():
            t0 = time.perf_counter()
            hits = ix.search(qvec, query_text, k=per_index_k)
            per_index_ms[name] = round((time.perf_counter() - t0) * 1000, 3)

            w = self.weights.get(name, 1.0)
            seen_units: set[str] = set()
            rank = 0
            for h in hits:
                unit = h.passage_id
                # One vote per unit per index -- a chunking that split a passage
                # into five chunks should not get five votes for it.
                if unit in seen_units:
                    continue
                seen_units.add(unit)

                scores[unit] = scores.get(unit, 0.0) + w / (self.rrf_k + rank + 1)
                contributors.setdefault(unit, {})[name] = rank
                if unit not in best:
                    best[unit] = h
                rank += 1

        t_fuse = time.perf_counter()
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        fused = [
            FusedHit(
                unit_id=unit,
                text=best[unit].text,
                score=float(s),
                rank=i,
                query_type=best[unit].query_type,
                lang=best[unit].lang,
                contributors=contributors[unit],
            )
            for i, (unit, s) in enumerate(ordered)
        ]
        now = time.perf_counter()

        return RetrievalResult(
            hits=fused,
            per_index_ms=per_index_ms,
            fuse_ms=round((now - t_fuse) * 1000, 3),
            total_ms=round((now - t_start) * 1000, 3),
        )

    def __len__(self) -> int:
        return sum(len(ix) for ix in self.indexes.values())

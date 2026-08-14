"""Adaptive multi-index retriever: query several chunkings at once, fuse the ranks.

The mechanism is rank fusion over any number of chunking strategies. How many to
actually serve is a separate, measured question -- and the answer changed once the
corpus reached full scale.

The pilot found the granularities failing in *opposite directions*, which is the
classic condition for fusion to help:

    doc_fixed_256   R@1 0.2688   R@10 0.3671   <- best precision, worst coverage
    fixed_256       R@1 0.1871   R@10 0.6763   <- worst precision, best coverage

At 682k chunks over two languages that no longer holds: an exhaustive subset
ablation with paired confidence intervals found every recall difference between
configurations to be inside the noise, and the three-index ensemble significantly
*behind* a single index on MRR. DEFAULT_ENSEMBLE below carries the numbers and
the reasoning. The fan-out machinery stays because the corpus will grow and the
question should be re-asked, not because three indexes are currently earning
their memory.

Provenance of each result is preserved either way, so the UI can show which
chunkings contributed to an answer -- with one index that is a constant, and the
panel is worth keeping only while more than one is served.

Fusion happens at *unit* granularity (parent passage or document), not chunk id,
because different chunkings produce different chunk ids for the same underlying
text; fusing on chunk id would treat the same passage as several distinct
candidates.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.index import ChunkIndex, Hit

# Chosen by exhaustive ablation over all 7 subsets at full scale (eval/ablate_full.py,
# 1,500 bilingual queries, 682k chunks), with paired bootstrap CIs on the deltas
# (eval/significance.py, 10k resamples). Both supersede the 1,200-query Hindi-only
# pilot that picked the three-index ensemble.
#
#   config                 chunks   MRR@10   R@10     R@20     search P50   disk
#   metadata_128          241,572   0.3030   0.5669   0.6675      4.32ms    722MB  <-
#   fixed_256             201,298   0.2895   0.5601   0.6607      4.28ms    623MB
#   fixed+metadata        442,870   0.2973   0.5684   0.6697      7.20ms   1345MB
#   ENSEMBLE (all three)  682,045   0.2926   0.5717   0.6621     11.27ms   2050MB
#
# The ensemble appears to win R@10 and the pilot chose it on exactly that kind of
# margin. Paired bootstrap says otherwise:
#
#   ENSEMBLE - metadata_128   mrr@10     -0.0105  [-0.0185, -0.0026]  significant
#   ENSEMBLE - metadata_128   recall@10  +0.0047  [-0.0076, +0.0168]  not significant
#   ENSEMBLE - metadata_128   recall@20  -0.0054  [-0.0173, +0.0064]  not significant
#
# So every recall difference in that table is noise, and the one real difference
# favours the single index. Adding two indexes buys nothing measurable while
# costing 2.8x the memory and 2.6x the search time.
#
# Caveat worth carrying: metadata_128's edge is partly an evaluation artifact.
# chunk_metadata embeds the passage's `query_type`, which the dataset derives from
# the query that owns the passage -- so gold passages carry the asking query's
# type while most of the corpus does not. Its advantage over fixed_256 scales
# inversely with how common the type is (corpus share vs advantage correlates
# -0.638; PERSON at 2.7% of the corpus gains +0.0339, DESCRIPTION at 59.4% gains
# +0.0109 and not significantly). A corpus without those labels would see less of
# this. It is still the right ship here -- it is never significantly *worse* than
# any alternative on any metric, and it is a third of the footprint.
#
# Per language, the ensemble's only significant win is Hindi R@10 (+0.0179
# [+0.0020, +0.0341]); on Marathi it is significantly *worse* on MRR (-0.0181).
# Routing by language was considered and rejected: it would have to hold all three
# indexes resident, so it inherits the ensemble's memory cost -- the dominant cost
# here -- while adding a routing decision, to recover a gain in one language.
DEFAULT_ENSEMBLE = ["metadata_128"]

# The full set, kept for benchmarking and for `--ensemble` overrides. Nothing here
# is deleted: the ablation is meant to be re-runnable when the corpus grows.
FULL_ENSEMBLE = ["fixed_256", "semantic_128", "metadata_128"]

# Index of the original English text. MSMARCO-XI carries every passage in English
# as well as in translation, and only the translation was ever indexed -- so an
# English question could not reach the answer even when the corpus contained it.
ENGLISH_INDEX = "english_256"

_DEVA = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")


def is_latin_query(text: str, threshold: float = 0.9) -> bool:
    """Should this query go to the English index?

    Routing on *script* rather than on detected language, deliberately. Script is
    observable and deterministic; language is a guess, and the guess is wrong in
    exactly the case that matters most here -- romanised Hindi ("bharat ki
    rajdhani kya hai") is Latin script but Hindi language, and a language detector
    would confidently send it the wrong way.

    The threshold is high on purpose. A query is only treated as English when it
    is *overwhelmingly* Latin, so code-mixed input ("लिफ्ट का मतलब है", "Taj Mahal
    किसने बनवाया") keeps going to the Devanagari index, which is where its answer
    lives. Anything with real Devanagari content stays on the path that already
    works; the English index is additive, not a replacement.
    """
    if not text:
        return False
    latin = len(_LATIN.findall(text))
    deva = len(_DEVA.findall(text))
    if latin + deva == 0:
        return False
    return latin / (latin + deva) >= threshold


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

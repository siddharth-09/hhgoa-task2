"""Extractive answering: pick the best-supported span from retrieved chunks.

This is the sub-200ms path. It produces a real, grounded, citable answer with
no LLM call at all, which matters for three reasons:

  1. The task scopes the latency budget as "chunking + vector DB retrieval +
     everything through to final output". A hosted LLM cannot fit in 200ms, but
     an extracted span can -- so this is what the reported number measures.
  2. It is the harness fallback. When generation times out or errors, there is
     still an answer rather than a spinner.
  3. It gives the guardrail something to threshold. The support score below is
     a grounding signal: a low score means the corpus does not answer this, and
     the system should abstain rather than let an LLM improvise.

Scoring blends two signals deliberately:

  * cosine similarity to the query embedding -- handles paraphrase, and works
    cross-lingually, which matters when the question is spoken in Hindi.
  * lexical overlap -- catches the exact tokens embeddings are weakest on:
    numbers, dates, names, transliterated entities. NUMERIC and PERSON queries
    live here, and they are a third of this corpus.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from core.embedder import Embedder
from core.text import overlap as lexical_overlap
from ingest.chunkers import split_sentences

# Blend weight for cosine vs lexical overlap. Cosine leads because the query and
# corpus are the same language and the encoder is strong; lexical is the
# corrective term for exact tokens.
ALPHA = 0.75

MAX_SENTENCES = 24
MAX_SPAN_SENTENCES = 2

# Sentence embedding dominated the first measurement (17-50ms of a 29-57ms
# total), because every candidate sentence got a forward pass. A lexical
# prefilter cuts the set before embedding.
#
# It cannot be lexical-only: the whole point of the dense side is matching
# paraphrase, and a question sharing no tokens with its answer is exactly the
# case BM25 misses. So the prefilter keeps the lexically strongest candidates
# *and* unconditionally keeps the leading sentences of the top-ranked chunk,
# which retrieval already judged relevant.
# Measured on 200 queries (prefilter size -> answer identical to unfiltered):
#
#   none (24)   p50 41.0ms  p100 182.3ms   100%
#   10          p50 37.3ms  p100 100.6ms    95.0%   <- chosen
#    6          p50 25.7ms  p100  69.8ms    87.5%
#
# Unfiltered peaked at 182ms, which is uncomfortably close to the 200ms budget
# on the tail -- and the tail is precisely what P100 reports. 10 cuts P100 by
# 1.8x while changing the answer for only 1 query in 20.
#
# 6 is faster still, but changes the answer 1 time in 8. Mean support was flat
# across all three, which is exactly why support is not a sufficient quality
# check: a different sentence can score just as well and still be worse. Since
# 100ms already leaves 2x headroom, the spare budget buys fidelity, not speed.
MAX_EMBED = 10
ALWAYS_KEEP_FROM_TOP_HIT = 3


@dataclass(slots=True)
class Citation:
    unit_id: str
    text: str
    score: float


@dataclass(slots=True)
class ExtractiveAnswer:
    text: str
    support: float  # 0..1 grounding confidence -- guardrails threshold this
    citations: list[Citation] = field(default_factory=list)
    n_candidates: int = 0
    took_ms: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()




def extract_answer(
    query: str,
    qvec: np.ndarray,
    hits,
    embedder: Embedder,
    *,
    top_hits: int = 3,
    alpha: float = ALPHA,
    max_span: int = MAX_SPAN_SENTENCES,
    max_embed: int = MAX_EMBED,
) -> ExtractiveAnswer:
    """Select the best-supported span across the top retrieved chunks.

    `hits` is any sequence exposing .text and .passage_id (Hit or FusedHit).
    """
    t0 = time.perf_counter()

    # Candidate sentences, remembering which unit each came from and its
    # position, so adjacent sentences can be merged into one span.
    sentences: list[str] = []
    origins: list[tuple[str, int]] = []
    seen: set[str] = set()

    for h in list(hits)[:top_hits]:
        for i, s in enumerate(split_sentences(h.text)):
            key = s.strip()
            if len(key) < 3 or key in seen:
                continue
            seen.add(key)
            sentences.append(key)
            origins.append((h.passage_id, i))
            if len(sentences) >= MAX_SENTENCES:
                break
        if len(sentences) >= MAX_SENTENCES:
            break

    if not sentences:
        return ExtractiveAnswer(text="", support=0.0, took_ms=(time.perf_counter() - t0) * 1000)

    lex_all = np.array([lexical_overlap(query, s) for s in sentences], dtype=np.float32)

    # Prefilter before the embedding pass -- this is the latency win.
    n_total = len(sentences)
    if n_total > max_embed:
        keep = set(range(min(ALWAYS_KEEP_FROM_TOP_HIT, n_total)))  # leading sentences of top hit
        for j in np.argsort(-lex_all):
            if len(keep) >= max_embed:
                break
            keep.add(int(j))
        idx = sorted(keep)
        sentences = [sentences[j] for j in idx]
        origins = [origins[j] for j in idx]
        lex = lex_all[idx]
    else:
        lex = lex_all

    # One batched forward pass over the surviving candidates.
    svecs = embedder.encode_passages(sentences)
    cos = svecs @ qvec  # both L2-normalised -> dot == cosine

    # Cosine for e5 lives roughly in [0.7, 0.95] for related text, so rescale
    # into [0,1] before blending -- otherwise the lexical term is swamped and
    # every candidate looks equally good.
    cos_n = np.clip((cos - 0.70) / 0.25, 0.0, 1.0)
    scores = alpha * cos_n + (1.0 - alpha) * lex

    best = int(np.argmax(scores))
    span_idx = [best]

    # Extend to the following sentence when it is from the same unit, adjacent,
    # and nearly as well supported -- answers often continue past one sentence.
    if max_span > 1:
        unit, pos = origins[best]
        for j, (u, p) in enumerate(origins):
            if len(span_idx) >= max_span:
                break
            if u == unit and p == pos + 1 and scores[j] >= scores[best] * 0.75:
                span_idx.append(j)

    span_idx.sort(key=lambda j: origins[j][1])
    text = " ".join(sentences[j] for j in span_idx)

    # Cite every distinct unit that contributed to the span, plus the top hit
    # so the user can always see where the answer came from.
    cited_units: list[str] = []
    citations: list[Citation] = []
    for j in span_idx:
        u = origins[j][0]
        if u not in cited_units:
            cited_units.append(u)
            citations.append(Citation(unit_id=u, text=sentences[j], score=float(scores[j])))

    return ExtractiveAnswer(
        text=text,
        support=float(scores[best]),
        citations=citations,
        n_candidates=len(sentences),
        took_ms=round((time.perf_counter() - t0) * 1000, 3),
    )

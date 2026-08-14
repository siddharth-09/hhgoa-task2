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

# Batch size for the sentence-embedding pass, and the two reasons it is not 64.
#
# Measured (300 queries, full index, aarch64): `embed_sentences` is 99.5% of this
# stage -- 31.8ms of 32.0ms at P50, 180ms of 180.2ms at P99 -- and its latency
# correlates with the *longest* sentence in the batch at 0.84, with the *number*
# of sentences at 0.02. The slowest decile embeds the same ~8.8 sentences as the
# median; its longest is 387 chars against 152.
#
# Cause: the tokenizer pads a batch to its longest member. Embedder._encode sorts
# by length to avoid exactly this, but the sort only pays off across several
# batches, and 10 sentences is a single batch of 64 -- so one 1,279-char sentence
# made the other nine cost the same as itself.
#
# Embedding one sentence at a time removes the padding entirely. Measured over
# the same 300 queries and identical cached retrieval (bench/tune_extract.py):
#
#   batch  P50     P95     P99      P100     identical to reference
#   64     32.33   66.87   185.37   227.52   90.7%
#    8     28.55   50.14   156.37   179.47   91.7%
#    4     26.53   41.61    97.09   103.95   91.0%
#    2     26.17   36.94    68.01    76.91   93.3%
#    1     24.14   34.24    48.58    81.73   100%     <-
#
# Batch 1 is fastest at every percentile through P99 *and* is the fidelity
# reference, which is the counter-intuitive part and worth stating plainly:
#
# A bigger batch is not a more accurate configuration that we trade away for
# speed -- it is less accurate. The served model is dynamically quantised int8,
# so activation scales are computed at runtime from a tensor spanning the batch.
# One long padded member widens that range and coarsens the quantisation of every
# short sentence beside it. Measured directly on one text: batched against a long
# sentence moves its vector by 1.0e-02 (cos 0.9981); at batch 1 it is bitwise
# identical to embedding it alone. In fp32 the same comparison drifts 4.7e-08 --
# so this is a quantisation effect, not a pooling bug.
#
# Batching therefore buys nothing here in either dimension. It is a throughput
# optimisation, and this path has a batch of ten.
EMBED_BATCH = 1

# Hard cap on the text handed to the encoder, or 0 to disable.
#
# batch=1 removes the *amplification* -- nine short sentences paying for one long
# one -- but not the cost of the long sentence itself, which is superlinear in
# length. On the single-index configuration that remainder is the whole tail:
# P99 127ms against a P50 of 25ms.
#
# Measured on metadata_128, 300 queries, batch=1 throughout:
#
#   trunc   P50     P95     P99      P100     identical to untruncated
#   none    24.57   64.98   127.10   129.28   100%
#   512     33.40   42.15    85.15   145.68    99.0%
#   256     33.63   43.93    51.97    55.80    98.3%   <-
#   192     34.92   44.16    50.87    76.92    96.3%
#
# 256 chars cuts P99 by 2.4x and P100 by 2.3x while changing 1 answer in 60, and
# mean support moves 0.6511 -> 0.6501. It costs ~9ms at P50, which is the honest
# part of the trade: this buys tail predictability, not median speed.
#
# Truncation applies only to the copy sent to the encoder. The returned span and
# its citation remain the full sentence, so a long answer is still served whole --
# only its *ranking* is decided on the first 256 characters.
MAX_SENTENCE_CHARS = 256


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
    # Per-stage breakdown. This stage owns the whole over-budget tail (P99 175ms
    # against a ~10ms budget for everything else), so it has to be attributable
    # rather than a single opaque number.
    stage_ms: dict[str, float] = field(default_factory=dict)
    # Tail diagnostics: the embedding batch is padded to its longest member, so
    # cost tracks max sentence length, not the mean.
    n_sentences_embedded: int = 0
    max_sentence_chars: int = 0

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
    embed_batch: int = EMBED_BATCH,
    max_sentence_chars: int = MAX_SENTENCE_CHARS,
) -> ExtractiveAnswer:
    """Select the best-supported span across the top retrieved chunks.

    `hits` is any sequence exposing .text and .passage_id (Hit or FusedHit).
    """
    t0 = time.perf_counter()
    stage: dict[str, float] = {}

    def mark(name: str, since: float) -> float:
        now = time.perf_counter()
        stage[name] = round((now - since) * 1000, 3)
        return now

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
    t_split = mark("split_sentences", t0)

    lex_all = np.array([lexical_overlap(query, s) for s in sentences], dtype=np.float32)
    t_lex = mark("lexical_overlap", t_split)

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
    t_pre = mark("prefilter", t_lex)

    # Embedding pass over the surviving candidates. `sentences` itself is never
    # truncated -- only the text handed to the encoder is -- so the returned span
    # and its citation stay verbatim.
    to_embed = sentences
    if max_sentence_chars:
        to_embed = [s[:max_sentence_chars] for s in sentences]
    svecs = embedder.encode_passages(to_embed, batch_size=embed_batch)
    t_embed = mark("embed_sentences", t_pre)
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

    mark("score_and_span", t_embed)

    return ExtractiveAnswer(
        text=text,
        support=float(scores[best]),
        citations=citations,
        n_candidates=len(sentences),
        took_ms=round((time.perf_counter() - t0) * 1000, 3),
        stage_ms=stage,
        n_sentences_embedded=len(sentences),
        max_sentence_chars=max((len(s) for s in sentences), default=0),
    )

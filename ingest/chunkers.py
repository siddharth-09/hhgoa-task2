"""Four chunking strategies, indexed side by side so they can be compared.

The point is not to pick one up front -- it is to build all four, score each on
held-out queries with `is_selected` gold labels (see eval/), and let the router
choose per query at serve time.

    1. fixed        -- fixed token window + overlap. The baseline to beat.
    2. sentence     -- pack whole sentences up to a token budget.
    3. semantic     -- split where consecutive-sentence similarity drops.
    4. metadata     -- passage-native, tagged with query_type / lang / provenance.

Devanagari note: sentence segmentation cannot key on "." alone. Hindi and
Marathi terminate sentences with the danda (U+0964) and double danda (U+0965).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, asdict
from typing import Literal

import numpy as np

Strategy = Literal["fixed", "sentence", "semantic", "metadata"]

# Sentence terminators: danda, double danda, and Latin punctuation.
_SENT_END = re.compile(r"(?<=[।॥?!])\s+|(?<=[.!?])\s+(?=[A-Zऀ-ॿ])")
_WS = re.compile(r"\s+")


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    passage_id: str
    query_id: int
    text: str
    strategy: Strategy
    lang: str
    query_type: str
    n_tokens: int
    # Metadata-aware strategy folds these into the embedded text; the others
    # keep them as filterable sidecar fields only.
    translator: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class Passage:
    """One passage from the corpus, as emitted by ingest/download.py."""

    passage_id: str
    query_id: int
    text: str
    lang: str
    query_type: str
    translator: str | None = None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Devanagari-aware sentence split. Falls back to the whole string."""
    parts = [p.strip() for p in _SENT_END.split(_WS.sub(" ", text.strip())) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


class TokenCounter:
    """Wraps a HF tokenizer; falls back to whitespace when none is supplied.

    Token counts drive every window size here, so using the *same* tokenizer as
    the embedding model matters -- Devanagari fragments into far more subword
    pieces than Latin script, and a whitespace estimate understates it badly.
    """

    def __init__(self, tokenizer=None):
        self._tok = tokenizer

    def count(self, text: str) -> int:
        if self._tok is None:
            return len(text.split())
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def encode_ids(self, text: str) -> list[int]:
        if self._tok is None:
            return list(range(len(text.split())))
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        if self._tok is None:
            raise RuntimeError("decode requires a real tokenizer")
        return self._tok.decode(ids)


# --------------------------------------------------------------------------
# 1. fixed window + overlap
# --------------------------------------------------------------------------


def chunk_fixed(
    p: Passage,
    tc: TokenCounter,
    size: int = 256,
    overlap: int = 64,
) -> Iterator[Chunk]:
    """Sliding token window. Deliberately naive -- this is the control group."""
    ids = tc.encode_ids(p.text)
    if not ids:
        return
    step = max(1, size - overlap)
    for i, start in enumerate(range(0, len(ids), step)):
        window = ids[start : start + size]
        if not window:
            break
        text = tc.decode(window) if tc._tok else " ".join(p.text.split()[start : start + size])
        yield Chunk(
            chunk_id=f"{p.passage_id}#fx{i}",
            passage_id=p.passage_id,
            query_id=p.query_id,
            text=text,
            strategy="fixed",
            lang=p.lang,
            query_type=p.query_type,
            n_tokens=len(window),
            translator=p.translator,
        )
        if start + size >= len(ids):
            break


# --------------------------------------------------------------------------
# 2. sentence packing
# --------------------------------------------------------------------------


def chunk_sentence(
    p: Passage,
    tc: TokenCounter,
    max_tokens: int = 256,
    min_tokens: int = 32,
) -> Iterator[Chunk]:
    """Greedily pack whole sentences, never splitting mid-sentence.

    Short trailing fragments are merged back into the previous chunk rather than
    emitted alone -- a 5-token orphan chunk is noise in the index.
    """
    sents = split_sentences(p.text)
    if not sents:
        return

    buf: list[str] = []
    buf_tokens = 0
    out: list[tuple[str, int]] = []

    for s in sents:
        n = tc.count(s)
        if buf and buf_tokens + n > max_tokens:
            out.append((" ".join(buf), buf_tokens))
            buf, buf_tokens = [], 0
        buf.append(s)
        buf_tokens += n

    if buf:
        if out and buf_tokens < min_tokens:
            prev_text, prev_n = out[-1]
            out[-1] = (prev_text + " " + " ".join(buf), prev_n + buf_tokens)
        else:
            out.append((" ".join(buf), buf_tokens))

    for i, (text, n) in enumerate(out):
        yield Chunk(
            chunk_id=f"{p.passage_id}#sn{i}",
            passage_id=p.passage_id,
            query_id=p.query_id,
            text=text,
            strategy="sentence",
            lang=p.lang,
            query_type=p.query_type,
            n_tokens=n,
            translator=p.translator,
        )


# --------------------------------------------------------------------------
# 3. semantic (similarity-drift) splitting
# --------------------------------------------------------------------------


def chunk_semantic(
    p: Passage,
    tc: TokenCounter,
    embed_fn: Callable[[list[str]], np.ndarray],
    threshold: float = 0.62,
    max_tokens: int = 320,
) -> Iterator[Chunk]:
    """Split where meaning shifts, not where the token counter runs out.

    Embeds each sentence, walks the sequence, and cuts whenever cosine
    similarity between consecutive sentences falls below `threshold` (a topic
    boundary) or the running window exceeds `max_tokens` (a safety valve).

    `embed_fn` must return L2-normalised row vectors.
    """
    sents = split_sentences(p.text)
    if len(sents) <= 1:
        yield from chunk_sentence(p, tc, max_tokens=max_tokens)
        return

    vecs = embed_fn(sents)
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)  # normalised -> dot == cosine

    groups: list[list[str]] = [[sents[0]]]
    running = tc.count(sents[0])
    for i, s in enumerate(sents[1:]):
        n = tc.count(s)
        if sims[i] < threshold or running + n > max_tokens:
            groups.append([s])
            running = n
        else:
            groups[-1].append(s)
            running += n

    for i, g in enumerate(groups):
        text = " ".join(g)
        yield Chunk(
            chunk_id=f"{p.passage_id}#sm{i}",
            passage_id=p.passage_id,
            query_id=p.query_id,
            text=text,
            strategy="semantic",
            lang=p.lang,
            query_type=p.query_type,
            n_tokens=tc.count(text),
            translator=p.translator,
            extra={"n_sentences": len(g)},
        )


# --------------------------------------------------------------------------
# 4. metadata-aware
# --------------------------------------------------------------------------

# Short natural-language hints. These get embedded with the passage, so the
# query-side embedding of "how many..." lands nearer NUMERIC-tagged chunks.
_TYPE_HINT = {
    "NUMERIC": "numeric quantity measurement",
    "LOCATION": "place location geography",
    "PERSON": "person name individual",
    "ENTITY": "named entity organisation thing",
    "DESCRIPTION": "explanation description definition",
}


def chunk_metadata(
    p: Passage,
    tc: TokenCounter,
    max_tokens: int = 256,
) -> Iterator[Chunk]:
    """Passage-native granularity with metadata prepended into the embedded text.

    MSMARCO passages are already retrieval-sized, so the win here is not in
    where we cut but in *what we embed*: a query-type hint plus language tag
    gives the encoder signal that pure passage text does not carry.

    Provenance (`translator`) is kept as a filterable field, not embedded --
    it says nothing about the passage's meaning.
    """
    hint = _TYPE_HINT.get(p.query_type, "")
    body_budget = max_tokens - tc.count(hint) - 8

    sub_chunks = (
        [p.text]
        if tc.count(p.text) <= body_budget
        else [c.text for c in chunk_sentence(p, tc, max_tokens=body_budget)]
    )

    for i, body in enumerate(sub_chunks):
        text = f"[{p.query_type.lower()}] {hint} | {body}"
        yield Chunk(
            chunk_id=f"{p.passage_id}#md{i}",
            passage_id=p.passage_id,
            query_id=p.query_id,
            text=text,
            strategy="metadata",
            lang=p.lang,
            query_type=p.query_type,
            n_tokens=tc.count(text),
            translator=p.translator,
            extra={"raw_text": body, "hint": hint},
        )


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


def to_documents(passages: Iterable[Passage]) -> Iterator[Passage]:
    """Group passages by query_id into multi-passage documents.

    MSMARCO ships pre-segmented passages, so chunking them barely does anything --
    at a 256-token budget every strategy emits one chunk per passage and the
    comparison measures nothing. Concatenating the ~10 passages that share a
    query_id reconstructs a document-sized unit (~2000+ tokens), which is the
    setting real RAG chunking operates in and where strategies actually diverge.
    """
    # Grouped by (lang, query_id), not query_id alone: the corpus is parallel, so
    # every language shard repeats the same query_ids. Grouping on the id alone
    # would concatenate a Hindi passage and its Marathi translation into a single
    # mixed-script document.
    groups: dict[tuple[str, int], list[Passage]] = {}
    for p in passages:
        groups.setdefault((p.lang, p.query_id), []).append(p)

    for (lang, qid), ps in groups.items():
        first = ps[0]
        yield Passage(
            passage_id=f"{lang}:doc{qid}",
            query_id=qid,
            text=" ".join(p.text for p in ps if p.text and p.text.strip()),
            lang=first.lang,
            query_type=first.query_type,
            translator=first.translator,
        )


def chunk_all(
    passages: Iterable[Passage],
    tc: TokenCounter,
    strategy: Strategy,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    *,
    max_tokens: int = 256,
    overlap_ratio: float = 0.25,
) -> Iterator[Chunk]:
    """Chunk a passage stream.

    `max_tokens` is the dominant variable: published benchmarks favour ~512 with
    10-20% overlap for English, but Devanagari fragments into 2-3x more XLM-R
    subword tokens, so the effective budget must be read in tokens, not words.
    """
    if strategy == "semantic" and embed_fn is None:
        raise ValueError("semantic chunking requires embed_fn")

    overlap = int(max_tokens * overlap_ratio)

    for p in passages:
        if not p.text or not p.text.strip():
            continue
        if strategy == "fixed":
            yield from chunk_fixed(p, tc, size=max_tokens, overlap=overlap)
        elif strategy == "sentence":
            yield from chunk_sentence(p, tc, max_tokens=max_tokens)
        elif strategy == "semantic":
            yield from chunk_semantic(p, tc, embed_fn, max_tokens=max_tokens)  # type: ignore[arg-type]
        elif strategy == "metadata":
            yield from chunk_metadata(p, tc, max_tokens=max_tokens)
        else:
            raise ValueError(f"unknown strategy: {strategy}")

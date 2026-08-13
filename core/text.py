"""Tokenisation that does not destroy Indic script.

Python's `\\w` matches characters that are alphanumeric by Unicode class.
Devanagari vowel signs (ि ा ो ी), the virama (्) and the chandrabindu (ँ) are
combining marks -- category Mn/Mc -- and are *not* alphanumeric. So the obvious
`re.findall(r"\\w+", text)` shatters every Hindi word at its matras:

    दिल्ली    -> ['द', 'ल', 'ल']
    विस्फोट   -> ['व', 'स', 'फ', 'ट']
    ताजमहल    -> ['त', 'जमहल']
    Delhi     -> ['Delhi']            (English is unaffected, which is how this hides)

Consequences before this module existed:

  * bm25s used its default token_pattern `(?u)\\b\\w\\w+\\b`, so the sparse half
    of hybrid retrieval indexed consonant fragments for every Hindi passage.
  * the extractive scorer's lexical term compared those fragments, which
    over-match badly -- unrelated words share consonants constantly.
  * the guardrail's grounding check inherited the same problem, which is why a
    factually wrong answer ("ताजमहल दिल्ली में है ... अकबर") still scored 0.67.

The fix is to define tokens by their separators rather than by character class:
split on whitespace, danda/double danda, and ASCII punctuation, and keep
everything else intact. Script-agnostic, and correct for Latin too.
"""

from __future__ import annotations

import re

# Whitespace, Devanagari danda (U+0964) and double danda (U+0965), and ASCII
# punctuation. Everything not a separator is part of a token.
_SEPARATORS = re.compile(r"[\s।॥!-/:-@\[-`{-~‐-‧‰-⁞]+")

# bm25s takes a regex *pattern* rather than a splitter, so this is the inverse:
# runs of two or more non-separator characters.
BM25_TOKEN_PATTERN = r"[^\s।॥!-/:-@\[-`{-~‐-‧‰-⁞]{2,}"

MIN_TOKEN_LEN = 2


def tokenize(text: str, *, lower: bool = True, min_len: int = MIN_TOKEN_LEN) -> list[str]:
    """Split on separators, preserving whole words in any script."""
    if not text:
        return []
    s = text.lower() if lower else text
    return [t for t in _SEPARATORS.split(s) if len(t) >= min_len]


def token_set(text: str, *, min_len: int = MIN_TOKEN_LEN) -> set[str]:
    return set(tokenize(text, min_len=min_len))


def overlap(query: str, candidate: str, *, min_len: int = MIN_TOKEN_LEN) -> float:
    """Fraction of the query's tokens present in the candidate.

    Recall-oriented: we care whether the candidate covers what was asked, not
    whether it is concise about it.
    """
    q = token_set(query, min_len=min_len)
    if not q:
        return 0.0
    return len(q & token_set(candidate, min_len=min_len)) / len(q)

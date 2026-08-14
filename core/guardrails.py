"""Guardrails: decide when the system must not answer.

Requirement #6 asks the system to know when *not* to answer, not just how to
answer. Three failure modes, and they need different mechanisms:

  1. UNSAFE INPUT   -- the question should not be answered at all, regardless of
                       what the corpus contains (credential harvesting, self-harm,
                       weapon synthesis).
  2. UNGROUNDED     -- the corpus does not cover this, so any answer would be
                       invention.
  3. HALLUCINATION  -- the generated answer asserts things the retrieved context
                       does not support.

Why the input check cannot be a score threshold. Measured on the pilot index:

    "मेरे बैंक खाते का पासवर्ड क्या है?"  (what is my bank password)  support 0.596
    "मैनहट्टन परियोजना का प्रभाव?"        (legitimate question)       support 0.649

The unsafe query scored *higher* than plenty of legitimate ones, because MSMARCO
genuinely contains bank-security passages and retrieval did its job. Grounding
says "the corpus discusses this"; it says nothing about whether answering is
appropriate. So intent is checked before retrieval, independently.

On the pattern lists below: regexes are not a safety classifier, and this is a
first line rather than a complete one. They are deliberately narrow -- keyed on
first-person possessive intent ("my password", "मेरा पासवर्ड") rather than bare
topic words -- because the corpus legitimately contains passages about passwords,
weapons and medication, and a topic-word blocklist would refuse real questions.
Recall is traded for precision on purpose: a false refusal is a visible failure,
a false allow on this corpus is bounded by the fact that we only ever return
retrieved text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from core.text import token_set


class Decision(str, Enum):
    ALLOW = "allow"
    REFUSE_UNSAFE = "refuse_unsafe"
    ABSTAIN_UNGROUNDED = "abstain_ungrounded"
    ABSTAIN_EMPTY = "abstain_empty"
    GREETING = "greeting"


# Support score below which we treat the corpus as not covering the question.
# Calibrated against the pilot: legitimate answered queries scored 0.65-0.84.
MIN_SUPPORT = 0.45

# Fraction of the answer's content tokens that must appear in retrieved context
# before a generated answer is accepted as grounded.
MIN_ANSWER_OVERLAP = 0.55

# Novel content tokens tolerated before an answer is treated as invented.
MAX_NOVEL_FACTS = 2

_TOKEN = re.compile(r"\w+", re.UNICODE)

# First-person credential / PII solicitation. Requires the possessive, so
# "how do I choose a strong password" is not caught but "what is my password" is.
# The possessive and the keyword are often separated by several words --
# "मेरे बैंक खाते का पासवर्ड" puts three between them, and Hindi's postpositions
# make this the norm rather than the exception. An early version allowed only
# one intervening word and passed the Hindi form while catching the English one,
# which is the worst possible failure on an Indic-language system.
_CREDENTIAL = re.compile(
    r"(?:\b(?:my|mera|meri|mere)\b|मेरा|मेरी|मेरे)"
    r"(?:\W+\S+){0,4}\W+"
    r"(?:password|passwd|\bpin\b|\botp\b|\bcvv\b|पासवर्ड|पिन|ओटीपी|"
    r"account\s*(?:number|no)|खाता\s*(?:संख्या|नंबर)|aadhaar|आधार|"
    r"credit\s*card|क्रेडिट\s*कार्ड|debit\s*card|डेबिट\s*कार्ड)",
    re.IGNORECASE,
)

_SELF_HARM = re.compile(
    r"(kill\s+myself|suicide|end\s+my\s+life|self\s*harm|"
    r"आत्महत्या|खुदकुशी|खुद\s*को\s*मार)",
    re.IGNORECASE,
)

_WEAPONS = re.compile(
    r"(how\s+to\s+(make|build|construct)\s+a?\s*(bomb|explosive|gun|weapon)|"
    r"बम\s*(कैसे\s*)?बना|विस्फोटक\s*बना|हथियार\s*बना)",
    re.IGNORECASE,
)

# Attempts to talk to the model rather than ask the corpus a question.
_INJECTION = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)|"
    r"disregard\s+(your|the)\s+(instructions?|rules?)|"
    r"you\s+are\s+now\s+|system\s*prompt|"
    r"पिछले\s*निर्देश|निर्देशों?\s*को\s*अनदेखा)",
    re.IGNORECASE,
)

UNSAFE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("credential_solicitation", _CREDENTIAL),
    ("self_harm", _SELF_HARM),
    ("weapons", _WEAPONS),
]

REFUSAL_TEXT = {
    "credential_solicitation": (
        "मैं आपके व्यक्तिगत खाते या पासवर्ड की जानकारी नहीं दे सकता। "
        "I can't help with personal account credentials."
    ),
    "self_harm": (
        "मैं इस विषय पर मदद नहीं कर सकता, लेकिन कृपया किसी से बात करें। "
        "भारत में टेली-मानस हेल्पलाइन: 14416. "
        "I can't help with this, but please reach out — Tele-MANAS: 14416."
    ),
    "weapons": (
        "मैं इस बारे में जानकारी नहीं दे सकता। I can't provide information on this."
    ),
    "prompt_injection": (
        "मैं केवल उपलब्ध दस्तावेज़ों से प्रश्नों के उत्तर दे सकता हूँ। "
        "I can only answer questions from the indexed documents."
    ),
}

ABSTAIN_TEXT = (
    "मेरे पास इसका उत्तर देने के लिए पर्याप्त जानकारी नहीं है। "
    "I don't have enough information in my sources to answer that."
)

GREETING_TEXT = (
    "नमस्ते! मुझसे दस्तावेज़ों के बारे में कोई प्रश्न पूछिए। "
    "Hello! Ask me a question about the indexed documents — try “भारत की राजधानी क्या है?”"
)

# Greetings and other conversational openers are not questions, and retrieval
# treats them terribly. Measured: "Hello" scored **0.602 support** against a
# passage reading "CIL is more complex than C# ... System.Console.WriteLine(Hello,
# World!)" -- a lexical collision on the literal string "Hello". That cleared the
# 0.45 grounding gate and was served as an answer.
#
# The grounding gate cannot catch this, because the passage genuinely *is* in the
# corpus and genuinely *does* contain the token. The defect is upstream: a
# greeting has no informational intent to satisfy, so the right move is to answer
# conversationally and never spend retrieval at all. Requirement 6 calls this
# "off-topic queries".
#
# Deliberately narrow -- it matches only when the *entire* input is a greeting, so
# "hello, who built the Taj Mahal?" still retrieves normally.
_GREETING = re.compile(
    r"^\W*(hi|hey+|hello+|yo|hola|namaste|namaskar|नमस्ते|नमस्कार|हॅलो|हैलो|"
    r"good\s+(morning|afternoon|evening)|"
    r"how\s+are\s+you|what'?s\s+up|kaise?\s+ho|कैसे\s+हो|test|testing|"
    r"thanks?|thank\s+you|dhanyavaad|धन्यवाद|ok|okay|bye)"
    r"[\s!.?,]*$",
    re.I,
)


def _tokens(s: str) -> set[str]:
    # core.text, not re.findall(r"\w+"): \w drops Devanagari matras and would
    # reduce every Hindi word to consonant fragments (see core/text.py).
    return token_set(s, min_len=3)


@dataclass(slots=True)
class InputVerdict:
    decision: Decision
    reason: str = ""
    message: str = ""
    flags: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.decision is not Decision.ALLOW


@dataclass(slots=True)
class OutputVerdict:
    decision: Decision
    reason: str = ""
    message: str = ""
    grounding: float = 0.0
    support: float = 0.0
    novel_tokens: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.decision is not Decision.ALLOW


# Same vocabulary as _GREETING, but matched as a *prefix* followed by real
# content, so "hi, भारत की राजधानी क्या है?" retrieves on the question alone.
_GREETING_PREFIX = re.compile(
    r"^\W*(hi|hey+|hello+|yo|hola|namaste|namaskar|नमस्ते|नमस्कार|हॅलो|हैलो|"
    r"good\s+(morning|afternoon|evening)|ok|okay|so|um+|uh+)"
    r"[\s,!.\-—:]+(?=\S)",
    re.I,
)


def strip_greeting_prefix(query: str) -> str:
    """Drop a leading greeting so it doesn't dilute retrieval.

    Spoken questions routinely open with one, and every extra token shifts the
    query embedding away from the passage that answers it. Measured on the live
    deployment: "भारत की राजधानी क्या है?" answers with support 0.787, while
    "hello, भारत की राजधानी क्या है?" fell under the 0.45 gate and abstained --
    the same question, refused because of the hello.

    Only strips when substantive text follows, so a bare "hello" still reaches
    the greeting branch of check_input rather than becoming an empty query.
    """
    stripped = _GREETING_PREFIX.sub("", query or "", count=1)
    return stripped if len(stripped.strip()) >= 3 else (query or "")


def check_input(query: str, *, min_chars: int = 2) -> InputVerdict:
    """Run before retrieval. Cheap, deterministic, no model call."""
    q = (query or "").strip()
    if len(q) < min_chars:
        return InputVerdict(
            Decision.ABSTAIN_EMPTY,
            reason="empty_or_too_short",
            message="मुझे प्रश्न सुनाई नहीं दिया। I didn't catch that — could you repeat?",
        )

    # Checked after the length guard but before the unsafe patterns, so a bare
    # greeting never reaches retrieval. Unsafe intent still wins over a greeting
    # because the patterns below are matched on the full string.
    if _GREETING.match(q):
        return InputVerdict(
            Decision.GREETING,
            reason="greeting_not_a_question",
            message=GREETING_TEXT,
        )

    flags: list[str] = []
    for name, pattern in UNSAFE_PATTERNS:
        if pattern.search(q):
            flags.append(name)
            return InputVerdict(
                Decision.REFUSE_UNSAFE,
                reason=name,
                message=REFUSAL_TEXT[name],
                flags=flags,
            )

    if _INJECTION.search(q):
        return InputVerdict(
            Decision.REFUSE_UNSAFE,
            reason="prompt_injection",
            message=REFUSAL_TEXT["prompt_injection"],
            flags=["prompt_injection"],
        )

    return InputVerdict(Decision.ALLOW)


def grounding_score(answer: str, contexts: list[str]) -> tuple[float, list[str]]:
    """(overlap fraction, novel content tokens) for an answer against its context.

    Runs in microseconds and needs no second model, which is what keeps it
    inside the latency budget.

    Measured limitation, not a theoretical one: the answer
    "ताजमहल दिल्ली में है और इसे अकबर ने बनवाया" (wrong city, wrong emperor)
    scores 0.67 overlap against correct context, because only two of its tokens
    are fabricated and the rest are shared. Overlap alone therefore cannot
    separate a false recombination from a true paraphrase.

    So the novel tokens are returned as well. Numbers and long content words
    absent from every retrieved chunk are the highest-signal hallucinations --
    a date or a name the sources never mention cannot have come from them.
    Proper entailment checking would need an NLI model the budget cannot absorb;
    this is the honest approximation.
    """
    a = _tokens(answer)
    if not a:
        return 0.0, []
    ctx: set[str] = set()
    for c in contexts:
        ctx |= _tokens(c)
    novel = sorted(a - ctx)
    return len(a & ctx) / len(a), novel


# Hindi inflects by suffix, so a faithful paraphrase of the context produces
# surface forms that exact-match nothing: लाभ (benefit) -> लाभों (of benefits),
# लिखना -> लिखा. An exact-match novelty check reads that grammar as invention.
# Measured: a correct generated answer was rejected for
# novel_facts(जिनमें, लाभों, शामिल) -- two connectives and one inflection.
#
# Prefix matching handles it: if a context token shares a long enough prefix with
# the candidate, treat it as a morphological variant rather than a new fact.
# Numbers are exempt -- "1653" and "1888" share no prefix and must never be
# excused as inflection, which is exactly the hallucination we most need to catch.
_INFLECTION_PREFIX = 4

# Prefix matching still flagged जाते / रहते / जिनमें -- inflected forms of common
# verbs and connectives whose stems differ from anything in the context. They
# carry no factual claim, so they cannot be a hallucination; a proper-noun of the
# same length (अकबर) can, which is why length alone cannot separate them.
# A small closed-class list does, and costs nothing at runtime.
_FUNCTION_WORDS = {
    # Hindi verbs/auxiliaries and connectives, in the inflected forms that recur
    "जाते", "जाता", "जाती", "जाना", "रहते", "रहता", "रहती", "रहना",
    "होते", "होता", "होती", "होना", "करते", "करता", "करती", "करना",
    "किया", "किये", "गया", "गये", "गयी", "दिया", "लिया", "सकते", "सकता", "सकती",
    "जिनमें", "जिसमें", "जिसके", "जिनके", "जिससे", "इसमें", "उसमें", "इसके", "उसके",
    "शामिल", "अनुसार", "द्वारा", "लेकिन", "इसलिए", "क्योंकि", "अथवा", "तथा", "एवं",
    "आदि", "यहाँ", "वहाँ", "जहाँ", "कहाँ", "बहुत", "अधिक", "सबसे", "कुछ", "कोई",
    # Marathi. The list above was built while the corpus was Hindi-only, and the
    # omission is not cosmetic: Marathi is half the index, and its connectives
    # were being counted as invented facts. Measured on 20 live queries, a
    # correct Marathi answer was rejected for novel_facts(किंवा, ज्यामध्ये) --
    # "or" and "in which". The verifier was penalising the language itself.
    "आणि", "किंवा", "म्हणून", "म्हणजे", "परंतु", "पण", "तसेच", "तथापि",
    "आहे", "आहेत", "होते", "होता", "होती", "नाही", "असे", "अशा", "असते", "असतो",
    "करतात", "करते", "करतो", "केले", "केली", "गेले", "गेली", "शकते", "शकतात",
    "ज्यामध्ये", "यामध्ये", "त्यामध्ये", "ज्यांना", "त्यांना", "यांचा", "त्यांचा",
    "मध्ये", "वरून", "पासून", "साठी", "नुसार", "बद्दल", "समाविष्ट", "इत्यादी",
    "येथे", "तेथे", "कोठे", "खूप", "जास्त", "काही", "सर्व",
    # English function words that survive the length filter
    "which", "that", "there", "these", "those", "their", "about", "would",
    "could", "should", "been", "were", "with", "from", "into", "also", "such",
}


def _has_novel_facts(novel: list[str], context_tokens: set[str]) -> list[str]:
    """Novel tokens that look like asserted facts rather than grammar."""
    out: list[str] = []
    for t in novel:
        if any(ch.isdigit() for ch in t):
            out.append(t)  # numerals are never inflection
            continue
        if len(t) <= 3 or t in _FUNCTION_WORDS:
            continue  # grammar, not an asserted fact
        stem = t[:_INFLECTION_PREFIX]
        if any(c.startswith(stem) or t.startswith(c[:_INFLECTION_PREFIX]) for c in context_tokens):
            continue  # morphological variant of something in context
        out.append(t)
    return out


def check_output(
    answer: str,
    contexts: list[str],
    support: float,
    *,
    min_support: float = MIN_SUPPORT,
    min_overlap: float = MIN_ANSWER_OVERLAP,
    max_novel_facts: int = MAX_NOVEL_FACTS,
) -> OutputVerdict:
    """Run after retrieval/generation. Abstain rather than emit an unsupported answer."""
    if not (answer or "").strip():
        return OutputVerdict(
            Decision.ABSTAIN_UNGROUNDED,
            reason="empty_answer",
            message=ABSTAIN_TEXT,
            support=support,
        )

    if support < min_support:
        return OutputVerdict(
            Decision.ABSTAIN_UNGROUNDED,
            reason=f"low_support({support:.3f}<{min_support})",
            message=ABSTAIN_TEXT,
            support=support,
        )

    g, novel = grounding_score(answer, contexts)
    if g < min_overlap:
        return OutputVerdict(
            Decision.ABSTAIN_UNGROUNDED,
            reason=f"low_grounding({g:.3f}<{min_overlap})",
            message=ABSTAIN_TEXT,
            grounding=g,
            support=support,
            novel_tokens=novel,
        )

    # Overlap can clear the bar while the answer still asserts facts the sources
    # never mention. Two or more novel content tokens is the practical cutoff:
    # one is usually inflection or a connective, several means invention.
    ctx_tokens: set[str] = set()
    for c in contexts:
        ctx_tokens |= _tokens(c)
    facts = _has_novel_facts(novel, ctx_tokens)
    if len(facts) >= max_novel_facts:
        return OutputVerdict(
            Decision.ABSTAIN_UNGROUNDED,
            reason=f"novel_facts({','.join(facts[:4])})",
            message=ABSTAIN_TEXT,
            grounding=g,
            support=support,
            novel_tokens=facts,
        )

    return OutputVerdict(Decision.ALLOW, grounding=g, support=support, novel_tokens=novel)

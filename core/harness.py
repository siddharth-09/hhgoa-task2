"""The harness: one orchestrated call from question to verified answer.

Requirement #5 asks for "structured orchestration around the model (tool calls,
retries, structured input/output handling, error recovery) rather than a single
raw prompt-in, text-out call". That is this module.

Pipeline, in order, with every stage timed:

    check_input        guardrail -- refuse unsafe intent before spending anything
    embed_query        local, ~2-5ms
    retrieve           ensemble fan-out, ~3ms
    extract            best-supported span, no LLM        <- FAST ANSWER, <200ms
    check_output       grounding gate -- abstain if unsupported
    generate           LLM rewrite, cited                 <- QUALITY ANSWER, ~1.3s
    verify             grounding gate again, on the generated text

Two design points worth stating plainly.

**The extractive answer is produced before generation and never depends on it.**
That is what makes the sub-200ms claim measurable: the fast answer is real,
grounded and citable on its own. Generation is an enhancement layered on top,
and if it times out, errors, returns malformed JSON, or produces something the
grounding check rejects, the extractive answer stands. Error recovery here is
not a try/except -- it is a second, already-computed answer.

**Guardrails run twice, on both sides of generation.** The input check is intent,
before retrieval. The output check is grounding, applied to the extractive span
*and* independently to the generated text. A prompt instructing the model to stay
grounded is a request; the post-check is the constraint.

Nothing raises. `AnswerResult.decision` says what happened and why.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.embedder import Embedder, EmbedderConfig
from core.extractive import ExtractiveAnswer, extract_answer
from core.guardrails import ABSTAIN_TEXT as ABSTAIN_MESSAGE
from core.guardrails import Decision, check_input, check_output, strip_greeting_prefix
from core.llm import GenerationResult, LLMChain, LLMClient
from core.retriever import AdaptiveRetriever

# Upper bound on the question text, in characters.
#
# The extractive path caps the *passage* sentences it embeds, but the query was
# unbounded -- and the query is embedded too. Measured on 300 sampled corpus
# queries: median 32 chars, P99 102 chars, and a single outlier at **2,717**,
# a machine-translation artifact where one phrase repeats for the whole string.
#
# That one query was the entire P100. It forces a full 512-token forward pass
# (the tokenizer's truncation limit) where a real question needs about twenty,
# and it showed up as embed_query P99 6ms / P100 143ms -- a 23x cliff caused by
# one row of the corpus.
#
# 512 characters is far above any real spoken question: STT caps audio at 30
# seconds, which is perhaps 300 characters of speech. So this bounds the tail
# without touching a single genuine query.
MAX_QUERY_CHARS = 512


@dataclass(slots=True)
class Source:
    unit_id: str
    text: str
    score: float
    contributors: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class AnswerResult:
    question: str
    answer: str = ""
    decision: str = ""
    reason: str = ""

    # Both tiers are reported so the demo can show the fast answer, then the
    # refined one, and the benchmark can measure them separately.
    extractive_answer: str = ""
    generated_answer: str = ""
    answer_source: str = ""  # "extractive" | "generated" | "refusal" | "abstain"

    support: float = 0.0
    grounding: float = 0.0
    sources: list[Source] = field(default_factory=list)
    citations: list[int] = field(default_factory=list)
    retrieval_provenance: dict[str, int] = field(default_factory=dict)

    timings_ms: dict[str, float] = field(default_factory=dict)
    fast_path_ms: float = 0.0  # question -> extractive answer (the <200ms number)
    total_ms: float = 0.0

    llm_ok: bool = False
    llm_error: str = ""
    llm_attempts: int = 0

    @property
    def answered(self) -> bool:
        return self.decision == Decision.ALLOW.value

    def to_dict(self) -> dict:
        return asdict(self)


class RAGHarness:
    def __init__(
        self,
        index_root: Path,
        ensemble: list[str] | None = None,
        llm: LLMClient | LLMChain | None = None,
        embedder: Embedder | None = None,
        *,
        threads: int = 0,
        top_k: int = 10,
        context_passages: int = 4,
        retriever: AdaptiveRetriever | None = None,
    ):
        self.embedder = embedder or Embedder(EmbedderConfig(threads=threads))
        # An injected retriever lets several harnesses share one set of loaded
        # indexes. The API serves one configuration but compares all of them side
        # by side, and loading each subset separately would hold the same 2GB of
        # index several times over.
        self.retriever = retriever or AdaptiveRetriever.load(index_root, ensemble)
        self.llm = llm or LLMChain.from_env()
        self.top_k = top_k
        self.context_passages = context_passages

    def warm(self) -> None:
        """Touch every hot-path component once.

        Without this the first real query pays ONNX graph init and HNSW page
        faults, and lands in the benchmark as P100 -- reporting warmup as
        latency. Called at API startup.
        """
        qv = self.embedder.encode_query("warmup")
        res = self.retriever.search(qv, "warmup", k=self.top_k)
        extract_answer("warmup", qv, res.hits, self.embedder)

    def answer(self, question: str, *, generate: bool = True) -> AnswerResult:
        t_start = time.perf_counter()
        t: dict[str, float] = {}
        r = AnswerResult(question=question)

        def mark(name: str, since: float) -> float:
            now = time.perf_counter()
            t[name] = round((now - since) * 1000, 3)
            return now

        # 1. Input guardrail -- before any compute is spent.
        t0 = time.perf_counter()
        verdict = check_input(question)
        t1 = mark("guardrail_in", t0)
        if verdict.blocked:
            r.answer = verdict.message
            r.decision = verdict.decision.value
            r.reason = verdict.reason
            # A greeting is not a refusal -- nothing was denied, there was simply
            # no question to answer. Collapsing the two would make the UI report
            # a blocked query every time someone says hello.
            r.answer_source = (
                "greeting" if verdict.decision is Decision.GREETING else "refusal"
            )
            r.timings_ms = t
            r.total_ms = r.fast_path_ms = round((time.perf_counter() - t_start) * 1000, 3)
            return r

        # 2-4. Fast path: embed -> retrieve -> extract.
        # Normalised here rather than inside the embedder so that retrieval,
        # lexical matching and extraction all see the same question text.
        question = strip_greeting_prefix(question)
        if len(question) > MAX_QUERY_CHARS:
            question = question[:MAX_QUERY_CHARS]

        qvec = self.embedder.encode_query(question)
        t2 = mark("embed_query", t1)

        retrieval = self.retriever.search(qvec, question, k=self.top_k)
        t3 = mark("retrieve", t2)
        r.retrieval_provenance = retrieval.provenance()
        r.sources = [
            Source(h.unit_id, h.text, round(h.score, 5), h.contributors)
            for h in retrieval.hits[: self.context_passages]
        ]

        extracted: ExtractiveAnswer = extract_answer(question, qvec, retrieval.hits, self.embedder)
        t4 = mark("extract", t3)
        r.extractive_answer = extracted.text
        r.support = round(extracted.support, 4)

        contexts = [s.text for s in r.sources]

        # 5. Output guardrail on the extractive span. If the corpus does not
        # support an answer, abstain here -- do not pay for generation, and do
        # not give a model the chance to improvise over weak context.
        out = check_output(extracted.text, contexts, extracted.support)
        t5 = mark("guardrail_out", t4)
        r.grounding = round(out.grounding, 4)
        r.fast_path_ms = round((time.perf_counter() - t_start) * 1000, 3)

        if out.blocked:
            r.answer = out.message
            r.decision = out.decision.value
            r.reason = out.reason
            r.answer_source = "abstain"
            r.timings_ms = t
            r.total_ms = r.fast_path_ms
            return r

        # The fast answer is valid from here on. Everything below can only
        # improve it, never remove it.
        r.answer = extracted.text
        r.answer_source = "extractive"
        r.decision = Decision.ALLOW.value

        if not generate:
            r.timings_ms = t
            r.total_ms = r.fast_path_ms
            return r

        # 6. Generation -- retries and timeout live in LLMClient; it returns a
        # typed result rather than raising.
        gen: GenerationResult = self.llm.generate(question, contexts)
        t6 = mark("generate", t5)
        r.llm_ok = gen.ok
        r.llm_error = gen.error
        r.llm_attempts = gen.attempts

        if gen.ok and gen.sufficient and gen.answer:
            # 7. Verify the *generated* text independently. The model was asked
            # to stay grounded; this checks that it did.
            gver = check_output(gen.answer, contexts, extracted.support)
            mark("verify", t6)
            if not gver.blocked:
                r.generated_answer = gen.answer
                r.answer = gen.answer
                r.answer_source = "generated"
                r.citations = gen.citations
                r.grounding = round(gver.grounding, 4)
            else:
                # Generated text failed verification -> keep the extractive span.
                r.reason = f"generation_rejected:{gver.reason}"
        elif gen.ok and not gen.sufficient:
            # The model read the *same* context and judged it inadequate. Keeping
            # the extractive span here treats that judgment as noise -- and the
            # span is drawn from exactly the context the model just rejected.
            #
            # Measured case that forced this: "Hello" retrieved a passage reading
            # "CIL is more complex than C# ... System.Console.WriteLine(Hello,
            # World!)". Lexical overlap on the literal token put support at 0.602,
            # clearing the 0.45 gate, so the span was served as an answer while
            # the LLM was correctly reporting the context did not answer anything.
            #
            # Two independent checks disagreeing is not a tie to be broken in
            # favour of the weaker one. The grounding gate measures token overlap;
            # the model measures whether the question is actually answered. When
            # the stronger signal says no, abstain.
            r.answer = ABSTAIN_MESSAGE
            r.answer_source = "abstain"
            r.decision = Decision.ABSTAIN_UNGROUNDED.value
            r.reason = "llm_reported_insufficient"
        elif not gen.ok:
            r.reason = f"llm_failed:{gen.error[:120]}"

        r.timings_ms = t
        r.total_ms = round((time.perf_counter() - t_start) * 1000, 3)
        return r

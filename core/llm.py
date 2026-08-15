"""Provider-agnostic LLM client with structured output, timeouts and retries.

Requirement #5 asks for "structured orchestration around the model (tool calls,
retries, structured input/output handling, error recovery) rather than a single
raw prompt-in, text-out call". This module is the structured-I/O and
error-recovery half of that; `core/harness.py` sequences it.

Two providers behind one interface, selected by LLM_PROVIDER. Not gratuitous:
the generation step is the only part of the pipeline that leaves the machine, so
it is the only part that can fail for reasons we do not control. Being able to
swap provider with one env var is error recovery at the deployment level, and it
keeps the demo alive if a key expires the night before submission.

Everything returns a typed object. The caller never sees raw text, and never
sees an exception it cannot act on -- `GenerationResult.ok` is the single check.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

# The model is instructed to answer *only* from context and to say so when it
# cannot. This is the generation-side half of the guardrail: core/guardrails.py
# verifies the output independently, because a prompt is a request, not a
# constraint.
SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided.

Rules:
- Use only facts stated in the context. Never add outside knowledge.
- If the context does not answer the question, set "answer" to "" and "sufficient" to false.
- Answer in the SAME language AND THE SAME SCRIPT as the question.
  A Devanagari question gets a Devanagari answer. Never mix scripts: write ने, not نے.
  Prefer wording that appears in the context over your own paraphrase.
- Answer in a COMPLETE sentence that restates what was asked, not a bare fragment.
  For "भारत की राजधानी क्या है?" answer "भारत की राजधानी नई दिल्ली है।", not "नई दिल्ली".
  A spoken answer has no question on screen beside it, so a fragment loses its meaning.
- Keep it to one or two sentences.
- Cite the passage numbers you used in "citations".

Return ONLY a JSON object, no markdown fence:
{"answer": str, "sufficient": bool, "citations": [int]}"""

# Used ONLY when retrieval found nothing and the system has already abstained.
# The answer it produces is never merged into the grounded answer, never cited,
# and never scored for grounding -- it is surfaced separately and labelled, so a
# reader can always tell corpus-backed text from model recall. Requirement 6 is
# about not passing off ungrounded text as grounded; keeping the two visibly
# apart is how this stays on the right side of that line.
UNSOURCED_PROMPT = """Answer the question from your own general knowledge.

Rules:
- Answer in the SAME language AND SCRIPT as the question. Never mix scripts.
- One or two sentences, a complete sentence, not a fragment.
- If you are genuinely unsure, set "sufficient" to false.

Return ONLY a JSON object, no markdown fence:
{"answer": str, "sufficient": bool, "citations": []}"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(slots=True)
class GenerationResult:
    answer: str = ""
    sufficient: bool = False
    citations: list[int] = field(default_factory=list)
    ok: bool = False
    error: str = ""
    provider: str = ""
    model: str = ""
    attempts: int = 0
    took_ms: float = 0.0
    raw: str = ""


def build_prompt(question: str, contexts: list[str], max_chars: int = 700) -> str:
    numbered = "\n\n".join(f"[{i + 1}] {c[:max_chars]}" for i, c in enumerate(contexts))
    return f"CONTEXT:\n{numbered}\n\nQUESTION: {question}"


def _parse(text: str) -> dict[str, Any]:
    """Extract the JSON object from a model response.

    Models wrap JSON in prose or code fences even when told not to, and a
    ValueError here would surface as a 500 rather than a degraded answer. So we
    locate the outermost object rather than trusting the whole response to parse.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = _JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass  # a JSON-ish block that does not parse -- fall through

    if not text:
        raise ValueError("empty response")

    # The model answered in prose instead of JSON. Raising here discarded a
    # perfectly usable answer and then burned the retry budget re-asking: measured
    # on 20 live queries, 2 failed this way and one spent **15.5 seconds** doing
    # it. Prose is a formatting miss, not a refusal.
    #
    # Salvaging it is safe because nothing downstream trusts this text -- the
    # grounding gate still verifies it against the retrieved context and rejects
    # it if unsupported. Citations are dropped rather than guessed, since an
    # invented citation is worse than none.
    return {"answer": text, "sufficient": True, "citations": [], "recovered_from_prose": True}


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
        # Gemini rejects short deadlines outright ("Manually set deadline 8s is too
        # short"), so this floor is a provider constraint, not a preference. The
        # harness does not wait on it anyway -- the extractive answer is already
        # returned by then, and generation replaces it when it lands.
        self.timeout_s = timeout_s or float(os.getenv("LLM_TIMEOUT_S", "30"))
        self._client: Any = None
        self._system: str = SYSTEM_PROMPT

        if self.provider == "gemini":
            # "-latest" aliases, not pinned versions: models.list() happily returns
            # ids that 404 on generateContent ("no longer available"), so listing is
            # not a capability check. gemini-flash-lite-latest measured 1529ms vs
            # 4029ms for gemini-flash-latest on identical correct output.
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
            self.api_key = os.getenv("GEMINI_API_KEY", "")
        elif self.provider == "anthropic":
            self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
            self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        elif self.provider == "openrouter":
            # OpenAI-compatible, so httpx is enough -- no extra SDK. One key
            # fronts ~500 models, which makes it the cheapest insurance against
            # a single provider rate-limiting us the night before submission.
            self.model = model or os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
            self.api_key = os.getenv("OPENROUTER_API_KEY", "")
            self.base_url = "https://openrouter.ai/api/v1"
        elif self.provider == "nvidia":
            # NVIDIA NIM speaks the same OpenAI dialect, so it reuses the same
            # transport -- only base_url and the key differ.
            self.model = model or os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
            self.api_key = os.getenv("NVIDIA_API_KEY", "")
            self.base_url = "https://integrate.api.nvidia.com/v1"
        elif self.provider == "groq":
            # Also OpenAI-dialect. Worth having in the chain for a reason the
            # others do not cover: Groq runs on LPUs and returns in a few hundred
            # ms, so it degrades the *quality* tier far less than a slower
            # fallback would. Free tier is quota-limited per minute and per day
            # (30 req/min on the models we use), which is exactly what the
            # cooldown logic in LLMChain exists to handle.
            #
            # Default is llama-3.3-70b-versatile: of the free Groq models it has
            # the strongest Devanagari output, and this corpus is Hindi/Marathi.
            self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            self.api_key = os.getenv("GROQ_API_KEY", "")
            self.base_url = "https://api.groq.com/openai/v1"
        elif self.provider == "bedrock":
            # Bedrock authenticates with the ambient AWS credential chain rather
            # than an API key, so `configured` is decided by boto3 finding
            # credentials, not by an env var being set.
            #
            # Note the "global." prefix: newer Anthropic models on Bedrock must be
            # invoked through an *inference profile*, not a bare model id. Passing
            # the raw id returns ValidationException "Operation not allowed", which
            # reads like a permissions problem and is not one.
            # `aws bedrock list-inference-profiles` lists the valid ids.
            self.model = model or os.getenv(
                "BEDROCK_MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0"
            )
            self.region = os.getenv("BEDROCK_REGION", os.getenv("AWS_DEFAULT_REGION", "ap-south-1"))
            self.api_key = "aws-credential-chain"
        else:
            raise ValueError(f"unknown LLM_PROVIDER {self.provider!r}")

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    # -- providers ---------------------------------------------------------

    def _gemini(self, prompt: str, max_tokens: int) -> str:
        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)

        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self._system,
                max_output_tokens=max_tokens,
                temperature=0.0,  # deterministic: this is extraction, not writing
                response_mime_type="application/json",
                http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)),
            ),
        )
        return resp.text or ""

    def _openai_compatible(self, prompt: str, max_tokens: int) -> str:
        """Shared transport for every OpenAI-dialect endpoint (OpenRouter, NVIDIA NIM, Groq).

        Only base_url and the key differ between them, so one method covers both
        and any future provider that speaks the same protocol. httpx rather than
        the openai SDK: it is already a dependency, and the surface we use here is
        one POST.
        """
        import httpx

        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s)

        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.provider == "openrouter":
            # Attribution headers; harmless elsewhere but only meaningful here.
            headers |= {
                "HTTP-Referer": "https://github.com/hhgoa-task2",
                "X-Title": "HH Goa Task2 Voice RAG",
            }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        # Not every NIM model implements response_format; the prompt already
        # demands JSON and _parse() tolerates prose around it. OpenRouter and
        # Groq both honour it, so ask for JSON where it is supported.
        if self.provider in ("openrouter", "groq"):
            payload["response_format"] = {"type": "json_object"}

        resp = self._client.post(
            f"{self.base_url}/chat/completions", headers=headers, json=payload
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        if "choices" not in body:
            raise RuntimeError(f"no choices in response: {str(body)[:300]}")
        return body["choices"][0]["message"]["content"] or ""

    def _bedrock(self, prompt: str, max_tokens: int) -> str:
        """Bedrock Converse API -- one shape across every model family it hosts."""
        import boto3
        from botocore.config import Config

        if self._client is None:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=Config(
                    read_timeout=self.timeout_s,
                    connect_timeout=min(self.timeout_s, 3),
                    retries={"max_attempts": 1},  # we do our own, with classification
                ),
            )

        resp = self._client.converse(
            modelId=self.model,
            system=[{"text": self._system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
        )
        return "".join(
            b.get("text", "") for b in resp["output"]["message"]["content"]
        )

    def _anthropic(self, prompt: str, max_tokens: int) -> str:
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout_s)

        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            system=self._system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    # -- public ------------------------------------------------------------

    def generate(
        self,
        question: str,
        contexts: list[str],
        *,
        max_tokens: int = 300,
        retries: int = 2,
        backoff_s: float = 0.4,
        rate_limit_backoff_s: float = 3.0,
        system: str | None = None,
    ) -> GenerationResult:
        # Per-call system prompt. Defaults to the grounded contract; the unsourced
        # path passes UNSOURCED_PROMPT explicitly and nothing else may.
        self._system = system or SYSTEM_PROMPT
        """Generate a grounded answer. Never raises -- check `.ok`.

        Retries cover transient faults (timeout, 429, 5xx) and malformed JSON.
        They deliberately do not cover a missing key or an unknown model: those
        will fail identically every time, and burning the retry budget on them
        just delays the fallback to the extractive answer.
        """
        t0 = time.perf_counter()
        base = GenerationResult(provider=self.provider, model=self.model)

        if not self.configured:
            base.error = f"{self.provider}: no credentials configured"
            base.took_ms = round((time.perf_counter() - t0) * 1000, 2)
            return base
        # Guards the *grounded* path: generating from zero passages there would be
        # ungrounded by construction. The unsourced path has no context by
        # definition, and says so via its own system prompt.
        if not contexts and self._system is not UNSOURCED_PROMPT:
            base.error = "no context passages"
            base.took_ms = round((time.perf_counter() - t0) * 1000, 2)
            return base

        prompt = build_prompt(question, contexts) if contexts else f"QUESTION: {question}"
        call = {
            "gemini": self._gemini,
            "openrouter": self._openai_compatible,
            "nvidia": self._openai_compatible,
            "groq": self._openai_compatible,
            "bedrock": self._bedrock,
            "anthropic": self._anthropic,
        }[self.provider]
        last_err = ""

        for attempt in range(1, retries + 2):
            base.attempts = attempt
            try:
                raw = call(prompt, max_tokens)
                data = _parse(raw)
                base.raw = raw[:2000]
                base.answer = str(data.get("answer", "") or "").strip()
                base.sufficient = bool(data.get("sufficient", False))
                base.citations = [int(c) for c in data.get("citations", []) if str(c).isdigit()]
                base.ok = True
                base.took_ms = round((time.perf_counter() - t0) * 1000, 2)
                return base
            except Exception as e:  # noqa: BLE001 -- classify, never propagate
                last_err = f"{type(e).__name__}: {e}"
                msg = str(e).lower()
                fatal = any(
                    s in msg
                    for s in (
                        "api key",
                        "unauthorized",
                        "permission",
                        "not found",
                        "invalid model",
                        "accessdenied",
                        "could not find credentials",
                        "don't have access to the model",
                        # Bedrock returns this for a malformed request or a model
                        # that needs an inference profile -- never transient.
                        "validationexception",
                        "operation not allowed",
                        # Gemini spells it NOT_FOUND / INVALID_ARGUMENT, which the
                        # space-separated forms above miss. A retired model burned
                        # 3 attempts and 5.8s before this was added.
                        "not_found",
                        "invalid_argument",
                        "no longer available",
                        "deadline",
                    )
                )
                if fatal or attempt > retries:
                    break
                # Free-tier quotas are per-minute, so 0.4s is useless against a
                # 429 -- it just burns the retry budget. Measured on Gemini's
                # free tier: 19 of 40 queries hit RESOURCE_EXHAUSTED.
                rate_limited = "429" in msg or "resource_exhausted" in msg or "quota" in msg
                delay = (rate_limit_backoff_s if rate_limited else backoff_s) * attempt
                time.sleep(delay)

        base.error = last_err
        base.took_ms = round((time.perf_counter() - t0) * 1000, 2)
        return base


class LLMChain:
    """Try providers in order; the first success wins.

    Motivated by measurement, not theory: on Gemini's free tier, 19 of 40 queries
    returned 429 RESOURCE_EXHAUSTED. The harness already degrades to the extractive
    answer when generation fails, but a fluent answer is better than a span, and a
    second provider is cheap insurance -- especially during a judged demo window
    where a quota reset is not something we control.

    Order matters and is measured, not assumed:

        gemini-flash-lite-latest (direct)   1542ms   <- primary, fastest
        google/gemma-4-31b-it:free          2150ms   <- fallback, no quota shared

    The fallback deliberately uses a *different vendor path*. A second Gemini model
    behind the same key would share the same quota and fail at the same moment.
    """

    # Cooldowns after a failure, so a dead provider is skipped rather than
    # re-tried on every subsequent query. Measured motivation: with no memory,
    # a degraded chain spent 7.5s per query walking two throttled providers
    # before reaching a working one -- every single time.
    RATE_LIMIT_COOLDOWN_S = 60.0  # quotas are per-minute; retry after one
    FATAL_COOLDOWN_S = 600.0  # retired model / bad key: will not fix itself soon
    TRANSIENT_COOLDOWN_S = 15.0

    def __init__(self, clients: list[LLMClient]):
        self.clients = [c for c in clients if c.configured]
        if not self.clients:
            raise ValueError("no configured LLM clients")
        # index -> unix ts before which this client is skipped
        self._cooldown_until: dict[int, float] = {}
        self._served: dict[str, int] = {}
        self._skipped: dict[str, int] = {}

    @classmethod
    def from_env(cls) -> LLMChain:
        """Primary from LLM_PROVIDER, then LLM_FALLBACK_CHAIN as provider:model pairs.

        Ordered by measured latency, and deliberately spanning *vendors* rather
        than models. Free tiers throttle, and a second model behind the same key
        shares the same quota -- it fails at the same moment the first one does.
        Three vendors means three independent pools.

            gemini   gemini-flash-lite-latest        1542ms   primary
            openrouter google/gemma-4-31b-it:free    2150ms
            openrouter google/gemma-4-26b-a4b-it:free 3154ms
            nvidia   meta/muse-glimmer-30b           4261ms   last resort

        Anything unconfigured is skipped silently, so the chain adapts to whichever
        keys a given deployment actually has.
        """
        chain = [LLMClient()]
        # Ordered by measurement (eval/compare_llms.py), and spanning vendors:
        # a same-vendor fallback shares the quota that just failed.
        #
        #   groq   llama-3.3-70b-versatile   219-420ms  primary, full sentences
        #   gemini gemini-flash-lite-latest  962-1836ms different vendor
        #   groq   openai/gpt-oss-20b        672-687ms  different model family
        #   openrouter gemma-4-26b:free      6.7-9.0s   slow, but a 4th vendor
        #
        # Dropped after measuring: qwen3.6-27b (HTTP 400 on Marathi) and
        # nvidia/muse-glimmer-30b (empty response on Marathi). Half this corpus is
        # Marathi, so a model that fails on it is not a fallback.
        spec = os.getenv(
            "LLM_FALLBACK_CHAIN",
            "gemini:gemini-flash-lite-latest,"
            "groq:openai/gpt-oss-20b,"
            "openrouter:google/gemma-4-26b-a4b-it:free",
        )
        for entry in (e.strip() for e in spec.split(",") if e.strip()):
            provider, _, model = entry.partition(":")
            if not model:
                continue
            try:
                c = LLMClient(provider=provider.strip(), model=model.strip())
            except ValueError:
                continue
            if c.configured:
                chain.append(c)
        return cls(chain)

    @property
    def provider(self) -> str:
        return "+".join(c.provider for c in self.clients)

    @property
    def model(self) -> str:
        return " -> ".join(f"{c.provider}:{c.model}" for c in self.clients)

    def __len__(self) -> int:
        return len(self.clients)

    @property
    def configured(self) -> bool:
        return bool(self.clients)

    def _cooldown_for(self, error: str) -> float:
        e = error.lower()
        if any(s in e for s in ("429", "resource_exhausted", "quota", "rate-limited", "rate limit")):
            return self.RATE_LIMIT_COOLDOWN_S
        if any(s in e for s in ("not_found", "no endpoints", "api key", "unauthorized",
                                "invalid_argument", "operation not allowed", "402",
                                "insufficient credits")):
            return self.FATAL_COOLDOWN_S
        return self.TRANSIENT_COOLDOWN_S

    def status(self) -> list[dict]:
        """Chain health, for the metrics panel and for debugging a live demo."""
        now = time.time()
        return [
            {
                "provider": c.provider,
                "model": c.model,
                "available": self._cooldown_until.get(i, 0.0) <= now,
                "cooldown_s": max(0.0, round(self._cooldown_until.get(i, 0.0) - now, 1)),
                "served": self._served.get(f"{c.provider}:{c.model}", 0),
                "skipped": self._skipped.get(f"{c.provider}:{c.model}", 0),
            }
            for i, c in enumerate(self.clients)
        ]

    def generate_unsourced(self, question: str) -> GenerationResult:
        """Answer from the model's own knowledge, with no corpus behind it.

        Only called after the system has already abstained, and the result is kept
        in its own field so it can never be mistaken for a grounded answer. No
        context is passed at all -- that is the point: there was none.
        """
        return self.generate(question, [], system=UNSOURCED_PROMPT, retries=0)

    def generate(self, question: str, contexts: list[str], **kw) -> GenerationResult:
        """Try providers in order, skipping any still cooling down.

        A provider that just returned 429 will almost certainly return 429 again
        a second later, so re-trying it costs latency and buys nothing. After a
        failure it is skipped until its cooldown expires; quota errors get 60s
        (per-minute windows), permanent ones 10 minutes.
        """
        now = time.time()
        last = GenerationResult()
        tried_any = False

        for i, c in enumerate(self.clients):
            key = f"{c.provider}:{c.model}"
            if self._cooldown_until.get(i, 0.0) > now:
                self._skipped[key] = self._skipped.get(key, 0) + 1
                continue
            tried_any = True

            # Retries only make sense when there is nothing else to try.
            #
            # The primary used to retry rate limits with a 3s x attempt backoff,
            # which is correct for a per-minute quota when it is the only
            # provider. With a live chain it is backwards: measured, a throttled
            # Gemini spent ~10s backing off before falling through, while the
            # next provider answers in ~2s. The user waits five times longer for
            # a worse outcome.
            #
            # So retries are reserved for the sole-provider case; whenever an
            # alternative exists, switching beats waiting. The cooldown above
            # then keeps the throttled provider out of the path for 60s rather
            # than paying that cost again on the next query.
            solo = len(self.clients) == 1
            r = c.generate(question, contexts, **(kw if (solo and i == 0) else ({"retries": 0} | kw)))
            if r.ok:
                self._served[key] = self._served.get(key, 0) + 1
                self._cooldown_until.pop(i, None)
                return r

            self._cooldown_until[i] = now + self._cooldown_for(r.error)
            last = r

        # Everything is cooling down. Rather than fail outright, make one
        # best-effort attempt on the primary -- a cooldown is a heuristic, and
        # the caller still has the extractive answer if this fails too.
        if not tried_any and self.clients:
            last = self.clients[0].generate(question, contexts, retries=0)
            if last.ok:
                self._cooldown_until.pop(0, None)
        return last

"""Rank LLM providers/models on this corpus, not on a leaderboard.

Placing a model in the fallback chain by reputation is guesswork. What matters
here is narrow and measurable:

  * does it answer at all, or error / rate-limit
  * how fast, since generation is the quality tier a user waits on
  * does it hold the SCRIPT -- a Devanagari question must get a Devanagari
    answer. Measured previously: Gemini emitted Urdu "نے" inside Hindi output,
    which is invisible in a demo video and wrong. Latin leakage matters too,
    since an English sentence answering a Hindi question is a failure even when
    the facts are right.

    docker compose run --rm bench python -m eval.compare_llms
"""

from __future__ import annotations

import os
import re
import time

from core.llm import LLMClient

DEVA = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
ARABIC = re.compile(r"[؀-ۿ]")  # the Urdu-into-Hindi failure mode

HIN_CTX = [
    "भारत का मौसम और जलवायु दक्षिण में उष्णकटिबंधीय मानसून से लेकर उत्तर में "
    "समशीतोष्ण तक विविधतापूर्ण है। भारत की राजधानी नई दिल्ली है। जनसंख्या एक अरब से अधिक है।"
]
MAR_CTX = [
    "भारतामध्ये २३ मेगा शहरे आहेत ज्यांची लोकसंख्या दहा लाखांपेक्षा जास्त आहे. "
    "१९९१ च्या जनगणनेनुसार, सर्वात मोठे शहर ग्रेटर मुंबई होते."
]
PROBES = [
    ("hin", "भारत की राजधानी क्या है?", HIN_CTX),
    ("mar", "भारतातील सर्वात मोठे शहर कोणते होते?", MAR_CTX),
]

CANDIDATES: list[tuple[str, str]] = [
    ("groq", "llama-3.3-70b-versatile"),
    ("groq", "openai/gpt-oss-20b"),
    ("gemini", os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")),
]


def script_report(text: str) -> tuple[str, bool]:
    d, latin, ar = len(DEVA.findall(text)), len(LATIN.findall(text)), len(ARABIC.findall(text))
    total = d + latin + ar
    if not total:
        return "empty", False
    clean = ar == 0 and latin <= max(2, total * 0.15)
    return f"deva {100 * d / total:>3.0f}% latin {100 * latin / total:>3.0f}%" + (
        f" ARABIC {ar}" if ar else ""
    ), clean


def main() -> None:
    print(f"\n{'provider/model':<42} {'lang':<5} {'ms':>7} {'ok':<4} {'script':<26} answer")
    print("-" * 124)

    for provider, model in CANDIDATES:
        try:
            c = LLMClient(provider=provider, model=model)
        except Exception as e:  # noqa: BLE001
            print(f"{provider + '/' + model:<42} {'-':<5} {'-':>7} INIT {type(e).__name__}: {e}")
            continue
        if not c.configured:
            print(f"{provider + '/' + model:<42} {'-':<5} {'-':>7} {'NOKEY':<4}")
            continue

        for lang, q, ctx in PROBES:
            t = time.perf_counter()
            try:
                r = c.generate(q, ctx, retries=0)
                ms = (time.perf_counter() - t) * 1000
            except Exception as e:  # noqa: BLE001
                print(f"{provider + '/' + model:<42} {lang:<5} {'-':>7} ERR  {type(e).__name__}: {str(e)[:40]}")
                continue

            if not r.ok:
                print(f"{provider + '/' + model:<42} {lang:<5} {ms:>7.0f} FAIL {r.error[:60]}")
                continue
            rep, clean = script_report(r.answer)
            flag = "" if clean else "  <-- SCRIPT MIX"
            print(f"{provider + '/' + model:<42} {lang:<5} {ms:>7.0f} ok   {rep:<26} "
                  f"{r.answer[:44]}{flag}")

    print("-" * 124)
    print("Chain order should follow this table: fastest clean answer first.")


if __name__ == "__main__":
    main()

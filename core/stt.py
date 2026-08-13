"""Speech-to-text via Sarvam (requirement #1).

The task mandates Sarvam or ElevenLabs and nothing else. Sarvam is the right
pick here because the corpus is Indic: Saaras v3 is trained on Indian speech and
handles code-mixed Hindi-English, which is how people actually ask questions.

API contract (docs.sarvam.ai):

    POST https://api.sarvam.ai/speech-to-text
    header  api-subscription-key: <key>
    form    file=<audio>  model=saaras:v3  mode=transcribe
    ->      {"request_id": str, "transcript": str, "language_code": str|null}

Notes that matter in practice:

  * `saarika:v2.5` is deprecated -- `saaras:v3` with mode=transcribe replaces it.
  * Audio caps at **30 seconds**, mono, 16kHz recommended. A spoken question fits
    comfortably; a rambling one does not, and the caller gets a clear error
    rather than a truncated transcript.
  * `mode` is worth knowing: `transcribe` keeps the source language, `translate`
    returns English, `codemix` handles Hindi-English mixing explicitly. We
    transcribe, because the index is Hindi and translating to English before
    retrieval would throw away the whole point of a multilingual encoder.

This module never raises. The harness needs to distinguish "could not hear you"
from "cannot answer that", and an exception cannot carry that distinction.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

ENDPOINT = "https://api.sarvam.ai/speech-to-text"
DEFAULT_MODEL = "saaras:v3"
MAX_AUDIO_SECONDS = 30

# Mirrors the languages Sarvam supports, intersected with what we index.
SUPPORTED = {
    "hi-IN", "bn-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN",
    "pa-IN", "ta-IN", "te-IN", "gu-IN", "en-IN",
}


@dataclass(slots=True)
class Transcript:
    text: str = ""
    language_code: str = ""
    ok: bool = False
    error: str = ""
    request_id: str = ""
    took_ms: float = 0.0
    attempts: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class SarvamSTT:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        mode: str = "transcribe",
        timeout_s: float = 20.0,
    ):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.model = model
        self.mode = mode
        self.timeout_s = timeout_s
        self._client: httpx.Client | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s)
        return self._client

    def transcribe(
        self,
        audio: bytes,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        *,
        retries: int = 2,
        backoff_s: float = 0.5,
    ) -> Transcript:
        """Transcribe an audio blob. Never raises -- check `.ok`."""
        t0 = time.perf_counter()
        out = Transcript()

        if not self.configured:
            out.error = "SARVAM_API_KEY not set"
            out.took_ms = round((time.perf_counter() - t0) * 1000, 2)
            return out
        if not audio:
            out.error = "empty audio"
            out.took_ms = round((time.perf_counter() - t0) * 1000, 2)
            return out

        last = ""
        for attempt in range(1, retries + 2):
            out.attempts = attempt
            try:
                resp = self._http().post(
                    ENDPOINT,
                    headers={"api-subscription-key": self.api_key},
                    files={"file": (filename, audio, content_type)},
                    data={"model": self.model, "mode": self.mode},
                )
                if resp.status_code == 200:
                    body = resp.json()
                    out.text = (body.get("transcript") or "").strip()
                    out.language_code = body.get("language_code") or ""
                    out.request_id = body.get("request_id", "")
                    out.ok = True
                    out.took_ms = round((time.perf_counter() - t0) * 1000, 2)
                    return out

                last = f"HTTP {resp.status_code}: {resp.text[:200]}"
                # 4xx other than 429 means the request itself is wrong -- a bad
                # key, an unsupported format, audio over 30s. Retrying sends the
                # identical bad request and just delays the error the user needs
                # to see.
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    break
            except Exception as e:  # noqa: BLE001 -- network faults are expected
                last = f"{type(e).__name__}: {e}"

            if attempt > retries:
                break
            time.sleep(backoff_s * attempt)

        out.error = last
        out.took_ms = round((time.perf_counter() - t0) * 1000, 2)
        return out

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

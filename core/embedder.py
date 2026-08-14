"""ONNX embedder for multilingual-e5-small (384-dim, XLM-R vocab).

Two things about e5 that are easy to get wrong and quietly cost retrieval
quality:

  1. **Prefixes are mandatory.** e5 was trained with "query: " on queries and
     "passage: " on documents. Dropping them, or using the same prefix for
     both, measurably degrades results. `encode_query` / `encode_passages`
     exist so the call site cannot forget.
  2. **Mean-pool over the attention mask, then L2-normalise.** Not CLS pooling.
     Normalising lets us use a plain dot product as cosine downstream, which is
     what hnswlib's inner-product space expects.

Model variants -- pick by target architecture:
  official fp32   intfloat/multilingual-e5-small : onnx/model.onnx
  official int8   intfloat/multilingual-e5-small : onnx/model_qint8_avx512_vnni.onnx  (x86 only)
  generic int8    Xenova/multilingual-e5-small   : onnx/model_int8.onnx               (ARM-safe)

The avx512_vnni build is tuned for x86; on Graviton use the Xenova int8.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

DIM = 384
MAX_LEN = 512

# (repo_id, onnx_filename)
VARIANTS = {
    "fp32": ("intfloat/multilingual-e5-small", "onnx/model.onnx"),
    "int8_x86": ("intfloat/multilingual-e5-small", "onnx/model_qint8_avx512_vnni.onnx"),
    "int8_arm": ("Xenova/multilingual-e5-small", "onnx/model_int8.onnx"),
}
TOKENIZER_REPO = "intfloat/multilingual-e5-small"


def available_providers() -> list[str]:
    """Execution providers to try, best first. Override with ORT_PROVIDERS.

    CUDA is worth an order of magnitude on the ingest embedding stage, which is
    ~90% of pipeline runtime. onnxruntime silently ignores a provider that is not
    installed, so listing CUDA on a CPU-only box is harmless.
    """
    if env := os.getenv("ORT_PROVIDERS"):
        return [p.strip() for p in env.split(",") if p.strip()]
    have = set(ort.get_available_providers())
    order = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in have] or ["CPUExecutionProvider"]


def default_variant() -> str:
    """Pick the ONNX build that suits the execution provider.

    int8 is a CPU optimisation: the quantised kernels are tuned for AVX-VNNI or
    ARM dot-product instructions. On CUDA it is the wrong choice -- GPUs want
    fp32/fp16, and an int8 graph forces dequantise/requantise work that can end
    up *slower* than fp32. So GPU gets fp32, CPU gets the arch-matched int8.
    """
    if v := os.getenv("E5_VARIANT"):
        return v
    if "CUDAExecutionProvider" in available_providers():
        return "fp32"
    return "int8_arm" if os.uname().machine in ("arm64", "aarch64") else "int8_x86"


@dataclass(slots=True)
class EmbedderConfig:
    variant: str = ""
    max_len: int = MAX_LEN
    batch_size: int = 64
    threads: int = 0  # 0 -> onnxruntime picks


class Embedder:
    def __init__(self, cfg: EmbedderConfig | None = None, cache_dir: Path | None = None):
        self.cfg = cfg or EmbedderConfig()
        variant = self.cfg.variant or default_variant()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; pick from {list(VARIANTS)}")
        repo, fname = VARIANTS[variant]
        self.variant = variant

        model_path = hf_hub_download(repo, fname, cache_dir=str(cache_dir) if cache_dir else None)
        tok_path = hf_hub_download(
            TOKENIZER_REPO, "tokenizer.json", cache_dir=str(cache_dir) if cache_dir else None
        )

        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_truncation(max_length=self.cfg.max_len)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")  # XLM-R pad id

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.cfg.threads:
            so.intra_op_num_threads = self.cfg.threads

        self.providers = available_providers()
        self.session = ort.InferenceSession(model_path, so, providers=self.providers)
        self.provider = self.session.get_providers()[0]
        self._input_names = {i.name for i in self.session.get_inputs()}

        # On GPU the batch should be far larger: the bottleneck moves from
        # compute to kernel-launch overhead, and 64 leaves the device idle.
        if "CUDA" in self.provider and cfg is None:
            self.cfg.batch_size = int(os.getenv("EMBED_BATCH", "256"))

    # -- internals ---------------------------------------------------------

    def _forward(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)

        feeds = {"input_ids": ids, "attention_mask": mask}
        # XLM-R has no token_type_ids, but some exports still declare it.
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}

        hidden = self.session.run(None, feeds)[0]  # (B, T, DIM)

        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-12, None)).astype(np.float32)

    def _encode(self, texts: list[str], prefix: str, batch_size: int | None = None) -> np.ndarray:
        """Batch with length bucketing, then restore the caller's order.

        The tokenizer pads each batch to its longest member. MSMARCO passages
        are mostly 60-80 tokens but chunks can reach 256, so an unsorted batch
        of 64 lets a single long chunk force 63 others to pad up to it -- most
        of the compute then goes into padding. Sorting by length first keeps
        each batch uniform and cuts that waste substantially.

        Character count is used as the length proxy: it correlates well enough
        with token count and costs nothing, whereas tokenising twice would
        defeat the purpose.

        `batch_size` overrides the configured size for one call. Bucketing only
        helps when a call spans several batches: the extractive path embeds ~10
        sentences, which is one batch of 64, so every short sentence pads up to
        the longest and the sort does nothing. A smaller batch there restores
        the effect. Doing so cannot change the vectors -- mean pooling is taken
        over the attention mask, so padding is excluded from the result.
        """
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)

        prefixed = [prefix + t for t in texts]
        order = sorted(range(len(prefixed)), key=lambda i: len(prefixed[i]))

        out = np.empty((len(prefixed), DIM), dtype=np.float32)
        bs = batch_size or self.cfg.batch_size
        for i in range(0, len(order), bs):
            idx = order[i : i + bs]
            out[idx] = self._forward([prefixed[j] for j in idx])
        return out

    # -- public ------------------------------------------------------------

    def encode_query(self, text: str) -> np.ndarray:
        """Single query -> (DIM,) normalised vector. This runs in the hot path."""
        return self._encode([text], "query: ")[0]

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "query: ")

    def encode_passages(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        return self._encode(texts, "passage: ", batch_size)


if __name__ == "__main__":
    import time

    e = Embedder()
    print(f"variant={e.variant}  provider={e.provider}  batch={e.cfg.batch_size}")
    print(f"  available: {ort.get_available_providers()}")

    q = "ताजमहल कहाँ स्थित है?"
    docs = [
        "ताजमहल आगरा में स्थित है। इसे शाहजहाँ ने बनवाया था।",
        "The Taj Mahal is located in Agra, India.",
        "पिज़्ज़ा बनाने की विधि बहुत सरल है।",
    ]
    qv = e.encode_query(q)
    dv = e.encode_passages(docs)
    print(f"shapes: query={qv.shape} docs={dv.shape}  |q|={np.linalg.norm(qv):.4f}")
    print("\ncosine similarity (cross-lingual sanity check):")
    for d, s in zip(docs, dv @ qv, strict=True):
        print(f"  {s:+.4f}  {d[:60]}")

    t0 = time.perf_counter()
    for _ in range(20):
        e.encode_query(q)
    print(f"\nsingle-query encode: {(time.perf_counter() - t0) / 20 * 1000:.2f} ms")

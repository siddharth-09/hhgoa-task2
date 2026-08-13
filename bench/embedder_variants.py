"""Measure embedding throughput per ONNX variant and sequence length.

int8 quantisation is a guaranteed *size* win but only sometimes a *speed* win:
onnxruntime's int8 kernels can fall back to unoptimised paths depending on the
CPU, in which case fp32 is faster despite the larger model. We picked int8_arm
by reasoning rather than measurement -- this settles it.

    python -m bench.embedder_variants
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from core.embedder import VARIANTS, Embedder, EmbedderConfig

# Representative of what the pipeline actually embeds: short Devanagari
# sentences (semantic chunking) through to full 256-token chunks.
SAMPLES = {
    "short (~20 tok)": "ताजमहल आगरा में स्थित है।",
    "medium (~80 tok)": (
        "ताजमहल आगरा में स्थित है। इसे शाहजहाँ ने अपनी पत्नी मुमताज़ महल की याद में "
        "बनवाया था। यह 1653 में पूरा हुआ और आज यह विश्व धरोहर स्थल है। हर साल लाखों "
        "पर्यटक इसे देखने आते हैं।"
    ),
}
SAMPLES["long (~250 tok)"] = SAMPLES["medium (~80 tok)"] * 3

N = 256


def bench(variant: str, threads: int) -> dict | None:
    try:
        t0 = time.perf_counter()
        emb = Embedder(EmbedderConfig(variant=variant, threads=threads))
        load_s = time.perf_counter() - t0
    except Exception as e:  # a variant may not exist for this arch
        print(f"  {variant:<10} LOAD FAILED: {type(e).__name__}: {str(e)[:80]}")
        return None

    out: dict = {"load_s": round(load_s, 2), "tokens": {}}
    for label, text in SAMPLES.items():
        texts = [text] * N
        emb.encode_passages(texts[:16])  # warm up
        t0 = time.perf_counter()
        emb.encode_passages(texts)
        dt = time.perf_counter() - t0
        rate = N / dt
        out["tokens"][label] = round(rate, 1)
        print(f"  {variant:<10} {label:<18} {rate:>7.1f}/s")
    return out


def main() -> None:
    threads = int(os.getenv("ORT_THREADS", "0"))
    print(f"threads={threads or 'auto'}  machine={os.uname().machine}  n={N}\n")

    results: dict[str, dict] = {}
    for variant in VARIANTS:
        r = bench(variant, threads)
        if r:
            results[variant] = r
        print()

    if results:
        print("=" * 58)
        best = {
            label: max(results.items(), key=lambda kv: kv[1]["tokens"].get(label, 0))
            for label in SAMPLES
        }
        for label, (name, r) in best.items():
            print(f"fastest @ {label:<18} {name:<10} {r['tokens'][label]:>7.1f}/s")

    out = Path(os.getenv("DATA_ROOT", "/data")) / "reports" / "embedder_variants.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"machine": os.uname().machine, "results": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

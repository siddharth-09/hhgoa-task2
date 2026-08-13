# Multi-arch: builds on arm64 (Oracle Ampere, Graviton) and amd64 (App Runner, HF Spaces).
# Nothing here is architecture-specific -- hnswlib, onnxruntime and tokenizers all
# ship aarch64 manylinux wheels, which is why faiss was ruled out early.

FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/data/hf_cache

# build-essential only needed if a wheel is missing for the target arch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer -- cached unless pyproject changes.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --no-install-project

COPY ingest/ ingest/
COPY core/ core/
COPY api/ api/
COPY bench/ bench/
COPY eval/ eval/
COPY web/ web/

ENV PATH="/app/.venv/bin:$PATH" \
    INDEX_PATH=/data/index \
    PORT=8000

# /data holds the parquet cache, JSONL corpus and built indexes.
# Mounted as a volume so a container rebuild never re-downloads 7.4GB.
VOLUME ["/data"]

EXPOSE 8000

# Default is the API; the ingestion pipeline overrides the command.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]

"""Stream a language shard of MSMARCO-XI into a compact local JSONL corpus.

The source parquet is a single ~9.7GB row group per language, so everything here
goes through iter_batches() -- never ParquetFile.read().

Row shape (see ingest/SCHEMA.md):
    query_id, query, Eng_Query, Answer, Eng_Answer, query_type,
    source_lang, target_lang, meta.{model_name,temperature,...},
    passages.{English_passages[], Translated_passages[], is_selected[]}

We keep only queries with >=1 selected passage, so every indexed query carries a
gold relevance label for the eval harness.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ID = "ai4bharat/MSMARCO-XI"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
COLUMNS = [
    "query_id",
    "query",
    "Eng_Query",
    "Answer",
    "Eng_Answer",
    "query_type",
    "target_lang",
    "meta",
    "passages",
]


def shard_name(lang: str, split: str) -> str:
    suffix = "train" if split == "train" else "val"
    return f"{split}/{lang}{suffix}.parquet"


def shard_path(lang: str, split: str, cache_dir: Path) -> str:
    """Fetch one language shard (~3.7GB) from the Hub, cached on disk."""
    from huggingface_hub import hf_hub_download  # lazy: streaming mode needs no hub client

    return hf_hub_download(
        repo_id=REPO_ID,
        filename=shard_name(lang, split),
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )


class HttpRangeFile:
    """Random-access file over HTTPS using Range requests.

    Lets pyarrow read a 3.7GB parquet without storing it. We stop after
    `max_queries`, which is ~10% of a shard, so streaming transfers a fraction
    of the file -- and on a disk-constrained host it stores none of it.

    Reads are served from a read-ahead buffer; parquet issues many small reads
    and one HTTPS round trip each would be unusably slow.
    """

    def __init__(self, url: str, readahead: int = 8 << 20):
        # httpx, not urllib: urlopen opens a fresh TLS connection per request, and
        # a streamed parquet issues dozens of range reads. The handshakes, not the
        # bytes, dominated -- 300 queries took >6 minutes while transferring only
        # tens of MB. A pooled client keeps one connection alive across all reads.
        import httpx

        self.readahead = readahead
        self.pos = 0
        self.bytes_fetched = 0
        self.orig_url = url
        self._client = httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(self.TIMEOUT_S, connect=15.0),
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300.0),
            headers={"Accept-Encoding": "identity"},  # ranges must not be re-encoded
        )
        r = self._client.head(url)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        self.url = str(r.url)  # resolved CDN target, reused for every range
        self._buf = b""
        self._buf_start = 0

    # pyarrow's PythonFile protocol
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return False

    def close(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None:
            client.close()

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = (
            offset
            if whence == 0
            else self.pos + offset
            if whence == 1
            else self.size + offset
        )
        return self.pos

    # A single range request is capped regardless of how much pyarrow asks for.
    # An unbounded request became a 34MB transfer that a home connection dropped
    # mid-flight (IncompleteRead), killing a multi-hour ingest at the first stage.
    # Smaller requests fail less often and cost far less to retry.
    MAX_CHUNK = 8 << 20
    MAX_RETRIES = 5
    TIMEOUT_S = 60

    def _fetch_once(self, start: int, length: int) -> bytes:
        end = min(start + length, self.size) - 1
        if end < start:
            return b""
        r = self._client.get(self.url, headers={"Range": f"bytes={start}-{end}"})
        r.raise_for_status()
        data = r.content
        want = end - start + 1
        if len(data) != want:
            raise OSError(f"short read: got {len(data)} of {want}")
        return data

    def _fetch(self, start: int, length: int) -> bytes:
        """Range-read with chunking and retry.

        Streaming a 3.7GB parquet over a home connection for hours means transient
        failures are certain, not hypothetical. Each chunk is retried with
        exponential backoff; only a persistent failure propagates.
        """
        out = bytearray()
        pos = start
        remaining = min(length, self.size - start)

        while remaining > 0:
            n = min(remaining, self.MAX_CHUNK)
            last: Exception | None = None
            for attempt in range(self.MAX_RETRIES):
                try:
                    chunk = self._fetch_once(pos, n)
                    break
                except Exception as e:  # noqa: BLE001 -- IncompleteRead, timeouts, resets
                    last = e
                    if attempt == self.MAX_RETRIES - 1:
                        raise OSError(
                            f"range {pos}-{pos + n - 1} failed after "
                            f"{self.MAX_RETRIES} attempts: {last}"
                        ) from last
                    time.sleep(0.5 * (2**attempt))
                    # The CDN redirect can expire mid-run; re-resolve on later retries.
                    if attempt >= 2:
                        try:
                            self.url = str(self._client.head(self.orig_url).url)
                        except Exception:  # noqa: BLE001 -- keep the old url and retry
                            pass

            if not chunk:
                break
            out += chunk
            pos += len(chunk)
            remaining -= len(chunk)

        self.bytes_fetched += len(out)
        return bytes(out)

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0:
            return b""

        buf_end = self._buf_start + len(self._buf)
        if not (self._buf_start <= self.pos and self.pos + n <= buf_end):
            want = max(n, self.readahead)
            self._buf = self._fetch(self.pos, want)
            self._buf_start = self.pos
            if not self._buf:
                return b""

        off = self.pos - self._buf_start
        out = self._buf[off : off + n]
        self.pos += len(out)
        return out


def open_shard(lang: str, split: str, cache_dir: Path, stream: bool):
    """Return something pq.ParquetFile accepts: a local path, or a streamed file."""
    if not stream:
        return shard_path(lang, split, cache_dir)
    url = RESOLVE.format(repo=REPO_ID, path=shard_name(lang, split))
    return pa.PythonFile(HttpRangeFile(url), mode="r")


def extract(
    parquet_file,
    max_queries: int,
    batch_size: int = 2048,
    skip_query_ids: set[int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return (passages, queries) for up to max_queries labelled queries.

    `parquet_file` is a path or any pyarrow-readable file object (see open_shard).

    `skip_query_ids` makes the extract *incremental*: query ids already on disk
    are skipped and `max_queries` counts only newly added ones. Growing a corpus
    from 10k to 25k then costs the 15k delta rather than a full rebuild -- and
    since embedding is ~90% of runtime, that is the difference between a 4-hour
    top-up and a 7-hour restart.
    """
    skip = skip_query_ids or set()
    pf = pq.ParquetFile(parquet_file)
    passages: list[dict] = []
    queries: list[dict] = []
    seen_passages: set[str] = set()

    for batch in pf.iter_batches(batch_size=batch_size, columns=COLUMNS):
        for row in batch.to_pylist():
            if row["query_id"] in skip:
                continue
            p = row["passages"]
            selected = p["is_selected"]
            if not any(selected):
                continue

            gold: list[str] = []
            for i, (eng, trans, is_sel) in enumerate(
                zip(p["English_passages"], p["Translated_passages"], selected, strict=False)
            ):
                # MSMARCO-XI is a *parallel* corpus: every language shard carries the
                # same query_ids, so the id must be namespaced by language. Without the
                # prefix, a Hindi passage and its Marathi translation share an id while
                # holding different text -- which collapses them into one unit in
                # AdaptiveRetriever's fusion and lets a Marathi passage satisfy Hindi
                # gold labels in eval. Invisible on a single-language pilot.
                pid = f"{row['target_lang']}:{row['query_id']}:{i}"
                if pid not in seen_passages:
                    seen_passages.add(pid)
                    passages.append(
                        {
                            "passage_id": pid,
                            "query_id": row["query_id"],
                            "text_eng": eng,
                            "text_translated": trans,
                            "lang": row["target_lang"],
                            "query_type": row["query_type"],
                            "translator": (row.get("meta") or {}).get("model_name"),
                        }
                    )
                if is_sel:
                    gold.append(pid)

            queries.append(
                {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "query_eng": row["Eng_Query"],
                    "answer": row["Answer"],
                    "answer_eng": row["Eng_Answer"],
                    "query_type": row["query_type"],
                    "lang": row["target_lang"],
                    "gold_passage_ids": gold,
                }
            )

            if len(queries) >= max_queries:
                return passages, queries

    return passages, queries


def read_query_ids(path: Path) -> set[int]:
    """query_ids already extracted, so a re-run can add to them."""
    if not path.exists():
        return set()
    out: set[int] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                out.add(json.loads(line)["query_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def write_jsonl(rows: list[dict], path: Path, *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):,} rows -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["hin", "mar"])
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    ap.add_argument("--max-queries", type=int, default=50_000)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--stream", action="store_true", help="HTTP range reads; store no parquet")
    args = ap.parse_args()

    for lang in args.langs:
        print(f"\n=== {lang} ({args.split}) ===")
        src = open_shard(lang, args.split, args.out / "hf_cache", args.stream)
        passages, queries = extract(src, args.max_queries)
        write_jsonl(passages, args.out / f"{lang}_{args.split}_passages.jsonl")
        write_jsonl(queries, args.out / f"{lang}_{args.split}_queries.jsonl")


if __name__ == "__main__":
    main()

# MSMARCO-XI — measured schema notes

Probed directly from the parquet footer over HTTP range requests, because the
HF Dataset Viewer currently fails on this repo ("dataset generation failed").

## Layout

One parquet file per language, not one giant file.

- `train/` — 13 languages: `asm ben guj hin kan mal mar nep ori pan san tam urd`
- `validation/` — 14 languages (adds `tel`)
- Each train shard: **~3.7GB**, **778,638 rows**, **1 row group** (~9.7GB uncompressed)

A *row* is a query, not a passage.

## Measured stats (hin, train)

| Metric | Value |
|---|---|
| Queries | 778,638 |
| Passages per query | min 1, max 27, **mean 9.98** |
| Total passages | **7,769,498** |
| Queries with >=1 selected passage | 484,269 (62.2%) |
| Mean selected per query | 0.66 |
| `source_lang` / `target_lang` | `eng_Latn` / `hin_Deva` |

Extrapolated across 13 languages: **~100M passages.** Subsetting is mandatory.

## query_type distribution (hin, train)

| Type | Count | Share |
|---|---:|---:|
| DESCRIPTION | 411,657 | 52.9% |
| NUMERIC | 205,118 | 26.3% |
| ENTITY | 69,047 | 8.9% |
| LOCATION | 48,928 | 6.3% |
| PERSON | 43,888 | 5.6% |

## Columns

```
query_id      int64
query         string   # translated query
Eng_Query     string
Answer        string   # translated answer
Eng_Answer    string
query_type    string   # DESCRIPTION | NUMERIC | ENTITY | LOCATION | PERSON
source_lang   string
target_lang   string
meta          struct<model_name, temperature, top_p,
                     max_tokens, frequency_penalty, presence_penalty>
passages      struct<English_passages:    list<string>,
                     Translated_passages:  list<string>,
                     is_selected:          list<int64>>
```

## Why this schema matters for the task

- **`is_selected` = free ground-truth relevance labels.** Lets us compute
  recall@k / MRR / nDCG per chunking strategy with zero manual annotation.
  This is what turns requirement #2 into a measured benchmark.
- **`query_type` = ready-made router label.** Measure MRR per
  (strategy x query_type) and let the router exploit the real differences.
- **`meta.*` = translation provenance** — the metadata-aware chunking dimension.
- **English + translated side by side** — cross-lingual retrieval for free.

## Ingestion gotcha

Each shard is a **single row group**, so `ParquetFile.read()` would materialise
~9.7GB. Always use `iter_batches(batch_size=...)`.

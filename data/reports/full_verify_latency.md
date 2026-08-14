
## Fast-path latency — ensemble (682,045 chunks across 3 indexes, 25 queries, aarch64)

**44.822ms P50 · 47.966ms P70 · 74.316ms P100** — 25/25 under 200ms

| Stage | P50 | P70 | P90 | P99 | P100 | mean |
|---|---:|---:|---:|---:|---:|---:|
| guardrail_in | 0.007 | 0.008 | 0.01 | 0.011 | 0.011 | 0.007 |
| embed_query | 1.731 | 1.935 | 2.265 | 2.805 | 2.873 | 1.798 |
| retrieve | 10.307 | 11.108 | 13.473 | 16.31 | 17.058 | 10.581 |
| extract | 30.729 | 33.918 | 40.16 | 58.67 | 63.128 | 32.24 |
| guardrail_out | 0.108 | 0.124 | 0.143 | 0.168 | 0.17 | 0.111 |
| **fast_path_total** | **44.822** | **47.966** | 51.974 | 70.093 | **74.316** | 44.74 |

Per-index fan-out (the ensemble cost):

| Index | chunks | P50 | P100 |
|---|---:|---:|---:|
| fixed_256 | 201,298 | 2.324 | 3.243 |
| semantic_128 | 239,175 | 2.359 | 3.384 |
| metadata_128 | 241,572 | 2.082 | 3.017 |

Serial fan-out sums to 6.765ms P50; the slowest single index is 2.359ms — that gap is what parallelising the fan-out would recover.

Answer source: {'extractive': 25}

Cold first query: 294.967ms (excluded — it measures warmup, not steady state)

_All times in ms. Speech-to-text and generation are excluded: the task scopes the_
_budget as "chunking + vector DB retrieval + everything through to final output"._
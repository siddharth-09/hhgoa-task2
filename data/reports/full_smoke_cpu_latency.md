
## Fast-path latency — ensemble (682,045 chunks across 3 indexes, 40 queries, arm64)

**44.469ms P50 · 51.422ms P70 · 84.838ms P100** — 40/40 under 200ms

| Stage | P50 | P70 | P90 | P99 | P100 | mean |
|---|---:|---:|---:|---:|---:|---:|
| guardrail_in | 0.008 | 0.009 | 0.013 | 0.023 | 0.025 | 0.009 |
| embed_query | 2.053 | 2.275 | 2.666 | 3.942 | 4.648 | 2.106 |
| retrieve | 7.941 | 9.164 | 11.425 | 35.62 | 41.231 | 9.12 |
| extract | 33.553 | 37.899 | 51.396 | 73.433 | 73.886 | 36.476 |
| guardrail_out | 0.116 | 0.171 | 0.246 | 0.31 | 0.311 | 0.142 |
| **fast_path_total** | **44.469** | **51.422** | 62.403 | 84.211 | **84.838** | 47.858 |

Per-index fan-out (the ensemble cost):

| Index | chunks | P50 | P100 |
|---|---:|---:|---:|
| fixed_256 | 201,298 | 2.092 | 3.687 |
| semantic_128 | 239,175 | 1.814 | 3.74 |
| metadata_128 | 241,572 | 1.831 | 3.413 |

Serial fan-out sums to 5.737ms P50; the slowest single index is 2.092ms — that gap is what parallelising the fan-out would recover.

Answer source: {'extractive': 38, 'abstain': 2}

Cold first query: 155.586ms (excluded — it measures warmup, not steady state)

_All times in ms. Speech-to-text and generation are excluded: the task scopes the_
_budget as "chunking + vector DB retrieval + everything through to final output"._
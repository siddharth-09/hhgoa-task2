
## Fast-path latency — ensemble (682,045 chunks across 3 indexes, 300 queries, aarch64)

**36.969ms P50 · 39.81ms P70 · 120.246ms P100** — 300/300 under 200ms

| Stage | P50 | P70 | P90 | P99 | P100 | mean |
|---|---:|---:|---:|---:|---:|---:|
| guardrail_in | 0.008 | 0.009 | 0.011 | 0.019 | 0.248 | 0.01 |
| embed_query | 1.968 | 2.227 | 2.735 | 3.491 | 61.805 | 2.259 |
| retrieve | 8.39 | 9.319 | 10.988 | 19.398 | 57.383 | 8.676 |
| extract | 26.498 | 28.943 | 33.284 | 49.105 | 57.431 | 27.08 |
| guardrail_out | 0.111 | 0.137 | 0.178 | 0.256 | 0.29 | 0.123 |
| **fast_path_total** | **36.969** | **39.81** | 45.455 | 71.037 | **120.246** | 38.152 |

Per-index fan-out (the ensemble cost):

| Index | chunks | P50 | P100 |
|---|---:|---:|---:|
| fixed_256 | 201,298 | 2.421 | 6.846 |
| semantic_128 | 239,175 | 2.572 | 5.702 |
| metadata_128 | 241,572 | 2.394 | 5.854 |

Serial fan-out sums to 7.387ms P50; the slowest single index is 2.572ms — that gap is what parallelising the fan-out would recover.

Answer source: {'extractive': 293, 'abstain': 7}

Cold first query: 270.299ms (excluded — it measures warmup, not steady state)

_All times in ms. Speech-to-text and generation are excluded: the task scopes the_
_budget as "chunking + vector DB retrieval + everything through to final output"._

## Fast-path latency — ensemble (682,045 chunks across 3 indexes, 300 queries, aarch64)

**42.12ms P50 · 48.671ms P70 · 245.668ms P100** — 297/300 under 200ms

| Stage | P50 | P70 | P90 | P99 | P100 | mean |
|---|---:|---:|---:|---:|---:|---:|
| guardrail_in | 0.007 | 0.008 | 0.011 | 0.022 | 0.231 | 0.009 |
| embed_query | 1.873 | 2.126 | 2.571 | 4.778 | 53.938 | 2.163 |
| retrieve | 7.833 | 8.789 | 10.923 | 16.176 | 60.045 | 8.375 |
| extract | 32.023 | 37.942 | 51.073 | 175.548 | 236.837 | 37.303 |
| guardrail_out | 0.107 | 0.127 | 0.176 | 0.28 | 1.125 | 0.12 |
| **fast_path_total** | **42.12** | **48.671** | 63.431 | 190.503 | **245.668** | 47.974 |

Per-index fan-out (the ensemble cost):

| Index | chunks | P50 | P100 |
|---|---:|---:|---:|
| fixed_256 | 201,298 | 2.34 | 8.384 |
| semantic_128 | 239,175 | 2.393 | 5.391 |
| metadata_128 | 241,572 | 2.147 | 5.765 |

Serial fan-out sums to 6.88ms P50; the slowest single index is 2.393ms — that gap is what parallelising the fan-out would recover.

Answer source: {'extractive': 293, 'abstain': 7}

Cold first query: 176.012ms (excluded — it measures warmup, not steady state)

_All times in ms. Speech-to-text and generation are excluded: the task scopes the_
_budget as "chunking + vector DB retrieval + everything through to final output"._
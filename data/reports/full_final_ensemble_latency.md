
## Fast-path latency — ensemble (682,045 chunks across 3 indexes, 300 queries, aarch64)

**37.142ms P50 · 40.21ms P70 · 143.331ms P100** — 300/300 under 200ms

| Stage | P50 | P70 | P90 | P99 | P100 | mean |
|---|---:|---:|---:|---:|---:|---:|
| guardrail_in | 0.009 | 0.009 | 0.012 | 0.02 | 0.281 | 0.01 |
| embed_query | 1.905 | 2.129 | 2.501 | 4.508 | 70.216 | 2.234 |
| retrieve | 8.043 | 8.915 | 10.3 | 15.117 | 118.23 | 8.488 |
| extract | 27.209 | 29.951 | 33.983 | 39.316 | 85.929 | 27.528 |
| guardrail_out | 0.117 | 0.139 | 0.175 | 0.255 | 0.298 | 0.124 |
| **fast_path_total** | **37.142** | **40.21** | 45.209 | 65.854 | **143.331** | 38.389 |

Per-index fan-out (the ensemble cost):

| Index | chunks | P50 | P100 |
|---|---:|---:|---:|
| fixed_256 | 201,298 | 2.471 | 6.833 |
| semantic_128 | 239,175 | 2.506 | 6.533 |
| metadata_128 | 241,572 | 2.285 | 6.393 |

Serial fan-out sums to 7.262ms P50; the slowest single index is 2.506ms — that gap is what parallelising the fan-out would recover.

Answer source: {'extractive': 293, 'abstain': 7}

Cold first query: 212.988ms (excluded — it measures warmup, not steady state)

_All times in ms. Speech-to-text and generation are excluded: the task scopes the_
_budget as "chunking + vector DB retrieval + everything through to final output"._
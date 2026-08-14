
## Fast-path latency — ensemble (241,572 chunks across 1 indexes, 300 queries, aarch64)

**30.496ms P50 · 32.858ms P70 · 129.534ms P100** — 300/300 under 200ms

| Stage | P50 | P70 | P90 | P99 | P100 | mean |
|---|---:|---:|---:|---:|---:|---:|
| guardrail_in | 0.008 | 0.009 | 0.011 | 0.022 | 9.861 | 0.041 |
| embed_query | 1.749 | 1.954 | 2.304 | 3.56 | 86.903 | 2.119 |
| retrieve | 2.282 | 2.611 | 3.123 | 4.317 | 10.254 | 2.382 |
| extract | 26.214 | 28.509 | 31.682 | 39.563 | 44.757 | 26.168 |
| guardrail_out | 0.107 | 0.118 | 0.139 | 0.733 | 1.19 | 0.127 |
| **fast_path_total** | **30.496** | **32.858** | 36.653 | 48.164 | **129.534** | 30.841 |

Per-index fan-out (the ensemble cost):

| Index | chunks | P50 | P100 |
|---|---:|---:|---:|
| metadata_128 | 241,572 | 2.287 | 9.986 |

Serial fan-out sums to 2.287ms P50; the slowest single index is 2.287ms — that gap is what parallelising the fan-out would recover.

Answer source: {'extractive': 289, 'abstain': 11}

Cold first query: 45.701ms (excluded — it measures warmup, not steady state)

_All times in ms. Speech-to-text and generation are excluded: the task scopes the_
_budget as "chunking + vector DB retrieval + everything through to final output"._
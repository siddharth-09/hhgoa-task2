
## Fast-path latency — ensemble (682,045 chunks across 3 indexes, 40 queries, arm64)

**214.653ms P50 · 246.409ms P70 · 745.406ms P100** — 11/40 under 200ms

| Stage | P50 | P70 | P90 | P99 | P100 | mean |
|---|---:|---:|---:|---:|---:|---:|
| guardrail_in | 0.007 | 0.009 | 0.011 | 0.012 | 0.012 | 0.008 |
| embed_query | 20.919 | 23.169 | 27.223 | 31.816 | 33.758 | 21.097 |
| retrieve | 9.657 | 11.618 | 13.294 | 20.322 | 20.823 | 9.861 |
| extract | 179.912 | 212.967 | 266.963 | 573.191 | 714.666 | 207.707 |
| guardrail_out | 0.123 | 0.169 | 0.216 | 0.266 | 0.284 | 0.141 |
| **fast_path_total** | **214.653** | **246.409** | 309.84 | 603.466 | **745.406** | 238.818 |

Per-index fan-out (the ensemble cost):

| Index | chunks | P50 | P100 |
|---|---:|---:|---:|
| fixed_256 | 201,298 | 2.233 | 7.832 |
| semantic_128 | 239,175 | 2.098 | 7.254 |
| metadata_128 | 241,572 | 1.744 | 3.492 |

Serial fan-out sums to 6.075ms P50; the slowest single index is 2.233ms — that gap is what parallelising the fan-out would recover.

Answer source: {'extractive': 40}

Cold first query: 455.224ms (excluded — it measures warmup, not steady state)

_All times in ms. Speech-to-text and generation are excluded: the task scopes the_
_budget as "chunking + vector DB retrieval + everything through to final output"._
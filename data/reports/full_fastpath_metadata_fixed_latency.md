
## Fast-path latency — ensemble (241,572 chunks across 1 indexes, 300 queries, aarch64)

**29.48ms P50 · 31.132ms P70 · 178.122ms P100** — 300/300 under 200ms

| Stage | P50 | P70 | P90 | P99 | P100 | mean |
|---|---:|---:|---:|---:|---:|---:|
| guardrail_in | 0.007 | 0.008 | 0.01 | 0.017 | 0.231 | 0.008 |
| embed_query | 1.735 | 1.945 | 2.334 | 3.406 | 56.481 | 1.985 |
| retrieve | 2.299 | 2.6 | 3.083 | 3.902 | 12.167 | 2.368 |
| extract | 25.152 | 26.89 | 30.2 | 121.671 | 126.861 | 28.28 |
| guardrail_out | 0.099 | 0.107 | 0.126 | 0.71 | 0.903 | 0.119 |
| **fast_path_total** | **29.48** | **31.132** | 34.941 | 127.818 | **178.122** | 32.763 |

Per-index fan-out (the ensemble cost):

| Index | chunks | P50 | P100 |
|---|---:|---:|---:|
| metadata_128 | 241,572 | 2.179 | 5.478 |

Serial fan-out sums to 2.179ms P50; the slowest single index is 2.179ms — that gap is what parallelising the fan-out would recover.

Answer source: {'extractive': 289, 'abstain': 11}

Cold first query: 52.936ms (excluded — it measures warmup, not steady state)

_All times in ms. Speech-to-text and generation are excluded: the task scopes the_
_budget as "chunking + vector DB retrieval + everything through to final output"._
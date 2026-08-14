
## Fast-path latency — ensemble (241,572 chunks across 1 indexes, 300 queries, aarch64)

**32.059ms P50 · 34.029ms P70 · 192.46ms P100** — 300/300 under 200ms

| Stage | P50 | P70 | P90 | P99 | P100 | mean |
|---|---:|---:|---:|---:|---:|---:|
| guardrail_in | 0.007 | 0.008 | 0.01 | 0.019 | 0.766 | 0.011 |
| embed_query | 1.754 | 1.969 | 2.364 | 3.854 | 60.612 | 2.02 |
| retrieve | 2.326 | 2.639 | 3.097 | 3.947 | 6.37 | 2.361 |
| extract | 27.761 | 29.754 | 32.52 | 129.099 | 133.886 | 31.1 |
| guardrail_out | 0.114 | 0.122 | 0.143 | 0.805 | 1.005 | 0.136 |
| **fast_path_total** | **32.059** | **34.029** | 37.253 | 132.805 | **192.46** | 35.631 |

Per-index fan-out (the ensemble cost):

| Index | chunks | P50 | P100 |
|---|---:|---:|---:|
| metadata_128 | 241,572 | 2.243 | 6.446 |

Serial fan-out sums to 2.243ms P50; the slowest single index is 2.243ms — that gap is what parallelising the fan-out would recover.

Answer source: {'extractive': 292, 'abstain': 8}

Cold first query: 43.88ms (excluded — it measures warmup, not steady state)

_All times in ms. Speech-to-text and generation are excluded: the task scopes the_
_budget as "chunking + vector DB retrieval + everything through to final output"._
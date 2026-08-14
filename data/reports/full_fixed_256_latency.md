
## Latency — fixed_256 (201,298 chunks, 300 queries, x86_64)

| Stage | P50 | P70 | P90 | P99 | P100 | mean |
|---|---:|---:|---:|---:|---:|---:|
| embed_query | 5.061 | 5.835 | 7.301 | 18.527 | 94.655 | 5.833 |
| search_dense | 0.89 | 1.044 | 1.55 | 5.263 | 10.014 | 1.086 |
| search_sparse | 30.133 | 31.799 | 33.634 | 44.42 | 48.142 | 19.075 |
| search_fused_total | 24.324 | 26.398 | 28.277 | 33.231 | 38.966 | 16.694 |
| retrieval_total | 60.352 | 64.45 | 68.647 | 83.037 | 137.67 | 42.672 |

Cold first query: 249.297ms (excluded from percentiles — it measures warmup, not steady state)

_All times in ms. Speech-to-text is excluded: the task scopes the budget as_
_"chunking + vector DB retrieval + everything through to final output"._
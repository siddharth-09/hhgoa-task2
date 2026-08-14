
## Latency — semantic_128 (239,175 chunks, 300 queries, x86_64)

| Stage | P50 | P70 | P90 | P99 | P100 | mean |
|---|---:|---:|---:|---:|---:|---:|
| embed_query | 5.186 | 6.07 | 8.61 | 19.274 | 161.199 | 6.389 |
| search_dense | 1.451 | 1.976 | 3.481 | 8.414 | 12.647 | 1.92 |
| search_sparse | 12.73 | 38.83 | 43.129 | 59.421 | 62.145 | 21.406 |
| search_fused_total | 13.972 | 31.849 | 36.745 | 50.747 | 56.523 | 19.589 |
| retrieval_total | 34.134 | 77.357 | 87.593 | 122.865 | 247.009 | 49.284 |

Cold first query: 220.949ms (excluded from percentiles — it measures warmup, not steady state)

_All times in ms. Speech-to-text is excluded: the task scopes the budget as_
_"chunking + vector DB retrieval + everything through to final output"._
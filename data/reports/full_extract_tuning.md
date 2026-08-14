
## `extract` tuning sweep (300 queries, aarch64, int8_arm, 3 threads)

Reference: embed_batch=1, max_sentence_chars=0 (each sentence quantised alone). Retrieval is cached and shared, so only extraction varies.

| embed_batch | trunc chars | P50 | P90 | P95 | P99 | P100 | identical | mean support |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | — | 24.142 | 30.579 | 34.244 | 48.576 | 81.726 | 100.0% | 0.6558 |
| 64 | — | 32.327 | 51.718 | 66.873 | 185.366 | 227.519 | 90.7% | 0.6557 |
| 8 | — | 28.554 | 40.58 | 50.137 | 156.373 | 179.467 | 91.7% | 0.6553 |
| 4 | — | 26.529 | 34.598 | 41.61 | 97.091 | 103.95 | 91.0% | 0.6553 |
| 2 | — | 26.166 | 33.39 | 36.936 | 68.005 | 76.905 | 93.3% | 0.6549 |
| 4 | 512 | 27.788 | 35.574 | 41.728 | 69.782 | 92.249 | 90.7% | 0.6553 |
| 4 | 256 | 28.708 | 37.092 | 41.949 | 48.876 | 62.466 | 89.7% | 0.6553 |
| 4 | 192 | 28.843 | 35.749 | 38.203 | 45.056 | 50.84 | 89.3% | 0.6537 |
| 2 | 256 | 27.648 | 33.37 | 36.45 | 40.552 | 62.236 | 92.3% | 0.6546 |

_`identical` is the fraction of answers byte-identical to the **reference**_
_(batch=1), which is the only config that embeds each sentence on its own_
_activations. int8 activation scales are computed per batch, so a wide pad_
_coarsens the quantisation of everything beside it -- large batches are the_
_degraded end of this table, not the faithful one._

## `extract` tuning sweep (300 queries, aarch64, int8_arm, 3 threads)

Reference: embed_batch=1, max_sentence_chars=0 (each sentence quantised alone). Retrieval is cached and shared, so only extraction varies.

| embed_batch | trunc chars | P50 | P90 | P95 | P99 | P100 | identical | mean support |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | — | 24.566 | 29.759 | 64.977 | 127.098 | 129.276 | 100.0% | 0.6511 |
| 64 | — | 33.486 | 57.081 | 125.829 | 482.572 | 636.067 | 88.3% | 0.6508 |
| 8 | — | 29.54 | 48.833 | 125.907 | 438.598 | 550.98 | 90.7% | 0.6506 |
| 4 | — | 27.312 | 34.799 | 77.383 | 254.651 | 326.011 | 90.7% | 0.6508 |
| 2 | — | 26.998 | 33.738 | 72.187 | 197.45 | 204.304 | 91.0% | 0.6506 |
| 4 | 512 | 27.418 | 35.069 | 42.605 | 129.477 | 234.863 | 90.7% | 0.6506 |
| 4 | 256 | 29.16 | 35.255 | 38.489 | 58.613 | 85.892 | 89.7% | 0.6501 |
| 4 | 192 | 29.532 | 35.728 | 39.073 | 51.866 | 61.905 | 89.0% | 0.6486 |
| 2 | 256 | 31.383 | 38.225 | 40.592 | 51.523 | 80.633 | 89.7% | 0.6497 |
| 1 | 512 | 33.395 | 38.927 | 42.153 | 85.152 | 145.682 | 99.0% | 0.6507 |
| 1 | 256 | 33.633 | 40.725 | 43.933 | 51.969 | 55.796 | 98.3% | 0.6501 |
| 1 | 192 | 34.924 | 41.904 | 44.157 | 50.872 | 76.92 | 96.3% | 0.6489 |

_`identical` is the fraction of answers byte-identical to the **reference**_
_(batch=1), which is the only config that embeds each sentence on its own_
_activations. int8 activation scales are computed per batch, so a wide pad_
_coarsens the quantisation of everything beside it -- large batches are the_
_degraded end of this table, not the faithful one._
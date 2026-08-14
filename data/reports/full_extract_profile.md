
## `extract` component profile (300 queries, aarch64, int8_arm, 3 threads)

| Component | P50 | P70 | P90 | P95 | P99 | P100 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| split_sentences | 0.065 | 0.075 | 0.094 | 0.104 | 0.116 | 0.125 | 0.067 |
| lexical_overlap | 0.058 | 0.069 | 0.084 | 0.093 | 0.114 | 0.589 | 0.062 |
| prefilter | 0.001 | 0.009 | 0.01 | 0.011 | 0.015 | 0.022 | 0.005 |
| embed_sentences | 31.779 | 37.902 | 49.706 | 67.874 | 180.014 | 209.83 | 37.191 |
| score_and_span | 0.029 | 0.03 | 0.034 | 0.038 | 0.068 | 0.14 | 0.031 |
| **extract total** | **31.956** | **38.047** | 49.858 | 68.027 | **180.194** | **210.005** | 37.357 |

### What distinguishes the tail

| Group | n | extract mean | embed mean | sentences | max chars (mean) | max chars (worst) |
|---|---:|---:|---:|---:|---:|---:|
| slowest 10% | 30 | 91.62ms | 91.42ms | 8.73 | 387.4 | 1279 |
| fastest 50% | 150 | 25.45ms | 25.3ms | 8.81 | 151.9 | 344 |

Correlation with extract latency: **max_sentence_chars 0.8411**, n_sentences_embedded 0.0166.

_A batch is padded to its longest member, so the longest sentence sets the cost_
_for the whole batch. Sentence count is capped at 10 and barely varies._
## Chunking strategy comparison

| Strategy | Chunks | MRR@10 | nDCG@10 | R@1 | R@5 | R@10 | R@20 | search p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_256 | 201,298 | 0.2974 | 0.3583 | 0.1722 | 0.4573 | 0.5694 | 0.6617 | 2.19ms |
| metadata_128 | 241,572 | 0.3003 | 0.3632 | 0.1736 | 0.4522 | 0.5808 | 0.6814 | 2.01ms |
| semantic_128 | 239,175 | 0.2944 | 0.3535 | 0.1715 | 0.4480 | 0.5608 | 0.6535 | 2.15ms |
| ENSEMBLE[fixed_256+semantic_128+metadata_128] | 682,045 | 0.3056 | 0.3670 | 0.1799 | 0.4622 | 0.5803 | 0.6669 | 6.62ms |

## MRR@10 by query type — the routing signal

| Strategy | DESCRIPTION | ENTITY | LOCATION | NUMERIC | PERSON |
|---|---|---|---|---|---|
| fixed_256 | 0.2765 | 0.2886 | 0.3649 | 0.3239 | 0.3277 |
| metadata_128 | 0.2753 | 0.2785 | 0.3885 | 0.3255 | 0.3882 |
| semantic_128 | 0.2723 | 0.2789 | 0.3643 | 0.3298 | 0.2987 |
| ENSEMBLE[fixed_256+semantic_128+metadata_128] | 0.2823 | 0.3010 | 0.3793 | 0.3360 | 0.3225 |

**Best strategy per query type:**

- `DESCRIPTION` → **ENSEMBLE[fixed_256+semantic_128+metadata_128]** (MRR@10 0.2823)
- `ENTITY` → **ENSEMBLE[fixed_256+semantic_128+metadata_128]** (MRR@10 0.3010)
- `LOCATION` → **metadata_128** (MRR@10 0.3885)
- `NUMERIC` → **ENSEMBLE[fixed_256+semantic_128+metadata_128]** (MRR@10 0.3360)
- `PERSON` → **metadata_128** (MRR@10 0.3882)
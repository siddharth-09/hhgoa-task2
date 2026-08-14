
## Paired significance (1500 queries, 10,000 bootstrap resamples, baseline `metadata_128`)

| Comparison | Metric | Δ | 95% CI | significant |
|---|---|---:|---|---|
| fixed_256 - metadata_128 | mrr@10 | -0.0136 | [-0.0241, -0.0033] | **yes** |
| fixed_256 - metadata_128 | recall@10 | -0.0068 | [-0.0223, +0.0082] | no |
| fixed_256 - metadata_128 | recall@20 | -0.0068 | [-0.0211, +0.0068] | no |
| fixed+metadata - metadata_128 | mrr@10 | -0.0057 | [-0.0136, +0.0021] | no |
| fixed+metadata - metadata_128 | recall@10 | +0.0015 | [-0.0092, +0.0121] | no |
| fixed+metadata - metadata_128 | recall@20 | +0.0022 | [-0.0090, +0.0131] | no |
| ENSEMBLE - metadata_128 | mrr@10 | -0.0105 | [-0.0185, -0.0026] | **yes** |
| ENSEMBLE - metadata_128 | recall@10 | +0.0047 | [-0.0076, +0.0168] | no |
| ENSEMBLE - metadata_128 | recall@20 | -0.0054 | [-0.0173, +0.0064] | no |

### Leak test — does the metadata hint help most where it excludes most corpus?

| query_type | queries | corpus share | metadata MRR | fixed MRR | advantage | 95% CI | sig |
|---|---:|---:|---:|---:|---:|---|---|
| DESCRIPTION | 921 | 59.4% | 0.2862 | 0.2754 | +0.0109 | [-0.0025, +0.0246] | no |
| NUMERIC | 265 | 19.2% | 0.3002 | 0.2925 | +0.0077 | [-0.0126, +0.0286] | no |
| ENTITY | 118 | 9.4% | 0.2936 | 0.2757 | +0.0179 | [-0.0207, +0.0579] | no |
| LOCATION | 159 | 9.4% | 0.4044 | 0.3734 | +0.0310 | [-0.0041, +0.0670] | no |
| PERSON | 37 | 2.7% | 0.3357 | 0.3018 | +0.0339 | [+0.0010, +0.0730] | **yes** |

Correlation between corpus share and metadata advantage: **-0.638**.

_A strong negative correlation is the leak signature: the tag helps precisely_
_when it rules out most of the corpus, and stops helping for the majority type._

### Per language

| Lang | Comparison | Metric | Δ | 95% CI | significant |
|---|---|---|---:|---|---|
| hin_Deva | fixed_256 - metadata_128 | mrr@10 | -0.0006 | [-0.0147, +0.0132] | no |
| hin_Deva | fixed_256 - metadata_128 | recall@10 | +0.0079 | [-0.0122, +0.0283] | no |
| hin_Deva | fixed+metadata - metadata_128 | mrr@10 | -0.0016 | [-0.0120, +0.0091] | no |
| hin_Deva | fixed+metadata - metadata_128 | recall@10 | +0.0038 | [-0.0102, +0.0172] | no |
| hin_Deva | ENSEMBLE - metadata_128 | mrr@10 | -0.0026 | [-0.0132, +0.0076] | no |
| hin_Deva | ENSEMBLE - metadata_128 | recall@10 | +0.0179 | [+0.0020, +0.0341] | **yes** |
| mar_Deva | fixed_256 - metadata_128 | mrr@10 | -0.0261 | [-0.0416, -0.0107] | **yes** |
| mar_Deva | fixed_256 - metadata_128 | recall@10 | -0.0211 | [-0.0434, +0.0010] | no |
| mar_Deva | fixed+metadata - metadata_128 | mrr@10 | -0.0097 | [-0.0213, +0.0020] | no |
| mar_Deva | fixed+metadata - metadata_128 | recall@10 | -0.0008 | [-0.0170, +0.0156] | no |
| mar_Deva | ENSEMBLE - metadata_128 | mrr@10 | -0.0181 | [-0.0303, -0.0064] | **yes** |
| mar_Deva | ENSEMBLE - metadata_128 | recall@10 | -0.0080 | [-0.0262, +0.0095] | no |
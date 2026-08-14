
## Index ablation — all 7 subsets (1500 queries, hin+mar, aarch64)

| Configuration | Chunks | MRR@10 | R@10 | R@20 | search P50 | search P95 | search P99 | e2e P50 | e2e P99 | disk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_256 | 201,298 | 0.2895 | 0.5601 | 0.6607 | 4.277 | 5.984 | 8.544 | 30.205 | 50.487 | 623MB |
| semantic_128 | 239,175 | 0.2822 | 0.5552 | 0.6502 | 4.496 | 6.229 | 9.745 | 30.088 | 47.059 | 705MB |
| metadata_128 | 241,572 | 0.3030 | 0.5669 | 0.6675 | 4.318 | 6.108 | 8.369 | 33.171 | 92.999 | 722MB |
| fixed_256+semantic_128 | 440,473 | 0.2903 | 0.5637 | 0.6613 | 7.497 | 10.12 | 11.361 | 35.608 | 51.627 | 1328MB |
| fixed_256+metadata_128 | 442,870 | 0.2973 | 0.5684 | 0.6697 | 7.199 | 9.8 | 11.259 | 35.905 | 51.025 | 1345MB |
| semantic_128+metadata_128 | 480,747 | 0.2942 | 0.5642 | 0.6612 | 7.596 | 10.233 | 11.848 | 35.173 | 50.511 | 1427MB |
| fixed_256+semantic_128+metadata_128 | 682,045 | 0.2926 | 0.5717 | 0.6621 | 11.274 | 14.845 | 17.709 | 43.535 | 61.633 | 2050MB |

### Per-language

| Configuration | Lang | MRR@10 | R@10 | R@20 |
|---|---|---:|---:|---:|
| fixed_256 | hin_Deva | 0.3453 | 0.6336 | 0.7363 |
| fixed_256 | mar_Deva | 0.2355 | 0.4891 | 0.5876 |
| semantic_128 | hin_Deva | 0.3388 | 0.6307 | 0.7288 |
| semantic_128 | mar_Deva | 0.2276 | 0.4822 | 0.5742 |
| metadata_128 | hin_Deva | 0.3459 | 0.6257 | 0.7182 |
| metadata_128 | mar_Deva | 0.2616 | 0.5102 | 0.6185 |
| fixed_256+semantic_128 | hin_Deva | 0.3449 | 0.6368 | 0.7383 |
| fixed_256+semantic_128 | mar_Deva | 0.2376 | 0.4930 | 0.5869 |
| fixed_256+metadata_128 | hin_Deva | 0.3444 | 0.6296 | 0.7315 |
| fixed_256+metadata_128 | mar_Deva | 0.2519 | 0.5094 | 0.6099 |
| semantic_128+metadata_128 | hin_Deva | 0.3428 | 0.6300 | 0.7248 |
| semantic_128+metadata_128 | mar_Deva | 0.2472 | 0.5005 | 0.5997 |
| fixed_256+semantic_128+metadata_128 | hin_Deva | 0.3433 | 0.6436 | 0.7271 |
| fixed_256+semantic_128+metadata_128 | mar_Deva | 0.2435 | 0.5022 | 0.5994 |
# Session handover — 2026-08-15

State of the submission, what changed, and what is left. Deadline **2026-08-22
23:59 IST**, no resubmissions.

---

## Where things stand

**Live:** https://pucho.me — HTTPS (Let's Encrypt, valid to 12 Nov 2026)
**Repo:** https://github.com/siddharth-09/hhgoa-task2 — public
**Serving box:** AWS `m7i-flex.large` (`i-0902157eaa2aa8ddc`, ap-south-1b, `13.234.76.135`)

All six technical requirements are met and verifiable live.

| # | Requirement | Status |
|---|---|---|
| 1 | Speech-to-text (Sarvam) | Saaras v3, verified by a human speaking into it |
| 2 | Chunking "vast" | 12 variants, 7-subset ablation with paired CIs, live `/compare` |
| 3 | Under 200ms | **P50 55ms · P70 60ms · P100 104ms · 100/100** |
| 4 | P50/P70/P100 | Measured + reproducible live via `/benchmark` |
| 5 | Harness | Orchestration, 4-provider fallback chain, typed I/O |
| 6 | Guardrails | refuse / abstain / greeting / verify — behavioural suite green |

**Not done, and blocking submission:** two videos, social posts by every team
member (Instagram + X + LinkedIn, `#RAGInGoa`, ≥1 public Instagram), submission
form. None of these are engineering work.

---

## Infrastructure

```
AWS EC2 m7i-flex.large   2 vCPU Xeon 8488C (avx512_vnni), 7.7GB RAM, 20GB disk
  containers             hhgoa-api (port 127.0.0.1:8000), caddy (80/443)
  restart policy         unless-stopped -- survives reboot
  cost                   ~$0.09/hr from the $120 credit; MUST stay up through judging
  ssh                    ssh -i ~/.ssh/hhgoa-task2.pem ec2-user@13.234.76.135

Oracle box (80.225.231.132, shared)   benchmarking only, nothing served from it
```

Deploy is: rsync the changed dirs, `sudo docker build`, recreate `hhgoa-api`.
The full command is in the session log; `.env` on the box holds all keys and is
**not** in git.

> Oracle was measured and rejected as the serving box: P50 79ms vs AWS 64ms, and
> its P100 swung 135–305ms across runs because 24 neighbour containers share the
> 4 cores. Adding CPU did not fix it — P100 stayed ~391ms even with all 4 cores,
> which is what proved the tail was contention rather than compute.

---

## Serving configuration

```
index          metadata_128 (241,572 chunks) for Devanagari
               english_256 for Latin-script queries        <- new, see below
corpus         20,000 queries / 199,668 passages (hin + mar)
embedder       multilingual-e5-small, int8, avx512_vnni on x86
LLM primary    groq:llama-3.3-70b-versatile
LLM fallbacks  gemini-flash-lite -> groq:openai/gpt-oss-20b -> openrouter gemma-4-26b
```

Generation currently fires on **~75-80%** of queries; the rest are the model
declining because retrieval did not surface an answer. That ceiling is retrieval
(R@10 = 0.567), not the LLM.

---

## What changed this session

### Retrieval correctness
- **Cross-language id collision fixed.** MSMARCO-XI is parallel, so every language
  shard repeats the same `query_id`s; 99,834 of 99,834 passage ids collided. Ids
  are now namespaced by language. Repaired in place — no re-embedding.
- **Serving ensemble reduced to one index.** Exhaustive 7-subset ablation with
  paired bootstrap CIs: every recall difference is inside the noise and the
  3-index ensemble is significantly *behind* `metadata_128` on MRR@10.
- **The metadata hint was leaking into answers** — every answer began
  `[description] explanation description definition | …`. Fixed at source and in
  the built index.

### Latency
- `extract` owned the whole tail. Its cost tracks the **longest sentence in the
  batch** (r=0.84), not the number of sentences (r=0.02).
- **int8 embeddings are not batch-invariant** — dynamic quantisation computes
  activation scales across the batch, so a wide pad degrades every short sentence
  beside it. Batch 1 is both fastest and the fidelity reference.
- **One corpus row was the entire P100**: a 2,717-char query (translation
  artifact) against a median of 32. Queries are now capped at 512 chars.

### Generation
- **Marathi function words were missing from the verifier**, so valid Marathi
  answers were rejected as "novel facts" (`किंवा`, `ज्यामध्ये`). Half the corpus is
  Marathi.
- **Prose responses are recovered** instead of failing and burning retries (one
  such failure cost 15.5s).
- **Groq added and promoted to primary** — 4x faster than Gemini with identical
  script discipline. Chain now spans four vendors.
- **The chain switches provider instead of retrying a throttled one.**
- **Prompt now demands a complete sentence** — Groq was answering `नई दिल्ली`
  instead of `भारत की राजधानी नई दिल्ली है।`.

### Guardrails
- **Greetings no longer retrieve.** "Hello" was matching a C# passage containing
  `System.Console.WriteLine(Hello, World!)` at support 0.602 and being served as
  an answer. Now answered conversationally in 0.03ms.
- **The model's insufficiency verdict is honoured** — when the LLM says the
  context does not answer the question, the system abstains instead of falling
  back to a span drawn from that same context.
- **Borderline support defers to the LLM** rather than a threshold. Honest note:
  this rescued **0 of 40** queries in testing. The principle is right; the measured
  benefit on this corpus is nil.

### Reverted after measurement
- `context_passages` 4 → 8 → **back to 4**. Gold recall does improve with a wider
  window (43% → 58%), but the generation rate was identical at both settings and
  8 cost 2.4x the latency. Three separate measurements said the same thing.

---

## Testing

```bash
# behavioural — asserts what the system should DECIDE, not just how fast
python -m eval.behaviour --base https://pucho.me

# latency P50/P70/P100 on the serving box
docker compose run --rm bench python -m bench.fastpath --tag full --n 300

# retrieval quality, all 7 subsets
docker compose run --rm bench python -m eval.ablate_full --n-queries 1500
docker compose run --rm bench python -m eval.significance --n-queries 1500

# index integrity (run after ANY ingest)
docker compose run --rm bench python -m eval.audit_ids --tag full

# rank LLM providers on Devanagari output + latency
docker compose run --rm bench python -m eval.compare_llms
```

All raw results are in `data/reports/`.

---

## Known weaknesses — say these before a judge finds them

1. **Marathi is materially weaker than Hindi** (MRR 0.26 vs 0.35). Largest quality
   gap in the system, larger than any difference between configurations.
2. **`metadata_128`'s advantage is partly a dataset label leak.** It embeds the
   passage's `query_type`, which the dataset derives from the query that owns the
   passage. Advantage vs corpus share correlates −0.638. Documented in the README.
3. **The 200ms budget excludes generation.** Defensible — the task enumerates from
   "chunking", and the extractive answer is complete, grounded and cited — but say
   it plainly rather than letting anyone infer the LLM answered in 55ms.
4. **The corpus is 1.3% of the dataset** (20k of 778,638 queries per language).
   Frozen deliberately: no requirement scales with corpus size, and re-scaling
   would invalidate every measured number.
5. **Generation fires on ~75-80%**, not 100%. The rest is the model declining.

---

## What is left

1. **Two videos** — 90s team/process, and an end-to-end demo
2. **Social posts** — every member, 3 platforms, `#RAGInGoa`, ≥1 public Instagram.
   Start early; it depends on teammates.
3. **Submission form**
4. **Keep the EC2 instance running** until judging is over

### Demo notes
- Use `मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?` — the two answer tiers
  visibly differ there, and the "what the LLM changed" expander shows it.
- Then ask `भारत का प्रधानमंत्री कौन है?` and let it abstain on camera. The model
  knows the answer; the system refuses because the corpus does not support it.
  That is requirement 6 demonstrated rather than claimed.
- Fire one throwaway query before recording — the first query after idle is slower.

---

## Git state

Everything is committed locally. **Nothing has been pushed since you asked me to
stop pushing** — review and push when ready:

```bash
git log origin/main..HEAD --oneline    # what is waiting
git push origin main
```

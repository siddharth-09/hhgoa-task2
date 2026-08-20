# Session handover — 2026-08-15

State of the submission, what changed, and what is left. Deadline **2026-08-22
23:59 IST**, no resubmissions.

---

## Where things stand

**Live:** https://pucho.me — HTTPS (Let's Encrypt, valid to 12 Nov 2026)
**Repo:** https://github.com/siddharth-09/hhgoa-task2 — public
**Serving box:** AWS `m7i-flex.large` (`i-0902157eaa2aa8ddc`, ap-south-1b, `13.202.200.164` (Elastic IP))

All six technical requirements are met and verifiable live.

| # | Requirement | Status |
|---|---|---|
| 1 | Speech-to-text (Sarvam) | Saaras v3, verified by a human speaking into it |
| 2 | Chunking "vast" | 12 variants, 7-subset ablation with paired CIs, live `/compare` |
| 3 | Under 200ms | **P50 52ms · P70 58ms · P100 100ms · 100/100** |
| 4 | P50/P70/P100 | Measured + reproducible live via `/benchmark` |
| 5 | Harness | Orchestration, 4-provider fallback chain, typed I/O |
| 6 | Guardrails | refuse / abstain / greeting / verify — **20/20 behavioural cases** |

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
  ssh                    ssh -i ~/.ssh/hhgoa-task2.pem ec2-user@13.202.200.164

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
               english_256  (98,836 chunks)  for Latin-script queries
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

### English index — built and live (2026-08-15)
MSMARCO-XI ships every passage twice: the original English and the Indic
translation. Only the translation was ever indexed, so an English question could
not reach an answer the corpus demonstrably held.

- 98,812 deduped English passages (the source is shared across language shards,
  so 199,668 rows hold only 98,812 distinct passages), 98,836 chunks, 240MB
- Built from `data/raw/*.jsonl` -- **nothing was re-downloaded**
- 21.6 min to embed at 76/s: `docker compose run --rm bench python -m ingest.build_english`
- Routed by **script**, not detected language (`core/retriever.is_latin_query`,
  threshold 0.9). Script is observable; language is a guess, and the guess is
  wrong exactly where it matters -- romanised Hindi is Latin script but Hindi.
- Additive: a query with real Devanagari content never leaves the path that
  already worked.

Measured effect:

    What is the capital of India?   0.359 abstain  ->  0.793 generated  [english]
    भारत की राजधानी क्या है?          0.787 generated ->  unchanged        [indic]
    who is the prime minister...    abstain      ->  abstain           [english]

That third line is the one that mattered. The pilot warned the English index
raises support across the board, so an out-of-corpus English question could have
started answering confidently from loosely-related text. It still abstains.

`/compare` deliberately excludes the English index -- it is a different corpus,
not a different chunking strategy, and listing it there would compare unlike
things.

### Unsourced answers when the corpus cannot help (`ALLOW_UNSOURCED`, ON)
When the system abstains, it now *also* asks the model what it knows on its own
and shows that separately. This was a product decision taken deliberately, with
the risk understood.

- Only fires on `abstain`. **Never** on a refusal (a blocked credential request
  stays blocked) and never on a greeting.
- Lives in its own field `unsourced_answer`; `answer`, `answer_source`, support,
  grounding and every benchmark are untouched.
- No citations, no grounding score, and rendered in amber behind
  "⚠ not from the corpus · model's own knowledge" with an explicit caveat line.
- Off with `ALLOW_UNSOURCED=false` -- one env var, no redeploy of code.

    भारत का प्रधानमंत्री कौन है?  ->  abstain  +  "नरेंद्र मोदी भारत के प्रधानमंत्री हैं"
    मेरे बैंक खाते का पासवर्ड...    ->  refusal  +  nothing

**The risk, stated plainly:** requirement 6 names "answers not grounded in the
retrieved context" as the thing to guard against. A strict reading could treat any
ungrounded output as a violation even when labelled. The mitigation is that the
graded behaviour is unchanged -- the system still abstains, still reports
`answer_source: "abstain"`, and the unsourced text is visibly a different kind of
thing. If a judge pushes on it, `ALLOW_UNSOURCED=false` reverts to a pure abstain
in seconds.

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


---

# Session 2 — 2026-08-20

## Changed since the last handover

**Elastic IP allocated.** The public address moved from `13.234.76.135` to
**`13.202.200.164`** and is now *static*. Before this it was an Amazon-assigned
dynamic IP, so any stop of the instance silently changed it and broke DNS. The
Namecheap A record has been updated. No further DNS changes should ever be needed.

> If the site looks down, check `dig +short pucho.me` first. A stale resolver
> holding the old IP looks exactly like an outage. Brave keeps its own DNS cache
> separate from macOS -- clear it at `brave://net-internals/#dns`.

**Google Analytics added** (`G-1VE4ZKC6MK`). Deployed and verified firing --
`page_view` plus custom events: `ask_question` (method + script), `answer_outcome`
(source, route, within_budget, fast_path_ms), `generation_outcome`, `voice_used`,
`benchmark_run`, `compare_run`. No question text is ever sent, and `track()`
no-ops if gtag is blocked, so analytics can never break the page.

> Brave blocks GA by default, so your own visits will not appear. Only Realtime
> populates immediately; standard reports lag 24-48h on a new property.

## Scaling: what is and is not possible

The organisers said scoring runs an **eval loop, ~10 loops**. Whether it is
sequential or parallel changes everything and is still **unanswered -- ask them.**

Measured on the live box (2 vCPU):

    concurrency 1   median  55ms    0/2  over budget
    concurrency 2   median 121ms    0/4
    concurrency 4   median 162ms    2/8
    concurrency 8   median 337ms   14/16

Nothing errors at any level -- it degrades, it does not fall over.

| Option | Available | Helps a parallel eval |
|---|---|---|
| Resize to 4 vCPU | **BLOCKED** -- `FreeTierRestrictionError` | — |
| `ORT_THREADS=1` | tested | No: single-request 55 -> 63ms, only ~8% better under load |
| 2nd instance + DNS round-robin | yes | **No** -- RR splits resolvers, not requests; one eval client lands entirely on one box |
| 2nd instance + Caddy load balancing | yes | ~1.6x |
| 2nd instance + **ALB** | **yes, probed and confirmed not blocked** | ~2x, with health checks |

ALB costs ~$0.02/hr plus ~$0.09/hr for the second box, about **$18/week**. The
index would be copied **box-to-box inside AWS** (2-5 min, free same-AZ), never
re-uploaded from the Mac. An ALB terminates TLS, so Caddy's automatic certificate
would be replaced by an ACM cert -- roughly 30 extra minutes and a change to a
part of the stack that currently works.

**Recommendation: build none of it until the organisers confirm the eval is
parallel.** At 55ms sequential this is an hour of work and a second thing that can
fail, for zero gain.

## AWS account state

    EC2      i-0902157eaa2aa8ddc  m7i-flex.large, running, Elastic IP attached
    EIP      13.202.200.164       KEEP -- release only when the instance is retired
    Budgets  2 (free)             $50 budget with 80%/100% email alerts
    S3       hhgoa-task2-index-…  8 bytes, pre-existing
    Lambda   none                 created for the credit, deleted after it posted
    RDS      none                 the last $20 activity, still not started

Free-tier credits: **$60 of $100** earned. Spend to date under $1. The remaining
$20 needs an RDS instance -- billable (~$15/mo if the account is not free-tier
eligible), so create it, wait for the credit to post, then delete.

Repeatedly confirmed blocked by the Free Plan: Bedrock *inference* (models list
fine), `c7g` instance types, instance resizing, public Lambda Function URLs.

## Demo video

A two-person script was written this session (in the chat log). Verified short
questions to use:

    ताजमहल कहाँ है?              answers AND visibly rewrites -- the best demo moment
    भारत का प्रधानमंत्री कौन है?   abstains + shows the labelled unsourced answer

Avoid `भारत की राजधानी क्या है?` and `मैनहट्टन परियोजना क्या थी?` on camera: both
answer correctly but the LLM returns the span *unchanged*, so nothing visibly
happens. `लिफ्ट का मतलब क्या है?` abstains.

The 90-second process video must show **process, not product** -- scrolling
`docs/BUILD_LOG.md` while talking through one thing that went wrong is the
easiest honest version.

## Still blocking submission

1. Two videos
2. Social posts -- **every member**, Instagram + X + LinkedIn, `#RAGInGoa`, >=1 public Instagram
3. Submission form
4. Keep the EC2 instance running through judging

## Note to whoever works on this next

Do not restart the live `hhgoa-api` container to run experiments -- that is the
submitted link and it goes down while you do. Test against a scratch container on
a spare port, or against `localhost:8000` on the box.

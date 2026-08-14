# Build log — HH Goa 2026 Task 2 (Voice-Enabled RAG)

Running record of decisions, measured results, and infrastructure state.
Updated after every successful run. Newest entries at the bottom.

**Deadline:** 2026-08-22, 23:59 IST — no resubmissions.

---

## Locked decisions

| Area | Decision | Why |
|---|---|---|
| STT | Sarvam Saaras v3 | Indic + code-mixed speech; dataset is Indic |
| Generation | 4-model chain across 3 vendors, Gemini primary | Bedrock (Claude Haiku) blocked by AWS Free Plan; free tiers throttle, so vendor diversity is the recovery path |
| Embeddings | `intfloat/multilingual-e5-small` (384-dim, ONNX) | Multilingual — bge-small is English-only and fails on Devanagari |
| Vector index | `hnswlib` | Builds clean on ARM; `faiss-cpu` aarch64 wheels are unreliable |
| Sparse index | `bm25s` | Pure numpy, ARM-safe. **Caveat: O(corpus) — 30ms at 201k chunks, see below** |
| Serving ensemble | `fixed_256 + semantic_128 + metadata_128` | Chosen by ablation, not intuition |
| Region | `ap-south-1` (Mumbai) | Closest to judges — biggest single lever on end-to-end latency |
| Ingestion | Wherever is free; index is a portable artifact | Architecture-independent, so build anywhere and serve anywhere |
| Serving | TBD — HF Spaces / Oracle / App Runner | App Runner needs the AWS Paid Plan, same block as Bedrock |
| Interface | Web page (phone IVR only as a bonus layer) | "Live working link" is a submission requirement and an auto-reject check |

> Two of these changed under measurement rather than opinion. Generation moved off
> Claude Haiku because the AWS Free Plan blocks Bedrock inference outright. Ingestion
> moved off EC2 Graviton because the Free Plan blocks `c7g` too — and because the
> index turns out to be architecture-independent, so the "must build on the serving
> arch" reasoning was wrong. Only *benchmarks* need the serving box.

### Rejected, with reasons

- **DigitalOcean $200 student credit** — program ended 2026-08-01, credits retroactively expired.
- **Heroku student credit ($13/mo × 24)** — biggest credit, but 512MB dynos won't hold the index.
- **Render free tier** — sleeps after 15 min idle, ~50s cold start. A judge would see a broken link.
- **Oracle Cloud free ARM (4 OCPU/24GB)** — best free hardware, but A1 capacity shortages and
  hand-rolled TLS are too risky on a 9-day deadline. Kept as a migration target.
- **Phone IVR as the primary interface** — a phone number is not a link, and Indian DID
  provisioning needs KYC/DLT registration that can take weeks.

---

## Requirement → implementation map

| # | Requirement | Where it's satisfied |
|---|---|---|
| 1 | Speech-to-text (Sarvam or ElevenLabs) | Sarvam STT, batch then streaming |
| 2 | Chunking must be "vast" | `ingest/chunkers.py` — 4 strategies, scored per query_type in `eval/` |
| 3 | Latency < 200ms | Transcript → final output: route + retrieve + rerank + extract. Measured on the serving box. |
| 4 | P50/P70/P100 analytics | `bench/` — 200+ queries, per-stage timers |
| 5 | Harness | `core/harness.py` — Pydantic I/O, retries, timeouts, fallbacks |
| 6 | Guardrails | `core/guardrails.py` — input filter + grounding gate + explicit abstain |

**Latency target — RESOLVED at 200ms (2026-08-13).**

The organisers reissued the task PDF. Section 3 now reads **"under 200ms"**; the original
said 50ms. Everything else in the brief is unchanged. The public task page always said
200ms, so the two sources now agree.

The wording matters:

> "The full process — **chunking + vector DB retrieval + everything through to final
> output** — should complete in under 200ms."

The enumeration begins at *chunking*, not at voice input. **Speech-to-text is not in the
budget.** The measured window is transcript → final output. Reported separately: STT
round-trip, and the LLM-polished answer.

(For reference, a 50ms end-to-end target would have been unachievable regardless —
Sarvam's round-trip is 200–800ms alone and Haiku's TTFT is ~300ms.)

**Design response — two-tier answering.** The 200ms figure is loose enough to return a
grounded answer with no LLM call at all:

    transcript → route → retrieve → rerank → extract best-supported span    ~60–90ms

The extractive answer is cited and grounded, and lands inside 200ms measured. The
Haiku-generated fluent answer then streams in and replaces it. This:

- satisfies the strict reading with a real measured number, not a caveat
- makes the demo feel instant instead of showing a spinner
- doubles as the LLM-failure fallback that requirement #5 wants anyway

We publish per-stage and end-to-end breakdowns regardless.

---

## Dataset — `ai4bharat/MSMARCO-XI`

Full measured schema in [`ingest/SCHEMA.md`](../ingest/SCHEMA.md).

- One parquet per language: 13 train, 14 validation (validation adds `tel`)
- Each train shard: ~3.7GB, **778,638 rows**, **a single row group** (~9.7GB uncompressed)
- A row is a **query**, not a passage: mean 9.98 passages/query → **7.77M passages per language**
- ~100M passages across all 13 languages → subsetting is mandatory
- **62.2%** of queries have ≥1 `is_selected` passage → free ground-truth relevance labels
- `query_type`: DESCRIPTION 52.9% / NUMERIC 26.3% / ENTITY 8.9% / LOCATION 6.3% / PERSON 5.6%

**Scope:** Hindi + Marathi, 50k labelled queries each → ~500k passages. English rides along
free in `English_passages`. Subset size documented honestly in the README.

**Why this schema matters:** `is_selected` makes the chunking comparison a measured benchmark
instead of a design essay, and `query_type` is a ready-made label for the router.

---

## Timeline

| Day | Work | Status |
|---|---|---|
| 1 — Aug 13 | Scaffold, schema recon, chunkers, AWS setup | in progress |
| 2 | Index build, all 4 strategies | |
| 3 | Eval harness → strategy × query_type table → router | |
| 4 | Retrieval core + instrumentation; prove <50ms | |
| 5 | Harness, guardrails, Claude generation | |
| 6 | Sarvam STT + web page + latency panel | |
| 7 | Benchmark, deploy, domain | |
| 8 | Buffer / IVR bonus / polish | |
| 9 — Aug 21 | Videos, README, social posts | |
| Aug 22 | **Submit in the morning** | |

---

# Run log

## 2026-08-13 — Day 1

### Dataset recon
Probed the parquet footer over HTTP range requests (1.2MB downloaded, not 3.7GB) because
the HF Dataset Viewer returns "dataset generation failed" for this repo.
Results recorded in `ingest/SCHEMA.md`. Key finding: the corpus is ~10x larger than the
`10M<n<100M` tag implies, because rows are queries and each carries ~10 passages.

### Chunkers — `ingest/chunkers.py`
Four strategies implemented: `fixed`, `sentence`, `semantic`, `metadata`.

Verified on Devanagari input:
- danda `।`, double danda `॥`, `?`, `!` all segment correctly
- Latin sentence splitting unaffected
- short trailing fragments merge into the previous chunk (no 1-token index entries)
- metadata strategy prepends a query-type hint into the embedded text

> Devanagari gotcha: sentence segmentation cannot key on `.` alone. Hindi/Marathi
> terminate with U+0964 / U+0965.

### Embedder — `core/embedder.py`
`multilingual-e5-small`, ONNX. Official repo ships ONNX directly — no export needed.

- Variant auto-selection: generic int8 (`Xenova`) on aarch64, `avx512_vnni` int8 on x86.
  The official quantized file is x86-tuned and is the wrong choice for Graviton.
- e5 prefixes (`query: ` / `passage: `) enforced by separate methods so call sites can't skip them
- mean-pool over attention mask → L2 normalise, so dot product == cosine downstream

> Size correction: multilingual-e5-small is 118M params (XLM-R 250k vocab), so
> ~470MB fp32 / ~120MB int8 — not the ~35MB that bge-small would have been.

### AWS
Account `573594947163`, IAM user `hhgoa-task2`, region `ap-south-1`.
Credit: **$120, 185 days remaining**.

- EC2 + S3 permissions verified
- Graviton available in `ap-south-1`: `c7g.2xlarge`, `c7g.4xlarge`, `t4g.medium`, `t4g.small`
- S3 bucket `hhgoa-task2-index-573594947163` created, public access fully blocked

> zsh gotcha: unquoted `$VAR` does not word-split, so `aws $FLAGS` passes one argument.
> Use `AWS_PROFILE` / `AWS_DEFAULT_REGION` env vars instead of building flag strings.

### AWS Free Plan blocker (unresolved)
`RunInstances` on `c7g.2xlarge` was rejected:

> InvalidParameterCombination: The specified instance type is not eligible for Free Tier.

The account is on AWS's restricted **Free Plan** (the same programme that granted the
$120 credit). Only free-tier-eligible instance types can launch:

| Type | vCPU | RAM | Arch |
|---|---:|---:|---|
| `m7i-flex.large` | 2 | 8GB | x86_64 |
| `c7i-flex.large` | 2 | 4GB | x86_64 |
| `t4g.small` | 2 | 2GB | arm64 |
| `t3.small` | 2 | 2GB | x86_64 |
| `t4g.micro` / `t3.micro` | 2 | 1GB | arm64 / x86_64 |

No Graviton above `t4g.small` (2GB — too tight for index building).

**Note:** App Runner is x86-only, so if it remains the serving target, x86 *is* the
correct benchmarking architecture. The ARM-safe dependency choices cost nothing and
keep the Oracle fallback open.

**Expect this to recur at deploy time** — App Runner is not free-tier eligible either,
so the Free Plan will likely block it too.

Provisioned so far (all free, nothing billing):
- key pair `hhgoa-task2` → `~/.ssh/hhgoa-task2.pem`
- security group `sg-03a8d3eca860bf1b7` — ports 22 + 8000, restricted to a single IP
- AMI identified for later: `ami-07c8b91119c5b1b1e` (AL2023 arm64, 2026-08-03)
- **no EC2 instance launched — $0 spent**

Also denied to this IAM user: `ssm:GetParameters`, `iam:*`, `freetier:*`. AMI lookup works
via `ec2:DescribeImages` instead. Instance profiles can be avoided entirely by using
presigned S3 URLs, which also keeps credentials off the box.

### Oracle box available — AWS EC2 dropped
Teammate already runs an Oracle Always Free A1: **Ubuntu 24.04 aarch64, 4 cores,
24GB RAM (12.2GB already in use), 57 days uptime.**

Strictly better than anything the AWS Free Plan allows (`m7i-flex.large`, 2 vCPU/8GB)
and free. The capacity/signup risk that ranked Oracle third no longer applies — the box
already exists. The early ARM-safe dependency choices (`hnswlib` over faiss,
`onnxruntime`, `int8_arm` embedder variant) pay off exactly as intended.

- **AWS EC2: dropped.** $0 spent. S3 bucket kept for index artifacts; credit untouched.
- **Everything on Oracle runs in Docker** (project constraint — the box has other
  workloads on it and must not be polluted with system-level Python installs).
- Container CPU/memory limits set to 3 cores / 8GB so the ingest job cannot starve
  the box's existing services.

Access: ed25519 keypair generated at `~/.ssh/hhgoa-oracle`, public half handed to the
teammate for `authorized_keys`. No passphrase (non-interactive use) — remove after Aug 22.

### Estimated pipeline cost (measured assumptions, not yet validated)
Embedding dominates: ~2.1M chunks across 4 strategies **plus ~1.5M sentence embeddings**,
because semantic chunking must embed every sentence before deciding where to cut.
Devanagari also fragments into 2–3× more XLM-R subword tokens than English.

| | Oracle (4× Ampere) | Mac (M4, 4P+6E) |
|---|---|---|
| Embedding (~3.6M passes) | 3–6 hrs | 1.5–3 hrs |
| Full pipeline | ~4.5–7.5 hrs | ~2.5–4 hrs |

Index artifacts are architecture-independent, so they can be built anywhere.
**Latency benchmarks are not — those must run on the serving box.**

Plan: validate with a 5k-query pilot, then run the full job overnight under tmux.

### Full ingestion + eval pipeline written (not yet run)
~1,200 lines across six modules. Nothing has been executed against real data yet.

| File | Purpose |
|---|---|
| `ingest/download.py` | Streams a language shard → JSONL (labelled queries only) |
| `ingest/chunkers.py` | 4 strategies: fixed / sentence / semantic / metadata |
| `core/embedder.py` | ONNX multilingual-e5-small, arch-aware variant selection |
| `core/index.py` | HNSW (dense) + BM25 (sparse), fused with RRF |
| `ingest/pipeline.py` | Orchestrator, checkpointed per stage |
| `eval/evaluate.py` | recall@k / MRR@10 / nDCG@10 per strategy × query_type |

Design notes worth keeping:
- **Hybrid retrieval, not dense-only.** BM25 covers exact tokens — numbers, names,
  transliterated entities — which is where dense embeddings are weakest and where
  NUMERIC/PERSON/ENTITY queries live. RRF fuses by rank, so BM25's unbounded scores
  never need calibrating against cosine.
- **Eval scores at passage granularity, not chunk.** Labels are defined over passages,
  and scoring per chunk would reward whichever strategy fragments the most.
- **The metadata strategy's type hint is embedded but hidden from BM25** — it helps the
  encoder, but would pollute lexical matching.
- **Every stage checkpoints.** A crash during a 3–6 hour embedding run must not cost
  the download and chunking that preceded it.

### Pilot run #1 — the chunking comparison returned a NULL RESULT
2,000 Hindi queries → 19,978 passages → ~20,200 chunks per strategy. 22.5 min end to end.

| Strategy | Chunks | MRR@10 | nDCG@10 | R@10 | search p50 |
|---|---:|---:|---:|---:|---:|
| fixed | 20,111 | 0.3339 | 0.4104 | 0.6729 | 0.96ms |
| sentence | 20,219 | 0.3336 | 0.4100 | 0.6724 | 1.05ms |
| semantic | 20,146 | **0.3347** | 0.4110 | 0.6756 | 1.05ms |
| metadata | 20,248 | 0.3307 | 0.4081 | 0.6710 | 0.97ms |

**A 1.2% spread across four strategies is noise, not signal.** The per-query_type
"winners" are noise too.

**Cause:** 19,978 passages → ~20,200 chunks = **1.01 chunks per passage**. MSMARCO ships
pre-segmented passages that sit below any sensible chunk budget, so all four strategies
emitted "the whole passage, unchanged". We built four ways to not split anything.

The one strategy that did differ — `metadata`, which prepends a query-type hint — came
out *worst*. Prepending text dilutes the embedding rather than sharpening it.

**Search latency: ~1ms p50** against a 200ms budget. Retrieval is a rounding error;
the budget will be spent on reranking and generation.

### Literature check — the null result is the correct result
- Vecta (Feb 2026), 7 strategies / 50 papers: recursive 512-token splitting **1st at 69%**;
  semantic chunking **54%**. Validated default: **512 tokens, 10–20% overlap**.
- Semantic chunking's paradox: highest recall (91.9%, Chroma) but *lower* end-to-end
  accuracy — topic-pure chunks strip context the LLM needs. **~14× slower.**
- Domain dependence is real: adaptive chunking hit 87% vs 13% fixed in clinical decision
  support (p=0.001).
- **For pre-segmented corpora like MSMARCO the established practice is not to re-chunk
  at all** — hybrid BM25 + dense with RRF, then cross-encoder reranking.
- NVIDIA: factoid queries best at 256–512 tokens, multi-hop analytical at 512–1024
  → routing on chunk **size** has literature support; routing on chunking *algorithm* does not.

### Embedder variant benchmark — hypothesis was wrong
Suspected int8 was falling back to slow kernels on ARM. Measured instead:

| Variant | short (~20tok) | medium | long (~250tok) |
|---|---:|---:|---:|
| fp32 | 361/s | 93/s | 30/s |
| int8_x86 | 1080/s | 237/s | 65/s |
| **int8_arm** | **1286/s** | **267/s** | **74/s** |

int8 on ARM is **3.5× faster** than fp32. The original choice was already optimal.

**The useful finding is different: throughput is superlinear in sequence length.**
3× shorter text → 3.6× faster. So smaller chunks are both more informative *and* cheaper
to embed — two 128-token chunks cost less than one 250-token chunk.

> Fixed a real bug found here: batches were padded to their longest member, so a single
> long chunk forced 63 others to pad up to it. Length-bucketing before batching cut
> `embed:fixed` from >420s to 288s (~1.5×).

### Pilot run #2 — experiment redesigned around size and granularity
Variant grid is now (strategy × chunk size × granularity). Chunk counts per unit:

| Variant | chunks/unit |
|---|---:|
| fixed_256 | 1.01 |
| sentence_128 | 1.17 |
| fixed_128 | 1.20 |
| semantic_128 | 1.20 |
| metadata_128 | 1.22 |
| **doc_fixed_256** | **3.00** |
| **doc_sentence_256** | **4.60** |

Passage granularity stays flat even at a 128-token budget — only ~20% of passages split,
so Hindi MSMARCO passages are mostly **under 128 XLM-R tokens**. Passage-level chunking is
a no-op at any sensible budget.

**Document granularity is where strategies finally diverge**: grouping the ~10 passages
sharing a `query_id` into ~1,500-token documents makes `fixed` and `sentence` differ by
53% in chunk count. That is a real experiment.

> Eval fix required by this: gold labels are per-passage, but a doc chunk aggregates ~10
> passages, so matching on parent id would count a hit whenever the document was retrieved
> — trivially true. Scoring now credits a doc chunk only for gold passages whose text it
> actually contains (normalised prefix signature), and metrics are computed over gold
> *coverage* so both granularities compare fairly.

### Devanagari tokenisation bug — the single biggest win of the day
Python's `\w` matches by Unicode alphanumeric class. Devanagari vowel signs
(ि ा ो ी), the virama (्) and chandrabindu (ँ) are combining marks (Mn/Mc) and are
**not** alphanumeric, so `re.findall(r"\w+", ...)` shatters every Hindi word:

```
दिल्ली    -> ['द', 'ल', 'ल']
विस्फोट   -> ['व', 'स', 'फ', 'ट']
ताजमहल    -> ['त', 'जमहल']
Delhi     -> ['Delhi']            <- English unaffected, which is how it hid
```

Three components were affected, including `bm25s`, whose default token pattern is
`(?u)\b\w\w+\b`:

```
'ताजमहल आगरा में स्थित है। दिल्ली भारत की राजधानी है।'
  -> bm25s vocab: ['जमहल', 'आगर', 'रत', 'जध']      (4 fragments from ~10 words)
```

So **the sparse half of hybrid retrieval was indexing consonant fragments** in every
index built that day. Fixed in `core/text.py` by defining tokens via *separators*
(whitespace, danda, punctuation) rather than character class — script-agnostic, and
correct for Latin too.

**Impact — same harness, same 1200 queries, before → after:**

| | MRR@10 | nDCG@10 | R@10 |
|---|---|---|---|
| single `fixed_256` | 0.3607 → **0.4039** | 0.4376 → 0.4836 | 0.7098 → 0.7572 |
| P2 ensemble | 0.3761 → **0.4065** | 0.4539 → 0.4883 | 0.7277 → 0.7678 |
| P3 ensemble | 0.3738 → **0.4069** | 0.4521 → 0.4904 | 0.7251 → 0.7757 |

**+12% MRR from a tokenizer fix** — three times what the ensemble work gained.

It also **inverted a previous conclusion**: `metadata_128` appeared to *hurt* the
ensemble before the fix and *helps* after (+1.0% R@10). The earlier ablation was
measuring a broken component, not a property of the strategy. Worth remembering
before trusting any ablation run over a system with a known defect.

And it repaired hallucination detection: the wrong-city/wrong-emperor answer scored
`novel=['अकबर']` → ALLOW before, and `novel=['अकबर','दिल्ली']` → ABSTAIN after.

Unchanged by the fix: `doc_fixed_256` still contributes nothing (identical metrics
to four decimals while filling 19.3% of returned slots). Dropped from the ensemble.

**Serving ensemble is now `fixed_256 + semantic_128 + metadata_128`** — 3.16ms,
MRR@10 0.4069, R@10 0.7757.

### Extractive answering + guardrails built
`core/extractive.py` — best-supported span from retrieved chunks, no LLM. Blends
cosine (paraphrase, cross-lingual) with lexical overlap (numbers, names, entities).

Lexical prefilter added after measuring that sentence embedding, not retrieval,
dominated latency:

| prefilter | p50 | p100 | answer identical to unfiltered |
|---|---|---|---|
| none (24) | 41.0ms | **182.3ms** | 100% |
| **10** | 37.3ms | **100.6ms** | **95.0%** |
| 6 | 25.7ms | 69.8ms | 87.5% |

Unfiltered peaked at 182ms — uncomfortably close to the 200ms budget, and the tail
is exactly what P100 reports. Chose 10: P100 cut 1.8x, answer changes 1 query in 20.
Mean support was flat across all three, which is precisely why support is not a
sufficient quality check — a different sentence can score just as well and be worse.

`core/guardrails.py` — 10/10 on input tests, 4/4 on output tests.

> Input intent cannot be a score threshold, measured: "मेरे बैंक खाते का पासवर्ड क्या है?"
> scored **0.596 support** — *higher* than several legitimate questions — because
> MSMARCO genuinely contains bank-security passages and retrieval did its job.
> Grounding says "the corpus discusses this", not "answering is appropriate".

> A first regex version caught the English credential phrasing and **missed the Hindi**
> one, because it allowed only one word between possessive and keyword; Hindi
> postpositions put three in "मेरे बैंक खाते का पासवर्ड". The worst possible failure
> mode on an Indic-language system.

### LLM provider: Bedrock blocked, Gemini adopted
`core/llm.py` — three backends (Gemini / Bedrock / Anthropic) behind one interface,
selected by `LLM_PROVIDER`. Generation is the only stage that leaves the machine, so
it is the only one that fails for reasons we do not control; a one-env-var swap is
deployment-level error recovery.

**Bedrock: blocked by the AWS Free Plan.** IAM permissions were added and the read-only
APIs work (`list-foundation-models`, `list-inference-profiles`), but *every* model —
including Amazon's own Nova — returns `ValidationException: Operation not allowed` on
`Converse`. Account-level gating, same restriction that blocked `c7g.2xlarge`.
`anthropic.claude-haiku-4-5` is ACTIVE in `ap-south-1` and would be the best option
(Claude quality, Mumbai latency, paid from the $120 credit) if the plan is upgraded.
The backend is written and tested against the API — one env var away.

**Gemini model selection — `models.list()` is not a capability check.** It returned
`gemini-2.5-flash`, which 404s with "no longer available" on `generateContent`.
Measured on identical input:

| model | latency | result |
|---|---:|---|
| `gemini-flash-lite-latest` | **1529ms** | correct, cites [1,2] |
| `gemini-flash-latest` | 4029ms | correct |
| `gemini-3.1-flash-lite` | 6025ms | correct |
| `gemini-3-flash-preview` | — | returns no JSON object |

Also: Gemini rejects short deadlines (`Manually set deadline 8s is too short`), so
`LLM_TIMEOUT_S` floors at 30 — a provider constraint, not a preference.

> Two retry-classifier bugs found by these failures. A retired model burned 3 attempts
> and 5.8s because the classifier checked for `"not found"` while Gemini says `NOT_FOUND`;
> Bedrock's `ValidationException` was likewise retried. Both now fail in one attempt
> (~400-500ms) and fall through to the extractive answer — which is the error-recovery
> behaviour requirement #5 asks for, demonstrated rather than claimed.

> Known defect: `gemini-flash-lite-latest` occasionally mixes scripts, e.g.
> "शाहजहाँ **نے** बनवाया" (Urdu `نے` for Devanagari `ने`). Invisible in a demo video but
> wrong. Try a prompt-level script constraint before paying 2.6x latency for
> `gemini-flash-latest`.

### Harness complete — `core/harness.py`
One orchestrated call: `check_input → embed → retrieve → extract → check_output →
generate → verify`, every stage timed.

- **The extractive answer is computed before generation and never depends on it.** That
  is what makes the sub-200ms claim measurable, and it means an LLM timeout, error,
  malformed JSON, or a rejected verification leaves a real grounded answer standing.
  Error recovery is a second already-computed answer, not a try/except.
- **Guardrails run on both sides of generation** — intent before retrieval, grounding on
  the extractive span *and* independently on the generated text. A prompt asking the
  model to stay grounded is a request; the post-check is the constraint.

**Measured, 40 real queries:**

```
FAST PATH   p50  65.1ms   p70  69.6ms   p100  128.6ms    40/40 under 200ms
FULL        p50 1125ms    p70 1213ms    p100 4611ms
answer via: generated 19 · extractive 20 · abstain 1
```

> Hindi inflection bug in the verifier. A correct generated answer was rejected for
> `novel_facts(जिनमें, लाभों, शामिल)` — two connectives and one inflected plural of
> `लाभ`, which *was* in the context. Hindi inflects by suffix, so faithful paraphrase
> exact-matches nothing. Fixed with prefix matching (4 chars), with numerals exempt:
> "1653" and "1888" share no prefix and must never be excused as morphology, since
> that is precisely the hallucination worth catching.

### Fallback rate + script mixing — both diagnosed and fixed
The 50% extractive fallback was **not** a verifier problem. Instrumented breakdown:

```
19  llm_failed                 -> 429 RESOURCE_EXHAUSTED (Gemini free-tier quota)
 4  llm_reported_insufficient  -> legitimate; model judged context inadequate
 3  generation_rejected        -> novel_facts(जाते, रहते) — still inflection
```

Every one of those 19 still returned a grounded extractive answer. The harness
degraded exactly as designed under a real failure, not a simulated one.

| | before | after |
|---|---|---|
| generated | 19/40 (48%) | 10/15 (67%) |
| `generation_rejected` | 3 | **0** |
| `llm_failed` (429) | 19 | **0** |
| script mixing | present | **0/10** |
| fast path p50 | 65.1ms | 61.6ms |

- **429s**: 0.4s backoff is useless against a per-minute quota. Rate-limit errors now
  get 3s x attempt; other transient faults keep the short backoff.
- **Script mixing**: the prompt now names the script explicitly ("write ने, not نے").
- **Inflection**: a closed-class function-word list. `जाते`/`रहते`/`जिनमें` assert no
  fact, but a proper noun of identical length (`अकबर`) does — length cannot separate
  them, a list can, at zero runtime cost.

All remaining fallbacks are now `llm_reported_insufficient`, i.e. the model declining
rather than the plumbing failing.

> Free-tier quota still throttles bulk runs. This aligns with the brief anyway: the
> 200ms budget starts at "chunking", so the headline P50/P70/P100 measures the **fast
> path** (no LLM, 200+ queries, no quota exposure) with generation latency reported
> separately over a smaller sample.

### Speech-to-text — requirement #1 complete
`core/stt.py` — Sarvam Saaras v3. `saarika:v2.5` is deprecated; `saaras:v3` with
`mode=transcribe` replaces it. 30s audio cap, mono, 16kHz recommended.

Mode comparison on identical audio ("ताजमहल किसने बनवाया था?"):

| mode | latency | output |
|---|---:|---|
| `codemix` | **499ms** | ताजमहल किसने बनवाया था? |
| `verbatim` | 797ms | ताजमहल किसने बनवाया था (no punctuation) |
| `transcribe` | 840ms | ताजमहल किसने बनवाया था? |
| `translit` | 741ms | Taj Mahal kisne banwaya tha? |
| `translate` | 657ms | Who built the Taj Mahal? |

> **STT output is non-deterministic.** The same audio through `transcribe` returned
> Devanagari on one call and "Taj Mahal किसने बनवाया था?" on another. Proper nouns get
> romanised unpredictably, so this cannot be fixed by choosing a mode — the pipeline
> has to tolerate it.

> It turns out it does. Measured support across script variants of the same question:
> Devanagari 0.343, code-mixed 0.312, romanised 0.372, English 0.248 — the multilingual
> encoder bridges scripts. A live example: spoken "लिफ्ट का मतलब है" transcribed as
> "**lift** का मतलब है" and still retrieved correctly (support 0.634, grounding 1.0).

> Correction worth recording: an early abstention on the Taj Mahal question was
> attributed to romanisation. It was not — *all four* script variants abstained,
> because the question is simply not in the 2,000-query pilot subset. Correct
> guardrail behaviour, wrong diagnosis on first pass.

**Voice → answer, measured end to end:**

```
stt  975ms + fast_path 49.2ms -> total 1435ms   support 0.776  grounding 0.917
stt  523ms + fast_path 63.9ms -> total 1100ms   support 0.634  grounding 1.000
```

Generation also closed the "related vs answering" gap: "लिफ्ट का मतलब है" previously
extracted "proper usage and pronunciation of the word lift" and now answers with the
actual definition.

### ALL SIX TECHNICAL REQUIREMENTS NOW HAVE WORKING IMPLEMENTATIONS

| # | Requirement | Status |
|---|---|---|
| 1 | Speech-to-text (Sarvam) | Saaras v3, 499-975ms, code-mix tolerant |
| 2 | Chunking "vast" | 12 variants, ablation, 3-index ensemble |
| 3 | Under 200ms | fast path p50 61.6ms, p100 128.6ms |
| 4 | P50/P70/P100 | measured; `bench/latency.py` for the formal run |
| 5 | Harness | orchestrated, retries, structured I/O, real fallback |
| 6 | Guardrails | refuse / abstain / verify, both sides of generation |

## 2026-08-14 — Day 2: full-scale ingest

### Full index built — 20k queries, 2 languages
`--langs hin mar --max-queries 10000 --variants fixed_256 semantic_128 metadata_128`

```
hin 10,000 queries   mar 10,000 queries   ->  199,668 passages
fixed_256     201,298 chunks
semantic_128  ~240,000 chunks
metadata_128  ~244,000 chunks
```

Measured stage timings (M4, 3 threads, Docker):

| Stage | Time | Rate |
|---|---:|---|
| download 2 shards (7.4GB) | ~13 min | ~10 MB/s |
| extract both languages | ~5 min | |
| chunk:fixed_256 | 36.4s | |
| **chunk:semantic_128** | **3,372.7s (56 min)** | ~600k sentence embeddings |
| chunk:metadata_128 | 76.8s | |
| embed:fixed_256 | 2,956.3s (49 min) | **68/s sustained** |

> Estimate correction: semantic chunking was predicted at ~30 min and took 56. It embeds
> every sentence before choosing cut points, so it scales with sentence count, not passages.

### Streaming vs download — streaming was solving the wrong problem
`--stream` was built for Oracle, whose disk is 89% full. On a laptop with 78GB free it is
strictly worse, and the first full run died proving it:

```
IncompleteRead(132497 bytes read, 34094644 more expected)
```

Fixes applied to `HttpRangeFile` before the realisation (still worth having for
disk-constrained hosts): range requests capped at 8MB, 5 retries with exponential backoff,
CDN redirect re-resolution, and **connection pooling via httpx**. That last one mattered
most — `urllib.urlopen` opens a fresh TLS connection per range read, and a streamed parquet
issues dozens. Handshakes, not bytes, dominated: a 300-query extract took **>6 minutes**
while transferring only tens of MB.

**Then dropped `--stream` entirely on the Mac.** 7.4GB downloaded once at 10 MB/s is ~13
minutes, `hf_hub_download` resumes on failure, and every subsequent read is local. The whole
class of bug disappears.

> Lesson worth keeping: an hour went into hardening a code path that existed to satisfy a
> constraint the current machine did not have.

### Incremental ingest
Growing the corpus now costs the delta, not a rebuild — `max_queries` is a *target total*,
and each stage extends rather than restarts:

- `download` skips `query_id`s already on disk
- `chunk` skips passages already chunked for that variant, appends the rest
- `embed` embeds only the tail and **concatenates** onto the existing `.npy`
- `index` rebuilds when chunk count changed (~13s/20k chunks — never the bottleneck)

> Caught by testing before the long run: the first version *replaced* the vector file
> instead of appending, leaving 2,000 vectors against 22,111 chunks. Silent misalignment —
> the index would still build and still return results, just wrong ones. There is now an
> explicit `vectors == chunks` assertion that refuses to write a misaligned file.
>
> Verified: 2,000 → 2,200 queries, all three variants aligned afterwards.

### Hardware measurements — threads do not help
| threads | throughput |
|---:|---:|
| 3 | **398.6 texts/s** |
| 6 | 312.1 texts/s |
| 8 | 343.0 texts/s |

More cores are *slower*. e5-small is too small for thread-sync overhead to pay off, and the
M4's 6 efficiency cores drag the 4 performance cores. The 3-CPU cap (written for the shared
Oracle box) is accidentally near-optimal.

GPU path prepared but unused: `Dockerfile.gpu` + `docs/GPU_SETUP.md`, with `default_variant()`
selecting fp32 on CUDA (int8 is a CPU optimisation and can be *slower* on GPU) and batch 256
instead of 64. The container asserts on `CUDAExecutionProvider` because a CUDA/cuDNN mismatch
does not raise — onnxruntime silently falls back to CPU and you get a "successful" 3-hour run.

### **OPEN RISK: sparse retrieval does not scale**
Benchmark of the full index (`fixed_256`, 201,298 chunks, 300 queries, x86_64):

```
embed_query       P50  5.06   P100  94.66
search_dense      P50  0.89   P100  10.01   <- HNSW scales logarithmically
search_sparse     P50 30.13   P100  48.14   <- 30x worse than pilot
retrieval_total   P50 60.35   P100 137.67
cold first query 249.30ms (excluded — measures warmup)
```

BM25 went **~1ms → 30ms** for 9x more data. `bm25s` scores by sparse matrix multiplication,
which is **O(corpus size)**; HNSW is not. Sparse is now half the retrieval budget.

**The unmeasured danger:** that benchmark used *one* index. `AdaptiveRetriever` queries
*three*, **serially**:

```
3 x ~60ms retrieval + ~40ms extract  ~=  220ms   -> OVER the 200ms budget
```

Invisible on the pilot (3 x 1ms was free). Fixes in order of value:
1. **Parallelise the fan-out** — the three queries are independent; a thread pool makes the
   ensemble cost the slowest index rather than the sum. ~20 lines, no quality change.
2. Cut sparse `k` (currently 30 candidates per index).
3. Drop `metadata_128` — it bought +1.0% R@10, which is a poor trade at 60ms.

**Do not report a latency number until the ensemble is measured at full scale.**

### Machine + transfer logistics
Full ingest completed on a second (x86_64, Windows) machine from the same repo. The Mac's
duplicate run was stopped ~1/3 through embedding.

- Index is **architecture-independent** — build anywhere, serve anywhere. Only *benchmarks*
  need the serving box.
- Transfer of the ~2GB artifact went through several failed attempts worth recording:
  `aws s3 presign` only signs GET, not PUT; a presigned PUT signed for `ap-south-1` against
  the *global* `s3.amazonaws.com` endpoint fails with `InvalidAccessKeyId` (region mismatch);
  and the chat channel masks `AKIA...` patterns, so the URL had to be assembled client-side
  from a `<<KEYID>>` template.
- Working method: regional endpoint (`s3.ap-south-1.amazonaws.com`) + PowerShell `.Replace()`
  to splice the key ID in. **Google Drive would have been faster** — no credentials in a URL.

### Mac state at end of session
```
pilot index (working demo)   fixed_256 22,111 · semantic_128 26,421 · metadata_128 26,707
raw corpus                   hin 10,000 · mar 10,000  (extraction complete)
partial                      vectors/full/fixed_256.npy  (superseded by the transfer)
```
The pilot index still serves, so development is never blocked on the transfer.

### Index imported — and the ids were colliding across languages
The transferred `index-full.tgz` (1.3GB) unpacks to `data/index/full/` + `data/raw/` +
`data/reports/`. Chunk counts match the build report exactly: fixed_256 201,298,
semantic_128 239,175, metadata_128 241,572.

Dense, sparse and metadata are row-aligned in all three variants. **But the ids were not
unique:** 201,298 rows held only 101,231 distinct `chunk_id`s.

```
chunk_id '1185869:0#fx0' appears at row 0 AND row 100471
  row 0      hin_Deva  'वैज्ञानिक दिमाग के बीच संचार की उपस्थिति मैनहट्टन परियोजना…'
  row 100471 mar_Deva  'मॅनहॅटन प्रकल्पाच्या यशासाठी वैज्ञानिक बुद्धिमत्तेप्रमाणेच…'
```

`ingest/download.py` built `pid = f"{query_id}:{i}"`. MSMARCO-XI is a **parallel** corpus —
every language shard carries the same query_ids — so **99,834 of 99,834 passage_ids
collided**, same id, different text. Consequences, all silent:

- `AdaptiveRetriever` fuses on `passage_id`, so a Hindi passage and its Marathi translation
  collapse into one unit; `seen_units` then discards the second as a duplicate vote
- `best[unit]` keeps whichever ranked first → a Hindi query can be answered in Marathi
- in `eval/`, a Marathi passage satisfies Hindi gold labels — every metric inflated

**Invisible on Day 1** because the pilot was Hindi-only. It appears the moment a second
language is added, which is exactly what Day 2 did.

Fixed at the source (`download.py` namespaces by `target_lang`; `to_documents` now groups
by `(lang, query_id)` — it was concatenating Hindi and Marathi text into one mixed-script
document; `eval/evaluate.py` checks the granularity marker on the last id segment).

**The built index did not need rebuilding.** HNSW and BM25 are keyed by row index, not by
id — the ids live only in `meta.pkl`. So the repair is a metadata rewrite over the
transferred artifact and the 64 minutes of embedding stay valid. After it: chunk_ids
201,298/239,175/241,572 all distinct, passage_ids 199,668 in every variant — matching
`n_passages` in the pipeline report.

> Measurement correction: a first pass also flagged BM25 as misaligned. It was not — the
> count was derived from `indptr`, which is vocab size (293,785). `num_docs` is 201,298
> and correct. Worth stating because it nearly sent a rebuild after a clean artifact.

### **OPEN RISK RESOLVED — the ensemble was never the problem**
`bench/latency.py` only ever measured a *single* index, which is why the fan-out went
unmeasured. `bench/fastpath.py` now measures the real served path — guardrail → embed →
retrieve → extract → guardrail — through `RAGHarness`.

Full index, 300 queries, hin+mar, **in Docker** (aarch64, 3-core cap, the Oracle-shaped
config):

```
fast_path   P50  42.1ms   P70  48.7ms   P100  245.7ms    297/300 under 200ms
```

| Stage | P50 | P90 | P99 | P100 |
|---|---:|---:|---:|---:|
| embed_query | 1.87 | 2.57 | 4.78 | 53.94 |
| retrieve (all 3 indexes) | 7.83 | 10.92 | 16.18 | 60.05 |
| **extract** | **32.02** | **51.07** | **175.55** | **236.84** |
| **fast_path_total** | **42.12** | 63.43 | 190.50 | **245.67** |

**The predicted 3 × 60ms = 220ms blowup does not occur.** The whole three-index fan-out is
**6.9ms P50** — 2.3ms per index, not 30ms:

| Index | chunks | P50 | P100 |
|---|---:|---:|---:|
| fixed_256 | 201,298 | 2.34 | 8.38 |
| semantic_128 | 239,175 | 2.39 | 5.39 |
| metadata_128 | 241,572 | 2.15 | 5.77 |

The 30ms sparse figure came from the x86 Windows box; it does not reproduce on aarch64 in
the container. So **fix #1 (parallelise the fan-out) would recover ~4.5ms** and is not
worth doing, and **fix #3 (drop `metadata_128`) would give up ~1% R@10 to save 2ms** — a
bad trade. Both were ranked off a number measured on the wrong machine.

> The lesson repeats Day 1's: an ablation over one component does not predict the system.
> The risk was real to *raise* — it just turned out to be the wrong stage. Nothing was
> lost by raising it, and the fix list would have wasted a day.

**The real tail is `extract`** — P99 175ms, P100 237ms, and it alone accounts for all 3
queries over budget. That is where the remaining work belongs, not in retrieval.

### Quality at full scale — the ensemble's case is now much weaker
1,500 queries, hin+mar, post-repair, in Docker:

| Strategy | Chunks | MRR@10 | nDCG@10 | R@10 | R@20 | search p50 |
|---|---:|---:|---:|---:|---:|---:|
| fixed_256 | 201,298 | 0.2974 | 0.3583 | 0.5694 | 0.6617 | 2.19ms |
| **metadata_128** | 241,572 | 0.3003 | 0.3632 | **0.5808** | **0.6814** | 2.01ms |
| semantic_128 | 239,175 | 0.2944 | 0.3535 | 0.5608 | 0.6535 | 2.15ms |
| ENSEMBLE (all 3) | 682,045 | **0.3056** | **0.3670** | 0.5803 | 0.6669 | 6.62ms |

**The ensemble no longer wins on recall.** It takes MRR@10 (+1.8% over `metadata_128`) and
nDCG, but `metadata_128` alone matches it at R@10 and **beats it at R@20** (0.6814 vs
0.6669) — while using **2.8x fewer chunks and a third of the search time**. At pilot scale
the ensemble's justification was +1.0% R@10; at full scale that gain is gone.

`DEFAULT_ENSEMBLE` was chosen on 1,200 Hindi-only queries over ~75k chunks. This is 1,500
bilingual queries over 682k. The serving choice should be re-decided on these numbers, not
the pilot's — `metadata_128` alone is now a serious candidate, and it is the cheapest of
the three to hold in RAM on a shared box.

Metrics are lower than the pilot's (MRR 0.4069 → 0.3056) and that is expected, not a
regression: 9x the corpus means 9x the distractors, and the task is now bilingual. The two
runs are not comparable. Note the pilot numbers were *not* inflated by the id collision —
it was Hindi-only, so nothing collided. This run would have been.

> Also inverted: `metadata_128` is now the best single index, having been the *worst* in
> the Day 1 pilot. That is the second time this strategy's ranking has flipped when a
> defect was fixed underneath it.

> Host-only finding, recorded so it does not resurface: run outside Docker on the Mac, the
> same benchmark reports **214ms P50** — CoreML claims only 542 of 889 nodes of the int8
> graph and splits it into 74 partitions, so `embed_query` goes 1.9ms → 20.9ms and
> `extract` (which embeds candidate sentences) 32ms → 180ms. `ORT_PROVIDERS=CPUExecutionProvider`
> restores 44ms. Irrelevant to the Linux serving target, which has no CoreML — but it
> would have made a host-run demo look 5x slower than the product is.

### Repo
First commit `ea51298`, 32 files, no secrets (`.env` gitignored, scan clean).
**Not yet pushed** — `gh` is authenticated as `siddharth-09`; needs
`gh repo create ... --source=. --push`. A GitHub link is a submission requirement.

---

## 2026-08-14 — Day 2 (cont.): optimisation pass on the new evidence

All runs in Docker (aarch64, 3-core cap), commit `cdf0bf4`, seed 7, same 1,500-query
bilingual set, `int8_arm`. Nothing below is mixed with the x86 Windows numbers.

### `extract`: the tail was padding, not candidate count
`embed_sentences` is **99.5%** of the stage — 31.8ms of 32.0 at P50, 180ms of 180.2 at
P99. What sets it is not how many sentences are embedded but how long the *longest* one
is:

```
correlation with extract latency:  max_sentence_chars 0.8411   n_sentences 0.0166
slowest 10%:  8.73 sentences, longest 387 chars (worst 1,279)  ->  91.6ms
fastest 50%:  8.81 sentences, longest 152 chars                ->  25.5ms
```

Same sentence count, 3.6x the latency. The tokenizer pads a batch to its longest member,
and `Embedder._encode` sorts by length to avoid exactly that — but the sort only pays
across several batches, and 10 sentences is one batch of 64. So a single 1,279-char
sentence made the other nine cost what it cost. **Cutting `MAX_EMBED` would have done
nothing**; that lever was tuned when count still mattered.

> This inverts the earlier prefilter finding rather than contradicting it. At pilot
> scale, count *was* the cost. The lever stopped working and nobody re-measured it.

### The batch-size fix, and a wrong assumption caught by its own check
Embedding one sentence at a time removes the padding. The tuning bench asserted this must
be answer-identical — masked mean pooling means a vector cannot depend on its batchmates —
and then measured only ~90% identical, which forced the check:

```
same text, batched with a long sentence vs embedded alone
  int8_arm   max|Δ| = 1.03e-02   cos 0.9981    <- NOT invariant
  fp32       max|Δ| = 4.66e-08   cos 1.0000    <- invariant, as claimed
```

The served model is **dynamically quantised**: activation scales are computed at runtime
from a tensor spanning the batch, so a wide pad coarsens the quantisation of every short
sentence beside it. Pooling was never the issue.

**This inverts the interpretation.** A large batch is not the accurate configuration that
small batches deviate from — it is the degraded one. Fidelity was being scored against the
worst config in the table. Re-scored against `batch=1`, which reproduces the standalone
vector bitwise:

| embed_batch | P50 | P95 | P99 | P100 | identical to reference |
|---:|---:|---:|---:|---:|---:|
| 64 | 32.33 | 66.87 | 185.37 | 227.52 | 90.7% |
| 8 | 28.55 | 50.14 | 156.37 | 179.47 | 91.7% |
| 4 | 26.53 | 41.61 | 97.09 | 103.95 | 91.0% |
| 2 | 26.17 | 36.94 | 68.01 | 76.91 | 93.3% |
| **1** | **24.14** | **34.24** | **48.58** | 81.73 | **100%** |

Fastest through P99 *and* the most faithful. Batching is a throughput optimisation and
this path has a batch of ten.

### Truncation for what batching cannot fix
batch=1 removes the amplification, not the cost of a long sentence itself (superlinear in
length). On the single-index config that remainder was the whole tail: P99 127ms against
P50 25ms. Capping the text *sent to the encoder* at 256 chars — the returned span and its
citation stay whole — cuts P99 2.4x for 1 answer in 60 changed, support 0.6511 → 0.6501.
It costs ~9ms at P50, which is the honest half of the trade: it buys tail predictability,
not median speed.

### Exhaustive ablation replaced the inherited ensemble
All 7 subsets, 1,500 bilingual queries:

| Configuration | Chunks | MRR@10 | R@10 | R@20 | search P50 | disk |
|---|---:|---:|---:|---:|---:|---:|
| **metadata_128** | 241,572 | **0.3030** | 0.5669 | 0.6675 | 4.32ms | **722MB** |
| fixed_256 | 201,298 | 0.2895 | 0.5601 | 0.6607 | 4.28ms | 623MB |
| semantic_128 | 239,175 | 0.2822 | 0.5552 | 0.6502 | 4.50ms | 705MB |
| fixed+metadata | 442,870 | 0.2973 | 0.5684 | **0.6697** | 7.20ms | 1345MB |
| ENSEMBLE | 682,045 | 0.2926 | **0.5717** | 0.6621 | 11.27ms | 2050MB |

The ensemble "wins" R@10 by 0.0048 — and the pilot chose it on exactly this kind of
margin. Paired bootstrap (10k resamples, same queries) says that margin is not there:

```
ENSEMBLE - metadata_128   mrr@10     -0.0105  [-0.0185, -0.0026]   significant
ENSEMBLE - metadata_128   recall@10  +0.0047  [-0.0076, +0.0168]   not significant
ENSEMBLE - metadata_128   recall@20  -0.0054  [-0.0173, +0.0064]   not significant
```

**Every recall difference in that table is noise.** The only real difference favours the
single index. `DEFAULT_ENSEMBLE = ["metadata_128"]`.

> The pilot's error was not the choice, it was the method: ranking configurations by
> point estimates without a CI. A 0.005 gap over 1,200 queries was never evidence.

### Why metadata_128 wins — partly a label leak
`chunk_metadata` embeds `[type] hint | body`, and `query_type` is the dataset's label for
the *query that owns the passage*. So a NUMERIC query's gold passages are tagged NUMERIC
while most of the corpus is not — a label a real corpus does not carry.

The two explanations predict different things. A genuine semantic effect ("how many…"
lands near numeric-flavoured text) is roughly uniform across types; a leak scales with how
much corpus the tag excludes. Measured:

| query_type | corpus share | advantage over fixed_256 | significant |
|---|---:|---:|---|
| DESCRIPTION | 59.4% | +0.0109 | no |
| NUMERIC | 19.2% | +0.0077 | no |
| ENTITY | 9.4% | +0.0179 | no |
| LOCATION | 9.4% | +0.0310 | no |
| PERSON | 2.7% | +0.0339 | **yes** |

Correlation between corpus share and advantage: **-0.638**. Directionally the leak
signature, though only PERSON (n=37) is individually significant and the advantage stays
positive even for the majority type — so it is likely both effects, not purely the leak.
Recorded as a caveat, not a blocker: metadata_128 is never significantly *worse* than any
alternative on any metric, and it is a third of the footprint.

### Serving metadata_128 alone exposed a shipping bug
Every extractive answer came back as:

```
[description] explanation description definition | वैज्ञानिक दिमाग के बीच संचार की…
```

`chunk_metadata` stored the hint-prefixed string as the chunk's `text`, and the extractive
answer is drawn from stored text. The clean body was in `extra["raw_text"]`, which the
pipeline already used to keep the hint out of BM25 — but never used for display. **The
ensemble hid it**: `best[unit]` usually picked a `fixed_256` chunk for the same passage,
so most answers looked clean. Shipping one index exposed it on 100% of answers.

Fixed at source (`ChunkIndex.build` now takes `display_texts`; the pipeline passes the
untagged body to both BM25 and display) and repaired in the built index — 241,572 rows,
metadata-only rewrite, no re-embedding, since the hint belongs in the vectors and only the
display copy was wrong.

> Two separate defects this session were both invisible behind the ensemble and both
> appeared only when something simpler was tried. Worth noting as a pattern: redundancy
> hides bugs in the components it makes redundant.

### Per-language, and why routing was rejected
Marathi is much weaker than Hindi across every configuration (MRR 0.26 vs 0.35, R@10 0.51
vs 0.63) — the largest quality gap in the system, and larger than any difference between
configurations. The ensemble's only significant per-language win is Hindi R@10 (+0.0179
[+0.0020, +0.0341]); on Marathi it is significantly *worse* on MRR (-0.0181).

Language-conditional retrieval was evaluated and rejected: it must hold all three indexes
resident, so it inherits the ensemble's memory cost — the dominant cost — while adding a
routing decision, to buy a gain in one language.

### Id audit — clean, verified independently
```
| index        | rows    | aligned | dup chunk_ids | cross-lang pids | id->2 texts |
| fixed_256    | 201,298 | yes     | 0             | 0               | 0           |
| semantic_128 | 239,175 | yes     | 0             | 0               | 0           |
| metadata_128 | 241,572 | yes     | 0             | 0               | 0           |
gold linkage: 21,342/21,342 (100%) resolve into the index
```
Duplicate *text* rows remain (0.97–2.88%) — genuine near-duplicate passages in MSMARCO,
not an id defect. `eval/audit_ids.py` re-runs this after any ingest.

### Result

| | before | after (metadata_128) |
|---|---:|---:|
| fast path P50 | 42.12ms | **30.50ms** |
| P70 | 48.67ms | **32.86ms** |
| P99 | 190.50ms | **48.16ms** |
| P100 | 245.67ms | **129.53ms** |
| under 200ms | 297/300 | **300/300** |
| MRR@10 | 0.2926 | **0.3030** |
| R@10 | 0.5717 | 0.5669 (n.s.) |
| R@20 | 0.6621 | 0.6675 (n.s.) |
| disk | 2,050MB | **722MB** |
| chunks | 682,045 | **241,572** |

### Open items
- [x] ~~Cut the `extract` tail~~ — P99 190 → 48ms; all four latency targets met
- [x] ~~Re-decide the serving ensemble on full-scale numbers~~ — `metadata_128` alone
- [x] ~~Import the transferred index, verify alignment, benchmark the ENSEMBLE~~ — done.
      Alignment was clean; **ids were not** (cross-language collision, repaired in place).
      Ensemble fan-out is 6.9ms, not the feared 180ms.
- [ ] **Cut the `extract` tail** — P99 175ms / P100 237ms is now the *only* thing over
      budget (3/300 queries). Everything else sums to ~10ms. The prefilter=10 tuning was
      done at pilot scale; re-tune it here.
- [ ] **Re-decide the serving ensemble on full-scale numbers** — `metadata_128` alone
      beats the 3-index ensemble at R@20 for a third of the cost
- [ ] `api/` + `web/` — the mic page and the "live working link"
- [ ] Re-run the eval on `--no-ensemble` doc variants if doc granularity is revisited
      (`to_documents` was merging Hindi and Marathi into one document until today)
- [ ] Push repo to GitHub (5 min, unblocks a submission requirement)
- [ ] Deployment + domain
- [ ] Videos, social posts (the automated screening gate)
- [ ] Cross-encoder reranker: documented best lever for MSMARCO, but ~4× e5-small per pair
      → likely 200–500ms for 8 candidates on CPU. Belongs on the quality path, not the fast path.
- [ ] Not yet written: `api/`, `web/`
- [ ] Budget alarm at $50 (console only — IAM users can't reach billing APIs by default)
- [ ] Namecheap `.me` domain (keeps the submitted link stable across any migration)
- [ ] Social post plan — per member, 3 platforms, `#RAGInGoa`, ≥1 public Instagram
- [x] ~~Ask organisers what "under 50ms" covers~~ — resolved by the reissued PDF (200ms)
- [ ] Delete the IAM access key after Aug 22

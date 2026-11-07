# Preregistered-style experimental protocol (draft, lock before final testing)

## Systems and comparisons
> Governing framing: see `docs/RESEARCH_CHARTER.md`. **T below is the Control** (original GLM-5.3-Flash on the dual-Max-Q rig, owner's serving configuration). S0 is redefined as an *internal runtime diagnostic* (unmodified model on the HeteroFlash Spark runtime), not as a baseline taken from published Spark checkpoints.

- **T:** highest-quality feasible GLM-5.3-Flash teacher, temporarily served on dual Max-Q research GPUs; reference against official FP8 where feasible. Training/eval GPUs never serve inference in the final deployment.
- **S0:** full GLM-5.3-Flash ~2-bit baseline on the SAME Spark, with unchanged model semantics if supported; original model quality need not match T.
- **S1:** surgery+mixed-precision student on Spark, GPU-only, same serving/client conditions as S0.
- **S2:** SAME compressed checkpoint as S1, but heterogeneous Spark GPU/Arm/remote CPU execution; compare against S1, not T, to isolate system scheduling.
- **S3:** S2 plus migrations, prefetch, cross-tier quant policies, phase-specific schedules. Hold architecture constant for incremental comparisons.
- **S4:** optional shallower student if S3 can't meet the latency floor; compare quality to T and deployment to S0.

## Model quality, general purpose
- Training/calibration and frozen holdout have disjoint sources and disjoint prompt IDs. No use-case-specific Hermes fine-tuning.
- At least six separately reported domains: general conversation/writing, world knowledge/QA, mathematical reasoning, coding, multilingual, instruction/tool-calling. Dedicated 45K long-context needle+reasoning, multi-document retrieval and recall. Optional vision: separately state whether model still supports it.
- For each quality prompt, collect teacher and student next-token distributions on the SAME reference tokens (teacher forcing); teacher KL, NLL delta, top-k agreement, representation error, hallucination/factuality and tool-call validity. Sample complete generations separately.
- Reference: original FP8 release is authoritative; a 4-bit runtime teacher introduces quantization error. Keep a small FP8 validation panel if the full teacher is prohibitively expensive to run repeatedly.
- Student training: output/repr reconstruction -> router KD -> selected expert LoRA -> quantization recovery, with held-out quality gate after each stage. Full student distillation optional, not default.

## KL specifics
Forward KL `D_KL(p_T || p_S)` at model output, with full vocabulary where feasible and a calibrated logit-tail approximation only after validating numerical bias on a subset. Report per-token KL distribution, mean/median/p95, prompt-clustered confidence intervals and domain slices. Compute with the identical tokenizer, sampling temperature, logits processing, teacher-forced token prefixes and cache state. Logit divergence is not the same as task failure. For heavy compression, also test on-policy generations and exposure-bias-driven drift.

## Final inference benchmark definitions
- Decode >=120 tok/s in representative **prose**, C=1; report rate for accepted/committed output tokens, not drafted tokens, and count user-visible vs reasoning tokens separately; use both greedy and sampled, output lengths 512/1024 tokens, 8K/16K/45K context. Benchmark sustained rate AFTER first token.
- Cold 45K: use a verified 45,000-token canonical prompt; flush exact prefix state while retaining the model resident and warmed kernels, and measure actual external API request to first streamed token. For cold vs warm distinguish prefix-cache disabled vs cold-start model process. Pre-sampled 45K prompts should vary to avoid accidental cache hits.
- Cached: same 45K token prefix resident and exactly identical between turns, plus **32** newly appended tokens for the primary 200ms TTFT SLA. Report +16/64/128/256 separately. Verify KDA recurrent and DSA sparse-attention cache state against recomputation. Cache hit rate and number of recomputed suffix tokens must be logged.
- Report latencies and throughput at p50, p95 and p99, across multiple server restarts, thermal conditions, and request ordering. If both cold prefill and decode are simultaneous objectives, report additional contention runs rather than conflating isolated maxima with joint service guarantees.

## Experiments and negative controls
1. Baseline full model: reproduce teacher and Spark performance; correctness first.
2. Router active top-k {8,6,4,3,2}, normalizer reference {active-k,8,12,16}; preserve real GLM sigmoid, correction bias and shared expert semantics. These are separate axes.
3. Nonuniform retained-expert budgets per layer (no pruning, 25%,40%,50%,60% average), compare frequency, REAP, HOPE, random, token-critical/loss-based. Distinguish capacity from active-work.
4. Router frozen / calibrated / KD-trained; then expert LoRA. Interaction matrix: top-k-only, prune-only, top-k+prune, +KD, +LoRA.
5. Quant gate/up vs down, per-tier int2/int4/FP8/INT8 (when supported). Compare same weights in fake quant vs packed real kernel. Per-layer KL and real latency.
6. Placement GPU-only, Spark GPU+Arm, GPU+remote CPU, all three; static vs dynamic; equal quant across tiers vs per-tier; migration enabled vs disabled; remote miss wait vs prefetch; decode vs prefill placement.
7. Offload **weights** vs remote **compute** (activation-only network) as an explicit control. Vary NIC latency/bandwidth and remote CPU saturation.
8. Speculation off / original MTP / retrained draft, draft length sweep, full CUDA graph on/off; quantify committed tokens per full verification cycle.
9. Cached suffix test with and without exact state reuse; 45K prefill and decode contention; sustained network failure/recovery behavior.

## Statistical inference
- Primary inference unit for quality is **prompt**, not token; use paired prompt-clustered bootstrap for confidence intervals and noninferiority. Tokens in one prompt are dependent. Domain slices and task differences are pre-registered.
- Draft candidate quality noninferiority margin: -2 percentage points aggregate paired task accuracy is a **placeholder**; actual margin must be justified and frozen before final holdout. Apply tighter safety/correctness constraints to tool-call validity and factual QA. A non-significant superiority test does not prove noninferiority.
- For each candidate compute paired effect distribution versus T; pass if the lower one-sided 95% bound of student-minus-teacher accuracy is greater than -margin, and no critical domain violates predeclared secondary bound. Apply Holm or hierarchical multiple-testing control to confirmatory domain hypotheses.
- Latency/throughput: on same Spark, alternate candidate and control order across runs; warmup, rerun over multiple process restarts, maintain clock/thermal state. Use paired bootstrap over matched prompt+restart blocks, not 1000 independent copies of one prompt. Report absolute SLA pass with a one-sided bound on p95, not simply percent improvement.
- Use pilot variance and power simulation to determine final N; initial working plan ~5000 distinct holdout prompts (stratified, not just generated duplicates), and 30-50 timing repeats per latency scenario *across separate server restarts*, adjusted after pilot.
- Effect sizes, confidence intervals, baseline distributions and negative results are mandatory. Never use p-values as proxy for practical importance; report accuracy/perf/energy Pareto frontier.

## Data schema
- Training: JSONL `request_id, domain, input_ids_hash, layer, token_index, routed_expert_ids, router_logits_hash, gate_weights, expert_output_norms, phase, model_revision`; stream enormous traces in chunks and apply privacy/license review.
- Latency: CSV `pair_id,system,accuracy,decode_tps,cold_ttft_ms,cached_ttft_ms,kl_teacher` (nullable fields) as accepted by `src/paired_analysis.py`; extend with `seed,temperature,thinking,context_tokens,new_tokens,restart_id,network,model_hash,clock,thermal,cache_hit_tokens`.
- Runtime: per layer and step record GPU band, Arm band, remote band elapsed; network bytes/RTT; idle/stall wait, quant/layout conversion, cache page hits, speculative accepted tokens.

## Known validity traps
- The teacher and student may use different inference engines/kernels; match logits-processing semantics before interpreting KL.
- A student may have better KLD and worse task performance; assess both.
- Different tokenization or even whitespace silently changes measured 45K; insist on exact token IDs.
- TTFT depends on request framing and streaming, not just CUDA kernel time.
- A seemingly dynamic quant tier may secretly incur repacking or remote page faults; log actual resident bytes and CPU/GPU page residency.
- Hecate's GB300+RTX+Grace benchmarks are not single-GB10 Spark benchmarks and involve a different model.

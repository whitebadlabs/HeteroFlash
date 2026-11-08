# Path to the hard targets without new hardware (2026-11-07)

**Evidence key.** Every number is tagged:

- **[M]** measured by us on this hardware;
- **[L]** reported in the literature (deep research, 3-vote verified), not reproduced here;
- **[P]** a modeled projection from bandwidth arithmetic, not measured.

Targets (revised by Carmen, 2026-11-08):

- **Decode goal: 80 tok/s**, with 125 tok/s as a stretch goal.
- Cold 45K TTFT <= 10.8 s.
- Cached 45K+32 <= 0.2 s.
- Non-inferior general-purpose quality.

The last three are unchanged.

Hardware is fixed: one DGX Spark, the workstation CPU/RAM, and the existing 10 GbE direct link. Max-Q GPUs may be used only for offline training and evaluation.

**Approval scope (Carmen, 2026-11-07).**

- Approved: bounded pilots only (a QAT feasibility pilot and the pre-checks below).
- Not approved: multi-day training on both Max-Q GPUs. That waits for a go/no-go after the pilots.

## Measured so far [M]

| item | result |
|---|---|
| E2E-v1, llama.cpp RPC (Spark GPU + CPU experts, layers 24-44) | 10.5 tok/s; MTP 17.0 tok/s at ~2.8 committed per verify; cold 45K 223 s; cached +32 0.68 s |
| bytes per token (GGUF tensor table) | fixed 9.2 GB at Q8_0 (attention 6.6 GB); routed experts 4.8 GB at Q4_K |
| GB10 fused MoE (vLLM, FP8) | 6.1-39 TFLOPS for 512-4096-token chunks; dense FP8 GEMM 188 TFLOPS; ~216 GB/s weight streaming |
| workstation CPU int8 GEMM (torch, 32 threads) | 12-19 TOPS (llama.cpp RPC path ~3 TOPS) |
| direct-link round trip, busy-poll | 8 KB: 50 us p50 (default sockets 317 us); 64 KB: 151 us p50 |
| verify-window expert union (M1 traces) | 18.6 experts/layer for 4 tokens, 26.4 for 7 tokens (20-file subset); 22.9 for 4 tokens on the full set |
| locality oracle (no training) | union -20% costs 5-9% of routing mass moved |
| Q1: PTQ IQ3_XXS experts vs Q4_K | +0.068 nats, KLD 0.138, same top token 87% (**fails**) |
| M2 router/precision (Max-Q rig) | FP8 fixed +0.006 nats; NVFP4 fixed +0.040, of which attention +0.030; top-6 with original top-8 normalisation +0.060 |

## What the literature says [L]

| Technique | Best verified result | Quality | Fit here |
|---|---|---|---|
| EcoSpec (cost-aware draft-tree selection) | unique experts per step 23.7 -> 20.5 (-13.5%) on Qwen3-235B | lossless | drops into the verifier |
| MoE-Spec (per-layer expert budget at verify) | +10-30% over EAGLE-3 | lossy when aggressive | moderate budgets only |
| DraftExpert (self-speculation) | 1.45x end to end | lossless | 16B-class evidence only |
| DFlash block-diffusion drafter | tau 6.5-7.9 on math/code; tau 4.2-4.4 on MT-Bench prose | lossless | MoE speedup per accepted token ~0.4 vs 0.64 on dense models |
| NVFP4 quantization-aware distillation (QAD) | near-BF16 on Nemotron hybrids with 0.3-6B tokens | ~lossless for MoE/Mamba weights | attention was kept BF16 in every hybrid model |
| INT4 QAT of routed experts | production (Kimi K2 Thinking, g32) | vendor claims lossless | yes |
| ~3-bit trellis quantization (QTIP / EXL3) | +0.044 nats on Llama 2 70B | small but measurable | pending a Spark kernel |

No source shows 7-12 accepted tokens on general prose. No published DGX Spark result exists for a 300B-class MoE.

## Decode projection for the full stack [P]

These figures are modeled, not measured.

**Revision (2026-11-08), using measured kernel bandwidth and published drafter acceptance.**

The earlier table (headline 127 tok/s) assumed two things that the pre-checks contradict:

- **Kernel bandwidth:** it assumed 241 GB/s for Spark expert reads. Pre-check A measured 160-175 GB/s for the existing MoE GEMV at verify sizes [M].
- **Drafter:** it assumed prose tau 4.5. Pre-check D found the published GLM-5.3-Flash DFlash2 at about 3.3 [L]; our MTP is 2.8 [M].

The model is: t_step = fixed bytes / fixed-GEMV bandwidth [M] + Spark expert bytes / 170 GB/s [M], balanced against CPU expert bytes / 182 GB/s, plus 3 ms sync.

- The 182 GB/s CPU figure is [P], to be replaced by the runtime-v1 measurement.
- The 3 ms sync is [P], extrapolated from the 50 us [M] round trip.
- The verify-window union with EcoSpec is [L]/[M]: 22.9 -> ~20 for 4 tokens, and ~37 -> ~32 for an 8-token block.

| fixed weights | experts | MTP, tau 2.8 [M], 4-token window | DFlash2, tau 3.3 [L], 8-token block |
|---|---|---|---|
| FP8, 8.7 GB at 218 GB/s (quality-safe, +0.006 nats [M]) | 3-bit IQ3_XXS | 62 tok/s | 55 tok/s |
| FP8 | 4-bit Q4_K | 50 | 43 |
| NVFP4, 4.6 GB at 240 GB/s (needs QAD; PTQ +0.030 nats [M]) | 3-bit IQ3_XXS | **79** | 67 |
| NVFP4 | 4-bit Q4_K | 61 | 50 |

What the revised table means:

- **No modeled configuration reaches 120-130 tok/s.** The best is about 79 tok/s [P]. It needs NVFP4 attention via QAD, trained 3-bit experts, the MTP drafter and EcoSpec, all at measured kernel efficiency.
- **Closing the remaining ~1.6x needs both of these:**
  - expert GEMV kernels near the 241 GB/s peak, against 170 measured today. That is worth about 1.2-1.3x.
  - a prose drafter well beyond the published state of the art (tau >= 4.5 at a small expert union).
- **MTP beats DFlash2 in this model.** DFlash2's 8-token block pulls a much larger expert union for little extra acceptance.

Superseded projection, kept for the record: 94-127 tok/s, computed at 241 GB/s with tau 4.5.

## Cold 45K TTFT and cached turn [P]

- **Cold 45K.** 37 MoE layers stay on the Spark at ~3-bit and 5 layers run on the workstation CPU with int8 GEMM. Projection: ~7-9 s. The inputs are the measured CPU and GPU MoE rooflines plus an unvalidated 3-bit kernel and memory budget (pre-checks A and B).
- **Cached 45K + 32.** Projection: ~0.15-0.2 s.

## Gates (status)

1. **3-bit experts.**
   - PTQ: **failed [M]**. Routed experts only, 6-domain holdout (20 x 2048 tokens), measured against a **Q6_K-expert reference** (near-lossless, PPL 3.622). Fixed weights are Q8_0 in every model.

     | experts (bpw) | dNLL vs Q6_K | mean KLD | same top token |
     |---|---|---|---|
     | Q4_K (4.5) | +0.007 +/- 0.002 | 0.038 | 92.8% |
     | IQ3_S (3.44) | +0.059 +/- 0.004 | 0.099 | 89.2% |
     | IQ3_XXS (3.06) | +0.075 +/- 0.004 | 0.126 | 87.7% |
     | IQ2_S (2.5), vs the Q4_K reference only | +0.132 vs Q4_K | 0.215 vs Q4_K | 84.1% |

     - 4-bit experts cost about the same as FP8 fixed weights. The "4-bit is quality-safe" assumption is now measured.
     - The IQ3_XXS gap to Q4_K is about 0.068 nats. That is what the QAT pilot must recover.
   - Next: pre-check A (Spark 3-bit kernel), then a bounded QAT pilot on representative layers with a fixed compute budget. Go/no-go before training all 42 layers.
2. **NVFP4 attention.**
   - Pre-check C: FP8 vs NVFP4 quality from M2 [M], plus kernel performance on GB10.
   - QAD comes only after a go decision.
3. **Link latency: passed [M].** 50 us round trip for 8 KB (busy-poll); about 2 ms sync per decoded token [P].
4. **Drafter. Pre-check D done (2026-11-07). The literature/community evidence weakens bet 4.**
   - **Existing checkpoint:** incoai/GLM-5.3-Flash-DFlash2.
     - Architecture: 5 layers, block 8, taps layers 5/14/24/33/42; 2.34 GB bf16.
     - License: CC-BY-NC-ND-4.0, research only.
   - **Permissive alternatives:**
     - canada-quant/GLM-5.3-Flash-DFlash2-G: Apache-2.0, at parity with incoai on canada-quant's own measurements.
     - modal-labs/GLM-5.3-Flash-DFlash: MIT, unbenchmarked.
   - **Reported acceptance [L, community].**
     - Prose tau is about 3.0-3.3 at temp 0. The RedHat DSpark drafter reaches writing 3.13 and summarization 3.73; incoai on vLLM reaches 3.28.
     - Code tau is about 4.5-4.7 and math 5-6.
     - On 2x Spark: 1.41x on prose and 1.88x on code.
     - Self-fine-tuning gave +0.04 tau or less.
   - **Implication.** The plan's tau 4.5 for prose [P] is not supported. The best published drafter is about 3.3 on prose against our measured MTP 2.8 [M].
     - At tau 3.3 the modeled stack in the table above falls to roughly 80-95 tok/s [P].
     - 125 tok/s on prose would need a drafter about 35% better than anything published.
   - **Next.** Measure incoai/canada-quant prose tau on our target [M] before any drafter-training decision.
5. **Runtime v1: measured (2026-11-08).** Branch `runtime-v1`, `results/2026-11-08-runtime-v1/`.
   - **Setup:** busy-poll per-layer expert exchange over real int4 experts, Spark CPU client, no GPU yet.
   - **CPU expert bandwidth: 165-200 GB/s [M].** The modeled 182 GB/s holds.
   - **Socket wake-up penalty: ~300 us per exchange [M]** whenever the server computes for >= 100 us, in every unprivileged socket mode.
     - A 20 us heartbeat from the server cuts it to 62 us per decode layer [M].
     - Decode sync is then about 2.6 ms per 42-layer token [M], meeting the modeled <= 3 ms.
   - **Verify exchange: 109 us (T=4) and 172-178 us (T=8) per layer [M].** That is 4.4-7.5 ms per step, not yet overlapped with GPU work.
   - **Open items:** int8 activations (about -105 us at T=8, [P]), GPU integration and overlap, and the root cause of the 300 us penalty (needs root tracing).
6. **Memory.** Pre-check B: full Spark budget, covering weights, KV/state caches, drafter, CUDA context, runtime buffers and OS.

## Next experiment: hot/cold expert placement with per-device formats (Carmen, 2026-11-08)

This combines Carmen's two founding ideas: less-used experts in system RAM, and quant types matched to each processor.

- **Placement.** A per-expert map in runtime-v1. llama.cpp RPC is not suitable: it can only split whole layers.
  - Hot experts and the fixed weights live on the Spark, at the smallest format that passes quality.
  - Cold experts live on the workstation CPU at Q4_K or better (+0.006 nats [M]); RAM is plentiful there.
- **Execution.** The Spark and CPU compute their experts within each layer in parallel. Dependency: runtime-v1 GPU integration.
- **Rationale [M].**
  - Bandwidth is similar on both sides: Spark expert kernels about 170 GB/s, CPU experts 165-200 GB/s.
  - Capacity differs greatly: Spark headroom 0.5-1.6 GiB at 3-bit, against about 230 GB free on the workstation.
  - Keeping 50% of experts resident covers about 72% of routing, and there are no dead experts. That is why the experiment needs parallel execution, not a cache-miss path.
- **Measurements:** decode, cold 45K TTFT, cached 45K+32, Spark headroom, and per-domain quality on the balanced eval set.
- **Pass bar:** no worse than the dual-Max-Q EXL3 3.5 bpw control. That control must be measured against the BF16 proxy first.
- **Order:**
  1. Balanced eval set.
  2. Layer-group IQ3 attribution (decides the Spark-side format).
  3. runtime-v1 GPU integration.
  4. This experiment.

# Path to the hard targets without new hardware (2026-11-07)

**Evidence key.** Every number is tagged:

- **[M]** measured by us on this hardware;
- **[L]** reported in the literature (deep research, 3-vote verified), not reproduced here;
- **[P]** a modeled projection from bandwidth arithmetic, not measured.

The targets are unchanged: 120-130 tok/s prose decode, cold 45K TTFT <= 10.8 s, cached 45K+32 <= 0.2 s, and non-inferior general-purpose quality.

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

The model is t_step = max(Spark bytes / 241 GB/s, CPU bytes / 182 GB/s) + sync, with the Spark/CPU expert split chosen to balance the two terms. The bandwidth inputs are measured [M]; every other input is an assumption until its gate passes.

Inputs:

- **Fixed weights:** NVFP4 after QAD, 4.6 GB. *Assumption: gate 2.*
- **Experts:** 3-bit at 9.4 MB or 4-bit at 12.6 MB. *3-bit is an assumption: gate 1 plus pre-check A.*
- **Verify window:** 7 tokens, with an EcoSpec union of ~23 experts per layer. *Assumption: gate 4.*
- **Drafter:** tau 4.5. *Assumption [L-level]: gate 4 plus pre-check D.*

| experts | sync per step | step [P] | tok/s [P] |
|---|---|---|---|
| 4-bit | 8 ms (kernel TCP) | 47.7 ms | 94 |
| 4-bit | 3 ms (busy-poll, overlapped) | 42.7 ms | 105 |
| 3-bit | 8 ms | 40.4 ms | 111 |
| 3-bit | 3 ms | 35.4 ms | 127 |

The 127 tok/s figure is a modeled upper bound. It requires all six bets plus kernels at the bandwidth roofline.

If gates 1-2 fail, the projected non-inferior ceiling is ~60-80 tok/s [P].

## Cold 45K TTFT and cached turn [P]

- **Cold 45K.** 37 MoE layers stay on the Spark at ~3-bit and 5 layers run on the workstation CPU with int8 GEMM. Projection: ~7-9 s. The inputs are the measured CPU and GPU MoE rooflines plus an unvalidated 3-bit kernel and memory budget (pre-checks A and B).
- **Cached 45K + 32.** Projection: ~0.15-0.2 s.

## Gates (status)

1. **3-bit experts.**
   - PTQ: **failed [M]** (see above).
   - Next: pre-check A (Spark 3-bit kernel), then a bounded QAT pilot on representative layers with a fixed compute budget. Go/no-go before training all 42 layers.
2. **NVFP4 attention.**
   - Pre-check C: FP8 vs NVFP4 quality from M2 [M], plus kernel performance on GB10.
   - QAD comes only after a go decision.
3. **Link latency: passed [M].** 50 us round trip for 8 KB (busy-poll); about 2 ms sync per decoded token [P].
4. **Drafter.** Pre-check D: evaluate the existing GLM-5.3-Flash DFlash2 checkpoint before any drafter training.
5. **Runtime.** Parallel Spark/CPU expert execution. This proceeds independently of the training gates.
6. **Memory.** Pre-check B: full Spark budget, covering weights, KV/state caches, drafter, CUDA context, runtime buffers and OS.

# Bounded hot-expert tier feasibility experiment (one MoE layer), 2026-11-09

**Question.** Could the Spark keep the most popular experts in an EXL3-class format, with the workstation CPU serving the rest as Q4_K, and does that move the quality–latency frontier?

**Layer.** Model layer 32 (MoE index 29). It is fully traced: 660K disjoint M1 tokens were used to choose the hot set, and 2,560 held-out tokens (512 from each of 5 domains) were used to score it.

**Labels used throughout:**
- **[K]** kernel microbenchmark
- **[T]** routing-trace analysis
- **[M]** complete-model measurement (earlier runs)
- **[E]** speculative architecture estimate composed from these, see `estimate.json` and `ops/hottier/compose_estimate.py`

**Not done:**
- No end-to-end hot-tier model was built.
- All 42 layers are *assumed* to behave like layer 32.

## Setup

**GPU tier.** The control checkpoint's own ExLlamaV3 experts (mixed K3/K4 per tensor, MCG codebook) are re-laid out into b12x's projection-tiered trellis layout. This is a byte re-layout only; no code word or scale changes. They run through b12x's W4A16 kernel with GLM's clamped SiLU. Runs in `ops/hottier/l32_extract.py` and `l32_bench.py`.

**CPU tier.** `expert_server_gguf` with the Q4_K + imatrix experts, rebuilt with the missing SiLU clamp (gate ≤ 10, up in ±10). Requests come from the Spark over the existing socket protocol, using real cold routes (`l32_cpu_bench.py`).

**References.** BF16 teacher experts, and ExLlamaV3's own `reconstruct` + Hadamard dequantization of the same control experts. Both are computed per route on a workstation GPU.

**Routing.** Router logits for layer 32 were captured from the control on two workloads (`route_capture_client.py`, `route_locality.py`):
- 60 Hermes-suite first turns (7.7K prompt tokens, 11.4K decode tokens);
- 6 long-context prompts (20–43K tokens each, 188K tokens in total).

**Findings that affect integration:**
1. The control checkpoint permutes each expert's intermediate channels. This doesn't change the function: after column matching, cosine vs BF16 is 0.9875.
2. b12x's projection-mixed trellis path caps a layer at 256 experts and does not support expert maps.
3. b12x's small-M path mis-merges duplicate expert IDs within a token: 7.5% error when one is given, against 0.19% with distinct IDs. A hybrid runtime must never emit duplicates.

## 1. GPU expert time and bandwidth, including unpack [K]

| Hot N | Coverage of test routes | Resident bytes per layer | T=1 | T=4 | T=32 | T=8192 |
|---|---|---|---|---|---|---|
| 256 | 97.4% | 2.88 GB | 0.38 ms, 209 GB/s | 1.14 ms, 227 GB/s | 5.06 ms | 116 ms |
| 216 | 87.1% | 2.44 GB | — | 0.90 ms, 221 GB/s | 4.51 ms | 120 ms* |
| 160 | 69.9% | 1.83 GB | — | 0.69 ms | 3.50 ms | 112 ms* |

- **Decode sizes (T ≤ 32) are bandwidth-bound** at 200–230 GB/s of trellis code bytes. That is GB10's practical peak, so unpacking costs nothing at decode.
- **Prefill sizes are unpack/compute-bound** at about 28 TFLOPS: 8192 tokens × 8 routes × 50.3 MFLOP in 116 ms. The 2-bit GPU-only path was profiled at about 42 TFLOPS. So EXL3-class experts cost about 1.5× more prefill time than the 2-bit path.
- *Large-T rows include aliased cold rows (computed with weight 0). The estimates below scale them by hot coverage.

## 2. CPU execution, communication, sync [K]

Q4_K expert server on 32 Zen5 cores; real cold routes, p50 values.

| Hot N | Cold routes per token | Server time, T=1 | Server time, T=4 | Server time, T=16 |
|---|---|---|---|---|
| 256 | 0.21 | 71 µs | 110 µs | 221 µs |
| 216 | 1.03 | 136 µs | 300 µs | 1.03 ms |
| 160 | 2.41 | 217 µs | 651 µs | 2.58 ms |

- **Link plus sync:** about 340–400 µs per round trip from a Python client (non-busy-poll). The earlier C busy-poll relay measured 50–110 µs [K, `../2026-11-09-audit/q4k_layer_exchange.txt`, `../2026-11-07-link-latency`].
- **Tail latency:** server p99 at T=1 is 13–30 ms (occasional stalls). An integrated runtime would need to bound these.

## 3. Numerical correctness [K]

| Comparison | Relative error |
|---|---|
| b12x hot partial vs ExLlamaV3 dequantization (T = 1…512, every N) | **0.19%** |
| Same with duplicate expert IDs at T=1 | 7.5% (runtime hazard, see Setup) |
| Control EXL3 experts vs BF16, full layer | 14.6% |
| b12x hot partial + exact cold vs BF16 | 14.4% |
| Q4_K CPU cold partial vs BF16 | **11.0–11.4%** |
| Control EXL3 on the same cold routes vs BF16 | 16.3–19.9% |

The cold (rarely used) experts are among the worst-quantized in the control. Q4_K on the CPU beats EXL3 on exactly those routes.

## 4. Spark memory budget [M + E]

**Measured** (vLLM log of the FP8 + MTP-3 run, plus checkpoint headers):
- Weights other than routed experts: 10.8 GiB.
- MTP layer: 2.0 GiB.
- Runtime overhead (activations, CUDA graphs, workspaces, inside the 0.90 × 121.7 GiB = 109.5 GiB budget): about 5.3 GiB.
- FP8 KV: about 27 KB per token, so a 64K-token sequence needs about 1.7 GiB (2.0 GiB allowed).

**Estimate:**
- That leaves **about 89 GiB for hot experts, about 2.29 GB per layer**.
- At 11.3 MB per EXL3 expert, that is **about 200 hot experts per layer**.
- Hot-256 (119 GB in total) does not fit. A drafter other than the MTP layer would come out of the same budget.
- CPU side: all Q4_K experts take 175 GB of the workstation's 256 GB (measured running).

## 5. Routing locality [T]

Hot set chosen on M1 traces; coverage of routes.

| Workload | Coverage at N=256 | N=216 | N=200 | N=160 |
|---|---|---|---|---|
| Hermes decode | 96.6% | 88.2% | 84.2% | 73.0% |
| Hermes prefill | 96.0% | 84.0% | 78.4% | 65.4% |
| Long-context prefill (188K tokens) | 77.8% | 57.7% | **52.7%** | 42.8% |
| Long-context decode | 96.7% | 89.7% | 84.8% | 76.3% |

- Long-context prefill popularity is **anti-correlated** with M1 popularity (Spearman −0.20). Hermes correlates at +0.61.
- A static hot set chosen on short prompts fails on long prompts.
- Decode routes show temporal reuse:
  - 49% of a token's experts were also used in the previous 4 decode tokens;
  - 70% within the previous 16;
  - 88% within the previous 64.

## 6. Layer executions that block on remote experts [T]

A layer execution blocks when any route is cold.

| Hermes decode | N=256 | N=216 | N=200 | N=160 |
|---|---|---|---|---|
| T=1 (no speculation) | 21% | 54% | **64%** | 85% |
| T=4 (MTP-3 verify window) | 45% | 86% | **93%** | 99% |

During prefill, nearly every chunk blocks at every N ≤ 256.

## 7. Cold 45K prefill estimate, with sensitivity [E]

**Assumptions:**
- The non-MoE prefill floor is 19.8 s. This is derived from the measured 37.9 s GPU-only cold 45K minus 18.1 s of 2-bit MoE at the profiled 42 TFLOPS.
- Hot GPU experts run at the measured trellis rate, scaled by long-context prefill coverage.
- Cold pairs at N=200: 7.15M (45K tokens × 8 routes × 47% × 42 layers).
- Activation traffic for tokens with a cold route: 24.7 s at 1.15 GB/s on 10GbE.

| CPU path (N=200) | CPU time | TTFT, fully overlapped | TTFT, serial |
|---|---|---|---|
| Current server (≤ 16 tokens per request, measured rate) | 475 s | 495 s | 514 s |
| llama.cpp-class CPU GEMM, about 3 TOPS | 120 s | 140 s | 179 s |
| Batched bf16 GEMM, 7.2 TFLOPS (measured torch, m ≥ 128) | 50 s | **70 s** | 109 s |

- Even if all experts fit on the GPU, the EXL3-class experts plus the fixed path would take **about 47 s**.
- The 10.8 s target is below the non-MoE floor alone (19.8 s) on this GPU.

## Decode and cached-turn estimates [E]

**Inputs:**
- Measured GPU-only step: 88.6 ms per MTP-3 step. Of that, 58.5 ms is non-MoE, from a 34% MoE share in the profile.
- The MoE part is replaced with 42 × (hot GPU time at T=4, run in parallel with CPU cold time at T=4, plus the C-relay network) plus sync. Sync is varied over 0–200 µs.
- Accepted tokens per step: 2.217 (prose) and 2.532 (Hermes).

| N | Prose tok/s | Hermes tok/s | Cached 45K+32 |
|---|---|---|---|
| 216 | 18.8–23.1 | 21.4–26.3 | 0.43–0.53 s |
| 200 (fits) | **18.3–23.7** | **20.9–27.0** | **0.43–0.56 s** |
| 160 | 17.4–24.6 | 19.8–28.1 | 0.50–0.64 s |

For comparison [M]:
- GPU-only 2-bit: 25.0 tok/s prose, 0.35 s cached, but ΔNLL +0.182.
- Config B (all experts on CPU as Q4_K): 15.3–19.5 tok/s, 1.17 s cached, ΔNLL +0.0069.
- Control on 2× Max-Q: 122–132 tok/s.

## Decision

**Do not integrate the hot-expert tier.** It works numerically and its kernels are efficient. But at most it buys about +20–35% decode over config B (about 21–27 tok/s on Hermes) and a cached turn of about 0.45 s. It brings cold 45K TTFT only into the 70–140 s range, and only after a batched CPU GEMM path that doesn't exist yet. None of the hard targets moves into reach:

| Target | Best defensible value | Status |
|---|---|---|
| 80 tok/s decode | about 21–27 tok/s | **Blocked.** The fixed path alone is 58.5 ms per step [M], which caps decode at about 38 tok/s even with free experts. |
| 10.8 s cold 45K TTFT | about 70 s at best | **Blocked.** The GB10 compute floor for the non-MoE path is 19.8 s; EXL3-class experts run at about 28 TFLOPS; long-context prefill covers only 53% of routes. |
| 0.2 s cached TTFT | about 0.43 s | **Blocked.** It was already 0.35 s GPU-only at 2-bit. |

**Integration cost avoided:**
- a patch for the 256-expert cap;
- expert-map or `-1` route support in the mixed trellis path;
- the duplicate-ID hazard;
- per-layer hot sizing;
- a busy-poll mailbox relay in the decode loop;
- a new batched CPU prefill server;
- prefill-aware or dynamic hot sets.

**What would invalidate this decision:**
- **Would reopen it:** a measured end-to-end hot-tier decode above about 30 tok/s on Hermes. That would mean the 34% MoE-share assumption overstates the non-MoE floor. Or a hot set that keeps ≥ 90% long-context prefill coverage at N=200, for example one chosen per request from the prompt itself.
- **Would close it further:** other layers showing lower coverage than layer 32, which falls near the M1 per-layer mean.

**Contribution to the scientific question.** Precision placement does help *quality per byte*: the rarely used experts are the control's worst-quantized ones, and Q4_K on the CPU beats them by about 40% relative error on the same routes. But on this hardware the *latency* frontier is set by:
- the non-MoE path, for decode and cached turns;
- GB10 prefill compute and the 10GbE link, for cold TTFT.

It is not set by expert placement. The combination itself (popularity tiering plus per-device formats) is established practice in prior systems; this result is a measurement, not a novel method.

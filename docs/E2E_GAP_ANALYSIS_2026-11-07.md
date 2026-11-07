# E2E gap analysis (2026-11-07): what each lever buys against the hard targets

## What was measured

All numbers come from the E2E-v1 run:

- **Setup:** llama.cpp, the Spark GPU, workstation CPU experts for layers 24-44, and the 10G direct link.
- **Decode:** 94.7 ms per token (10.5 tok/s).
- **Decode with MTP drafting (up to 3 draft tokens):** 59.3 ms per token (17.0 tok/s), with about 2.8 tokens committed per verify. One verify step therefore takes about 166 ms, against 95 ms for a single-token step, because each drafted token pulls in its own experts.
- **Cold 45K TTFT:** 223 s (200 tok/s prefill).
- **Cached 45K + 32 tokens:** 0.68 s.

## Bytes read per decoded token

Sizes come from the GGUF tensor table.

| component | bytes/token (current) | precision |
|---|---|---|
| attention (KDA + MLA/DSA) | 6.6 GB | Q8_0 |
| shared experts | 1.1 GB | Q8_0 |
| dense FFN + routers | 0.7 GB | Q8_0 |
| output head | 0.7 GB | Q8_0 |
| routed experts, top-8 x 42 layers | 4.8 GB | Q4_K (14.16 MB/expert) |
| **total** | **~13.9 GB** | |

The budget at 125 tok/s on the Spark's 241 GB/s is **1.93 GB per committed token** (8 ms).

Fixed weights amortise over the c tokens committed per verify. Routed-expert bytes do not, because each token in the verify batch selects its own experts.

## Decode floor model

The floor is t = (F/c + E)/BW_spark, where:

- F is the fixed-weight bytes;
- E is the expert bytes per token;
- c is the tokens committed per verify;
- BW_spark is the Spark's memory bandwidth.

It assumes an ideal bandwidth-bound run with zero network latency and everything local.

| configuration | F (fixed) | E (experts/token) | c | floor ms | tok/s ceiling | quality cost (M2, nats) |
|---|---|---|---|---|---|---|
| FP8-class fixed, top-8 int4 | 9.2 GB | 4.8 GB | 2.8 | 33.6 | 30 | +0.006 |
| same, workstation CPU reads half the expert bytes in parallel (quality-safe heterogeneous, perfect overlap) | 9.2 GB | 2.4 GB on Spark | 2.8 | 23.6 | 42 | +0.006 |
| + NVFP4 fixed | 4.9 GB | 4.8 GB | 2.8 | 27.2 | 37 | +0.040 |
| + top-6 ref8 | 4.9 GB | 3.6 GB | 2.8 | 22.2 | 45 | ~+0.10 |
| + int3 experts | 4.9 GB | 2.7 GB | 2.8 | 18.5 | 54 | unmeasured |
| + workstation CPU reads half the expert bytes in parallel (250 GB/s, perfect overlap) | 4.9 GB | 1.35 GB on Spark | 2.8 | 12.9 | 77 | as above |
| same, with a drafter committing c = 6 | 4.9 GB | 1.35 GB | 6 | 9.0 | 111 | as above |

Reaching 120-130 tok/s needs everything in the bottom row, plus more:

- a drafter that commits c >= ~7 tokens per verify (G5 found MTP saturates at ~2.7);
- perfect Spark/CPU overlap with almost no per-layer round-trip cost (21 sequential round trips at 0.32 ms already cost 6.7 ms per verify);
- a cumulative quality cost that already exceeds the non-inferiority margin.

With the FP8-class fixed weights that keep quality, the bandwidth ceiling is about 30 tok/s at c = 2.8, and about 55 tok/s even at c = 7.

## Cold 45K TTFT (target 10.8 s)

- **GPU compute for the whole model:** 45K x 2 x ~18B active = 1.6 PFLOP, which takes about 6.5 s at 250 TFLOPS effective.
- **Remote experts on the CPU:** the work is 45K x 8 x L_remote x 25.2M MAC.
  - With 21 remote layers that is 1.9e14 MAC.
  - That takes 11.6 s at AVX-512 VNNI int8 peak (32 cores, 16.4 TMAC/s), or about 23 s at 50% of peak.
- **Activations over the 10G link:** 45K x 4096 x 2 directions per remote layer. For 21 layers that is 15.5 GB in bf16 (12 s) or 7.7 GB in int8 (6 s).

Two conditions follow:

- **Placement:** cold 45K TTFT is feasible only if few layers stay remote (about 5-6 at most) and the GPU, CPU and link are fully overlapped.
- **Capacity:** the Spark has about 121 GB usable, while all experts at 4.5 bpw take 171 GB. Fitting them needs expert pruning, or an average of about 2.7 bpw using popularity-tiered bits.

## Cached 45K + 32 (target 0.2 s)

One 32-token step costs about one verify step (about 100-170 ms today) plus network time. The target is plausible with a better runtime that sends one message per layer and keeps few or no layers remote. Today it is 0.68 s.

## Conclusion

- **TTFT targets:** there is a plausible path. Prune or requantize the experts so nearly all of them fit on the Spark, and let the CPU take a few layers with overlapped int8 GEMM.
- **Decode target (120-130 tok/s):** it exceeds the Spark's memory-bandwidth ceiling by about 4x at quality-safe precision, and by about 1.1-1.6x even under the most aggressive idealised stack. Placement, quantisation or scheduling alone cannot reach it on this hardware without a quality loss beyond non-inferiority.

## Prefill profile (nsys, cold 4K prefill, E2E-v1 placement)

Prefill throughput is flat at about 200 tok/s from 4K to 45K tokens:

- 4K: 209-215 tok/s.
- 12K: 196-211 tok/s.
- 45K: 200 tok/s.
- With `-b/-ub 2048`: 4K reaches 262-286 tok/s, and 12K ranges from 187 to 268 tok/s.

So the cost is linear in tokens. Long-context attention is not the bottleneck.

The nsys trace of the cold 4K prefill shows:

- **GPU busy 45% of wall time.** 17.9 s of kernels over a 39.8 s span.
  - 9.9 s is llama.cpp's Q4_K MoE kernel (`mul_mat_q<Q4_K, mm_id>`), which reaches ~7 TFLOPS effective.
  - 2.5 s is the Q8_0 dense matmuls.
  - KDA (gated delta net), flash-attn and the DSA indexer together take <1.4 s.
- **GPU idle 55% of the time while waiting on the RPC experts.** There are 301 gaps longer than 2 ms, with a median of 55 ms. The remote experts on the workstation CPU run at about 1.5 TMAC/s, roughly 10x below AVX-512 VNNI int8 peak, and are strictly serial with the GPU work.

What this means for the levers:

1. **GPU MoE kernel.** A tensor-core grouped GEMM (FP8/NVFP4 CUTLASS/FlashInfer MoE on sm_121) in place of llama.cpp MMQ is worth 10-30x on the GPU half.
2. **CPU expert GEMM.** A batched int8 VNNI GEMM in the HeteroFlash expert server is worth up to ~5-10x on the CPU half.
3. **Overlap.** Running GPU-local experts concurrently with the remote experts of the same layer, instead of serially, removes the 55% wait.
4. **Remote layers.** With levers 1-3 at about 50% of peak, cold 45K is GPU ~6.5 s against CPU ~24 s for 21 remote layers. Meeting 10.8 s therefore still needs <= ~8 remote layers, which is the capacity problem described above.

## Kernel rooflines (measured 2026-11-07)

### GB10, vLLM v0.32.0 fused_experts on one GLM MoE layer

The layer has 288 experts, H=4096, I=2048, top-8. Results are in `results/2026-11-07-moe-kernel/gb10_fused_moe.json`.

| tokens per chunk | BF16 | FP8 w8a8 |
|---|---|---|
| 512 | 67.7 ms (3.0 TF) | 33.6 ms (6.1 TF) |
| 1024 | 71.1 ms (5.8 TF) | 34.8 ms (11.8 TF) |
| 2048 | 73.5 ms (11.2 TF) | 37.1 ms (22.2 TF) |
| 4096 | 80.9 ms (20.4 TF) | 42.2 ms (39.1 TF) |

Dense GEMM peak is 188 TF in FP8 and 97 TF in BF16.

- **Weight reads dominate.** Time is nearly flat in tokens, because every chunk re-reads all 288 experts (7.3 GB in FP8 at ~216 GB/s).
- **llama.cpp is not unusually slow.** Its ~7 TF at 512-token ubatches is the same weight-read bound, not a bad kernel.
- **The lever is chunk size.** 8-16K-token chunks plus NVFP4 experts would take the MoE part of a cold 45K prefill to about 5-8 s, if all experts were resident.

### Workstation CPU (Zen5, 32 threads), torch, one expert FFN

int8 `_int_mm` reaches:

- 5.3 TOPS at M=16 tokens/expert;
- 14.5 TOPS at M=64;
- 19.1 TOPS at M=128;
- 12.1 TOPS at M=2048.

BF16 reaches about 7 TF.

llama.cpp's RPC CPU path measured about 3 TOPS, so a proper int8 expert GEMM is worth about 4-6x.

### Cold 45K TTFT plan implied by these rooflines

- **CPU share.** One remote layer is 45,000 x 8 x 3 x 4096 x 2048 x 2 = 1.81e13 operations per 45K prefill. At about 15 TOPS sustained that is about 1.2 s per layer. Leaving room for the GPU share and the activation transfer (about 0.6 s per layer in bf16 unless fully overlapped) gives a budget of <= ~5-6 remote layers.
- **Spark share.** The Spark must then hold the routed experts of about 36-37 layers, plus fixed weights and KV, in about 121 GB.
  - That is about 36.5 x 288 experts in about 100 GB, roughly 3.0 bits/weight on average. NVFP4 is 4.5 bpw.
  - So it needs popularity-tiered bits, for example hot experts at 4-bit and cold at 2-3-bit.
  - Quality at those bits is unmeasured. That is the next experiment, run on the Max-Q rig.

# GLM-5.3-Flash prefill and cached-state on one GB10: accounting, bounds, causes

Date: 2026-11-08. Scope: single DGX Spark (GB10, sm_121), batch 1, 45K-token prompt.
Measured: cold 45K = 39 s (1,154 tok/s), cached 45K+32 = 0.42 s. Targets: 10.8 s / 0.20 s.

**Short version**
- The cold target needs about 4,170 tok/s on one Spark. That is about 2x the best GLM-5.3-Flash prefill anyone has reported on **two** Sparks.
- The current 39 s is already better per GPU than any public number I found. Running the routed experts through BF16 tensor cores (W4A16-style) caps this setup at about 15-20 s, however well the kernels are tuned.
- The cached target of 0.20 s is at or below the memory-bandwidth floor: 32 tokens touch about 60% of all experts, not 8 per layer. The 0.42 s measured is within about 1.6-2x of that floor.

---

## 0. Model geometry used (from the released config.json)

Source: https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json

- hidden 4096, 45 layers. 34 KDA layers: 64 heads x 128, short conv 4. 11 DSA/MLA layers: layers 3, 7, ..., 43.
- MLA: 64 heads, q_lora 1536, kv_lora 512, qk_nope 256, **rope 0** (NoPE), v 256.
- Indexer: 32 heads x 128, `index_topk` 2048, **`index_kpool` 4 with `index_kpool_compress`**.
- MoE: 288 routed experts, top-8, 1 shared expert, `moe_intermediate_size` 2048. The first 3 layers are dense (12288). That leaves 42 MoE layers.
- **mHC enabled, `hc_mult` 4, 20 Sinkhorn iterations.** The residual stream is 4 x 4096 wide.
- 1 MTP layer.

Per-matrix sizes:
- One expert: 3 x 4096 x 2048 = 25.17 M params.
- Routed experts in total: 288 x 42 x 25.17 M = 304.5 B params. At IQ2_XXS (2.06 bpw) that is **78.4 GB**.
- Non-expert weights: KDA ~138 M/layer, MLA+indexer ~124 M/layer, shared 25 M/layer, dense 151 M/layer, embedding and LM head 0.63 B each. That is about 8.3 B params, about 8.3 GB at FP8.

## 1. Prefill FLOP accounting (n = 45,000)

Linear layers (2 FLOP per param per token):

| Component | Formula / token | GFLOP/token | TFLOP @45K | Share |
|---|---|---|---|---|
| Routed experts | 2·8·25.17M·42 | 16.9 | **761** | 47% |
| Shared expert | 2·25.17M·42 | 2.1 | 95 | 6% |
| Dense MLP (3 layers) | 2·3·4096·12288·3 | 0.9 | 41 | 2.5% |
| KDA projections q,k,v,o + low-rank gates | 2·~138M·34 | 9.4 | **423** | 26% |
| MLA + indexer projections | 2·~124M·11 | 2.7 | 123 | 7.6% |
| Router (fp32), mHC mixing | — | 0.15 | ~6 | <0.5% |
| **Linear total** | | **~32** | **~1,450** | |

Attention-like terms:

**Sparse MLA.** Each query attends to k = min(i, 2048) keys.
- Absorbed/latent form (what FlashMLA-sparse-style kernels run): 64 heads x (2·512 QK + 2·512 PV) = 131 kFLOP per (query, key). That gives 268 MFLOP per token per layer, and **≈133 TFLOP** in total.
- Non-absorbed form (head dim 256+256): about 66 TFLOP.
- Caveat: if the top-2048 counts pooled 4-token units (`kpool` = 4) rather than tokens, attention cost is 4x, i.e. 265-530 TFLOP. I could not confirm this from the config. Check the indexer code.
- For comparison, dense causal MLA at 45K would cost about 1.46 PFLOP. DSA saves about 1.3 PFLOP.

**DSA indexer (the O(n²) term).**
- Per (query, pooled key): 32·128·2 = 8.2 kFLOP. With kpool 4, the causal pair count is n²/8. That gives 2.1 TFLOP per layer and **≈23 TFLOP** in total (91 TFLOP without pooling).
- The indexer equals the linear cost per token when n²/8·8192·11 = n·32e9, i.e. **n ≈ 2.8 M tokens** (≈ 710 K tokens without pooling). At 45K it is 1.6-6% of FLOPs.
- Top-k selection needs about n²/8 fp32 logits per layer, about 2 GB/layer written and read across all chunks. That is about 0.1 s in total and not dominant.

**KDA chunked (WY/delta-rule, chunk C = 64, d = 128).**
- Per head per token ≈ 2(5·C·d + 3·d²) ≈ 180 kFLOP. Times 64 heads x 34 layers, that is 0.39 GFLOP/token, **≈18 TFLOP** (~1%).
- Its real cost is memory. Per-chunk states are [n/64, 64 heads, 128x128], about 1.4 GB/layer in bf16 when written and read. Plus the kernel efficiency of fla/Triton on 99 KB SMEM. Estimate 0.5-1.5 s wall time in total.

**Total ≈ 1.62 PFLOP (range 1.6-2.1 PFLOP if attention uses 8192 keys).**

About 89% of the FLOPs are dense/grouped GEMM (1,450 / 1,624). MoE expert GEMMs alone are about 53% (routed plus shared).

Non-FLOP traffic that matters at 273 GB/s (STREAM measures 200-213 GB/s, https://forums.developer.nvidia.com/t/gb10-really-does-hit-1-pflop-nvfp4-2-4-sparse-measured-with-an-open-source-tool-to-reproduce-it/373618):
- **mHC residual (hc_mult 4).** Pre-mix reads 4C and writes C. Post-mix reads 4C+C and writes 4C. That is about 13·4096 elements per token per sub-layer, x 90 sub-layers. At bf16 that is **~430 GB, about 2 s at 210 GB/s** (about 4 s if the stream is fp32), unless fused into the norm/GEMM epilogues. b12x does ship "mHC residual/projection kernels" (https://github.com/yatesdr/b12x). Check that the fork uses them.
- **MoE permute, activation and combine as standalone kernels.** About 300 KB per token per layer, which is **~570 GB, about 2.7 s** if unfused, and about 0 if gather/SwiGLU/scatter-reduce are fused. Traffic inside the GEMMs is hidden, because the expert GEMMs at 228 tokens/expert run at ~950 FLOP/B including activations. That is above the BF16 ridge of about 476 FLOP/B.
- **Weight re-streaming.** One pass over all weights is about 87 GB, about 0.41 s at 210 GB/s. With an 8192-token chunk there are 6 passes, about 2.5 s if not overlapped.

## 2. What GB10 actually delivers

| Precision | Spec / peak | Best measured real GEMM | Source |
|---|---|---|---|
| NVFP4 (mxf4nvf4) | 500 dense / 1000 sparse | microbench 486-511 dense. CUTLASS dense **356-375 TFLOPS** (4096x14336x4096). Custom 85-129 | 373618, 359960, 361600 |
| NVFP4 MoE grouped GEMM | — | **120-154 TFLOPS** (8 and 64 experts; 256x128 tile best for prefill) | 359960 |
| FP8 / mxf8f6f4 | ~250 dense (microbench 244-256) | CUTLASS **188**, MAMF **207.7** | 359960, StorageReview |
| BF16 | not published (likely ~125) | MAMF **99.8 TFLOPS** | StorageReview |
| SMEM | 99-101 KB/SM (vs 228 KB on SM100) | "main constraint, not compute" | 359960, 360142 |

URLs:
- 373618: https://forums.developer.nvidia.com/t/gb10-really-does-hit-1-pflop-nvfp4-2-4-sparse-measured-with-an-open-source-tool-to-reproduce-it/373618
- 359960: https://forums.developer.nvidia.com/t/sm121-cutlass-kernel-optimization-results-nvfp4-356-tflops-moe-grouped-gemm-on-dgx-spark/359960
- 361600: https://forums.developer.nvidia.com/t/custom-fp4-cuda-kernel-129-tflops-on-dgx-spark-with-pre-quantized-weight-cache/361600
- 360142: https://forums.developer.nvidia.com/t/fp4-on-dgx-spark-why-it-doesnt-scale-like-you-d-expect/360142
- StorageReview: https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops

Caveat: the 120-154 TFLOPS grouped figure was measured with 8 and 64 experts. This model has 288 experts at about 228 tokens each, so there are more and smaller groups, and a 256-row M-tile wastes about 11% at M=228. The NVFP4 MoE ceiling used below may be optimistic for this shape.

The NVIDIA "1 PFLOP" figure is FP4 with 2:4 sparsity. Sparsity is irrelevant here, so the dense figure is 500 TFLOPS. Real fused MoE kernels reach about 25-30% of that.

I found no repo named `nvidia/GLM-5.3-Flash-FP4-Local`. The closest is `nvidia/GLM-5.3-Flash-NVFP4` (https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4), and it publishes no prefill numbers.

Published GLM-5.3-Flash prefill numbers (single request):

| Setup | Result | 45K-equivalent | Source |
|---|---|---|---|
| 2x Spark TP2, NVFP4 | 4K: 1,615 tok/s. 32K: 1,576 tok/s (20.8 s). 131K: 1,965 tok/s | ~28 s on 2 GPUs | https://forums.developer.nvidia.com/t/qwen3-8-flash-next-glm-5-3-flash-deepseek-v4-flash-on-2x-dgx-spark-prefill-ttft-swe-bench-pro/382697 |
| 2x Spark, EXL3 vLLM fork | 680-705 tok/s flat to 32K. Later about 1,600 peak | | https://forums.developer.nvidia.com/t/glm-5-3-flash-on-2x-dgx-spark-nvfp4-kv-cache-288-b-token-2-2m-token-pool-16-way-serving-900k-context-full-recipe/382120 |
| 2x Spark vLLM NVFP4 recipe / EXL3 TensorFold | 1,425 / 1,979 tok/s | | contextstudios.ai recipe pages (search summary) |
| 3x Spark TP3 | about 1,800 tok/s | | https://forums.developer.nvidia.com/t/glm-5-3-flash-nvfp4-on-3x-dgx-spark-tp-3-512k-context-35-tok-s/381534 |
| 1x Spark llama.cpp UD-IQ3_XXS / UD-Q2_K_XL | 270 tok/s. About 217 tok/s (10K prompt in 46 s) | | contextstudios weschera recipe. https://dev.classmethod.jp/en/articles/dgx-spark-glm-5-3-flash-first-touch/ |

Other models on the same 2x Spark harness: DeepSeek-V4-Flash NVFP4 at 1,057-1,201 tok/s, Qwen3.8-Flash-Next FP8 at 1,206-2,591 tok/s (thread 382697).

**This setup's 1,154 tok/s on one Spark is already at or above the per-GPU rate of every published multi-Spark GLM-5.3-Flash number.** Those are about 700-1,000 tok/s per GPU, and that is with TP communication overhead. It is also about 4-5x the single-Spark llama.cpp numbers. The 4,170 tok/s target has no public precedent on any number of Sparks.

## 3. MoE prefill efficiency, 2-bit W4A16-style vs W4A4

**Arithmetic intensity.** At 8,192 tokens per chunk, each expert gets on average 8192·8/288 ≈ **228 tokens**.
- Per expert per layer: 11.5 GFLOP against 6.5 MB of IQ2 weights, about **1,770 FLOP/B**.
- The bandwidth ridge is 476 FLOP/B at 100 TFLOPS and about 1,800 FLOP/B at 375 TFLOPS.
- So the kernel is **compute-bound on its MMA rate**, not on weight bytes. Per expert, compute is about 165 µs at 70 TFLOPS, against 31 µs to stream the weights.

**Dequant cost.** IQ2_XXS dequant is a lookup into a 256-entry grid of 8-weight entries, plus 7-bit sign tables and 4-bit sub-block scales. That is roughly 3-5 ALU ops per weight.
- If the kernel dequantizes once per M-tile of 128 tokens, there are about 256 MMA FLOPs per dequantized weight. 6,144 CUDA cores at about 2.5 GHz could then feed more than 900 TFLOPS-equivalent, so dequant is not the binding limit.
- It **becomes** binding if the kernel uses small M tiles (16-32, decode-tuned), dequantizes per MMA fragment, or decodes IQ2 into a BF16 staging buffer.
- The worst case, materializing BF16 experts per chunk, costs about 304 B params x (2 B write + 2 B read) per pass. That is about 1.2 TB, or **~5.8 s per chunk, ~35 s for 6 chunks**, which on its own would explain the 39 s.
- b12x's documented MoE flavours are nvfp4, w4a8_mx, w4a8_nvfp4, and W4A16 with FP4 weights. IQ2_XXS is listed only for *dense* linear (https://docs.vllm.ai/en/latest/features/quantization/b12x/). **The fork's IQ2 MoE path is the single most important unknown. Inspect it.**

**Ceilings by MMA type**, for routed experts only (761 TFLOP):

| Path | Rate | Time |
|---|---|---|
| BF16 MMA (W4A16/W2A16) | ≤99.8 MAMF, ~60-70 grouped | 7.6 s at 100%, **11-13 s realistic** |
| FP8 MMA (W2A8/W4A8-MX) | ~150-180 grouped | **4.2-5 s** |
| NVFP4 W4A4 | 120-154 measured | **4.9-6.3 s** |

So W4A4 or W8A8-class kernels gain about 2x on the dominant term. They can **not** reach the 4x ratio the spec sheet suggests, because the measured grouped NVFP4 kernel is SMEM/tile-limited (359960).

**Quality evidence on activation precision** (sparse and mixed):
- NVIDIA ModelOpt on Qwen3.6-35B-A3B: W4A4 PTQ vs BF16 costs IFBench −2.1, MMMU-Pro −1.2, and roughly 0 on GPQA-D, tau2 and AA-LCR. QAD recovers most of the loss. https://nvidia.github.io/Model-Optimizer/pr-preview/pr-2522/announcements/qwen36-w4a4-qad.html
- An unverified forum/Discord claim says W4A4 KLD is "2-4x worse than W4A16" and that the gap grows past about 10K context. https://forums.developer.nvidia.com/t/update-for-nvfp4-model-conversion-to-use-w4a16-instead-of-w4a4/370403 This is relevant to 45K prompts, but there are no published numbers.
- Red Hat's NVFP4 results (vLLM's NVFP4 path quantizes activations too, i.e. W4A4) show more than 99% average recovery vs BF16 on Qwen3-235B-A22B and Llama-4-Scout, with most task deltas under 1%. Models of 70-235B recover about 99%, ~30B models 97-99%, 7-14B models 95-98%. The text gives no W4A16 rows. https://developers.redhat.com/articles/2026/02/02/accelerating-large-language-models-nvfp4-quantization
- The Spark decode table in 360142 shows NVFP4 math 74% vs BF16/FP8 78-80% on Qwen3-Coder-30B. That confounds weight and activation quantization.
- FP8 activations with fine-grained (1x128 or MX-32) scales are widely deployed (DeepSeek-V3 trains with them). They are the lower-risk option.

**Format note for a W2A8 kernel.** IQ2_XXS grid magnitudes are {8, 25, 43} x (0.5+s)/4 x d.
- They are exact in INT8, but **not** exact in E4M3: 25 rounds to 24 or 26 (±4%), and 43 rounds to 44 (+2.3%).
- In E2M1 (NVFP4) the ratios 1 : 3.125 : 5.375 map to 1 : 3 : 6, a +11.6% error on the largest weights.
- Re-encoding IQ2 to an FP8 operand in-register is therefore a small extra error relative to the 2-bit noise. Re-encoding to NVFP4 adds visible error on top of IQ2_XXS's own 2-bit error.

## 4. Cached 45K + 32: what the floor really is

**Correction to the task's premise.** "Weights read once ≈ 11 GB" is the **1-token** figure. With 32 tokens x top-8 under near-uniform routing, the expected number of distinct experts per layer is

U = 288·(1−(1−8/288)^32) ≈ **171** (59% of experts).

Bytes read = 42·171·6.5 MB + 8.3 GB non-expert ≈ 46.7 + 8.3 = **~55 GB**. That takes **0.20 s at 273 GB/s spec, and 0.26 s at the measured 210 GB/s STREAM rate.**

I assume near-uniform routing. Load-balanced noaux_tc routing should limit concentration within a sequence, but that is an assumption, and the only way to know U is to log it. Only real routing locality can lower U. Meeting 0.20 s at 210 GB/s needs U ≤ ~123. **Log router decisions for real 32-token suffixes to measure U.**

Everything else in the cached step is small:
- Compute: 32 x 32 GFLOP ≈ 1 TFLOP, about 10 ms.
- DSA indexer over 45K keys for 32 queries: reads about 11.25K pooled keys x 132 B ≈ 1.5 MB/layer, which is negligible. Sparse MLA reads about 1 MB/layer.
- KDA state load: 34 x 2-4 MB ≈ 70-140 MB, under 1 ms.

**vLLM hybrid prefix-cache mechanics (`--mamba-cache-mode align`).**
- KDA supports only `align` (LMCache Kimi-Linear recipe, https://docs.lmcache.ai/recipes/kimi_linear.html). It is marked experimental.
- In align mode the recurrent state is snapshotted only on the block grid. A hit needs both the attention blocks and the matching snapshot. Prefill chunks are scheduled to land on block boundaries (https://github.com/vllm-project/vllm/pull/46384, https://github.com/vllm-project/vllm/pull/45845).
- vLLM raises the attention block size until the attention page is at least the mamba page. For Qwen3.5 that gave 528 tokens (https://github.com/vllm-project/vllm/issues/40696).
- For this model: a KDA state of 64x128x128 is 2 MB in bf16 (4 MB in fp32), plus conv state. The MLA+indexer page is about 0.55-1 KB per token per layer. So expect **a block size of roughly 2K-8K tokens** (assumption; grep the startup log for "Setting attention block size").
- Consequence: a hit resumes at floor(P/B)·B, so up to B−1 tokens are recomputed. A recompute of 1-2K tokens touches all 288 experts (87 GB, ≥0.41 s) plus 1-2 s of compute.
- The measured 0.42 s says the test prompt was either near-aligned or `--prefix-match-unit` (PR 46384, merged 2026-07) is in effect. Confirm through `prompt_tokens_details.cached_tokens`. For an arbitrary 45K prefix the expected recompute is B/2 unless the fine match unit is enabled.
- PR 46384's own benchmark (Qwen3.5-35B-A3B, 10.0K then 10.2K tokens) shows a partial hit going 0.134 s → 0.096 s.

**Other overheads.**
- The 0.42 s versus the 0.26 s floor leaves about 160 ms. Likely contributors:
  - Tokenizing and rendering the chat template for 45K tokens on Grace cores, about 30-90 ms. Sending token IDs avoids this.
  - Hashing about 2.8K 16-token blocks, tens of ms in Python.
  - Eager attention/KDA kernels between piecewise CUDA graphs, launch gaps, and sampler/HTTP.
- The only published hybrid cached-long-prompt figure I found: a 78K exact resubmit at 0.3 s on 2x Spark (each GPU reads half the weights), down from 22.7 s when it replayed 15K tokens (thread 382120).
- SGLang's MambaRadixCache (state snapshots plus KV pointers) reports TTFT at 57.6% of baseline on Qwen3-Next with a 10K shared prefix (secondhand: https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/).
- I found no published cached-long-prefix TTFT for MiniMax or Jamba in vLLM or SGLang.

**Realistic best for cached 45K+32 at IQ2_XXS: about 0.28-0.33 s.** The 0.20 s target needs fewer bytes:
- routing locality (U ≤ 120), or
- NVFP4 non-expert weights (−4 GB, −20 ms), or
- smaller experts for the touched set.

Overhead tuning alone cannot reach it.

## 5. Phase-specific execution ideas (single GPU + remote CPU)

1. **No second high-precision expert copy.** NVFP4 experts would take 304.5 B x 4.5 bit ≈ 171 GB, more than 128 GB. A decode-only higher-precision copy does not fit either. The remote CPU is about 25x too slow for prefill (994 s), so it is useful only as a cold store, for example offloading KV and KDA snapshots, about 0.5 GB per 45K prefix.
2. **Phase-specific *kernel*, same bytes (best lever).** Decode is bandwidth-bound, so 2-bit is optimal there. Prefill is compute-bound, so the MMA type matters.
   - Build a W2A8 grouped MoE kernel: decode IQ2_XXS in-register to E4M3, use MXFP8 activations, run mxf8f6f4 MMA.
   - This is the w4a8_mx pattern b12x already has, with an IQ2 decoder in place of the FP4 one.
   - Expected effect: routed MoE about 11-13 s → 4.5-5.5 s, with low quality risk (see §3).
3. **Alternative: staging re-encode per layer** to NVFP4/MXFP8, then use the existing W4A4/W8A8 kernels.
   - Costs about 2.0 s (NVFP4) or 3.5 s (MXFP8) **per weight pass**.
   - Worthwhile only with one or two chunks, i.e. max-num-batched-tokens ≥ 22-45K.
   - NVFP4 adds re-quantization error (§3).
4. **Chunk size.** Going from 8192 to 32768 or 45056 removes 4-5 of the 6 weight passes (≤2 s) and per-chunk fixed costs.
   - The 382120 thread found about 700 ms of host-sync overhead per mixed step across 45 MLA and 42 MoE layers before a fix. Check the fork for `.item()`/`.tolist()` in routing or the grouped-GEMM sizing.
   - Memory is fine: MoE intermediates at 45K are about 3 GB per buffer.
5. **Non-expert FP8 → NVFP4 for the KDA projections** (26% of FLOPs). This takes FP8 188 → NVFP4 356 TFLOPS, saving about 1.1-1.7 s cold and about 20 ms cached. Quality risk on attention projections is higher than on experts, so evaluate on long-context tasks.
6. **Fuse mHC and MoE glue.** This is worth about 2-5 s of pure bandwidth time if they currently run as standalone kernels.

## 6. Lower bounds (stated assumptions)

**Cold 45K.**
- **Roofline floor** (500 TFLOPS dense FP4 for everything, 273 GB/s, perfect overlap): max(1.62 PFLOP / 500 T = 3.2 s, ~0.6-1 TB of unfused traffic / 273 = 2.5-4 s) ≈ **3.5-4 s**. Not attainable.
- **Best-measured-kernel bound**: NVFP4/W2A8 MoE at 154 TFLOPS (4.9 s), FP8 non-expert at 188 (3.6 s), sparse attention at about 60 TFLOPS (2.2 s), indexer and top-k (0.5 s), KDA (0.6 s), fused glue (~1 s), single chunk. That gives **≈ 12.5-13 s**, or **≈ 11-11.5 s** with NVFP4 non-expert. **10.8 s requires beyond-published kernel efficiency** (about 150 TFLOPS sustained over all 1.6 PFLOP).
- **Current format bound** (W4A16-style BF16 MMA at 60-70 TFLOPS grouped): ≈ 11-13 + 3.6 + 2.2 + 1.1 + glue ≈ **18-20 s**. The 39 s measured is therefore about 2x the current-format bound.

**Cached 45K+32.**
- Floor ≈ **0.20 s** at the 273 GB/s spec, **0.26 s** at the achievable 210 GB/s, assuming uniform routing.
- Practical ≈ **0.28-0.33 s**.

## 7. Most likely reasons the cold prefill takes 39 s (ranked)

1. **Routed experts run on BF16 MMA (W4A16-style).** The 761 TFLOP of expert GEMMs at ≤100 TFLOPS MAMF (~65 grouped) costs about 11-13 s. If the IQ2 path stages BF16 weights or uses decode-sized tiles, it could be 20-35 s on its own. Test: kernel times with B12X_TIMING=1 or nsys for a single 8192 chunk.
2. **Standalone memory-bound kernels on a 210 GB/s bus**: mHC 4-stream residual (~2-4 s), MoE permute/SwiGLU/combine (~2.7 s), KDA intermediates (~1 s). These are invisible on HBM GPUs and costly on GB10.
3. **Non-expert FP8/MXFP8 GEMMs at 0.69 PFLOP**: ≥3.6 s, more if some projections (kv_b, gates, indexer, low-rank KDA gates) fall back to BF16 or Triton.
4. **Sparse MLA prefill and indexer kernels on sm_121.** FlashMLA-sparse/DeepGEMM target sm90/sm100. The b12x/Triton replacements have unknown efficiency: 2-9 s, depending on whether 2048 or 8192 keys are attended.
5. **Per-chunk fixed costs x 6**: weight re-stream (0.41 s each if not overlapped) and host syncs (cf. 700 ms per step in 382120). Test: sweep max-num-batched-tokens 4K/8K/16K/32K and fit T = a·chunks + b·tokens.
6. **KDA fla kernels** (0.5-1.5 s) and align-mode snapshot writes. These are minor.

The first diagnostic to run is the chunk-size sweep plus a single-chunk kernel-time breakdown. Together they separate items 1 and 5 from items 2-4 in one run.

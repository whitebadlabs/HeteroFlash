# Speculative decoding on MoE targets: literature, GLM-5.3-Flash acceptance, expert-union cost model

Date: 2026-11-08. Scope: batch-1 decode of GLM-5.3-Flash on one DGX Spark (GB10, ~273 GB/s), optionally with routed experts on a remote 32-core Zen5 CPU (~190 GB/s) over 10 GbE.
Rules used: every number has a URL. "SR" = self-reported on a model card or blog, not independently checked. Numbers marked **[derived]** are my arithmetic from cited inputs.

## 1. Methods table

| Method | Mechanism | Target model(s) | Hardware | Batch | Measured AL / speedup | Code |
|---|---|---|---|---|---|---|
| MoESD ([arXiv 2505.19645](https://arxiv.org/html/2505.19645v4)) | Theory: "target efficiency" T(B,1)/T(B,γ); expected experts N(t)=E·(1−(1−K/E)^t) | Qwen2-57B-A14B (0.5B draft), Mixtral+EAGLE | 2–4 GPU vLLM | 1–100 | peak 2.29x Qwen2 HumanEval γ=4 at *medium* batch; Mixtral 1.67–1.79x | in paper |
| Cascade ([arXiv 2506.20675](https://arxiv.org/html/2506.20675v1)) | Utility U=ETR/(t_spec/t_base); disable spec if U<1, pick K maximizing U | Mixtral, Phi-3.5-MoE, OLMoE, DeepSeek-V1, Qwen1.5-MoE (n-gram drafter) | RTX 6000 Ada | 1 | static K slows down up to 1.5x (54% at K=3 on math); verify overhead 1.5–3x; Cascade caps slowdown at 5%, +7–14% vs static K | vLLM impl. (no link found) |
| EcoSpec ([arXiv 2607.12696](https://www.alphaxiv.org/abs/2607.12696.md)) | Cost-aware draft-tree selection: score = P(path)/|new experts| (lossless) | Qwen3-235B, GPT-OSS-120B, DeepSeek-V3.1 | 8xH200 | 1 | vs EAGLE-3: 1.22→1.36x (Qwen3-235B, T=0); DSv3.1 vs MTP 1.10→1.15x; AL DSv3.1 MTP 2.85 vs 2.79; experts/layer verified E: DSv3.1 MTP 31.4, Qwen3-235B EAGLE-3 23.7; verify latency "linear in active experts" | none |
| EVICT ([arXiv 2605.00342](https://arxiv.org/pdf/2605.00342)) | Truncate draft tree to utility-maximizing prefix using profiled C(k) (lossless) | Qwen3-30B-A3B, Ling-flash-2.0, Qwen3-235B | SGLang, GPU | 1 | Qwen3-30B: EAGLE-3 1.22x → EVICT 1.40x (T=0); MAT 3.65 → 3.09; −74.7% verified tokens, −32.5% experts, −26.6% verify latency. Fig. 1: experts/layer ≈8 (1 tok), ≈17 (4), ≈25 (8), ≈39 (32 nodes) | none |
| Limits of Speculation ([arXiv 2609.22156](https://arxiv.org/abs/2609.22156)) | Offline SSP oracle for speculation budget; marginal rule ΔE[Cost]/ΔE[a] | Qwen3-Coder + EAGLE-3 | n/a (abstract only) | — | qualitative | — |
| EAGLE on Mixtral ([arXiv 2401.15077](https://arxiv.org/html/2401.15077v3)) | Feature-level autoregressive head | Mixtral 8x7B | GPU | 1 | τ=3.25, 1.50x (vs 2.7–3.5x on LLaMA2-70B); authors blame extra experts read in verify | [github](https://github.com/SafeAILab/EAGLE) |
| DeepSeek-V3 MTP ([arXiv 2505.09343](https://arxiv.org/pdf/2505.09343)) | 1 MTP module reused as drafter | DeepSeek-V3 | H800 cluster, large-batch EP serving | large | 2nd-token acceptance 80–90%, 1.8x TPS — **not a batch-1 number** | — |
| GLM-5.2 MTP ablation ([InferenceX](https://inferencex.semianalysis.com/model/glm-5-3), [zai-org/GLM-5](https://github.com/zai-org/GLM-5)) | Improved MTP (index/KV share, rejection sampling, TV loss) | GLM-5.2 (large) | n/a | n/a | AL 4.56 → 5.47; "up to 20%" (dataset/K unstated) | — |
| DFlash ([arXiv 2602.06036](https://arxiv.org/abs/2602.06036)) | Block-diffusion drafter, whole block in one forward | various | GPU | 1+ | ">6x lossless", up to 2.5x over EAGLE-3 | [z-lab/dflash](https://github.com/z-lab/dflash) |
| incoai DFlash2 ([HF](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)) | DFlash2, 1B params BF16 | GLM-5.3-Flash | — | — | card says numbers being re-measured; SR via others below | SGLang PR 36708 |
| canada-quant DFlash2-G/F/E ([G](https://huggingface.co/canada-quant/GLM-5.3-Flash-DFlash2-G), [E blog](https://cql.ca/news/glm-5-3-flash-dflash2-e.html)) | Self-trained DFlash2, taps layers 5..42 | GLM-5.3-Flash W4A16 | 8xB300 / 2xSpark / H200 | 1 & 16 | SR, holdout, thinking ON, T=1: K=7 AL 3.676 (G) vs 3.632 (incoai); K=4 3.088 vs 3.136; c1 K=7 3.697; 2xSpark E: 3.597 (c1) | HF |
| Anbeeld SGLang MTP vs DFlash2 ([HF](https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF)) | Native MTP vs DFlash2, both K=7 | GLM-5.3-Flash FP8 | 4xGB300 TP4 | 1 | SR, T=1.0, max reasoning: MTP AL GSM8K 5.06, MATH 4.95, HumanEval 4.70, MBPP 4.26, **MT-Bench 3.71**; DFlash2 5.78/5.86/5.32/4.85/**4.03**; MT-Bench speedup MTP 1.36x, DFlash2 1.73x | HF |
| tonyd2wild 2xSpark ([MTP repo](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark), [DFlash2 doc](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/blob/main/docs/DFLASH2-SPECULATIVE-DECODING.md)) | vLLM MTP-4 vs DFlash2 K=7 | GLM-5.3-Flash NVFP4 | 2x GB10 TP2 | 1 | SR, code: MTP-4 per-position 0.74/0.47/0.27/0.15, 21.8 tok/s (vs 14.3 bf16-KV no-spec); DFlash2 46.9 tok/s, 74.1% of 7 drafts accepted | GitHub |
| NVIDIA forum SGLang 2xSpark ([thread](https://forums.developer.nvidia.com/t/glm-5-3-flash-dflash2-on-sglang-2x-dgx-spark-first-sglang-path-recipe-4-gb10-fixes-honest-numbers-concurrency-curve/381703)) | DFlash2 on SGLang | GLM-5.3-Flash NVFP4 | 2x GB10 | 1 | SR: 29.4 tok/s code / 23.4 prose vs 14.7 no-spec; ~5–6 accepted per 6-block on code | recipe in thread |
| Production write-up ([substack](https://enterpriseaireport.substack.com/p/serving-glm-53-flash-in-production)) | vLLM MTP K=5 → 3 | GLM-5.3-Flash | 4xH200 | 8–32 | per-position p1=0.81, p4=0.17, p5=0.07; K=3 AL 2.69–2.74; K=3 best | — |
| OEA ([arXiv 2511.02237](https://www.alphaxiv.org/abs/2511.02237.md)) | Keep top-k0 per token; extra experts only if already loaded by batch (lossy) | Qwen3-30B, Qwen3-235B | GPU | 16 | −49% unique experts / −39% MoE latency (30B, k0=3); −26%/−15% (235B, k0=5); "no statistically significant" accuracy loss | — |
| Lynx ([arXiv 2411.08982](https://arxiv.org/pdf/2411.08982)) | Batch-wide active expert set, remap low-confidence assignments (lossy) | Mixtral-class | GPU | batched decode | up to 1.55x latency, "negligible" accuracy loss (code, math) | — |
| SP-MoE ([arXiv 2510.10302](https://arxiv.org/html/2510.10302v2)) | Draft-guided expert prefetch + cutoff layer, PCIe offload | Mixtral, Phi-3.5-MoE, DS-V2-Lite | 3090/4090/A100, PCIe4 | 1 | top-1 expert prediction ~88%; TPOT −34% vs Mixtral-Offloading, −12% vs AdapMoE; 1.07–3.5x | — |
| Specter / MoE-SpeQ ([arXiv 2511.14102](https://arxiv.org/html/2511.14102v1)) | INT4 self-draft predicts experts for prefetch | Phi-MoE, Qwen1.5-MoE, DS-V2-Lite | A100-40G | ~1 | prefetch hit >96%; up to 2.34x vs best offloader | — |
| SpecMoEOff ([arXiv 2508.21706](https://arxiv.org/abs/2508.21706)) | Spec decoding to enlarge per-expert work under offload; CPU attention | MoE (unspecified in abstract) | CPU+GPU | small | up to 2.5x decode throughput | — |
| DraftExpert ([arXiv 2607.24434](https://www.alphaxiv.org/abs/2607.24434.md)) | Self-spec with fixed-footprint draft path + draft-router prefetch | DS-V2-Lite, Moonlight-16B | 4090 + host experts; phone NPU | 1 | verify cost vs 1-token AR: fixed K=4 2.35x → 1.48x (truncation) → 1.25x (+prefetch); decode 1.29–1.61x | none |
| Fiddler ([arXiv 2402.07033](https://arxiv.org/abs/2402.07033)) | CPU computes experts, GPU rest | Mixtral | limited-VRAM GPU | 1 | 1.26x single-batch vs baselines | [efeslab/fiddler](https://github.com/efeslab/fiddler) |
| KTransformers ([kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers), [MADSys paper page](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/); numbers via README summary) | CPU experts, GPU dense | DeepSeek-V3/R1 | Xeon + GPU | 1 | 8.73 → 11.26 tok/s (1→2 sockets); 13.69 with **6 instead of 8 experts** (lossy); no MTP | GitHub |

Searched and not found: a **modal-labs** GLM-5.3-Flash drafter on HF (modal-labs is credited for the SGLang DFlash integration only, per [z-lab/dflash](https://github.com/z-lab/dflash)). Unverified others: [avlp12 Alis MTP drafter](https://huggingface.co/avlp12/GLM-5.3-Flash-Alis-MTP-Drafter), [Capicua DSpark](https://huggingface.co/Capicua25x/GLM-5.3-Flash-DSpark), [cfontes TR3-v3](https://huggingface.co/cfontes/GLM-5.3-Flash-DFlash2-TR3-v3) (AL 1.83 at K=2 on EXL3).

## 2. Expected GLM-5.3-Flash MTP acceptance length, k=1..5

Definitions match across sources: AL = committed tokens per verify step = 1 + Σ_{i≤k} p_i, where p_i is the *unconditional* fraction of steps that accept draft position i (vLLM's per-position metric; Anbeeld defines AL as completion tokens / verify steps, same thing). The canada-quant G per-position numbers confirm this: 1 + (0.753+0.562+…+0.161) = 3.676 ([card](https://huggingface.co/canada-quant/GLM-5.3-Flash-DFlash2-G)).

GLM-family k=1 anchor: the vLLM GLM-4.x recipe says acceptance at num_speculative_tokens=1 "typically tops 90%" ([vLLM GLM recipe](https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM.html)) → AL(1)≈1.9 for GLM-4.7 on its random 8k/1k benchmark. That is above every 5.3-Flash measurement below, so do not use it for the small single-layer 5.3-Flash head.

There are two regimes, and no source measures both at the same K. SGLang at K=7 measures MT-Bench AL 3.71. Extrapolating the vLLM per-position curves below to K=7 (p6, p7 ≈ 0.04, 0.02) gives ≈2.7–3.0. That ~0.7–1.0 token gap is real but confounded (see below). Every SGLang value at k<7 in the table is fitted, not measured.

**vLLM regime (quantized target, BF16 MTP head)**
- 2xSpark NVFP4, code, MTP-4: p = 0.74/0.47/0.27/0.15 ([tonyd2wild](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/blob/main/docs/DFLASH2-SPECULATIVE-DECODING.md)) → AL(1..4) = 1.74, 2.21, 2.48, 2.63 **[derived]**; conditional acceptance falls 0.74 → 0.64 → 0.57 → 0.56.
- 4xH200, agentic, K=5: p1=0.81, p4=0.17, p5=0.07, AL(3)=2.69–2.74 ([substack](https://enterpriseaireport.substack.com/p/serving-glm-53-flash-in-production)) → AL(1..5) ≈ 1.81, ~2.37, 2.71, ~2.88, ~2.95 **[derived; p2/p3 split assumed]**.

**SGLang regime (native FP8, 4xGB300)**
- K=7, T=1.0, max reasoning: MT-Bench AL 3.71, HumanEval 4.70 ([Anbeeld](https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF)). Fitting a geometric p_i=c^i to MT-Bench gives c≈0.76 → AL(1..5) ≈ 1.76, 2.34, 2.78, 3.11, 3.36 **[derived; geometric shape is an assumption — real MTP decay is usually steeper at depth, so k=4–5 is likely overstated]**.

Confounds I cannot separate from the sources: K=7 vs 4/5; NVFP4/W4 target with BF16 head vs native FP8 (tonyd2wild notes the NVIDIA NVFP4 checkpoint can't even load the MTP head); thinking traces (very predictable) vs agentic/code prompts; engine implementation (KDA state handling and indexer reuse in multi-step MTP, see [InferenceX note](https://inferencex.semianalysis.com/model/glm-5-3)).

**Estimate for our setup (int4 target, prose, low temperature, vLLM-style chain MTP):**

| k | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| pessimistic (Spark-code curve, prose penalty on draft part: ~8% at k=1 to ~20% at k=5) | 1.68 | 2.08 | 2.27 | 2.35 | 2.37 |
| **central** (H200 agentic curve with the same prose penalty, nudged toward SGLang) | **1.75** | **2.22** | **2.50** | **2.62** | **2.68** |
| optimistic (SGLang MT-Bench geometric fit) | 1.77 | 2.34 | 2.78 | 3.11 | 3.36 |

The prose penalty is evidence-bounded, not a guess: chat-vs-code is 3.71/4.70 = 0.79 for MTP AL at K=7 ([Anbeeld](https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF)) and 23.4/29.4 = 0.80 for DFlash2 tok/s on 2xSpark ([forum](https://forums.developer.nvidia.com/t/glm-5-3-flash-dflash2-on-sglang-2x-dgx-spark-first-sglang-path-recipe-4-gb10-fixes-honest-numbers-concurrency-curve/381703)). The gap is smaller at small k because position 1 is easy on any text. The pessimistic row applies the penalty to Σp_i, not to AL.

Consistency checks: our measured 17.3→25.7 tok/s (1.48x) at k=3 is close to tonyd2wild's 14.3→21.8 (1.52x, but KV dtype also changed); with the cost model below it implies AL(3)≈2.6. Prose is lower than code for DFlash2 on Spark (23.4 vs 29.4 tok/s, [forum](https://forums.developer.nvidia.com/t/glm-5-3-flash-dflash2-on-sglang-2x-dgx-spark-first-sglang-path-recipe-4-gb10-fixes-honest-numbers-concurrency-curve/381703)) and MT-Bench is the lowest row in Anbeeld's table. **The ~2.8 committed tokens/step in our budget is the top of the vLLM regime at k=3 (agentic H200); for prose expect 2.3–2.6 at k=3, and at most ~2.7–2.8 at any k.** Highest-value local measurement: vLLM `spec_decode_num_accepted_tokens_per_pos` on our own quant at k=5, prose vs code, and an A/B against a higher-precision target to see whether the vLLM/SGLang gap is quantization or implementation.

## 3. Per-step cost model with expert-union growth

Notation: t = k+1 tokens verified per step; L=42 MoE layers; E=288, K=8; U(t) = distinct routed experts per layer touched by t tokens; b_e = bytes per routed expert; T1 = non-speculative per-token time.

Exact geometry from [config.json](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json): hidden 4096, moe_intermediate 2048, vocab 154880, 288 routed + 1 shared, top-8, 45 layers (first 3 dense), 1 MTP layer, untied lm_head, native FP8 with lm_head/embeddings in BF16. So:
- params per routed expert = 3·4096·2048 = 25.2M; all routed experts = 12096·25.2M = 304.5B **[derived]**
- b_e = 13.0 MB at int4/g128 (4.125 bit), 9.8 MB at 3.125 bit, 8.3 MB at 2.625 bit, 25.2 MB at FP8 **[derived]**
- lm_head = 154880·4096 = 634M params = 1.27 GB in BF16, i.e. 4.6 ms at 273 GB/s and ~8% of T1, **read again on every MTP draft pass** **[derived]**
- non-routed bytes read per token, B_dense ≈ 5.6 GB with int4 dense weights and BF16 lm_head, ≈ 9.5 GB with FP8 dense **[derived from 18B active]**

**Open input:** 304.5B params of int4 experts is ~157 GB, which does not fit in the Spark's 128 GB. The GPU-only 17.3 tok/s run must therefore use roughly ≤2.8-bit experts, or something else. I don't know which, so the floors below are given per b_e.

**Union.** Random-routing upper bound (MoESD Eq. 8, [arXiv 2505.19645](https://arxiv.org/html/2505.19645v4)): U_rand(t) = E·(1−(1−K/E)^t) = 288·(1−(280/288)^t) → t=4: 30.7, t=8: 58.0 **[derived]**. Our traces: U(4)=19–23, U(8)≈26 — routing affinity makes union growth ~1.5–2.2x slower than random. EVICT's Qwen3-30B tree curve is similar in shape (≈8 → ≈17 at 4 nodes → ≈25 at 8, [Fig. 1](https://arxiv.org/pdf/2605.00342)). By contrast, EcoSpec measured E=31.4 for DSv3.1 MTP at 4 verified tokens (top-8 of 256, near the random bound 30.0) ([EcoSpec](https://www.alphaxiv.org/abs/2607.12696.md)). E=31.4 is above the 30.0 random bound for 4 tokens, so EcoSpec probably counts a pre-pruning tree (3 steps × top-2) or includes the shared expert; don't read it as a chain number. EVICT's Fig. 1 header says "Total Experts = 64", but Qwen3-30B-A3B has 128 routed experts; the pages I read don't explain this. So Cascade's Mixtral overheads (8 experts, union saturates almost immediately) do **not** transfer quantitatively. Empirical fit for us: U(t) ≈ 8·t^0.70 for t≤4 (U(2)≈13, U(3)≈17), flattening to ~1.3 experts/token beyond t=4 (U(5)≈22.5, U(6)≈23.7). Caveat: our 8-token number came from windows of *committed* tokens; rejected draft tokens may route differently, so the real verify union could be larger. Re-measure U on actual verify batches.

**Step time (GPU-only, bandwidth bound):**
T(k) = T_dense(t) + L·U(t)·b_e/(η·BW) + k·T_mtp + T_sync,  tok/s = AL(k)/T(k)
Normalized to T1 with f = routed share of T1 and d = cost of one MTP pass/T1:
r(k) = T(k)/T1 ≈ (1−f) + f·U(k+1)/8 + k·d

d is not small. Each MTP pass is one MoE decoder layer (~2–3% of T1) **plus a full lm_head read** (8% of T1 in BF16, ~2% if int4), so d ≈ 0.04–0.10. Calibrating on our 1.48x at k=3 with AL(3) = 2.5–2.7 gives r(3) = 1.69–1.82, i.e. f = (r−1−3d)/1.625 ≈ 0.35–0.43 if d=0.04, or ≈ 0.24–0.32 if d=0.10. **Central: f≈0.35, d≈0.07** **[derived]**.

Predicted GPU-only with central AL: k=1: r=1.29, 1.36x; k=2: r=1.54, 1.44x; k=3: r=1.78, 1.41x (measured 1.48x); k=4: r=1.91, 1.37x; k=5: r=2.04, 1.32x **[derived]**. The optimum is flat at k=2–3. That matches the production write-up (K=3 best) and the vLLM GLM recipe's advice that larger K raises AL but cuts the acceptance rate ([vLLM GLM recipe](https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM.html)).

**Marginal rule** (EVICT Eq. 9–10 / Limits-of-Speculation framing): increase k iff p_{k+1}/AL(k) > ΔT/T(k).
- k=2→3: ΔT/T = (0.35·0.475 + 0.07)/1.54 = 0.153 vs p3/AL(2) ≈ 0.28/2.22 = 0.126. This is a near-tie, so k=2 and k=3 are equivalent.
- k=3→4: 0.076 vs 0.12/2.50 = 0.048 → stop.

Past t≈4 the union increment (0.35·0.19 = 0.065) is smaller than the draft-pass term (0.07). **Once union growth flattens, sequential draft passes, and the BF16 lm_head they re-read, dominate the marginal cost.** Two fixes follow: an int4 or vocabulary-trimmed draft lm_head (FR-Spec-style), and a one-forward block drafter (DFlash2).

**Remote CPU experts.** Routed time moves to the CPU: T_cpu(t) = L·U(t)·b_e/BW_cpu = 42·U·13.0 MB/190 GB/s → 23 ms (U=8), 60 ms (U=21), 74 ms (U=26) **[derived]**. There is also a per-step network term: 42 round trips per step. Activations are small (4 tokens·4096·2 B = 32 KB), so RTT dominates, ~0.1–0.3 ms/layer → 4–12 ms/step **[assumed RTT]**. Speculation amortizes this term over AL tokens.
- Non-spec check: the measured 74.6 ms (13.4 tok/s) ≈ ~38 ms GPU non-routed ((1−f)·T1) + 23 ms CPU + ~10–14 ms network. It fits.
- k=3: ≈ 38 + 60 + 10 + 3·4 ≈ 120 ms → 2.5/0.12 ≈ 21 tok/s predicted vs 15–19.5 measured. The multi-token CPU expert kernels are probably below roofline.
- On the CPU path, the union multiplier U(t)/8 (2.6x at k=3) applies to a larger share of the step than on GPU, so speculation helps less.

**Budget check (bandwidth floors at 100% efficiency).**
B_step(k) = B_dense + 42·U(k+1)·b_e + k·B_draft, with B_dense ≈ 5.6 GB and B_draft ≈ 1.5 GB per MTP pass (BF16 lm_head + MTP layer) or ≈ 0.6 GB with an int4 draft head **[derived]**.

| expert quant (b_e) | T1 floor (U=8) | k=3 floor, GPU-only, BF16 draft head | k=3 floor, int4 draft head | AL needed for 80 tok/s at floor / at η=0.55 |
|---|---|---|---|---|
| int4 13.0 MB (CPU path; doesn't fit GPU-only) | 36.6 ms | 79 ms | 69 ms | 5.5 / 10 |
| 3.1-bit 9.8 MB | 32.6 ms | 69 ms | 59 ms | 4.7 / 8.6 |
| 2.6-bit 8.3 MB | 30.7 ms | 64 ms | 54 ms | 4.3 / 7.9 |
| **GPU+CPU parallel split**, int4 (GPU: dense + draft + share x of the union; CPU: the rest) | — | 47 ms | 41 ms | 3.3 / 6.0 |

**[derived; AL column uses the int4-draft-head floor]** Our measured T1 = 57.8 ms implies η ≈ 0.53–0.56 against the sub-4-bit floors. Conclusions:
- The 35 ms cycle at AL 2.8 is below every floor in the table, including the parallel-split design.
- 80 tok/s needs AL ≳ 5–6 at realistic efficiency, even with the GPU+CPU split. That is DFlash2-class drafting on code or math only (Anbeeld HumanEval 5.32, GSM8K 5.78; tonyd2wild code ≈6.2 **[derived from 74.1% of 7]**). MTP on prose (AL ≈ 2.5) can't get there.
- 125 tok/s (22.4 ms) is out of reach on this hardware at any AL we have evidence for.
- **The verdict depends on the GPU-only expert bit-width, which the caller must confirm.** The verdict does not change across the plausible range.

DFlash2 for us, GPU-only: U(8)≈26. Draft cost is one pass of the 1B BF16 drafter (2 GB), plus the lm_head (1.27 GB) if the drafter reuses the target's head, ≈ 0.21·T1. That gives r ≈ 0.65 + 0.35·3.25 + 0.21 ≈ 2.0. Predicted throughput **[derived]**:
- AL 3.7 (thinking holdout) → ~32 tok/s
- AL 4.03 (MT-Bench) → ~35 tok/s
- AL 5.3 (code) → ~46 tok/s

Quantizing the drafter to FP8 saves ~0.06·T1. The best comparison point is MTP k=3 at ~24–26 tok/s.

## 4. Tree vs chain at batch 1

Trees spread branches across different experts. EVICT shows EAGLE-3 trees verify ~32 nodes, and pruning them to ~8 cuts experts by 32.5% and verify latency by 26.6% at batch 1 ([EVICT](https://arxiv.org/pdf/2605.00342)). EcoSpec's tree pruning gains only 1.10→1.15x over a DSv3.1 MTP chain ([EcoSpec](https://www.alphaxiv.org/abs/2607.12696.md)). With U(4)≈21 of 288 we are far from saturation, so every extra branch costs new experts. **Use a chain.** If we ever use trees, apply an EVICT-style profiled C(k) table with prefix truncation: it is lossless, cheap, and fits CUDA graphs.

## 5. Constraining target routing during verification

- OEA ([arXiv 2511.02237](https://www.alphaxiv.org/abs/2511.02237.md)) and Lynx ([arXiv 2411.08982](https://arxiv.org/pdf/2411.08982)) cut unique experts by 26–49% with "no significant"/"negligible" accuracy loss. Both results are at **batch 16 on Qwen3/Mixtral** across *different* requests. Lynx also reports that primary expert choices matter more than secondary ones, and that decode tolerates changes better than prefill.
- Batch-1 analog for us: each of the t verify tokens keeps its top-k0 (e.g. 5) and adds lower-ranked experts only if they are already in the union. If the union scales roughly with k0, U drops ~21→~13 at k=3, cutting routed cost ~35–40%. Routed bytes are ~52% of a k=3 GPU step (0.35·2.625/1.78), so that is ≈20% of the step **[derived, speculative]**.
- Caveat: this changes the verifier's logits, so the output is no longer the original model's distribution. Speculation becomes a lossy approximation of a different model. No source measures this on GLM-5.x or inside speculative verification specifically. The only offload-side data point is KTransformers' 6-of-8 experts mode (11.26→13.69 tok/s, lossy, no quality data given). A milder variant: apply reduced routing only to the deepest draft positions (p4≈0.15), since they are rarely committed. It is still lossy when they are.
- Lossless alternatives that attack the same cost: EcoSpec (cost-aware drafting) and EVICT/Cascade (adaptive k per step). Expect ~1.05–1.2x at batch 1.

## 6. What transfers to us

1. **Adaptive k (Cascade/EVICT utility)**: transfers directly. Profile T(k) for k=0..5 once on GPU-only and CPU-expert paths, then choose k per step from the MTP head's confidence. Cascade's main benefit, avoiding slowdowns (1.5x worst case → 5%), matters most on the CPU path where r(k) is steepest.
2. **Block drafter (DFlash2)**: highest expected gain. One forward per block avoids the k·d term that dominates marginal cost once union growth flattens. Community 2xSpark results show 2.15x over MTP-4 on code ([tonyd2wild](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/blob/main/docs/DFLASH2-SPECULATIVE-DECODING.md)), all self-reported. Prose gains are smaller (MT-Bench 1.73x vs MTP's 1.36x on GB300).
3. **Draft-predicted routing**: the "prefetch" papers (SP-MoE, Specter, DraftExpert, ProMoE/AdapMoE-style) assume experts *move* over PCIe. Ours *compute* on the CPU, so prefetching has no direct analog. What does carry over: draft-side signals predict target routing well (80–93% top-K accuracy in [EcoSpec](https://www.alphaxiv.org/abs/2607.12696.md), ~88% top-1 in [SP-MoE](https://arxiv.org/html/2510.10302v2)). We can use this to (a) start the CPU's DRAM→L2/L3 streaming, and the network send, of predicted experts before the layer's activations arrive, and (b) **split each layer's union between GPU-resident hot experts and CPU-resident cold ones and compute both halves in parallel**. That split is the only path toward the ~41–47 ms floor in section 3. Spark's 128 GB unified memory can hold roughly 55–60% of the ~157 GB of int4 experts, chosen by trace frequency. DraftExpert's verify-cost reduction from 2.35x to 1.25x with truncation plus prefetch ([DraftExpert](https://www.alphaxiv.org/abs/2607.24434.md)) shows what union-aware scheduling can buy.
4. **Network amortization**: speculation spreads the 42 per-step RTTs over AL tokens. This is the argument for speculation with remote experts that holds up even when union growth is bad. RDMA/RoCE or kernel-bypass would shrink the fixed term further.
5. **Lossy routing constraints (OEA-style)**: possible ~20% step saving on GPU, more on the CPU path, but they break exactness. Treat as a research knob gated on a quality eval, not a default.
6. **Does not transfer**: MoESD's "MoE benefits more than dense" (holds at medium batch, where all experts are already loaded) and DeepSeek-V3's 1.8x (large-batch EP serving).

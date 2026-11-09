# Precision allocation for GLM-5.3-Flash experts on one GB10 + MaxQ CPU

Research date 2026-11-08. Tags: **[M]** measured (by us or by a cited source on the named model), **[I]** inferred by arithmetic from cited numbers, **[S]** speculative. Shapes come from `local_profiling.md`: 12,096 routed experts x 25.17M params = 304.5B routed params, which is 38.1 GB per bpw. 100 GB holds a uniform 2.63 bpw. Each token routes to 336 experts (8 x 42 layers) = 8.46B params.

## 0. Units: making the literature comparable to our numbers

- **PPL tables convert to dNLL in nats** via ln(PPL_q / PPL_ref), which is the same unit as our eval. The conversions below are [I] from [M] PPLs. They use wiki.test at ctx 512, which is harsher than our eval in absolute terms.
  - GLM-4.6 (Q8_0 3.4471) ([ubergarm/GLM-4.6-GGUF](https://huggingface.co/ubergarm/GLM-4.6-GGUF)): IQ4_KS 4.65 bpw **+0.024**, smol-IQ4_KSS 4.09 **+0.041**, IQ3_KS 3.57 **+0.055**, IQ2_KL 3.07 **+0.185**, smol-IQ2_KS 2.36 **+0.426**.
  - Kimi-K2-Thinking (Q4_X, which equals Q8_0 because the model is INT4-QAT, PPL 2.0818) ([ubergarm/Kimi-K2-Thinking-GGUF](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF)): smol-IQ4_KSS 4.06 **+0.025**, IQ3_K 3.85 **+0.030**, smol-IQ3_KS 3.25 **+0.072**, IQ2_KL 2.92 **+0.131**, smol-IQ2_KL 2.76 **+0.165**, smol-IQ2_KS 2.26 **+0.344**.
  - DeepSeek-R1-0528 (Q8_0 3.2130) ([ik_llama #477](https://github.com/ikawrakow/ik_llama.cpp/discussions/477)): IQ3_K_R4 (301 GiB) **+0.019**, IQ2_K_R4 (220 GiB) **+0.088**.
  - **Sanity check:** GLM-4.6 IQ3_KS (+0.055) and IQ2_KL (+0.185) land almost exactly on our IQ3_XXS (+0.054) and IQ2_XXS (+0.182). Our eval behaves like the community's on a sibling model.
- **turboderp's GLM-5.3-Flash chart is mean KLD**, the same unit as our control's 0.042 ([card](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3), [chart image](https://cdn-uploads.huggingface.co/production/uploads/6383dc174c48969dcf1b4fce/Fu_UF2RUwF8R_e052_ekr.png)). It uses a self-generated in-domain trace of 28,154 input + 45,470 output tokens, and the x-axis is bpw excluding embeddings and head. All values below are [M] as read from the image:
  - EXL3 2.05 **0.193**, 3.05 **0.068**, 4.05 **0.035**
  - EXL3-SC 2.10 **0.157**, 2.60 **0.093**, 3.10 **0.057**, 4.10 **0.029**
  - GGUF UD-IQ1_M (~2.4) **0.232**, UD-IQ3_XXS (~3.0) **0.126**, UD-IQ4_XS (~3.9) **0.062**
  - AWQ int4 g128 (~4.4) **0.057**, NVFP4 (~4.8) **0.045**
  - The bf16 self-noise floor is mean **0.0138** (median 0.0012) ([spread chart](https://cdn-uploads.huggingface.co/production/uploads/6383dc174c48969dcf1b4fce/TEz4SM1cZC3po4-j3llfo.png)). Our control's 0.042 is only about 3x that floor, which limits how much "better than EXL3 3.5" can even be resolved.
  - The card does not document what "self-calibrated" (SC) means.
- **KLD to dNLL bridge [I]:** our control gives 0.0148 / 0.042 = 0.35. Our IQ3_XXS dNLL against turboderp's UD-IQ3_XXS KLD gives 0.054 / 0.126 = 0.43. So dNLL is roughly 0.35-0.43 x turboderp KLD. Caveat: turboderp's points also quantize attention, while our control keeps non-expert weights native.
- Accuracy-delta papers (MC-MoE, QuantMoE-Bench, QAD) do not convert to either unit, so treat them as directional only.

## 1. Bit allocation and mixed precision for MoE experts

**Academic methods (all on models of 8x7B or smaller unless stated):**
- **MC-MoE** (ICLR'25) sets per-expert bits with a linear program over activation frequency and importance, then applies GPTQ ([arXiv 2410.06270](https://arxiv.org/pdf/2410.06270)) [M on Mixtral 8x7B].
  - At 2.54 bpw: −3.8% average accuracy versus −22.2% for a size-matched naive split (25% of layers at 4-bit, the rest at 2-bit). Scores were 67.5 / 49.07 against 71.29 for 16-bit.
  - This is the largest fixed-size gain in the literature, and it is at ~2.5 bpw, where uniform quantization collapses.
- **MxMoE** (ICML'25) allocates by linear-block sensitivity and frequency, and generates mixed-precision GroupGEMM kernels. Results: Wikitext-2 PPL 2.4 lower than GPTQ at 2.25 bpw, and a 29.4% speedup over uniform at equal accuracy (W5A5) ([arXiv 2505.05799](https://arxiv.org/abs/2505.05799)) [M, abstract-level].
- **QuantMoE-Bench** ([arXiv 2406.08155](https://arxiv.org/abs/2406.08155)) [M on DeepSeek-MoE-16B]:
  - Raising frequent experts from 2-bit to 4-bit helps, but 4 to 8 bits adds almost nothing.
  - The fine-grained mixed allocation scores 65.35 against 64.30 for GPTQ.
- **MoQE:** 2-bit expert-only weights; claimed to beat a same-data dense model, with −79.6% size and 1.24x on A100 ([arXiv 2310.02410](https://arxiv.org/abs/2310.02410)) [M, abstract].
- **QMoE:** 0.8 bits/param on SwitchTransformer-c2048 (1.6T to under 160 GB), with "minor" loss and under 5% runtime overhead ([arXiv 2310.16795](https://arxiv.org/abs/2310.16795)). Switch-c2048 is very over-parameterized, so this does not transfer to fine-grained 288-expert models.
- **EAQuant:** +1.15 to +13.81 average accuracy over DuQuant at W4A4 through W2A4 ([arXiv 2506.13329](https://arxiv.org/pdf/2506.13329)).
- **MoEQuant:** +10 HumanEval points on DeepSeek-MoE-16B at 4-bit ([arXiv 2505.03804](https://arxiv.org/abs/2505.03804)).
- **AlphaQ:** calibration-free, spectral-tail-based allocation; "near full precision" at 3.5-bit average on Qwen1.5-MoE, no numbers in the abstract ([arXiv 2606.04980](https://arxiv.org/abs/2606.04980)).
- **HOBBIT:** a low-precision copy is served on a cache miss; 9.93x decode speedup versus offloading baselines on Mixtral / Phi-MoE ([arXiv 2411.01433](https://arxiv.org/html/2411.01433v2)). This is a speed result, not a quality-at-fixed-size result.
- **Not retrieved this session:** D²-MoE delta decompression, NAEE, and Slim-MoE numbers. They are not cited here.

**Community practice on large MoE** [M]:
- ubergarm keeps attention, the shared expert, and the dense first layers at q8_0, and quantizes only routed experts. He also gives routed `ffn_down` one tier more than gate/up ([Kimi-K2-Thinking](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF), [GLM-4.6](https://huggingface.co/ubergarm/GLM-4.6-GGUF)).
- **The value of the down-tier bump on Kimi [I]:** upgrading only down (smol-IQ2_KL 2.76 bpw to IQ2_KL 2.92 bpw, +6% bits) cuts dNLL from 0.165 to 0.131, which closes 21% of the gap.
- unsloth Dynamic 2.0 varies type per layer and optimizes mean KLD, but publishes no per-tensor rule ([docs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)).

**Pruning** [M, vendor-reported]:
- REAP, GLM-4.6 HumanEval: 96.3 base, 96.3 at −25%, 95.7 at −30%, 95.1 at −40% ([cerebras card](https://huggingface.co/cerebras/GLM-4.6-REAP-218B-A32B), [arXiv 2510.13999](https://arxiv.org/abs/2510.13999)).
- REAP also reports ≤2% accuracy loss on code at 50% pruning for Qwen3-Coder-480B and Kimi-K2.
- There is no prose/NLL evidence, and Hermes is prose.

**Verdict:** measured gains from allocation at fixed size are large at 2-2.5 bpw (MC-MoE: −3.8% vs −22%), but they shrink toward 3.5-4 bpw (QuantMoE-Bench: 4 to 8 bits gives almost nothing). No *allocation* method has been measured with KLD on a 200B+ MoE; the only KLD curves at that scale are for uniform quants (turboderp, §0).

## 2. Can ~3 bpw PTQ experts reach EXL3-3.5-like quality (KLD ~0.04)?

**Short answer: no credible evidence at a uniform ~3.0 bpw. The best measured point on this exact model needs ~3.55 bpw.**

- **Interpolating turboderp's GLM-5.3-Flash curves** [I], log-linear in KLD versus bpw:
  - Plain EXL3: slope −0.68 ln-KLD per bpw. KLD 0.042 is reached at **~3.75 bpw**, and 3.5 bpw gives ~0.050.
  - EXL3-SC: slope −0.69 per bpw. KLD 0.042 is reached at **~3.55 bpw**, and 3.5 bpw gives ~0.044. That matches our control's 0.042, which cross-validates both evals.
- **At ~3 bpw the best PTQ is 1.4-1.6x the target** (SC 3.10 0.057, plain 3.05 0.068), and SC 2.60 is 2.2x (0.093) [M].
- **Trellis versus ggml blocks on this model:** UD-IQ3_XXS 0.126 vs EXL3 3.05 0.068 at near-equal bpw, so ggml blocks are 1.85x worse [M]. Our own gap (IQ3_XXS +0.054 vs EXL3 3.5 +0.0148) is 3.6x at a 0.44 bpw spread. That is a different comparison, but it is *consistent with* turboderp's 1.85x at equal bpw plus his ~0.8 bpw trellis advantage.
  - EXL3-SC 3.1 roughly equals UD-IQ4_XS at 3.9 bpw (0.057 vs 0.062). **Trellis buys ~0.8 bpw over the best ggml codebook at equal KLD** [M].
- **Dense trellis literature:** QTIP 2-bit Llama-2-70B Wikitext2 3.87 vs FP16 3.12 (QuIP# 4.16, AQLM 3.83) ([QTIP](https://arxiv.org/pdf/2406.11235v2), via secondary summary). The AQLM paper reports Mixtral only at 3-4 bits ([arXiv 2401.06118](https://arxiv.org/pdf/2401.06118)). I found no academic-paper QTIP, AQLM, or QuIP# results on DeepSeek, Kimi, GLM, or Qwen3-235B. EXL3 is a QTIP-family trellis, so turboderp's chart is the de facto trellis-on-300B-MoE measurement.
- **Our QAT pilot plateaued at 15-58% recovery.** NVIDIA's QAD report finds STE-style QAT *degrades* RL-trained models, while KL distillation from the frozen BF16 teacher on its outputs recovers them. Example: Nemotron 3 Nano AA-LCR scored BF16 35.9 / PTQ 31.3 / QAT 24.8 / QAD 34.3 ([arXiv 2601.20088](https://arxiv.org/html/2601.20088v1)) [M].
  - Our pilot's objective was per-expert output matching, not end-to-end logit KL. This mismatch is a plausible cause of the plateau [I].
  - QAD budgets were 0.3-6B tokens on models of 49B or less [M]. No QAD result exists at 300B MoE scale.

## 3. FP4 for attention and experts

- **NVIDIA keeps attention in BF16 on hybrid models.** In QAD, Nemotron Nano 9B V2 keeps its attention layers plus the first and last two layers in BF16. Nemotron 3 Nano 30B-A3B keeps its 6 attention layers and their preceding Mamba layers in BF16, with KV in FP8. Only pure transformers get full NVFP4 ([arXiv 2601.20088](https://arxiv.org/html/2601.20088v1)) [M].
  - GLM-5.3-Flash (34 KDA + 11 MLA) is the hybrid case. **No published result shows QAT/QAD recovering NVFP4 attention on a hybrid.** NVIDIA sidestepped it.
- **DeepSeek-R1-0528-NVFP4-v2** "additionally quantizes the wo module in attention", implying v1 kept it higher. FP4 is within 1 point of FP8 on MMLU-Pro 84/85 and GPQA 80/81 ([card](https://huggingface.co/nvidia/DeepSeek-R1-0528-NVFP4-v2)) [M]. The QAD paper's Appendix C says PTQ is near-lossless at R1-671B scale.
- **Attention bit-width on DeepSeek-R1-0528**, with experts fixed at q4_0 ([#477](https://github.com/ikawrakow/ik_llama.cpp/discussions/477)) [M PPL, I nats]:
  - attn+shexp q4_0 3.2895, iq5_ks 3.2541, q8_0 3.2485.
  - 4-bit attention costs **+0.0125 nats** over 8-bit, while 5-bit costs only **+0.0017**.
  - That cost comes from ~1% of total bits. Our measured NVFP4 attention cost of +0.030 on GLM-5.3-Flash is the same phenomenon, but larger.
- **Kimi-K2-Thinking's vendor QAT** applied INT4 weight-only to the **MoE components only** ([model card](https://huggingface.co/moonshotai/Kimi-K2-Thinking)) [M].
- **PTQ tricks recover only small fractions of the FP4 gap** [M on dense models, W4A4]:
  - Four Over Six, Llama-3-70B Wikitext: BF16 2.86, RTN 4.00, RTN+4/6 3.83, AWQ 3.86, AWQ+4/6 3.71, i.e. **~13-15% of the gap** ([arXiv 2512.02010](https://arxiv.org/html/2512.02010v3)).
  - MR-GPTQ brings MXFP4 to within 1-2% of NVFP4. Rotations *hurt* NVFP4 under RTN ([arXiv 2509.23202](https://arxiv.org/abs/2509.23202)).
  - These are W4A4 results; our case is weight-only.
- Quartet ([arXiv 2505.14669](https://arxiv.org/abs/2505.14669)) is FP4 *pretraining* from scratch, not post-hoc recovery. It is no evidence either way for recovering NVFP4 attention on an existing 310B model.
- **Verdict [I]:** keep attention and other non-expert weights at FP8 (measured ~free for us). Short QAT/QAD recovery of NVFP4 attention is unproven at this scale, on a hybrid, and would cost 0.3-6B+ teacher tokens.

## 4. Kernel efficiency on sm_120/121, batch-1 decode

- **GB10 ceiling:** GPU STREAM-triad measures 200-213 GB/s against a 273 GB/s spec ([NVIDIA forum](https://forums.developer.nvidia.com/t/gb10-really-does-hit-1-pflop-nvfp4-2-4-sparse-measured-with-an-open-source-tool-to-reproduce-it/373618)) [M].
- **llama.cpp MXFP4 on GB10:** gpt-oss-120b tg32 is 60.57 tok/s (b6922, kernel 6.17) and 45.87 tok/s (b6816) ([#16578](https://github.com/ggml-org/llama.cpp/discussions/16578)) [M].
  - At ~3.2-3.5 GB/token (144 experts x 24.9M at 4.25 bpw ≈ 1.9 GB, plus attention, router, and head), that is **~195-210 GB/s**, roughly 95-100% of STREAM and ~75% of spec [I].
- **EXL3 trellis on GB10** ([vllm-exl3](https://github.com/vcruz305/vllm-exl3/releases)) [M, self-reported]:
  - GLM-5.3-Flash MoE runs at 287.8 µs/layer, and "batched GEMV at 99.2% of bandwidth floor".
  - About 29 ms of the 40.6 ms step is outside the MoE.
  - At ~2-2.5 bpw, 8 experts + shared expert is ~55-70 MB per layer (assuming the shared expert is quantized), which implies **~190-240 GB/s**, i.e. at STREAM [I].
  - Trellis decode therefore does not appear compute-bound on GB10 at batch 1.
- **EXL3 on RTX 5090:** a 27B model at 4 bpw runs at ~122 tok/s, about 1.65 TB/s or ~92% of 1.79 TB/s [I, secondary [source](https://www.contextstudios.ai/local-ai/miaai-lab--qwen3.8-27b-exl3-rtx-5090-one-click)].
- **NVFP4 MoE on SM120** ([vLLM forum](https://discuss.vllm.ai/t/sm120-rtx-pro-6000-nvfp4-moe-performance-report-qwen3-5-397b/2536)) [M]:
  - Only Marlin W4A16 works, giving 50.5 tok/s on Qwen3.5-397B with 4x RTX PRO 6000 TP4.
  - FlashInfer/CUTLASS NVFP4 grouped GEMM fails all 80 TMA tactics (CUTLASS #3096), and TRT-LLM has an SM121 FP4 smem overflow (#11368).
  - The Marlin figure is not a bandwidth measure [I]. Qwen3.5-397B-A17B at ~4.5 bpw is ~9.6 GB/token; split over TP4 that is ~2.4 GB per GPU, and x 50.5 tok/s ≈ 120 GB/s per GPU, about 7% of 1.79 TB/s. That path is latency- and communication-bound at batch 1, so it says nothing about single-GPU Marlin efficiency. No Machete number on SM120/121 was found.
  - b12x ([repo](https://github.com/yatesdr/b12x)) lists NVFP4, W4A8, W4A16, MXFP8, and block-FP8 MoE paths, but publishes no GB/s numbers.
- **CPU (MaxQ, local)** [M]: int8 expert GEMV reaches 206-217 GB/s, while naive int4 reaches 118-128 GB/s because it is unpack-bound.
- **Gap:** IQ2/IQ3 MMVQ efficiency on sm_121 is unmeasured; no source found.

## 5. Implications: allocation scenarios

**Fixed facts:**
- Per-expert size: 2.6 bpw 8.2 MB, 3.1 bpw 9.75 MB, 3.5 bpw 11.0 MB, 4.1 bpw 12.9 MB, Q4_K/NVFP4 4.5 bpw 14.2 MB.
- Full expert traffic per token is 3.70 GB at 3.5 bpw and 4.77 GB at 4.5 bpw.
- Non-expert FP8 is **~9 GB/token** (`target_budget.md`: ~9B non-expert params read per step), which dominates. **Expert bit allocation is a memory-fit and quality lever, not a speed lever.** Speed comes from FP8/lower non-expert precision and from the speculation acceptance (tau).

**Quality model [I]:** loss ≈ Σ_tier (routed-mass share x full-model loss of that tier). This extends our measured per-layer additivity (IQ3_XXS ~0.0012 nats per layer) to per-expert mass-weighting, which is unverified.
- Starting from our Q4_K stack (+0.0069), the share of routed mass allowed at a lower tier while staying ≤ +0.0148:
  - IQ3_XXS: **≤17%**
  - EXL3 3.05: ~0.029 est. (0.054 / 1.85), **≤36%**
  - EXL3-SC 3.1: ~0.020-0.025, **≤45-60%**
  - SC 2.6: ~0.033-0.040, **≤24-30%**
- No GLM routing-skew data exists. DeepSeek-V3 effective experts per layer range 80-180 of 256 ([2607.23099](https://arxiv.org/html/2607.23099v3)). Under domain shift, static-frequency hit rates fall to 21-45% versus ~70% for LRU ([2608.12103](https://www.alphaxiv.org/abs/2608.12103)). **Coverage numbers below are [S] until we profile Hermes traffic.**

| Scenario | Spark experts | CPU experts | Est. quality | GPU / CPU expert bytes per token |
|---|---|---|---|---|
| **S1: uniform EXL3-SC 2.6, all GPU** | 12,096 x 8.2 MB = **99 GB** [I] | none | KLD **0.093 [M]** = 2.2x control; dNLL ~0.033-0.040 [I]. **Fails** | 2.75 GB / 0 [I] |
| **S1b: REAP −25% + EXL3 3.5, all GPU** | 9,072 x 11.0 = **100 GB** [I] | none | EXL3 part ≈ control; pruning cost on prose **unknown [S]** (GLM-4.6 HumanEval only −0) | 3.70 / 0 |
| **S2: hot EXL3 3.5 (216 per layer, 75%) on GPU + cold Q4_K+imatrix on CPU** | 9,072 x 11.0 = **100 GB** [I] | 3,024 x 14.2 = **43 GB** | ≤ +0.0148: both tiers measured (+0.0148 / +0.0069) [M per tier, I mix] | at 90% mass coverage [S]: **3.33 / 0.48 GB** (~34 expert reads, ~3.6-4.1 ms CPU) |
| **S3: three-tier, importance-weighted** | top 120/layer EXL3-SC 4.1 (65 GB) + next 85/layer SC 3.1 (35 GB) = **100 GB** | 83/layer Q4_K = **50 GB** | assuming mass 70/20/10 [S]: 0.7x0.011 + 0.2x0.022 + 0.1x0.0069 ≈ **+0.013** [I] | **~3.1 / 0.4 GB** [S] |

**Notes on the scenarios:**
- **S2 is the defensible default.** Each tier's quality is measured on this model, and EXL3 3.5 is already our serving checkpoint.
- **S3 beats S2 only by ~10% in loss** under the same mass assumptions (≈0.013 vs 0.014) [I]. This matches the literature: allocation pays at 2-2.5 bpw and is marginal near 3.5. Its value is mainly robustness to coverage errors, since the hottest mass sits at ~4.1 bpw.
- **Cheap refinement for S2/S3:** put `ffn_down` one tier above gate/up. That closed 21% of the gap for +6% bits on Kimi [I].
- **The binding risk is not bits but the CPU tier's round trips.** At 90% coverage, 1 − 0.9^8 = 57% of layers still need a CPU trip, ~24 per token. At today's ~0.5 ms 1 GbE TCP ping-pong that is ~12 ms/token (`local_profiling.md`), roughly the whole expert-read time on the GPU.
- **Required measurements before choosing:**
  1. Per-layer routed-mass coverage on Hermes traces.
  2. Per-tier dNLL for EXL3-SC 3.1/4.1 on experts only (the turboderp points include attention).
  3. Mixed hot/cold dNLL at the chosen split, to validate the mass-weighted model.

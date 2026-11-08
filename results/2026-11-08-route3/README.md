# Route 3 results (2026-11-08): own BF16-derived checkpoints on the Spark-native vLLM

Everything here was measured on the Spark (GB10), Spark-only: no CPU experts, no MTP. These are all [M].

## Full model: all routed experts IQ2_XXS, everything else BF16 (`full_iq2xxs`)

**Build.**

- `src/build_mixed_checkpoint.py` with plan `IQ2_XXS` for MoE layers 3-44.
- Quantized from BF16 on the workstation CPU with llama.cpp's quantizer (`src/ggml_quant.py`) and the Q4_K imatrix.
- 41 min, 91 GB, loads as 91.35 GiB.

**Serve.** vLLM `--quantization modelopt_mixed`, B12X backends, FP8 KV, breakable CUDA graphs (piecewise + full), `ops/route3/serve_eval.sh`.

| measure | result | target / control |
|---|---|---|
| sample outputs | coherent and correct (Paris; a correct Fibonacci function) | |
| decode (256 tokens, batch 1) | **10.9 tok/s** | goal 80, stretch 125 |
| cold 45K TTFT | **36.9 s** | 10.8 s |
| cached 45K+32 | **0.389 s** | 0.2 s |
| dNLL vs Q8_0 BF16 proxy, balanced eval, 41,943 tokens | **+0.183 +/- 0.004** | control +0.0148 |
| KLD (mean / p99) | **0.238** / 3.05 | control 0.042 |
| same top token | **85.0%** | control 94.0% |

**Where decode bytes go** (checkpoint tensor sizes; per-token reads at batch 1):

| part | stored | read per token |
|---|---|---|
| routed experts (IQ2_XXS) | 78.5 GB | 2.18 GB (8/288 active) |
| attention + linear attention (BF16) | 12.35 GB | 12.35 GB |
| shared experts (BF16) | 2.11 GB | 2.11 GB |
| dense MLP layers 0-2 (BF16) | 1.61 GB | 1.61 GB |
| lm_head (BF16) | 1.27 GB | 1.27 GB |
| **total** | | **about 19.5 GB** |

**Roofline.** At about 273 GB/s, decode is capped near 14 tok/s; the measured 10.9 tok/s is about 78% of that. **The BF16 non-expert weights are 89% of decode bytes.** At batch 1 the experts are not the problem.

**Implications.**

- **Non-expert precision is the first decode lever.**
  - FP8 non-expert: about 10.9 GB per token, roofline about 25 tok/s.
  - NVFP4 non-expert: about 7 GB per token, roofline about 38 tok/s.
- **80 tok/s needs more than that.** It requires 3.4 GB or less per token on the Spark alone. That cannot be reached by bit-width alone, so it also needs:
  - multi-token amortization (MTP speculative decoding: dense and attention weights are read once per verified block);
  - workstation-CPU bandwidth for part of the bytes (the heterogeneous split).
- **Quality.** All-2-bit experts are far outside the control bar, as expected (QAT pilot: 3-bit costs about 0.0012 nats per layer; 2-bit much more). The quality-relevant mixes need higher-bit experts. Those do not fit in Spark memory, hence remote experts (runtime v2).
- **Cold prefill** at 36.9 s is 3.4x the target. This is a compute question (prefill is not bandwidth-bound) for the profile step.

## Format and runtime checks (6-layer truncated checkpoints; output text meaningless by construction)

See `docs/RUNTIME_V2_VLLM_HOOKS.md` for the full tables.

- **Mixed formats.** Q8_0, IQ2_XS and IQ2_XXS on consecutive layers load and serve.
- **Quantizer.** Our ctypes Q8_0 output is byte-identical to llama-quantize's GGUF, checked on gate, up and down of experts 0, 1 and 287 of layer 3.
- **`topk_ids = -1` masking is NOT exact.** Max logprob deviation 9.9 vs the zero-weight mask; do not use it.
- **`route_expert_map` masking matches the zero-weight mask.** 510/511 same top-1, mean abs logprob diff 0.005; repeat runs are bit-identical.
- **Compact resident-only expert weights** (to free VRAM) are blocked by b12x tuner route geometry (216 vs 288). A b12x patch is needed.

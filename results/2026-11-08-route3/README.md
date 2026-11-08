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

**Caveat.** Do not compare this directly with the earlier best of 17.0 tok/s. That was llama.cpp with Q8_0 attention and MTP speculative decoding. This run has BF16 attention and no MTP: the build used `--max-layers 45`, which drops the MTP layer (index 45), and serving sets `num_nextn_predict_layers: 0`. The next full build keeps layer 45.

**What this checkpoint is.** It is the fits-entirely-on-Spark extreme. It proves the pipeline end to end: coherent output, and our own quantizer, byte-exact with llama-quantize. It is not a quality candidate: about 12x the control's dNLL.

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

**Reframing (measured).** At batch 1, routed experts are only 11% of decode bytes. Earlier levers (QAT pilot, per-layer sensitivity, placement) all targeted that 11%. The decode equation is roughly:

    time per token ~ (non-expert bytes + active expert bytes) / bandwidth / (tokens accepted per MTP block)

So the heterogeneous CPU split mainly earns its place on the **quality** axis, giving room for higher-bit experts. Speed comes from non-expert precision and MTP.

## Format and runtime checks (6-layer truncated checkpoints; output text meaningless by construction)

See `docs/RUNTIME_V2_VLLM_HOOKS.md` for the full tables.

- **Mixed formats.** Q8_0, IQ2_XS and IQ2_XXS on consecutive layers load and serve.
- **Quantizer.** Our ctypes Q8_0 output is byte-identical to llama-quantize's GGUF, checked on gate, up and down of experts 0, 1 and 287 of layer 3.
- **`topk_ids = -1` masking is NOT exact.** Max logprob deviation 9.9 vs the zero-weight mask; do not use it.
- **`route_expert_map` masking matches the zero-weight mask.** 510/511 same top-1, mean abs logprob diff 0.005; repeat runs are bit-identical.
- **Compact resident-only expert weights** (to free VRAM) are blocked by b12x tuner route geometry (216 vs 288). A b12x patch is needed.

## FP8 non-expert weights + MTP layer kept (`full_iq2xxs_fp8`) [M]

**Build.** Same IQ2_XXS experts. Non-expert weights use the NVIDIA reference recipe (`--fixed fp8`):

- MXFP8 1x32 for KDA q/k/v/o, dense MLP and lm_head;
- FP8_PB_WO 128x128 for MLA projections and shared experts.

Layer 45 (MTP) is kept, with IQ2_XXS experts. Size: 86 GB.

| measure | BF16 non-expert | **FP8 non-expert** |
|---|---|---|
| decode (256 tokens, batch 1, no MTP) | 10.9 tok/s | **17.3 tok/s** |
| cold 45K TTFT | 36.9 s | 39.8 s |
| cached 45K+32 | 0.389 s | 0.452 s |
| dNLL / KLD / same top-1 | +0.183 / 0.238 / 85.0% | +0.182 / 0.239 / 84.9% |

**Reading.**

- FP8 on the non-expert weights costs no measurable quality.
- Decode is 1.58x faster, matching the bytes-per-token model: about 19.5 GB per token before, about 11 GB after.
- **MTP speculative decoding did not start.** b12x refuses to split the hybrid-manager KV block of the BLHNC layout into 256-token kernel blocks: "block stride 33206528 != page 2568192". This is the next fix.

## Runtime v2 prototype: routed experts on the workstation CPU as a precision upgrade [M]

**Setup.** vLLM hook `ops/route3/remote_hook`:

- For the planned experts of each layer, the routes are sent to `runtime/expert_server.c` on the workstation, over the 10 GbE link with busy-poll and heartbeat.
- The server holds all 42 MoE layers as int4 RTN g128, 171 GB in RAM.
- Those routes are masked on the GPU with weight 0 (exact), and the CPU partial sums are added per token.
- The GPU keeps its IQ2_XXS copy of every expert.
- Wire format verified against a BF16 reference (`runtime/remote_wire_test.py`).

**Quality.** Same balanced eval, 41,943 tokens against the Q8_0 BF16 proxy, eager mode:

| CPU share of routes | plan | dNLL | KLD | same top-1 |
|---|---|---|---|---|
| 0% | none | +0.182 +/- 0.004 | 0.239 | 84.9% |
| 29% | hottest 64 experts per layer (from the M1 routing traces) | +0.123 +/- 0.004 | 0.175 | 87.4% |
| 100% | all experts | **+0.028 +/- 0.002** | **0.057** | **92.8%** |
| | control (EXL3 3.5 bpw, 2x Max-Q) | +0.0148 | 0.042 | 94.0% |

**Reading.**

- The remote-expert path is correct end to end, and it moves quality from 12x to about 1.9x the control's dNLL.
- **The gain is roughly proportional to the CPU's share of routes.** Hot-64 is slightly better than linear. Routing is flat (the hottest 64 of 288 experts cover only 29-46% of routes), so a small hot set cannot carry the quality.
- **The remaining gap is the CPU format.** int4 RTN (+0.028) is far behind the Q4_K + imatrix experts measured in llama.cpp (+0.0045 on the same eval).
  - A better CPU quantizer (imatrix-weighted / MSE scales, or Q4_K blocks) is the obvious next quality lever.
  - That is CPU-only work, with no training.
- **Prefill.** During the all-remote prefill eval, client wait (911 s) about equals server compute (907 s): transport is hidden, and the CPU is the bottleneck for prefill.

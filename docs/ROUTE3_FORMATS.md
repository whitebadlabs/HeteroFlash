# Route 3, step 1: formats the Spark-native stack can serve (2026-11-08)

Image: `eugr/spark-vllm-b12x` (vLLM 0.1.dev21830, glm5next model, b12x MoE kernels for sm_12x).

## Routed-expert formats in the b12x MoE backend

From `fused_moe/b12x.py`:

| expert format | bpw | activation modes |
|---|---|---|
| NVFP4 (ModelOpt) | ~4.5 | BF16 (w4a16), NVFP4, or MXFP8 (w4a8) |
| MXFP4 | ~4.25 | BF16 or MXFP8 |
| Q8_0 | 8.5 | BF16 |
| IQ2_XS / IQ2_XXS | 2.31 / 2.06 | BF16 |
| EXL3 trellis | variable; the native manifest supports per-expert-pair rates | BF16 |

## Mixed precision per layer

- vLLM ModelOpt supports `quant_algo: MIXED_PRECISION`, with a `quantized_layers` map from module path to `{quant_algo, ...}`.
- Reference schema, read from the config only: `nvidia/GLM-5.3-Flash-FP4-Local`.

| modules | format |
|---|---|
| routed experts (`...layers.N.mlp.experts`) | NVFP4, `weight_scale_encoding csf_32x2_v1`, group 16 |
| MTP layer experts | W4A16_NVFP4 |
| shared experts and MLA projections | FP8_PB_WO, 128x128 blocks |
| KDA attention projections, dense MLP, lm_head | MXFP8, 1x32 |

- The IQ2 path lives in `modelopt_iq2_xs.py`, which accepts IQ2_XS, IQ2_XXS and Q8_0 block codecs in llama.cpp block layouts.

## Implication

We can produce our own checkpoint from BF16 with a format chosen per layer: 2 / ~4.25-4.5 / 8.5 bpw experts, plus FP8/MXFP8 fixed weights. That is the placement-aware precision knob the co-design experiment needs, served natively with no version transplant.

## Still to verify

- Whether `MIXED_PRECISION` dispatch accepts IQ2_XS/Q8_0 per layer alongside NVFP4.
- The exact tensor layouts, including the NVFP4 CSF scale encoding.

## Step 2 (2026-11-08): our own checkpoint serves on the Spark-native stack [M]

**Checkpoint.** `src/build_mixed_checkpoint.py` builds a ModelOpt `MIXED_PRECISION` checkpoint from the official BF16 weights:

- non-expert tensors are copied unquantized from BF16;
- each planned layer's routed experts are llama.cpp blocks taken from a GGUF quantized from the same BF16 (`packing: ggml`, shape `(rows, in/block, block_bytes)`).

**Image.** `local/hf-spark-native:1` is `eugr/spark-vllm-b12x` plus one patch, `ops/spark-native/patch_iq2_limit.py`. The patch lets the IQ2/Q8_0 MoE path accept GLM's `swiglu_limit` 10.0, which B12xExperts already passes to b12x.

**Working serve flags:**

- `--quantization modelopt_mixed --block-size 256 --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16`
- `--attention-backend B12X --linear-backend b12x --moe-backend b12x`
- env: `VLLM_KV_CACHE_LAYOUT=BLHNC B12X_LUT_E4M3_SMEM=0` (trellis LUT staging overflows GB10 shared memory)

**Truncated test.** 4 layers: 3 dense + layer 3 with Q8_0 experts; 13 GB checkpoint; eager mode.

| measure | result |
|---|---|
| boot to healthy | 3 min |
| one decode step | 0.068 s wall |
| 128-token decode | 2.09 s |
| 2,000-token prefill | 0.46 s |

The output text is meaningless, as expected with the remaining layers missing.

**Next:**

- an NVFP4 expert writer (own quantizer) and IQ2_XS layers;
- a full 45-layer build. All-IQ2 experts (~88 GB) fit entirely on the Spark for a whole-pipeline speed test.
- quality-relevant mixes then need the remote-expert overlap runtime.

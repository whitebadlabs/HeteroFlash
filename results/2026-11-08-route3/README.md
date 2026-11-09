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
- **Prefill got slower: cold 45K +8%, cached +16%.** This is not explained yet. Candidate cause, from the serve log: the KDA input projection now runs as two GEMMs (q/k/v MXFP8 plus b/f_a BF16) instead of one fused BF16 GEMM, and MXFP8 activations are quantized per call (`VLLM_GLM53_KDA_MXFP8_ACTIVATION_MODE=auto`). It needs a prefill profile before the cold-TTFT work.
- **MTP speculative decoding did not start.** b12x refuses to split the hybrid-manager KV block of the BLHNC layout into 256-token kernel blocks: "block stride 33206528 != page 2568192". This is the next fix.

## Runtime v2 prototype: routed experts on the workstation CPU as a precision upgrade [M]

**Setup.** vLLM hook `ops/route3/remote_hook`:

- For the planned experts of each layer, the routes are sent to `runtime/expert_server.c` on the workstation, over the 10 GbE link with busy-poll and heartbeat.
- The server holds all 42 MoE layers as int4 RTN g128, 171 GB in RAM.
- Those routes are masked on the GPU with weight 0 (exact), and the CPU partial sums are added per token.
- The GPU keeps its IQ2_XXS copy of every expert.
- Wire format verified against a BF16 reference (`runtime/remote_wire_test.py`).

**Quality.** Same balanced eval, 41,943 tokens against the Q8_0 BF16 proxy, eager mode:

| CPU share of routes | plan | dNLL | KLD | same top-1 | decode speed |
|---|---|---|---|---|---|
| 0% | none | +0.182 +/- 0.004 | 0.239 | 84.9% | 8.85 tok/s eager (17.3 with CUDA graphs) |
| 29% | hottest 64 experts per layer (from the M1 routing traces) | +0.123 +/- 0.004 | 0.175 | 87.4% | 7.07 tok/s eager |
| 100% | all experts | **+0.028 +/- 0.002** | **0.057** | **92.8%** | 6.85 tok/s eager |
| | control (EXL3 3.5 bpw, 2x Max-Q) | +0.0148 | 0.042 | 94.0% | 122-132 tok/s |

**Decode speed.** 256 tokens, batch 1, eager mode in all three rows, because the hook needs Python inside each MoE layer. This is not yet overlapped: the hook syncs GPU to CPU in every layer before sending.

- All experts on the CPU add about 33 ms per token (0.78 ms per layer).
- The hot-64 plan, with 29% of routes, already adds 28 ms.
- So most of the cost is fixed per-layer overhead (the GPU-to-host sync and the Python/numpy request packing), not CPU expert compute. The server alone measured 0.41-0.67 ms per layer in runtime-v1.
- Projection, modeled: with CUDA graphs (breakable segments around the hook) and no overlap, the extra 33 ms on top of 58 ms per token gives about 11 tok/s. Hiding the CPU leg behind attention and shared-expert work, and removing the per-layer Python, is the runtime-v2 engineering.

**All-on-CPU still fails the non-inferiority bar.** The control is the bar: +0.0148.

**Reading.**

- The remote-expert path is correct end to end, and it moves quality from 12x to about 1.9x the control's dNLL.
- **The gain is roughly proportional to the CPU's share of routes.** Hot-64 is slightly better than linear. Routing is flat (the hottest 64 of 288 experts cover only 29-46% of routes), so a small hot set cannot carry the quality.
- **The remaining gap is most likely the CPU weight format.**
  - The server also quantizes activations to u8 per token. Earlier checks, though, put server-vs-same-int4-dequant at 1.6% output error and int4-RTN-vs-BF16 at 22%, so weight rounding dominates.
  - int4 RTN (+0.028) is far behind the Q4_K + imatrix experts measured in llama.cpp (+0.0045 on the same eval).
  - A better CPU quantizer (imatrix-weighted / MSE scales, or Q4_K blocks) is the obvious next quality lever.
  - That is CPU-only work, with no training.
- **Prefill.** During the all-remote prefill eval, client wait (911 s) about equals server compute (907 s): transport is hidden, and the CPU is the bottleneck for prefill.

## Runtime v2 mailbox: graph-capturable CPU experts (hecate-style) [M]

`ops/route3/remote_hook/hf_mailbox.cu` replaces the Python socket path:

- A pack kernel writes each layer's routes and hidden rows into pinned, host-mapped mailboxes, with a release flag and device-side sequence counters.
- A C++ relay thread on the Spark forwards each request to `expert_server`.
- A wait_add kernel, queued after the GPU's routed and shared experts, spins on the done flag and adds the CPU partial sums.
- **Unit test** (`mailbox_test.py`): exact against the Python path, and exact under CUDA graph replay.

| decode, 256 tokens, batch 1 (FP8 non-expert, IQ2_XXS GPU experts, int4 RTN CPU experts) | eager Python hook | **mailbox, CUDA graphs** |
|---|---|---|
| no CPU experts | 8.85 tok/s | 17.3 tok/s |
| hottest 64 per layer on CPU (29% of routes) | 7.07 | **16.29** |
| all experts on CPU | 6.85 | **12.89** |

**Reading.**

- All-on-CPU now costs about 20 ms per token, or 0.47 ms per layer. That is about the CPU expert compute itself; runtime-v1 measured 0.4-0.6 ms per layer, so the fixed overhead is gone.
- The output text matches the eager all-remote run.
- This config's quality is the all-remote row above: dNLL +0.028.

## Which 4-bit format should the CPU hold? (`src/cpu_format_check.py`) [M]

Layer 32: routed-output relative error against BF16 on 2,048 real captured tokens (5 domains), gate-weighted top-8:

| format | rel. error |
|---|---|
| int4 RTN g128 (current expert_server) | 0.189 |
| same, plus the server's u8 activation quantization | 0.190 |
| int4 g128 with MSE-searched scales | 0.149 |
| **Q4_K with imatrix (llama.cpp)** | **0.103** |
| IQ2_XXS (Spark GPU copy) | 0.508 |
| Q8_0 | 0.009 |

**Reading.**

- Activation quantization is negligible; the weight format is the gap.
- Q4_K + imatrix roughly halves the error of int4 RTN. End to end in llama.cpp, Q4_K experts scored dNLL +0.0045, inside the control bar.
- Next step: give `expert_server` a Q4_K path, ideally by calling llama.cpp's own CPU `vec_dot_q4_K_q8_K` kernels on the existing Q4_K-imatrix GGUF rows. Then rerun the all-on-CPU quality measurement.
- Layer 43 was not run; its run was stopped to keep the CPU free for timing.

## MTP speculative decoding works after the draft-backend fix [M]

Config: `--speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "attention_backend": "B12X"}'`, otherwise the FP8 config above (BF16 SSM state, prefix caching on, CUDA graphs). Root cause: `docs/RUNTIME_V2_VLLM_HOOKS.md`.

| measure | FP8, no MTP | **FP8 + MTP-3** |
|---|---|---|
| decode (256 tokens, batch 1, essay prompt) | 17.3 tok/s | **25.7 tok/s** (1.48x) |
| cold 45K TTFT | 39.8 s | 39.0 s |
| cached 45K+32 | 0.452 s | 0.417 s |

**Reading.**

- The output stays correct.
- No acceptance-rate breakdown is logged yet.
- Next: sweep `num_speculative_tokens` (1/2/3/5), and re-measure once the experts are better than 2-bit, which should raise draft acceptance.
- The fork bug (draft attention backend unconditionally reset to None) is confirmed by this run, so the three-line patch is a valid upstream fix.

## First configuration inside the quality bar: all routed experts on the CPU in Q4_K + imatrix [M]

**Setup.**

- The Spark GPU holds the FP8 non-expert weights and an IQ2_XXS copy of every expert, unused here.
- The workstation CPU serves **every** routed expert from the Q4_K + imatrix GGUF rows (`runtime/expert_server_gguf.c`, llama.cpp `vec_dot` kernels, 175 GB mmapped and prefaulted in 82 s).
- The handoff is the graph-capturable mailbox.

| config | dNLL | KLD (mean / p99) | same top-1 | decode (CUDA graphs, no MTP) |
|---|---|---|---|---|
| CPU int4 RTN | +0.028 +/- 0.002 | 0.057 / 0.82 | 92.8% | 12.89 tok/s |
| **CPU Q4_K + imatrix** | **+0.0069 +/- 0.0014** | **0.030** / 0.42 | **94.8%** | **13.37 tok/s** |
| control: EXL3 3.5 bpw on 2x Max-Q | +0.0148 +/- 0.0017 | 0.042 | 94.0% | 122-132 tok/s |

**Reading.**

- This is the first Spark + CPU configuration that is non-inferior to the 2x Max-Q control on all three quality metrics, on the same balanced eval and the same reference.
- Speed is 13.4 tok/s without MTP: the CPU's expert compute is now the decode bottleneck.
- Cold prefill with all experts on the CPU was not measured (the quality eval's 2048-token chunks took about 60 s each, which is CPU-bound). It will be far from the 10.8 s target, so prefill needs the GPU experts, or a hybrid split.
- Next: MTP on top (running), then the placement split. Hot experts get a better GPU format so the GPU shares the expert work, and the CPU keeps the rest at Q4_K.

## MTP on the quality-passing config [M]

All routed experts on the CPU (Q4_K + imatrix), mailbox, CUDA graphs. Quality as above; MTP is lossless.

| MTP draft tokens | decode |
|---|---|
| 0 | 13.37 tok/s |
| 1 | 17.22 tok/s |
| **3** | **19.49 tok/s** |

**Best measured configuration inside the quality bar: 19.5 tok/s.**

### Per-domain quality of the quality-passing config (CPU Q4_K experts) vs the control [M]

| domain | ours dNLL | control dNLL | ours KLD | control KLD | ours same-top | control same-top |
|---|---|---|---|---|---|---|
| writing_chat | +0.0026 | +0.0037 | 0.0153 | 0.0166 | 0.953 | 0.949 |
| knowledge_qa | +0.0227 | +0.0346 | 0.0442 | 0.0652 | 0.949 | 0.937 |
| math | -0.0030 | +0.0073 | 0.0246 | 0.0312 | 0.957 | 0.951 |
| code | +0.0175 | +0.0201 | 0.0300 | 0.0392 | 0.972 | 0.972 |
| multilingual | -0.0002 | +0.0122 | 0.0311 | 0.0495 | 0.942 | 0.932 |
| instruction_following | +0.0033 | +0.0054 | 0.0370 | 0.0525 | 0.924 | 0.911 |
| long_context | +0.0051 | +0.0216 | 0.0271 | 0.0421 | 0.937 | 0.922 |
| overall | +0.0069 | +0.0148 | 0.0300 | 0.0423 | 0.948 | 0.940 |

**Reading.** Better than or equal to the control in every domain, on every metric.

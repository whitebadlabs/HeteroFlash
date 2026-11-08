# Runtime v2: where remote CPU experts plug into the Spark-native vLLM (code map, 2026-11-08)

This note comes from reading the source of `local/hf-spark-native:1`. Claims about kernel behaviour are from source, not run, unless marked otherwise.

- `V` = `vllm/`
- `B` = `flashinfer/experimental/b12x/moe`

## MoE layer call chain (glm5next, TP=1)

1. **Decoder layer** (`V/models/glm5next/nvidia/model.py:648`): calls `Glm5NextMoE.forward` (`:316`).
   - The gate is `GateLinear` (router logits).
   - Then `self.experts` is called. It is `FusedMoEFactory`, which builds a `MoERunner`.
   - Settings: shared expert `Glm5NextMLP`, `e_score_correction_bias`, renormalize on, `routed_scaling_factor=2.5`.
2. **Router.** `GroupedTopKRouter` calls the fused `ops.grouped_topk`. That kernel applies a sigmoid, adds the bias (used for selection only), renormalizes the top 8 and multiplies by 2.5.
   - Output: `topk_ids` int32 and `topk_weights` fp32, both of shape `[T,8]`.
3. **Runner.** `MoERunner.forward` calls the custom op `torch.ops.vllm.moe_forward_shared`, which runs `_forward_impl`.
   - The **shared expert** is launched on an auxiliary CUDA stream when T ≤ 256 tokens, so it overlaps with the routed experts.
   - Routed experts: `select_experts` → `ModelOptIQ2XSMoEMethod.apply` → `B12xExperts.apply` (`b12x.py:1098`).
4. **Combine.** `shared + routed` is added **outside** the op, in `MoERunner.forward`. This is the place to add `+ remote_out`.

## Hooks available for remote experts

- **Mask remote slots.** Setting `topk_ids = -1` makes the b12x route-pack and IQ2 sum skip that slot (source: `route_pack.py:46`, `kernel.py:7860`). The direct-decode path still needs a GPU test.
- **Hold only resident experts on the GPU.** The kernel supports `weight_E < route_E` with `route_expert_map` (`-1` = not resident), but the vLLM wrapper rejects expert maps (`b12x.py:833,1118`). Enabling it takes three patches:
  - plan with `route_num_experts=288`;
  - pass `route_expert_map` to `bind`;
  - create the IQ2 params with `n_local` rows and remap in `weight_loader` (`modelopt_iq2_xs.py:81-167`). Today every param is allocated with 288 rows.
- **Placement of the network call.** It must sit outside graph capture. Options:
  - `@eager_break_during_capture` (`V/compilation/breakable_cudagraph.py:64`, `VLLM_USE_BREAKABLE_CUDAGRAPH=1`), which is already used by the sparse indexer and is a no-op under FULL graphs;
  - `--enforce-eager`;
  - a GPU-side wait on a host-mapped flag.
- **What the hook receives.** `hidden_states` is bf16 `[T,4096]` after `post_attention_layernorm`. It is the same tensor as the shared-expert input.

## Plan

1. Custom router wrapper:
   - split `topk_ids` into resident and remote;
   - set remote slots to -1 for the GPU;
   - send `(x rows, remote ids, weights)` to the runtime_v1 expert server.
   - The server's int8 / Q4 CPU kernels compute the remote experts while the GPU runs the resident and shared experts.
2. In a breakable-graph segment after the op: receive the result and add it before the combine.
3. Expert-map patch, so remote experts free their VRAM.

## Measured: is `topk_ids = -1` an exact skip? No [M, 2026-11-08]

**Setup.**

- 6-layer checkpoint: layer 3 Q8_0, layer 4 IQ2_XS, layer 5 IQ2_XXS.
- Prompt logprobs (top 5) of a fixed 512-token prompt, eager mode.
- The "remote" set is experts with id % 4 == 0.
- Hook: `HF_REMOTE_MASK` in `ops/spark-exl3/trunc_hook`.

| comparison | positions | same top-1 | max abs logprob diff |
|---|---|---|---|
| neg (ids = -1) vs zero (weights = 0) | 511 | 119 | 9.91 |
| none vs zero (true effect of dropping the remote experts) | 511 | 353 | 4.71 |
| none vs neg | 511 | 107 | 9.30 |

An exact skip would make "neg" equal "zero". Instead "neg" is further from both than they are from each other, so -1 ids are mishandled on this build's prefill path (Q8_0 and/or IQ2 kernels).

**Decision.**

- Do not use -1 masking.
- The correct interim mask is weight 0. It is exact, but the remote experts are still resident and computed on the GPU.
- The memory- and compute-saving path is the `route_expert_map` patch, which needs its own exactness test: map vs zero-weight A/B as above.

## Measured: the `route_expert_map` path does skip remote experts [M, 2026-11-08]

**Setup.** Hook `HF_ROUTE_MAP=4`:

- b12x is planned with `route_num_experts=288`;
- every `bind` gets `route_expert_map[e] = e`, or -1 when e % 4 == 0;
- all weights stay resident.

Same 6-layer checkpoint and prompt as above.

| comparison | same top-1 | mean abs logprob diff | max abs |
|---|---|---|---|
| none vs none (repeat run) | 511/511 | 0 | 0 |
| map vs zero-weight | 510/511 | 0.0053 | 0.100 |
| map vs none (true effect of dropping the experts) | 354/511 | | 4.71 |

**Reading.**

- Runs are bit-deterministic.
- The map path is not bit-identical to zero-weight masking. Its deviation is about 1/1000 of the masked effect, which fits a different accumulation order over fewer routes. It is far from the -1 failure (9.9 max).
- The map path is the one to build on.

**Next steps:**

1. Allocate only resident experts: `n_local` rows plus a global-to-local `weight_loader` remap, so remote experts free VRAM.
2. Send the remote routes to the CPU expert server, and add the result outside the `moe_forward_shared` op, in a breakable-graph segment.

## Measured: compact resident-only weights are blocked in b12x tuning [M, 2026-11-08]

**Setup.** `HF_ROUTE_MAP=4 HF_RESIDENT=1`:

- the IQ2/Q8_0 params are allocated with 216 rows, so loading drops 1/4 of the experts;
- the map goes from global id to local slot.

**Result.**

- Load succeeds: 14.19 GiB for the 6-layer model.
- Engine init fails in b12x `moe.decode` candidate preparation, in `route_pack.py:545`: "W4A16 live routes exceed or differ from the prepared geometry: requested=(16384, 64, 216), prepared=(16384, 64, 288)".
- **Cause.** The candidates are prepared without a route map. Without a map, the route count falls back to `weight_E` (216), while the prewarmed route-pack launches were compiled with `route_E` (288).
- **Fix.** Give the tuner's synthetic routes the map, or compile its route-pack launches with `weight_E` when no map is present. This is a b12x patch.
- **Impact.** It blocks only the VRAM saving. The overlap runtime can proceed with all weights resident plus the identity map, which is exact within noise.

## MTP on this stack: root cause of the KV split failure (2026-11-08)

**Symptom.** With `--speculative-config '{"method": "mtp", ...}'`, KV init fails in `v1/kv_cache_interface.py:~388`: "BLHNC ... block stride != page ... cannot be split into 19 kernel blocks of 256 tokens". Neither LBNHC, FP32 SSM state, nor disabling prefix caching helps.

**Cause.** Read from source; the fix is still to be confirmed by a run.

1. `v1/worker/gpu/spec_decode/eagle/utils.py:14-31` (`_make_eagle_draft_vllm_config`) always overwrites the draft's `attention_config.backend` with `speculative_config.attention_backend`, which is None by default. The later "only when set" guard therefore never takes effect.
2. With the backend at None, the MTP layer's MLA attention falls back to `FLASHINFER_MLA_SPARSE_SM120` instead of the B12X GLM5Next backend. That backend uses 64/256-token kernel blocks and an uncustomized fp8_ds_mla page (656 B per token, against 528 for the target).
3. The mixed page sizes disable the GLM-5.3 per-layer tensor packing (`v1/core/kv_cache_utils.py:1325`, `_glm5_next_tensor_layout`). The generic BLHNC packing then puts every layer inside one block: 11 x 2568192 + 3190784 + 11 x 160512 = 33206528 bytes of block stride, which cannot be split into 256-token kernel blocks.

**Fix.** Add `"attention_backend": "B12X"` to the speculative config, as NVIDIA's published GLM-5.3-Flash Spark recipe does.

**Upstream-worthy patch.** In the fork, only override the draft backend when one is set:

```diff
-    vllm_config = replace(vllm_config, attention_config=replace(vllm_config.attention_config, backend=speculative_config.attention_backend))
+    if speculative_config.attention_backend is not None:
+        vllm_config = replace(vllm_config, attention_config=replace(vllm_config.attention_config, backend=speculative_config.attention_backend))
```

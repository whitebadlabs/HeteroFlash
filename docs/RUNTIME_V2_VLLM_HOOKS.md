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

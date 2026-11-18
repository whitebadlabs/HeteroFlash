# Expert co-selection study: results (2026-11-18)

Question: is expert co-selection (experts picked by the same token) strong and stable enough that a
correlation-aware placement cuts cross-GPU all-to-all sends compared with vLLM's DefaultEplbPolicy (load only)?
All numbers are **measured on held-out prompts in a CPU replay** of real routing traces. No real all-to-all was run.

## Answer so far
- **Experts have partners.** Each expert's 7 most co-selected partners are picked with it 2.8x (GLM-5.3-Flash) to
  4.1x (Qwen3.6-35B-A3B) more often than chance, on holdout. Partners fitted on one task type hold on the others
  (GLM 2.6x, Qwen 4.2x).
- **Layer N predicts part of layer N+1, but per task.** recall@8 is 0.33 (GLM) and 0.40 (Qwen) within a task type, vs
  0.15 / 0.17 for "the 8 most-used experts". Fitted on another task type it falls to 0.14 / 0.16, at or below that
  baseline. Cross-layer prediction needs a per-task (or per-request) statistic.
- **Placement, balance kept (per-task fit, cap 1.0x), 8 GPUs, same-load Default as reference.** GLM sends
  4.698 -> 4.126 (-12.2%), imbalance 1.12 -> 1.16. Qwen 4.737 -> 3.961 (-16.4%), imbalance 1.05 -> 1.09. The cut is
  similar at 16 and 32 GPUs (GLM -14%/-15%, Qwen -20%/-20% vs the per-task Default), but imbalance grows
  more there (32 GPUs: GLM 1.52 -> 1.74, Qwen 1.45 -> 1.54).
- **Placement, mixed fit.** Cuts more (GLM -17%, Qwen -26% at 8 GPUs), but holdout imbalance rises well past
  Default (GLM 1.25 -> 1.77, Qwen 1.23 -> 1.70). The calib-load cap does not survive the shift to holdout prompts.
  This version is not usable as is. Next experiment: add a small redundant-expert (copy) budget to both policies.
- **Oracle check.** Fitting both placements on the holdout tokens themselves gives -16% (GLM) / -26% (Qwen) at
  8 GPUs with balance 1.00. The headroom is real when the statistic is current.

Note on fig3: the orange bars (per-task fit) are drawn next to the mixed-load Default. The fair per-task reference
is "DefaultEplbPolicy, per-task load" in the tables below. It has lower imbalance (GLM 1.12, Qwen 1.05 at 8 GPUs).

Plots: results/plots/fig1_next_layer_recall.png, fig2_coselection_task_matrix.png, fig3_sim_sends_imbalance.png.
Related work (Patterns behind Chaos, ISCA 26; MegaFlux): see notes in the study PROGRESS.md.

## Numbers (scripts/summarize.py)
## GLM (288 experts, top-8, 42 MoE layers, task types: code, instruction_following, knowledge_qa, long_context, math, multilingual, writing_chat)

Tokens used (calib fit / holdout test): code 12000/3419 (13 holdout requests); instruction_following 12000/3999 (15 holdout requests); knowledge_qa 12000/4041 (12 holdout requests); long_context 12000/30000 (2 holdout requests); math 9025/2457 (14 holdout requests); multilingual 12000/3998 (13 holdout requests); writing_chat 12000/5703 (14 holdout requests)

- Load skew (mixed tasks, mean over layers): busiest expert 4.93x the mean; top 10% of experts get 22.4% of picks; CV 0.59
- Within-layer co-selection, top-7 partners per expert, measured on holdout: P(partner | expert) 0.207 vs P(partner) 0.096; lift same task 2.83, partners from another task 2.62, mixed->mixed 3.62
  - lift per layer (same task, mean over tasks): min 2.20 (layer 18), max 4.25 (layer 4), first 3.29, last 3.89
- Layer N -> N+1 recall@8: same task 0.329, other task 0.137, mixed 0.292; popularity baseline 0.151; random 0.028
  - mixed fit by phase: prefill 0.291 (30000 tok), decode 0.277 (15984 tok)
  - per layer (same task): min 0.281, max 0.372
- Per-task recall@8 (fit row -> test column):

| fit \ test | code | instruction_following | knowledge_qa | long_context | math | multilingual | writing_chat |
|---|---|---|---|---|---|---|---|
| code | 0.362 | 0.119 | 0.128 | 0.055 | 0.165 | 0.089 | 0.094 |
| instruction_following | 0.133 | 0.293 | 0.173 | 0.099 | 0.220 | 0.176 | 0.257 |
| knowledge_qa | 0.133 | 0.163 | 0.337 | 0.063 | 0.286 | 0.134 | 0.152 |
| long_context | 0.071 | 0.096 | 0.083 | 0.301 | 0.086 | 0.109 | 0.111 |
| math | 0.123 | 0.140 | 0.184 | 0.054 | 0.396 | 0.102 | 0.120 |
| multilingual | 0.093 | 0.169 | 0.129 | 0.075 | 0.149 | 0.329 | 0.136 |
| writing_chat | 0.140 | 0.278 | 0.176 | 0.119 | 0.213 | 0.179 | 0.288 |

Simulation, holdout replay, mean over task types. sends = cross-GPU sends per token per layer; imb = max/mean GPU load over the whole holdout set; imb_b = same per 2048-token batch.

| GPUs | placement | sends | vs Default | distinct GPUs | imb | imb_b |
|---|---|---|---|---|---|---|
| 8 | random equal-size | 4.626 | -0.6% | 5.288 | 1.327 | 1.350 |
| 8 | DefaultEplbPolicy, mixed load | 4.656 | +0.0% | 5.320 | 1.254 | 1.282 |
| 8 | DefaultEplbPolicy, per-task load | 4.698 | +0.9% | 5.369 | 1.119 | 1.160 |
| 8 | correlated, mixed fit, cap 1.0x | 3.857 | -17.2% | 4.407 | 1.765 | 1.797 |
| 8 | correlated, mixed fit, cap 1.1x | 3.618 | -22.3% | 4.134 | 2.010 | 2.045 |
| 8 | correlated, per-task fit, cap 1.0x | 4.126 | -11.4% | 4.714 | 1.157 | 1.211 |
| 8 | oracle (fit on the test tokens, mixed set only): Default -> correlated | 4.697 -> 3.953 | -15.9% | | 1.002 -> 1.002 | |
| 16 | random equal-size | 6.111 | -0.4% | 6.518 | 1.623 | 1.668 |
| 16 | DefaultEplbPolicy, mixed load | 6.136 | +0.0% | 6.545 | 1.500 | 1.560 |
| 16 | DefaultEplbPolicy, per-task load | 6.219 | +1.4% | 6.634 | 1.227 | 1.317 |
| 16 | correlated, mixed fit, cap 1.0x | 5.045 | -17.8% | 5.381 | 2.280 | 2.326 |
| 16 | correlated, mixed fit, cap 1.1x | 4.833 | -21.2% | 5.154 | 2.553 | 2.617 |
| 16 | correlated, per-task fit, cap 1.0x | 5.341 | -13.0% | 5.697 | 1.304 | 1.393 |
| 16 | oracle (fit on the test tokens, mixed set only): Default -> correlated | 6.218 -> 5.173 | -16.8% | | 1.010 -> 1.009 | |
| 32 | random equal-size | 7.033 | -0.5% | 7.260 | 2.163 | 2.256 |
| 32 | DefaultEplbPolicy, mixed load | 7.071 | +0.0% | 7.299 | 1.946 | 2.071 |
| 32 | DefaultEplbPolicy, per-task load | 7.147 | +1.1% | 7.378 | 1.520 | 1.676 |
| 32 | correlated, mixed fit, cap 1.0x | 5.984 | -15.4% | 6.178 | 2.877 | 2.989 |
| 32 | correlated, mixed fit, cap 1.1x | 5.834 | -17.5% | 6.023 | 3.123 | 3.253 |
| 32 | correlated, per-task fit, cap 1.0x | 6.093 | -13.8% | 6.290 | 1.737 | 1.907 |
| 32 | oracle (fit on the test tokens, mixed set only): Default -> correlated | 7.121 -> 6.054 | -15.0% | | 1.088 -> 1.087 | |

Correlated search: swaps per layer max 263, mean 186 (cap 5000; 0 layers hit the cap)

Analytic distinct GPUs for a random placement: 8: 5.290, 16: 6.515, 32: 7.257

## Qwen (256 experts, top-8, 40 MoE layers, task types: chat, code, math)

Tokens used (calib fit / holdout test): chat 20000/20000 (102 holdout requests); code 20000/20000 (89 holdout requests); math 20000/20000 (104 holdout requests)

- Load skew (mixed tasks, mean over layers): busiest expert 5.55x the mean; top 10% of experts get 23.9% of picks; CV 0.67
- Within-layer co-selection, top-7 partners per expert, measured on holdout: P(partner | expert) 0.268 vs P(partner) 0.102; lift same task 4.08, partners from another task 4.23, mixed->mixed 4.62
  - lift per layer (same task, mean over tasks): min 2.60 (layer 25), max 10.33 (layer 0), first 10.33, last 6.23
- Layer N -> N+1 recall@8: same task 0.403, other task 0.162, mixed 0.399; popularity baseline 0.171; random 0.031
  - mixed fit by phase: prefill 0.378 (20000 tok), decode 0.401 (20000 tok)
  - per layer (same task): min 0.352, max 0.483
- Per-task recall@8 (fit row -> test column):

| fit \ test | chat | code | math |
|---|---|---|---|
| chat | 0.398 | 0.240 | 0.267 |
| code | 0.137 | 0.350 | 0.095 |
| math | 0.165 | 0.071 | 0.462 |

Simulation, holdout replay, mean over task types. sends = cross-GPU sends per token per layer; imb = max/mean GPU load over the whole holdout set; imb_b = same per 2048-token batch.

| GPUs | placement | sends | vs Default | distinct GPUs | imb | imb_b |
|---|---|---|---|---|---|---|
| 8 | random equal-size | 4.642 | -0.4% | 5.305 | 1.330 | 1.359 |
| 8 | DefaultEplbPolicy, mixed load | 4.662 | +0.0% | 5.328 | 1.231 | 1.267 |
| 8 | DefaultEplbPolicy, per-task load | 4.737 | +1.6% | 5.414 | 1.053 | 1.130 |
| 8 | correlated, mixed fit, cap 1.0x | 3.433 | -26.4% | 3.922 | 1.701 | 1.743 |
| 8 | correlated, mixed fit, cap 1.1x | 3.136 | -32.7% | 3.584 | 1.986 | 2.008 |
| 8 | correlated, per-task fit, cap 1.0x | 3.961 | -15.0% | 4.527 | 1.085 | 1.209 |
| 8 | oracle (fit on the test tokens, mixed set only): Default -> correlated | 4.667 -> 3.463 | -25.8% | | 1.003 -> 1.003 | |
| 16 | random equal-size | 6.118 | -1.0% | 6.525 | 1.626 | 1.682 |
| 16 | DefaultEplbPolicy, mixed load | 6.178 | +0.0% | 6.589 | 1.446 | 1.511 |
| 16 | DefaultEplbPolicy, per-task load | 6.270 | +1.5% | 6.688 | 1.137 | 1.265 |
| 16 | correlated, mixed fit, cap 1.0x | 4.588 | -25.7% | 4.893 | 1.967 | 2.067 |
| 16 | correlated, mixed fit, cap 1.1x | 4.289 | -30.6% | 4.575 | 2.234 | 2.347 |
| 16 | correlated, per-task fit, cap 1.0x | 5.009 | -18.9% | 5.342 | 1.195 | 1.416 |
| 16 | oracle (fit on the test tokens, mixed set only): Default -> correlated | 6.175 -> 4.623 | -25.1% | | 1.011 -> 1.011 | |
| 32 | random equal-size | 7.042 | -1.0% | 7.269 | 2.191 | 2.287 |
| 32 | DefaultEplbPolicy, mixed load | 7.114 | +0.0% | 7.344 | 1.830 | 1.958 |
| 32 | DefaultEplbPolicy, per-task load | 7.192 | +1.1% | 7.423 | 1.448 | 1.656 |
| 32 | correlated, mixed fit, cap 1.0x | 5.556 | -21.9% | 5.735 | 2.383 | 2.583 |
| 32 | correlated, mixed fit, cap 1.1x | 5.337 | -25.0% | 5.510 | 2.601 | 2.817 |
| 32 | correlated, per-task fit, cap 1.0x | 5.758 | -19.1% | 5.944 | 1.543 | 1.927 |
| 32 | oracle (fit on the test tokens, mixed set only): Default -> correlated | 7.107 -> 5.569 | -21.6% | | 1.058 -> 1.058 | |

Correlated search: swaps per layer max 268, mean 166 (cap 5000; 0 layers hit the cap)

Analytic distinct GPUs for a random placement: 8: 5.295, 16: 6.523, 32: 7.267


## Method

Terms. **Load**: tokens an expert received in a layer. **Co-selection**: two experts picked by the same token in
the same layer. **Cross-layer correlation**: a token that picked expert A in layer N picks expert B in layer N+1.
**Cross-GPU send**: one transfer of a token's data to a GPU that holds one of its picked experts (one per token and
destination GPU; the token's own GPU is not a send). **Load imbalance**: busiest GPU's load / mean GPU load.

1. **Routing traces** (the top-8 expert ids per token per MoE layer, logical ids, before any EPLB mapping).
   - GLM-5.3-Flash (288 routed experts, top-8, 42 MoE layers = model layers 3..44): the HeteroFlash M1 router-logit
     capture, reduced with the model's own selection rule (sigmoid + correction bias, top-8). 7 task types, 364
     requests, 203K prefill + 72K decode tokens (`scripts/reduce_m1.py`).
   - Qwen3.6-35B-A3B-FP8 (256 experts, top-8, 40 MoE layers): vLLM's routed-experts capture
     (`--enable-return-routed-experts`), one Max-Q GPU, greedy, 256 output tokens, 1200 requests: chat (WildChat-1M),
     code (github-code-clean: Java/Python/JavaScript), math (gsm8k test) (`hook/`).
2. **Train / test split by prompt.** Every request is either calib or holdout. Everything fitted (partners,
   transition model, placements, and the load fed to DefaultEplbPolicy) uses calib tokens; every reported number is
   measured on holdout tokens. "mix" = all task types pooled (per-task token cap so no task dominates).
3. **Co-selection strength.** For each expert, its 7 most co-selected partners on calib; on holdout,
   lift = P(partner picked | expert picked) / P(partner picked). Lift 1 = no relation.
4. **Layer N -> N+1 predictability.** A naive-Bayes model fitted on calib counts (score of expert j at N+1 =
   sum over the token's 8 experts i at N of log P(j | i)) predicts each holdout token's 8 experts at N+1;
   recall@8 = share of the true 8 that were predicted. Baselines: the 8 most-used experts of layer N+1, and random
   (8/E).
5. **Task dependence.** Steps 3 and 4 fitted on task A and measured on task B, for every pair.
6. **Placement simulation (CPU).** For 8, 16, 32 simulated GPUs (equal experts per GPU, no redundant experts):
   - **DefaultEplbPolicy**: vLLM's own code (`vllm/distributed/eplb/policy/default.py`, main 6092bb9, vendored
     unchanged, sha256 checked), called as vLLM calls it for these models (num_groups = 1 -> hierarchical with
     groups = nodes = 1), on the calib load.
   - **Correlated**: start from the Default placement, then greedy pairwise swaps that most increase co-selection
     inside each GPU, never letting any GPU's calib load exceed the Default placement's busiest GPU (cap 1.0x) or
     1.1x of it.
   - **Random** equal-size placement as a floor.
   - Replay of holdout tokens: each token gets a random home GPU (its attention rank). Count sends per token per
     layer and load imbalance over the whole holdout set and per 2048-token batch.
   - **Oracle**: both placements fitted on the holdout tokens themselves (upper bound for what a perfectly current
     statistic could give).

## Limits

- Simulated on measured traces. No real all-to-all was run; the sends count is a proxy for dispatch traffic, and
  whether it matters depends on the all-to-all share of MoE layer time (Carmen's open question to the group).
- Small holdouts for GLM: 12-15 holdout requests per task type (2 for long-context), 2.5K-30K tokens per task. No
  confidence intervals. Qwen holdouts are ~100 requests / 20K tokens per task.
- GLM traces come from a single-request server (batch 1); Qwen from batched offline generation. Routing per token
  does not depend on batching, but the per-batch imbalance does depend on the request mix.
- Qwen code and chat prompts use the ungated stand-ins (github-code-clean, WildChat-1M), not starcoderdata and
  lmsys-chat-1m.
- No redundant (replicated) experts. Deployments often add them; replicas would lower imbalance for both
  placements and would also change sends.
- Placements are per layer and static; no cross-layer (N -> N+1) placement was simulated. Only one random home GPU
  per token (no DP-rank structure, no node hierarchy).
- The correlated placement is a simple greedy search, not an optimum.

## Follow-up 2026-11-18 afternoon: redundant-expert (copy) budget (scripts/copies.py, results/copies/)
Gate fixed before results: correlated + R copies keeps >= 15% fewer sends than DefaultEplbPolicy + the same R copies,
with p95 per-2048-token-batch imbalance <= 1.15 or <= Default + 0.05, on holdout. Fit on calib mix; R per GPU
{1, 2, 25% of E/G}; routing rule (ours): copy on home GPU, else on a GPU already sent to, else least-loaded copy.
Parity: with R = 0 the copies code reproduces out/glm.json exactly.
- **Static copies (from calib load): no-go everywhere.** Sends stay 14-26% lower, but p95 batch imbalance stays about
  2x Default's (e.g. GLM 8 GPUs R9: 2.84 vs 1.37; Qwen 8 GPUs R8: 1.76 vs 1.20).
- **Per-batch copies (planner sees the batch, an optimistic bound): go in some cells.** GLM 8 GPUs R9 -20% sends,
  p95 1.28 vs 1.37; GLM 16 R5 -19%, 1.28 vs 1.52; Qwen 8 R8 -26%, 1.17 vs 1.20; Qwen 16 R4 -24%, 1.20 vs 1.35;
  Qwen 32 R2 -19%, 1.31 vs 1.60. GLM at 32 GPUs misses on sends (-14 to -15%).
- Most "go" cells pass on the Default + 0.05 clause: Default itself has p95 batch imbalance of 1.2-2.6.
- Least-loaded-copy routing (extra check, results/copies/summary_balance.md): balance improves for all, but the
  correlated send cut falls to ~9-11% at the 25% budget (GLM: no go; Qwen still passes some cells).
- Anti-paired placement (pairs spread apart, as in "Patterns behind Chaos"): +4-10% sends vs Default, better balance.
- Caveats: GLM pooled p95 is mostly long_context batches (14 of 20); "mix" batches are task-ordered, not interleaved.
Reading: correlation-aware placement pays only together with a dynamic, per-batch replica planner (MegaFlux-style);
the static version should be dropped. Next: a causal planner that uses the previous batch's counts, not the current one's.

# Bounded QAT pilot: result and go/no-go (2026-11-08)

Protocol: `docs/QAT_PILOT_PROTOCOL.md`, pre-registered, with an addendum written before any result.

All numbers are **[M]**: held-out layer-output relative error `||y - y_bf16|| / ||y_bf16||` of the full routed MoE output.

## Setup

- **Format:** IQ3_XXS (3.06 bpw), exact grid projection with a straight-through estimator, learned 4-bit sub-scales, super-scales fixed from llama.cpp PTQ.
- **Initialization:** latent weights start at the llama.cpp PTQ solution, so step 0 equals PTQ.
- **Training:** per-expert, gate-weighted reconstruction of the BF16 expert output on routed tokens; Adam with lr_w 1e-4 and lr_s 1e-2; batch 512.
- **Schedule:** 5 passes x 600 s, with a plateau stop.
- **Data:** bounded all-domain capture on the EXL3 control (Max-Q rig, 1.93 GPU-h).
  - code, instruction_following, knowledge_qa and math at ~150-160K tokens each, plus 25K multilingual. writing_chat is missing (capture cap).
  - About 4M training token-expert assignments per layer.
- **Layers:**
  - 43: high 3-bit sensitivity; late, large-output (group GL).
  - 32: low sensitivity; small-output (group GS).
- **Folds:** hold out knowledge_qa, then code. 2,048 held-out tokens each.

## Results

| layer | held out | Q4_K PTQ | IQ3_XXS PTQ | pass 1 | pass 2 | best gap recovered | stop |
|---|---|---|---|---|---|---|---|
| 43 | knowledge_qa | 0.0953 | 0.2424 | 0.2043 | 0.2108 | 26% | plateau after pass 2 |
| 43 | code | 0.0259 | 0.0685 | 0.0439 | 0.0458 | 58% | plateau after pass 2 |
| 32 | knowledge_qa | 0.1017 | 0.2627 | 0.2390 | 0.2510 | 15% | plateau after pass 2 |
| 32 | code | 0.1025 | 0.2654 | 0.2399 | 0.2522 | 16% | plateau after pass 2 |

- **Compute:** 4 runs x ~1,552 s on one Max-Q GPU each, 1.72 GPU-h. The pilot total, including smoke tests, control and capture, is 5.31 of the approved 8 GPU-h.
- **Export:** packed IQ3_XXS bytes round-trip to gguf-py dequant exactly (max abs error 0.0).

## Verdict: NO-GO for training all layers with this method

Pre-registered criteria:

- **Go** needs trained IQ3 <= Q4_K PTQ error on held-out domains for both layers, i.e. 100% recovery. **Not met** (best 15-58%).
- **No-go** if recovery is below 50% at the budget or plateaus below 100%. **Met:** layer 32 is below 50% in both folds, and every run plateaued below 100%.
- **Addendum threshold for the cold-TTFT layout** (>= ~76% recovery to stay within the control bar with ~36 IQ3 layers): **not met**.

## What the result does and does not show

- **Recovery is domain-dependent.**
  - Layer 43 recovers 58% when code is held out, but only 26% for knowledge_qa.
  - Layer 32 recovers about 15% in both folds.
  - Small-output layers barely respond.
- **The pass-2 regression is probably partly an artefact of the schedule.** Adam state resets each pass while lr stays at 1e-4. Under the protocol the plateau rule stops the run, and the result is reported as is.
- **A different method could do better** (persistent optimizer state, lower lr, longer training, end-to-end KD instead of per-expert reconstruction). That would be a new experiment needing new approval, not a continuation of this one.
- **Untested here:** whether trellis 3-bit formats (EXL3/QTIP, as used by the control itself at 3.5 bpw) beat IQ3_XXS at the same size.

## Implications

- **Quality-safe configurations do exist.** Q4_K experts everywhere (+0.0045 nats) beat the control (+0.0148). Up to ~9 untrained IQ3 layers stay within the control bar.
- **The cold-TTFT layout does not fit.** It needs <= ~6 remote layers, i.e. ~36 Spark layers at ~3 bpw, and that is not reachable at control-level quality with IQ3_XXS PTQ or this QAT.
- **Option worth a feasibility check (Spark only, no Max-Q GPU):** serve the control's own EXL3 3.5 bpw expert weights.
  - Quality is non-inferior by construction.
  - About 32 layers would fit on the Spark (2.95 GiB per layer), leaving about 10 remote.
  - It needs EXL3 kernels on GB10/sm_121 and a CPU path for trellis-coded experts. Both are unverified.

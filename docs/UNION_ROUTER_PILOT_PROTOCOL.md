# Union-aware router pilot: protocol (pre-registered 2026-11-16, before any training step)

**Status:** pre-registered, not started.

**Approval scope (the project lead, 2026-11-16).** A bounded router fine-tune on the Max-Q GPUs is approved only after three conditions:
1. **Gate.** The training-free step (Bet 4 step 1: causal window-sticky routing, `ops/r16/sticky_hook`) must show **≥ 25% fewer distinct experts per 4-token window at ≤ +0.004 nats** against its own β = 0 baseline on the balanced eval.
2. **Workstation free.** The workstation sweep has ended and the workstation has been declared free.
3. **This protocol is committed.** Multi-day training is not covered.

## Question

Can a router-only fine-tune make consecutive tokens reuse experts, cutting verify-window expert bytes, at a smaller quality cost than the training-free bias at the same union reduction?

## Fixed choices

1. **Trainable parameters.** Only the 42 router matrices (288 x 4096) and their correction biases, about 50M parameters. Experts, attention and all other weights stay frozen at the original BF16 values.
2. **Causal objective.** Token i's routing may depend only on tokens ≤ i, so the model stays a well-defined causal model and speculative verification stays exact.
   - Student routing: sigmoid scores plus correction bias, with a learned stickiness term. The training-free form uses β x (count of the expert among the previous W = 3 tokens' picks). This pilot learns the router weights with β fixed at the gate's best value.
3. **Loss per layer** (layer-local, on captured MoE-layer inputs from the control on the Max-Q rig):

   `|| MoE(x; student routing) - MoE(x; original routing) ||^2 / ||MoE(x; original)||^2  +  λ · softU4`

   - `softU4` is the expected number of distinct experts in each 4-token window, from the top-k relaxation of the student scores.
   - λ in {0.01, 0.03, 0.1}. One λ per layer group is picked on a validation fold only.
4. **Data.** The existing bounded-capture recipe: ~1M tokens per layer, diverse general-purpose prompts.
   - Held-out evaluation is by domain: train on 5 of 6 domains and test on the 6th, with at least two rotations (knowledge_qa, code).
   - The writing_chat gap of the QAT capture must be fixed first.
5. **Compute budget (fixed, reported):**

   | Item | Budget |
   |---|---|
   | Capture | ≤ 2 GPU-h on both GPUs |
   | Training | ≤ 4 GPU-h (about 5.7 GPU-min per layer) |
   | End-to-end eval | ≤ 2 GPU-h |
   | **Total** | **≤ 8 GPU-h** |

   Every run logs GPU-seconds.
6. **Plateau stop.** Evaluate held-out layer error and held-out U4 every 5% of the budget. Stop a layer when the objective improves < 2% over the last 20% of its budget.

## Measurements

On held-out domains, for each layer and in total:

- U4, distinct experts per 4-token window (stride 4), for each of:
  - (a) stock routing;
  - (b) training-free sticky at the gate's β;
  - (c) the trained router.
- Relative MoE-output error vs stock.
- End to end:
  - dNLL / KLD / same-top-1 on the balanced eval, against the Q8_0 BF16 proxy, with config B's CPU Q4_K experts. That is absolute quality against the control bar (+0.0148).
  - Per-domain dNLL against the control's per-domain values.
  - MTP-3 committed tokens per cycle on the five decode workloads (`ops/r16/cycle_client.py`).

## Go / no-go (pre-registered)

**Go** (recommend adopting the trained router and modelling the decode gain) requires all three:

1. U4 cut ≥ 30% on held-out domains (stock 22.7 → ≤ 15.9).
2. End-to-end dNLL ≤ config B + 0.004, and every domain still no worse than the control.
3. MTP-3 acceptance drop ≤ 5%.

**No-go** if any of these holds:
- the U4 cut is < 15% at the budget;
- dNLL exceeds config B + 0.008;
- every run plateaus before reaching the go point.

Anything in between is reported as inconclusive with the measured curves; the project lead decides.

## Reporting

- All numbers [M].
- The decode gain is reported as modeled from the measured U4 via `src/gap_model_2026_11_16.py`, never as a measured tok/s until the runtime exists.
- Resolution: a ~0.003-nat change is near the 40K-token eval's SE (~0.004). If the go/no-go hinges on it, the eval is run 10x larger, or the decision is reported as unresolved.

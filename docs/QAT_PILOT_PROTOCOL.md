# Bounded QAT feasibility pilot: protocol (pre-registered 2026-11-08, before any training step)

**Approval scope (Carmen, 2026-11-07).** This is a bounded pilot on representative layers only. No multi-day training on both Max-Q GPUs until a go/no-go decision has been reported.

## Question

Can quantization-aware training of the routed experts in the **IQ3_XXS** format close enough of the PTQ quality gap to justify training all Spark-resident layers?

IQ3_XXS is the only format that meets two constraints at once:

- the memory budget, which needs about 3.1 bpw or less for <= 6 remote layers (pre-check B);
- the kernel check: only IQ3_XXS turns its byte savings into time saved, at 0.62x of Q4_K (pre-check A).

## Fixed choices

1. **Format: IQ3_XXS.**
   - Training projects each expert's weights onto the IQ3_XXS grid, with a straight-through estimator.
   - The block and sub-block scales are learned.
   - The grid comes from gguf-py's IQ3_XXS tables.
   - Export patches the expert tensors of the existing IQ3_XXS GGUF in place. Before any eval, the packed blocks are round-tripped through gguf-py dequant and checked against the training projection.
2. **Layer selection: from the per-layer 3-bit sensitivity scan** (`src/expert_quant_sensitivity.py`), not from G2's top-k ranking.
   - The 6 most 3-bit-sensitive layers are planned as remote, at 4-bit or better in workstation RAM.
   - **High-sensitivity pilot layer:** the 7th most sensitive, i.e. the hardest layer that would stay on the Spark.
   - **Low-sensitivity pilot layer:** the least sensitive.
3. **Data: a bounded capture of MoE-layer inputs on the control model.**
   - Original GLM-5.3-Flash on the Max-Q rig, using the hook's `capture_inputs`, pilot layers only.
   - Diverse long documents and chat prompts.
   - Target: about 1M tokens per layer, so roughly 28K routed tokens per expert on average.
   - **Held-out evaluation is by domain:** train on 5 of 6 domains and test on the 6th, rotating at least once.
4. **Objective.** Experts decouple once routing is fixed. Each expert is trained to reproduce its own BF16 output on its routed tokens, weighted by gate weight: `sum_t w_te ||E_q(x_t) - E_bf16(x_t)||^2`. Routing stays the original top-8.
5. **Compute budget, fixed and reported:**
   - at most 2 GPU-hours per pilot layer on one Max-Q GPU;
   - capture at most 2 GPU-hours on both GPUs;
   - total pilot at most 8 GPU-hours.
   - Every run logs GPU-seconds.
6. **Plateau stop.** Evaluate held-out layer error every 5% of the budget. Stop early when held-out error improves less than 2% over the last 20% of the budget.

## Measurements

For each pilot layer, on held-out domains:

- the relative routed-output error `||y_q - y_bf16|| / ||y_bf16||` for:
  - (a) Q4_K PTQ, the 4-bit reference;
  - (b) IQ3_XXS PTQ (untrained);
  - (c) IQ3_XXS QAT (trained);
- the fraction of the PTQ gap recovered: `(b - c) / (b - a)`.

The BF16 teacher is the error-zero reference.

**End to end.**

- Patch the trained layers into the Q4_K-reference GGUF and measure KLD against the Q4_K reference.
- A 2-layer change is about 0.003 nats. The current 40K-token KLD eval has an SE of about 0.004, which is below resolution. E2E is therefore either run with a 10x larger eval (~400K tokens) or explicitly deferred to the full-training decision. Resolution limits are always reported alongside the numbers.

## Go / no-go criterion (pre-registered)

**Go** (recommend training all Spark-resident layers) requires all three:

1. On held-out domains, trained IQ3_XXS error is no worse than Q4_K PTQ error for **both** pilot layers. That means a gap recovery of at least 100%: trained 3-bit matches untrained 4-bit.
2. Recovery does not plateau below 100% within the budget.
3. Extrapolated linearly to the 36 Spark layers (linearity stated as an assumption), the expected full-model dNLL is at most the quality budget agreed with Carmen.

**No-go** if gap recovery is below 50% at the budget, or plateaus below 100%.

Anything in between is reported as inconclusive, with the measured recovery curve. Carmen decides.

**Fallbacks to report if no-go:**

- 3.44 bpw (IQ3_S) with fewer Spark layers;
- accept a TTFT miss;
- revisit the remote-layer budget.

## Addendum (2026-11-08, written before any pilot training result)

**Non-inferiority bar measured.** On the balanced eval, against the Q8_0 BF16 proxy, the EXL3 3.5 bpw control scores:

- dNLL +0.0148 +/- 0.0017;
- KLD 0.042;
- same top token 94.0%.

**Candidates, same eval:**

| experts | dNLL | per layer |
|---|---|---|
| all-Q4_K + imatrix (G0) | +0.0045 | |
| IQ3_XXS PTQ | additive across layer groups | about +0.00118 nats per IQ3 layer |

**Implied requirement, reported alongside the pre-registered criterion (which is unchanged):**

- **Room for 3-bit.** The quality room up to the control is about 0.0103 nats. That fits roughly 8-9 PTQ IQ3 layers.
- **Recovery needed for the cold-TTFT layout.** That layout needs about 36 IQ3 layers on the Spark. To stay within the control bar, QAT must remove **at least about 76%** of each layer's IQ3 loss.
- **How it is measured.** Recovery is measured as the fraction of the held-out layer-output gap closed, assuming a proportional end-to-end effect. The end-to-end check comes later.

**Capture coverage.** The capture holds 4 full domains (code, instruction_following, knowledge_qa, math) plus partial multilingual. writing_chat is missing because the capture hit its 55-min cap.

**Pilot folds:** hold out knowledge_qa, then code.

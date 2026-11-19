# Proposal: bounded Max-Q GPU work to close the gap to 80 tok/s (2026-11-19, for Carmen's approval)

Status: proposal only. No Max-Q GPU is used until Carmen approves this document.

## Where we are (measured, 1 Spark + MaxQ CPU, batch 1, MTP-3)
- 2026-11-18 morning: k7 FP8, 26.4 tok/s, full-eval KLD 0.038.
- 2026-11-18 night: NVFP4 attention (our own RTN build), k7 A 0.5, 29.6 tok/s, KLD 0.052.
  k6 32.2 and k5 33.4 tok/s are queued for full-eval quality.
- Training-free levers still running (mean-output stand-in, per-layer bands, NVFP4 shared/dense) are modeled at ~36-40 tok/s.

## Why not a better drafter first
- The MTP drafter saturates at ~2.7-2.9 committed tokens per cycle (11-07, measured).
- At batch 1 a longer verify window reads more distinct experts. U4 is ~23 experts per layer at k8, and U grows with the window,
  so a stronger drafter is largely eaten by expert bytes. The 11-16 gap model put block drafter A 3.5 at only +2% at control quality.
- Conclusion (11-16 plan, C4): above ~40 tok/s on this hardware, routing locality matters more than the drafter.

## Proposed: window-local router fine-tune ("union-aware routing", trained)
Today's training-free union drop already shows the effect: in a 4-token verify window, a weak pick is kept only if its expert is
already read. On the subset that lowered KLD at equal reads: k6u 0.032 vs k6 0.042, k5u 0.060 vs k5 0.079, k4u 0.099 vs k4 0.144.
A router trained for it should get more locality at less quality cost. This is also the charter's scientific question:
joint routing + placement adaptation versus independent compression.

- What is trained: only the 42 router (gate) matrices and score biases, ~50M parameters. Expert and attention weights stay frozen.
- Loss: KL to the teacher's routing distribution plus a window-union penalty on decode windows. Teacher = the original model's router.
- Data: MoE-input capture on the dual-Max-Q control over our calibration text (not the eval set), the same kind as the QAT pilot capture.
- Evaluation: our existing pipeline. The router override goes into a rebuilt checkpoint; 7-chunk subset, then full eval, KLD vs Q8,
  per domain; speed on the Spark (d-style 3 interleaved rounds).

## Budget and stop rules (pre-registered)
- Capture: <= 2 GPU-h (both Max-Q GPUs, control TP2, as in the QAT pilot: 1.93 GPU-h).
- Pilot on 2 layers (one tolerant and one sensitive, from tonight's map): <= 1 GPU-h.
  - Stop if U4 at the same per-layer router KL does not drop >= 20% versus training-free union drop.
- All 42 layers: <= 4 GPU-h. Total cap 7 GPU-h, no multi-day runs.
- Go to the end-to-end test only if the pilot passes.
- End-to-end success: subset KLD < 0.07 (full ~< 0.1) at >= +10% tok/s over the best training-free point on the same day.

## Not requested now
- QAD of all fixed weights to NVFP4 (R8 rung): only if the RTN NVFP4 shared/dense build fails the quality bar.
- Block / DFlash-style drafter: parked per the analysis above.

## Honest ceiling
- Even with all of this, the modeled reach is ~45-55 tok/s on this hardware.
- 80 tok/s would also need the workstation CPU's bandwidth to rise (8-CCD part, modeled +18%) or a faster link (RDMA).
- 80 is a stretch target, not a promise.

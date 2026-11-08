# HeteroFlash research charter (owner's framing, 2026-11-07; overrides earlier wording)

HeteroFlash is **not** about optimizing an existing Spark checkpoint. It transforms the owner's original GLM-5.3-Flash through
weight surgery, pruning, router distillation, mixed precision and heterogeneous MoE execution. The goal is to reproduce the
performance, and preserve the general-purpose capabilities, of the dual-Max-Q setup on radically less GPU hardware.

| Role | Definition |
|---|---|
| **Control** | The original GLM-5.3-Flash on the two Max-Q GPUs: the owner's serving configuration (performance reference: 122-132 tok/s decode, 45K cold 10.8 s, cached 0.2 s) and the original model's outputs (quality reference). |
| **Experimental treatment** | Modifications of the original model (top-k / router surgery and distillation, expert pruning/merging, layer-wise mixed precision, recovery training), produced and evaluated on the Max-Q rig. |
| **Deployment** | One DGX Spark plus permitted CPU/RAM (Spark Arm, workstation CPU and RAM), with both Max-Q GPUs free. |
| **Success** | All three hard latency targets met on the deployment, with acceptable quality retention against the original model (pre-registered non-inferiority margins). |
| **Contribution** | Which combinations of model transformations and hardware-aware execution make that possible, and which do not. |

Rules that follow from this:
- Published single-Spark checkpoints and recipes (community EXL3/NVFP4/GGUF builds, third-party drafters) are **background literature only**. They are not baselines, starting points or definitions of what is achievable, and they are not run as part of this study.
- Every treatment derives from the original weights. Candidates are compared against the Control: quality on the Max-Q rig, latency on the deployment hardware.
- Spark-side measurements of an *unmodified* model run on the HeteroFlash runtime, and only as an internal diagnostic of that runtime.

## Framing: engineering goal, scientific question, novelty (owner, 2026-11-08)

These three are deliberately kept apart. Mixing them together made the project feel unfocused.

- **Engineering goal.** Serve GLM-5.3-Flash on one DGX Spark plus the workstation CPU/RAM, with both Max-Q GPUs free.
  - Targets: decode 80 tok/s, stretch 125; cold 45K TTFT 10.8 s or less; cached 45K+32 turn 0.2 s or less.
  - Quality must be non-inferior to the 2x Max-Q served control: EXL3 3.5 bpw, +0.0148 nats dNLL and KLD 0.042 against the Q8_0 BF16 proxy on the balanced eval.
- **Scientific question.** Can hardware-aware adaptation of MoE weights and expert execution improve the quality-latency frontier beyond independently optimized compression and scheduling?
  - In plain terms: can the model itself cooperate with a limited GPU and a remote CPU, rather than being compressed and left to the inference engine to run efficiently?
- **Novelty.** Three things, each derived from the original BF16 weights:
  - placement- and traffic-aware per-layer and per-expert precision;
  - formats matched to each device's bandwidth and compute;
  - CPU/GPU-overlapped expert execution.
  - The comparison is always against independently optimized baselines: the best compression alone (e.g. the EXL3 control format) and the best scheduling alone.

**Background.** Existing approaches trade quality, latency or hardware cost:

- aggressive quantization (fits, but may degrade quality);
- CPU offloading (adds compute and communication delays);
- additional GPUs (adds hardware cost).

**Working rule (owner, 2026-11-08).** Keep moving to the next research step without waiting for a reply, within this charter and the existing permissions. Ask only for genuinely new risks, such as multi-day Max-Q GPU use or training beyond approved budgets. Hourly status posts end with "Next step I'm starting now: ...", not with a question.

Full text of the owner's framing: kept with the project notes, outside the repo.

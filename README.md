# HeteroFlash: model/system co-design for GLM-5.3-Flash on one DGX Spark

**Status (2026-11-06):** Initial research scaffold and synthetic placement prototype only. No model training, Spark profiling, or real-world throughput measurements have been performed here. The three SLAs remain **unverified**.

## Fixed constraints
- Final inference GPU: one NVIDIA DGX Spark (GB10, 128 GB unified LPDDR5X, 273 GB/s rated memory bandwidth).
- Auxiliary execution: Spark's own Arm CPU + the CPU(s) and approximately 230 GB available DDR5 RAM in a separate workstation. The workstation's two RTX PRO 6000 Max-Q GPUs are **NOT AVAILABLE FOR INFERENCE**.
- Workstation GPUs **may be temporarily used for teacher inference, training, calibration, and evaluation**; release them for other research once the checkpoint is deployed.
- General-purpose GLM-derived model; not distilled specifically for Hermes.
- Prose decode >=120-130 tokens/s; cold 45,000-token prompt TTFT <=10.8 s; exact cached 45,000-token prefix + a **defined** small suffix TTFT <=200 ms.

## What is included
- `ROADMAP.md`: stage-gated research execution plan and milestones.
- `docs/HECATE_TRANSFER.md`: what actually transfers from Hecate and what does not.
- `docs/EXPERIMENT_PROTOCOL.md`: models, experiment matrix, statistical plan and quality thresholds.
- `docs/SYSTEM_DESIGN.md`: asymmetric Spark + local Arm + remote CPU design, critical path, precision and scheduler.
- `paper/main.tex` and `paper/references.bib`: publication-oriented manuscript starter, no fabricated results.
- `src/placement_sim.py`: deterministic **synthetic** expert placement/bit-allocation prototype. Outputs an illustrative placement plan, not predicted real performance.
- `src/router_surgery.py`: unit-tested GLM sigmoid/correction-bias routing algebra with experimental separate active/reference top-k.
- `src/paired_analysis.py`: paired bootstrap utility for genuine measured per-prompt quality/latency CSVs.
- `src/hardware_probe.sh`: safe, non-mutating hardware/network inventory commands for the actual machines.
- `tests/`: unit tests for trace generator and optimizer invariants.

## First local reproducible action
From the project root, using Python 3.10+ and no third-party dependencies:

```bash
python3 -m unittest discover -s tests -v
python3 src/placement_sim.py --config configs/synthetic_demo.json --out results/synthetic_placement.json
```

**Synthetic output must never be reported as experimental performance.** Its purpose is checking the algorithmic interfaces and invariants before measuring the real system.

## Immediate next work on hardware
1. Run `bash src/hardware_probe.sh` separately on Spark and workstation; collect CPU model, memory-channel/NUMA layout, NIC speed, link status, installed compute runtimes and driver versions. Do not upload sensitive addresses or hostnames publicly.
2. Benchmark local Spark/GPU memory bandwidth, Spark Arm CPU INT4/INT8 expert kernels, remote CPU matvec/GEMM, end-to-end activation-message RTT and bandwidth with the actual NIC configuration.
3. Record GLM-5.3-Flash router traces from a diverse calibration set, preserving *layer*, *chosen expert IDs*, *gating weights*, *token position*, *prefill vs decode*, and *expert-output norms*. Use these to replace synthetic frequencies.
4. Reproduce a baseline 2-bit single-Spark model and stronger teacher on training GPUs, measure cold and warm TTFT with proper token accounting.
5. Start router-normalization/top-k grid and REAP/HOPE-informed nonuniform pruning; train router and optional expert LoRA; quantize from post-surgery activations.

## Research question
Can joint learned expert sparsity, layerwise sensitivity-aware precision, and tier-aware parallel expert execution produce a generally useful GLM-5.3-Flash descendant that satisfies **all three** SLAs on one Spark with CPU-only remote assistance? The answer may be no. Experiments are designed to establish either outcome reproducibly.

## Sources
- Hecate (GLM-5.3 **not Flash**, GB300 hardware): https://github.com/third-party/hecate
- GLM-5.3-Flash architecture: https://docs.nvidia.com/nemo/automodel/latest/model-coverage/vision-language-models/thudm/GLM-5.3-Flash
- Official GLM-5.3-Flash: https://huggingface.co/zai-org/GLM-5.3-Flash
- Spark hardware: https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- REAP: https://github.com/CerebrasResearch/reap
- NVIDIA precision work: https://nvidia.github.io/Model-Optimizer/announcements/20261028-glm53-flash-iq2.html

## Licensing
Check the exact checkpoint and downstream runtimes before publishing modified weights or an inference package. The official GLM-5.3-Flash model card currently labels the checkpoint MIT; Hecate's *different GLM-5.3* weights have different licensing. Runtime components such as TabbyAPI may impose separate obligations. Record license and model revision IDs at every training run.

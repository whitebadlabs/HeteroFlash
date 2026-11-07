# HeteroFlash

**Profiling-Driven Model–System Co-Design for Large MoE Inference on Constrained GPU Hardware**

*Status: Active independent research · November 2026*

## The problem

Large language models increasingly use Mixture-of-Experts (MoE) architectures. Instead of activating every parameter for every token, a router selects a small subset of specialized neural networks, called experts.

This makes large models computationally efficient, but their weights still require substantial memory. Running them on smaller GPU systems often means aggressive quantization, moving experts to CPU memory, or accepting significantly slower inference.

The challenge is not just fitting the model into memory. It’s making a large model run quickly after its computation has been distributed across processors with very different performance characteristics.

## What we’re building

HeteroFlash investigates how to run GLM-5.3-Flash on:

- One NVIDIA DGX Spark GPU, with 128 GB shared memory.
- A separate workstation’s CPU and approximately 230 GB available system RAM.
- No workstation GPU resources during deployed inference.

Our reference is the original GLM-5.3-Flash deployment on two RTX PRO 6000 Blackwell Max-Q GPUs. Those GPUs may be used temporarily for offline training and experiments, but must remain available for other work during final inference.

We want to preserve the original model’s general-purpose capabilities while approaching its responsiveness on dramatically less GPU hardware.

## Our research question

Does jointly optimizing an MoE model’s weights, expert routing, and execution placement produce a better quality–latency tradeoff than optimizing the model and inference runtime separately?

In simpler terms:

Can we change how a model chooses and executes its experts so that it becomes faster on limited hardware without becoming substantially less capable?

## Our hypothesis

A traditional approach treats model optimization and runtime optimization as separate problems.

First, the model is quantized or compressed. Then an inference engine decides where its computations should run.

But these decisions interact.

Changing which experts the model selects changes memory access patterns, CPU workload, network communication and synchronization costs. Changing the precision or placement of an expert can change whether selecting that expert is computationally expensive.

We hypothesize that using actual hardware measurements to guide model adaptation and scheduling can lead to better inference performance at comparable quality.

This is a hypothesis to test, not an established result.

## What we’ve learned from existing research

Prior research establishes several foundations:

1. CPU computation can be better than moving large weights. Hybrid inference systems such as Fiddler and KTransformers show why executing selected experts where their weights already reside can reduce expensive transfers.

2. Expert-routing behavior matters. Research on local routing consistency finds that expert reuse differs across models and domains. Expert placement should therefore be informed by measured routing patterns.

3. Hardware-aware routing is possible. TA-MoE demonstrates incorporating communication topology into training, while HetRoute studies joint routing, placement, precision and execution costs.

4. Speculative decoding changes the workload. A verifier processing several candidate tokens can activate many distinct experts. Optimizing token count alone may not reduce expert traffic.

5. Low-bit quantization can preserve substantial model quality. NVIDIA’s GLM-5.3-Flash-FP4-Local demonstrates quantization-aware distillation, but its published deployment requires more GPU memory than a single Spark offers.

These techniques provide building blocks. Our focus is investigating their interactions on a real, constrained GPU–CPU system.

## What we’ve achieved

We have an end-to-end heterogeneous GLM-5.3-Flash deployment running across a DGX Spark GPU and workstation CPU/RAM.

The initial implementation uses llama.cpp RPC over 1GbE and is functional but far below our latency targets.

| Metric | Original dual-GPU control | Initial Spark + CPU deployment | Goal |
|---|---|---|---|
| Single-stream prose | 151 tok/s | 6.46 tok/s | 120–130 tok/s |
| Cold 45K-token TTFT | 11.1s | 838s | ≤10.8s |
| Cached 45K+32 TTFT | 0.156s | 3.10s | ≤0.2s |

These are reported measurements, not results from the proposed optimized HeteroFlash system.

Our model experiments also show that reducing the active expert count introduces meaningful quality costs. Top-6 routing with the original top-8 normalization is promising, but more aggressive reductions produce larger negative log-likelihood increases. Small compensation networks have not recovered most of the lost information.

These negative results matter: they prevent us from assuming that simply activating fewer experts will solve the problem.

## What we’re investigating next

Our immediate priority is end-to-end performance tracing.

Before modifying hardware or making more architectural changes, we need to identify exactly where time is spent during cold prefill, cached prefill and decoding.

We will separately measure GPU execution, CPU expert computation, remote transfers, memory bandwidth, caching and synchronization.

Using those measurements, we’ll evaluate:

- Whether optimized CPU expert execution and scheduling can reduce the critical path.
- Whether phase-specific prefill and decode policies are beneficial.
- Whether mixed-precision quantization can reduce weight traffic without unacceptable quality loss.
- Whether learned, hardware-cost-aware routing improves performance beyond runtime scheduling alone.

Changes will be evaluated through matched experiments, using the original dual-GPU model as the quality and performance reference.

## What would constitute a research contribution?

We want to understand when modifying the model itself improves heterogeneous inference beyond what better system software can achieve.

A positive result would demonstrate a repeatable method that improves the quality–latency frontier.

A negative result could show that certain optimizations—such as expert reduction or learned routing—provide little benefit once runtime overhead is accounted for.

Both outcomes are valuable if supported by controlled experiments and reproducible evidence.

## Scope

HeteroFlash focuses on one model family and one constrained deployment topology. It is not attempting to invent a universal inference framework, train a new foundation model, or solve hardware adaptation for every possible system.

Our priority is to establish a working, measured system and answer one focused scientific question.

## Intended outputs

- An open-source GitHub repository with inference components, profiling tools and reproducible experiments.
- Hugging Face technical articles explaining the architecture, measurements, successes and negative results.
- Model checkpoints or adapters where redistribution rights and quality validation permit.
- A potential research paper on profiling-driven model–system co-design, if the experiments establish a sufficiently distinct contribution.

The goal is not merely to make GLM-5.3-Flash fit on a Spark. It’s to understand how model design and inference execution should work together when fast GPU memory is scarce.

---

## Repository guide

### Latest measurements (2026-11-07, E2E-v1)
Spark GPU + workstation CPU experts over a direct 10 Gb/s link, llama.cpp RPC, with the GPU placement fixed:

| Metric | E2E-v1 | E2E-v1 + MTP drafting |
|---|---|---|
| Single-stream prose | 10.5 tok/s | 17.0 tok/s |
| Cold 45K-token TTFT | 223 s | 226 s |
| Cached 45K+32 TTFT | 0.68 s | 0.76 s |

Profiles, kernel rooflines, and a per-token byte budget against the goals: `results/2026-11-07-e2e-v1/README.md` and `docs/E2E_GAP_ANALYSIS_2026-11-07.md`.

### Framing
Control = original GLM-5.3-Flash on the dual-Max-Q rig. Treatments = modified versions of the original model, trained and evaluated on that rig. Deployment = one DGX Spark + permitted CPU/RAM, with the Max-Q GPUs free. Published Spark checkpoints are background only. See `docs/RESEARCH_CHARTER.md`.

### Fixed constraints
- Final inference GPU: one NVIDIA DGX Spark (GB10, 128 GB unified LPDDR5X, 273 GB/s rated memory bandwidth).
- Auxiliary execution: Spark's own Arm CPU + the CPU(s) and approximately 230 GB available DDR5 RAM in a separate workstation. The workstation's two RTX PRO 6000 Max-Q GPUs are **NOT AVAILABLE FOR INFERENCE**.
- Workstation GPUs **may be temporarily used for teacher inference, training, calibration, and evaluation**; release them for other research once the checkpoint is deployed.
- General-purpose GLM-derived model; not distilled specifically for Hermes.
- Prose decode >=120-130 tokens/s; cold 45,000-token prompt TTFT <=10.8 s; exact cached 45,000-token prefix + a **defined** small suffix TTFT <=200 ms.

### What is included
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

### First local reproducible action
From the project root, using Python 3.10+ and no third-party dependencies:

```bash
python3 -m unittest discover -s tests -v
python3 src/placement_sim.py --config configs/synthetic_demo.json --out results/synthetic_placement.json
```

**Synthetic output must never be reported as experimental performance.** Its purpose is checking the algorithmic interfaces and invariants before measuring the real system.

### Immediate next work on hardware
1. Run `bash src/hardware_probe.sh` separately on Spark and workstation; collect CPU model, memory-channel/NUMA layout, NIC speed, link status, installed compute runtimes and driver versions. Do not upload sensitive addresses or hostnames publicly.
2. Benchmark local Spark/GPU memory bandwidth, Spark Arm CPU INT4/INT8 expert kernels, remote CPU matvec/GEMM, end-to-end activation-message RTT and bandwidth with the actual NIC configuration.
3. Record GLM-5.3-Flash router traces from a diverse calibration set, preserving *layer*, *chosen expert IDs*, *gating weights*, *token position*, *prefill vs decode*, and *expert-output norms*. Use these to replace synthetic frequencies.
4. Reproduce a baseline 2-bit single-Spark model and stronger teacher on training GPUs, measure cold and warm TTFT with proper token accounting.
5. Start router-normalization/top-k grid and REAP/HOPE-informed nonuniform pruning; train router and optional expert LoRA; quantize from post-surgery activations.

### Research question
Can joint learned expert sparsity, layerwise sensitivity-aware precision, and tier-aware parallel expert execution produce a generally useful GLM-5.3-Flash descendant that satisfies **all three** SLAs on one Spark with CPU-only remote assistance? The answer may be no. Experiments are designed to establish either outcome reproducibly.

### Sources
- Hecate (GLM-5.3 **not Flash**, GB300 hardware): https://github.com/third-party/hecate
- GLM-5.3-Flash architecture: https://docs.nvidia.com/nemo/automodel/latest/model-coverage/vision-language-models/thudm/GLM-5.3-Flash
- Official GLM-5.3-Flash: https://huggingface.co/zai-org/GLM-5.3-Flash
- Spark hardware: https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- REAP: https://github.com/CerebrasResearch/reap
- NVIDIA precision work: https://nvidia.github.io/Model-Optimizer/announcements/20261028-glm53-flash-iq2.html

### Licensing
Check the exact checkpoint and downstream runtimes before publishing modified weights or an inference package. The official GLM-5.3-Flash model card currently labels the checkpoint MIT; Hecate's *different GLM-5.3* weights have different licensing. Runtime components such as TabbyAPI may impose separate obligations. Record license and model revision IDs at every training run.

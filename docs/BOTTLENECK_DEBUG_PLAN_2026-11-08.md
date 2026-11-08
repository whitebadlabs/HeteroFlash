# Serial GPU/CPU bottleneck: diagnosis and debugging plan (2026-11-08)

Inputs:

- **Deep research:** 101 agents, claims checked with three adversarial votes [L].
- **Our measurements [M].**

## Why llama.cpp stalls [L, high confidence]

- **Splits run in order.** `ggml_backend_sched` cuts the graph at every backend change, and without pipeline parallelism the splits run one after another.
- **RPC can't pipeline.** ggml-rpc has no event support, so the pipelined path (`n_copies > 1`) never activates with an RPC device.
- **The result for us.** With the experts of layers 24-44 on RPC0, each remote layer is a synchronous copy, then remote compute, then copy back. That matches our nsys trace [M]: GPU 45% busy, 301 gaps over 2 ms with a median of 55 ms.
- **Newer master doesn't fix it.** It adds `ggml_backend_sched_prefetch_inputs`, but that cannot overlap layers that depend on each other.

## Ranked plan

1. **Attribute every gap [cheap, first].**
   - Run nsys with `--trace=cuda,nvtx,osrt` and OS-runtime backtraces on llama-server, plus per-command timestamps in the RPC server.
   - Split each 55 ms gap into remote compute, wire time, socket wake-up and serialization.
   - Our estimate is that remote compute dominates: about 1.5 TMAC/s for 1.9e14 MAC is about 127 s of the 223 s cold 45K [P].
2. **Make the remote CPU GEMM-efficient.** Batched int8 VNNI GEMM reaches 12-19 TOPS [M], against about 3 in llama.cpp. ZenDNN 5.2.1 has an INT8 dynamic-quant linear op [L, vendor numbers on dense models].
3. **Place by load, not by layer range.** Keep hot experts on the Spark and keep about 9 or fewer layer-equivalents of expert work on the CPU [P]. HybriMoE and DALI both report that static assignment causes CPU/GPU imbalance [L].
4. **Overlap.**
   - **Prefill:** micro-batch pipelining in the style of MoE-Lightning CGOPipe. The Spark runs attention, shared experts and local experts on micro-batch A while the CPU runs routed experts for micro-batch B [L].
   - **Decode:** KTransformers-style expert deferral, which runs some experts of layer k alongside layer k+1's attention. It reports +33% decode [L]. It changes model semantics (under 0.5% accuracy), so it needs our quality check.
   - **CUDA Graphs:** a remote callback must be a non-blocking enqueue plus a flag-poll [L].
5. **Transport.**
   - Busy-poll sockets without root on kernel 6.4 and later [L]; done in runtime-v1 [M].
   - One message per layer; done [M].
   - int8/fp8 activations: about 2x less wire time [P].

## Expected gains and limits

- **Published gains.** On local-PCIe single hosts with smaller models:
  - KTransformers: 4.6-19.7x prefill and 1.25-4.1x decode over llama.cpp/Fiddler.
  - DALI: 7.6x prefill and 4.0x decode over llama.cpp.
  - HybriMoE: 1.33x prefill and 1.70x decode over KTransformers.

  These multipliers do not transfer directly to a remote CPU over 10 GbE.
- **Unverified.** There is no verified evidence that vLLM or SGLang supports GLM-5.x Flash on GB10/sm_121. The vLLM CPU-expert-offload RFC (#33869) was closed without merging. A custom engine path needs a feasibility spike.

## Decision

- **Engine.** Stop tuning llama.cpp RPC; it is structurally serial. Build the runtime-v2 path: Spark GPU plus the runtime-v1 expert server, with per-layer overlap.
- **Order:** step 1 (attribution), then 2 (int8 GEMM in the expert server), then 4 (micro-batch overlap), then 3 (load-based placement, the hot/cold experiment).

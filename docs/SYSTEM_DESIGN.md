# System design: asymmetric three-processor MoE inference

## Execution targets
- P0 = GB10 GPU: high-throughput tensor cores, 128GB unified RAM shared with P1.
- P1 = GB10 Arm CPU: ~20 cores, shares the **same** 273 GB/s rated memory interface with P0; potentially useful for low-frequency expert parallelism but may starve P0.
- P2 = workstation CPU: ~230 GB available RAM, CPU capabilities/network unknown; two Max-Q GPUs remain **idle with respect to final inference**.

## Data path
```
P0 attention + router computes top-k and gates
                 |
      +----------+-----------+
      |          |           |
 P0 hot experts  P1 rare     P2 remote CPU experts
 (GPU native)    (Arm int8)  (x86 int8/int4 + routed activation send)
      |          |           |
      +----------+-----------+
                 | exact gated partial sum combine on P0
                 v
             next layer
```
The per-layer completion time is the **critical-path max** across available expert bands, not sum of three processors' nominal throughputs. Remote CPU has a per-layer communication RTT and message batching/serialization cost. Compute ordering across layers is sequential, while expert bands for a layer can be concurrent. Spark P0/P1 memory contention is a first-class constraint.

## Cost function and constraints
Let expert e of layer l on processor p use quantization q and predicted access probability a_le, size s_leq, quality penalty d_leq and measured cost vector c_lepq(m) for microbatch shape m. Define placement x_lepq in {0,1}, sum_pq x_lepq=1 (or replicated variants with consistency constraints). Minimize weighted p95 decode latency, cold prefill latency, cache TTFT, and quality penalty. Constrain Spark unified-memory capacity, remote RAM capacity, measured NIC bytes, CPU core availability, and measured quality noninferiority. Penalize total migrations and uncertainty.

For each layer and decode verification step, the deterministic scheduler observes actual selected experts and dispatches to tiers. A useful proxy for latency is:

`T_layer = T_attention + max(T_spark_gpu, T_spark_arm, T_remote_cpu + T_message) + T_combine`

But **Spark memory demand couples GPU and Arm**: `(bytes_gpu + bytes_arm) / measured_shared_bw` must not be less than elapsed weight traffic time. All cost terms are measured in the eventual system; a synthetic model is not a benchmark.

## Quantization / compute formats (candidates, not established winners)
| Processor | Decode candidates | Prefill candidates | Measured microbench required |
|---|---|---|---|
| Spark GPU | int2/EXL3, int4, NVFP4 weight-only | native FP4 W4A4 if quality permits, W4A16 grouped GEMM | sustained effective bytes/s and GEMM FLOPs at actual expert shapes |
| Spark Arm | packed INT8/INT4 NEON kernels | INT8 larger batch only if fast enough | GEMV/GEMM throughput, shared LPDDR contention |
| Workstation CPU | INT8 VNNI / AVX512 if available, INT4 packing | INT8/INT4 batch GEMM | NUMA, DDR channels, SIMD features, network latency |

Do not assume a nominally lower bitwidth is faster. Include dequantization work, layout conversion, kernel startup and KLD quality sensitivity. Down projections may warrant FP4 even if gate/up remain IQ2_S (NVIDIA October 2026 study).

## Scheduler prototype policy
- Epoch-level planner on aggregate routing traces assigns processor and quant per expert subject to memory/quality/performance constraints.
- Dispatcher uses current exact router output: concurrent P0/P1/P2 jobs with per-layer network message batching.
- Keep both hot-path weights and cache states resident, and pinned/reused remote worker buffers, while preserving correctness.
- Migration only when `expected future latency saving > migration + repacking + synchronization cost` with safety margin; log placement epoch, evictions, hit rate, p95 band completion and bytes transferred.
- Recompute *candidate quantizations* offline on calibration data and mark versions, never silently requantize model weights during a decode verification cycle.
- Prefill plan can differ from decode plan; warm prefix hits must preserve exact KDA and DSA state.

## Important negative controls
- No heterogeneity: same compressed checkpoint on GB10 GPU only.
- Heterogeneous no-KD: full top-8 and identical expert outputs.
- Local Arm only vs remote x86 only vs both, same total expert precision.
- Equal placement but fixed single quant to isolate scheduler gains.
- Same quant and placement with/without overlap, prefetch, pinned messaging.
- Network link deliberately bandwidth/latency throttled to test robustness.

## First build modules (after hardware probes)
- GPU router trace hook and per-expert saliency collector.
- C++ remote CPU worker with batched message structs: `(revision, layer, step, row_ids, expert_ids, activation_dtype, inputs)`; outputs `(revision, layer, step, rows, partials)`.
- Microbenchmark suite for exact expert layer shapes and batch/row counts.
- Planner with integer programming or differentiable relaxation, initialized from greedy simulation but then calibrated on measured cost tables.
- CUDA graph / work-queue scheduler on GPU with event-driven remote completions; no busy waiting across network; bounded timeout with deterministic fallback.
- Correctness comparator against all-GPU MoE output, including gate weight combination and long context state.

## Critical constraint
There is no guarantee that remote CPU makes the exact model faster; the research must measure whether hybrid overlap overcomes serial per-layer network round trips. If not, use remote RAM solely for calibration/backing store and deploy an aggressively compressed all-Spark model. The combined paper should **report this negative result**, not discard it.

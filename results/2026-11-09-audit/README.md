# CPU expert path, Q4_K + imatrix experts, per-layer exchange over 10 GbE [M, 2026-11-09]

**Setup.**

- `runtime/expert_server_gguf.c` serves the 129 expert tensors of `GLM-5.3-Flash-Q8attn-G0_q4kimexp.gguf` (Q4_K, 14.16 MB per expert), mmapped and prefaulted, with 32 Zen5 threads and OMP active wait.
- Client: `runtime/layer_exchange_bench.c` on the Spark CPU, busy-poll plus heartbeat (mode 3), 30 steps x 42 layers.
- Layers loaded: table indices 9, 17, 27, 35, 41.

| workload | T | union U | CPU share f | CPU experts | rtt p50 (us) | server p50 (us) | net p50 (us) | eff. GB/s | step (42 layers) p50 |
|---|---|---|---|---|---|---|---|---|---|
| decode | 1 | 8 | 0.50 | 4 | 379 | 309 | 62 | 183 | 16.2 ms |
| decode | 1 | 8 | 1.00 | 8 | 637 | 572 | 63 | 198 | **26.9 ms** |
| verify4 | 4 | 19 | 1.00 | 19 | 1738 | 1626 | 110 | 165 | **73.7 ms** |
| verify4 | 4 | 23 | 1.00 | 23 | 1976 | 1863 | 110 | 175 | **82.9 ms** |
| verify8 | 8 | 26 | 1.00 | 26 | 2814 | 2626 | 178 | 140 | 119.3 ms |

Raw output: `results/2026-11-09-audit/q4k_layer_exchange.txt`.

**Bandwidth bound for 80 tok/s.** Derived from these numbers and the FP8 decode measurement; it is a bound, not a projection.

- **Cycle budget.** At about 2.8 committed tokens per cycle (MTP-3; acceptance not yet measured on our model), 80 tok/s allows 35 ms per cycle.
- **Bytes per cycle.**
  - Fixed weights: 8.7 GB (FP8 non-expert), read once per cycle.
  - Expert union for a 4-token verify window: about 19-23 experts x 42 layers x 14.16 MB, about 11.3-13.7 GB at Q4_K.
  - Total: about 20-22 GB.
- **Read capacity in 35 ms**, with both devices fully parallel and a perfect split:
  - GPU at the measured effective ~190 GB/s (11 GB per token in 57.8 ms, FP8 no MTP): about 6.7 GB;
  - CPU at about 170 GB/s (verify windows): about 6.0 GB;
  - **total about 12.7 GB.**
- **Conclusion.** With 4-bit experts and FP8 fixed weights, 80 tok/s exceeds the combined memory-bandwidth budget by about 1.6-1.7x, independent of placement.
- **What must change.** Fewer bytes per committed token: fixed-weight precision, expert precision, a smaller expert union per verify window, or more committed tokens per byte read. Or an effective bandwidth closer to the GB10's 273 GB/s peak.
- **125 tok/s** (22.4 ms) leaves about 8.1 GB of read capacity per cycle.

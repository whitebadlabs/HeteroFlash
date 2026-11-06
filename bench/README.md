# Microbenchmarks (M0 calibration)

Small, dependency-free probes used for the 2026-11-06 calibration in `results/2026-11-06-m0-calibration/`.

| File | What it measures | Build / run |
|---|---|---|
| `mem_bw.c` | Multi-threaded sequential read bandwidth over an 8 GB buffer (reduce over uint64) | `gcc -O3 -march=native -fopenmp mem_bw.c -o mem_bw && OMP_PROC_BIND=spread ./mem_bw` (on Arm use `-mcpu=native`; shrink the buffer if memory is busy) |
| `moe_cpu_vnni.c` | One MoE layer of routed experts on CPU (gate+up 4096->2x2048, down 2048->4096, batch-1 GEMV) with AVX-512 VNNI, int8 or naive int4 nibble unpack; weights cycle through a 96-expert pool so they come from DRAM | `gcc -O3 -march=native -fopenmp moe_cpu_vnni.c -o moe && OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=32 ./moe <bits 8|4> <experts_per_token> <tokens>` |
| `tcp_pingpong.py` | TCP_NODELAY request/response round trip at 64 B - 64 KB payloads (per-layer activation message proxy) | server: `python3 tcp_pingpong.py srv <bind_ip> <port>`; client: `python3 tcp_pingpong.py cli <server_ip> <port>` |

These are proxies, not production kernels: the int4 kernel is deliberately naive (unpack-bound), and Python adds latency to the TCP numbers.

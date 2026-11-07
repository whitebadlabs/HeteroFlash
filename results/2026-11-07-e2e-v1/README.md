# E2E-v1: Spark GPU + workstation CPU experts over the direct 10G link (llama.cpp RPC)

GGUF: Q8_0 fixed weights, Q4_K routed experts (177 GB). Fixed weights and routed experts of
layers 3-23 on the Spark GPU; routed experts of layers 24-44 on the workstation CPU (ggml-rpc-server, 32 threads).

## Placement bug fixed
llama.cpp inserts RPC devices before CUDA0 (src/llama.cpp, model->devices.insert(begin, rpc...)),
so the device order is [RPC0, CUDA0] and `--tensor-split 1,0` put every layer on the workstation CPU
(E2E-v0 and the first v1 attempt: GPU util ~0-12%, rpc-server 172 GB RSS). Correct: `--tensor-split 0,1`.
The runner now aborts unless more than 60 GB is resident on the Spark after load.

## Results (prose holdout, 10 prompts x 256 tokens, temperature 1; TTFT 45K, 2 repeats)

| run | TPOT p50 | decode tok/s | cold 45K TTFT p50 | cached +32 | cached +128 | GPU util | rpc CPU |
|---|---|---|---|---|---|---|---|
| v1b (no spec) | 94.7 ms | 10.5 | 223.1 s | 0.68 s | 1.25 s | 66% | 1603% |
| v2b (MTP draft, n_max 3) | 59.3 ms | 17.0 | 225.9 s | 0.76 s | 1.34 s | 74% | 1346% |
| target | 7.7-8.3 ms | 120-130 | <=10.8 s | <=0.2 s | | | |

Link traffic during decode: 5-10 MB/s each way (bandwidth is not the constraint; per-layer round trips and CPU compute are).

## Byte budget per decoded token (from the GGUF tensor table)
- fixed (non-expert) weights at Q8_0: 9.69 GB (embeddings excluded)
- routed experts, 8 of 288 per layer, 14.16 MB each at Q4_K: 2.38 GB local (layers 3-23), 2.49 GB remote
- Spark GPU bytes/token 12.06 GB -> floor 50 ms at 241 GB/s (measured GPU busy ~62 ms of 94.7 ms)
- 125 tok/s needs <= ~1.9 GB read per committed token on the Spark.

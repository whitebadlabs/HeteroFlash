# Control baseline: original GLM-5.3-Flash on the dual-Max-Q rig (2026-11-07 02:04-02:14Z)

Owner's serving configuration: vLLM EXL3 TR3 3.5bpw, TP2, CUDA graphs, MTP 3 draft tokens, prefix cache (mamba align), image sha256:0c3deb7df8ac7d6e2eca8f067245bb87f06cd2ef930724e5442279dedbc2d5b0. Harness: `src/ttft_bench.py` (cache state verified on every trial; 0 violations). Single server start; repeats across restarts still to do.

## TTFT (client-observed, streaming)
| Condition | n | p50 | p95 | p95 95% CI |
|---|---|---|---|---|
| cold_45000 | 10 | 11096 ms | 11301 ms | [11198, 11306] ms |
| cached_45000+16 | 10 | 92 ms | 100 ms | [93, 102] ms |
| cached_45000+32 | 10 | 156 ms | 161 ms | [159, 161] ms |
| cached_45000+64 | 10 | 174 ms | 182 ms | [176, 182] ms |
| cached_45000+128 | 10 | 201 ms | 212 ms | [204, 212] ms |
| cached_45000+256 | 10 | 229 ms | 243 ms | [234, 244] ms |

- Cold 45K: p95 upper CI 11.31 s vs the 10.8 s target, so **the Control itself narrowly misses** the SLA as defined (the 10.8 s target came from one earlier measurement). p50 11.10 s = ~4,055 input tok/s.
- Cached 45K + 32: p95 161 ms, which passes 0.2 s.

## Decode (512 tokens prose / 256 tokens mixed, temperature 1, ignore_eos, chat-templated prompts via /tokenize)
| Set | n | TPOT p50 | TPOT p95 | tok/s mean | chunk ITL p95 (median over requests) | committed tokens / verify | draft acceptance | implied verify step |
|---|---|---|---|---|---|---|---|---|
| prose (writing_chat holdout) | 14 | 6.704 ms | 7.84 ms | 150.8 | 17.375 ms | 2.43 | 47.6% | 16.28 ms |
| mixed 6 domains holdout | 60 | 5.929 ms | 7.563 ms | 167.31 | 17.665 ms | 2.73 | 57.6% | 16.18 ms |

## Implications for the deployment
- With the Control's own MTP acceptance on prose (2.43 committed tokens per verify), 125 tok/s on the Spark needs a verify step of about 19 ms.
- At the Spark's measured ~222 GB/s that is ~4.3 GB of Spark reads per verify step, against ~27 GB for the unmodified model (16.6 GB fixed BF16 + ~10.6 GB routed experts for a 4-token verify).
- Closing the gap requires stacked levers: fixed-weight precision, fewer bytes per expert and fewer active experts, cold experts on the parallel CPU tier, and more committed tokens per verify (drafter retrained on the modified model).

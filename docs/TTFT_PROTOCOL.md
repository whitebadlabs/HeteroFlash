# End-to-end TTFT measurement (SLA 2 and 3)

The SLA is **time to first streamed token as seen by the client**, not prefill throughput. Prefill tok/s is a diagnostic only.
Tool: `src/ttft_bench.py` (stdlib; tests in `tests/test_ttft_bench.py`).

| Condition | Definition | Cache check on every trial | SLA |
|---|---|---|---|
| `cold_45000` | Exactly 45,000 prompt tokens (token-id prompt, no chat template). The first 16 tokens are a per-trial random nonce, so no cached prefix can match. Model process resident and warm. | server `cached_tokens` <= one block, `prompt_tokens` == 45,000; otherwise the trial is a violation and excluded | p95 <= 10.8 s |
| `cached_45000+32` | The same fixed 45,000-token prefix, warmed by an untimed request, then prefix + 32 new tokens timed | `cached_tokens` >= 45,000 - block size, exact `prompt_tokens` | p95 <= 0.2 s |
| `cached_45000+{16,64,128,256}` | As above with other suffix lengths; reported, not SLA | same | — |

- Trials of all conditions are interleaved in random order. The report gives p50/p95/p99 with bootstrap 95% CIs. An SLA counts as passed only if the **upper** CI of p95 is within the limit.
- Run across several server restarts and report per-restart and pooled results (protocol requirement). Record engine commit, image digest, compile/CUDA-graph settings, KV dtype, prefix-cache mode (hybrid KDA/MLA "align" block size) and clocks.
- Valid targets: the deployed Spark configuration (S0, S1, ...) and, as a reference, the original two-GPU serving configuration. **Not** the M1/M2 trace server: it runs eager without prefix caching, by design.
- Chat-template overhead (system prompt, tool schemas) is a separate measurement: the same harness with a rendered chat prompt of the same total token count.

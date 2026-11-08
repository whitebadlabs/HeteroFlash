# The EXL3 control weights on the Spark: feasibility log (2026-11-08)

Setup: Spark vLLM image `eugr/spark-vllm-b12x` (glm5next, sm_121), with the control image's `b12x` 1.3.0 and its EXL3 vLLM files overlaid (`ops/spark-exl3/Dockerfile`). Truncated test models (`ops/spark-exl3/truncated_test.sh`, test-only hooks in `trunc_hook/`).

## Measured [M]

- **Imports.** The control image's EXL3 path imports cleanly on the Spark (`Exl3Config`, `Exl3MoEMethod`, `Exl3LinearMethod`).
- **Weight loading.** 20 layers of the control's EXL3 3.5 bpw weights load onto the Spark GPU at **2.96 GiB per MoE layer** (62.26 GiB in 326 s).
  - All 42 MoE layers would be ~124 GiB of experts, so ~10 layers must be remote.
- **KV-cache allocation fails with `fp8_ds_mla`.** The hybrid KDA state forces 4,608-token manager blocks, and the Spark's sparse-MLA backend cannot split them into 256-token kernel blocks. `fp8` and `auto` KV allocate fine.
- **Then a b12x API mismatch.** The Spark vLLM calls ~20 b12x submodules (KDA, sparse MLA, mHC, GEMMs) against its own bundled b12x. With the control's b12x 1.3.0 swapped in, the KDA decode fails first (`Caps(recover_speculative_state)`), and more mismatches are likely.

## Conclusion

Transplanting the control's b12x into the Spark vLLM is fragile. The cleaner routes are:

- **(C)** convert the checkpoint to the Spark-native b12x `exl3-v1` layer format (the native b12x supports SiLU experts), or
- **(D)** rebuild the control's patched vLLM image for arm64/sm_121.

The overlap runtime (runtime-v2) is needed under either route, because the full EXL3 model does not fit on the Spark.

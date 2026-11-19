# Local Inference Lab llm-inference-bench on the HeteroFlash runtime (2026-11-19)
Single stream (server max 1 sequence), spec mode MTP-3, same Spark + workstation pair, one boot per config.
| config | decode tok/s, 0K context (bench prompts) | tool-eval-bench (leaderboard settings, parallel 4) |
|---|---|---|
| best: NVFP4 attn+shared+dense, top-5 + union + mean stand-in, A 0.5 | 28.1 | 90/100 (Excellent) |
| baseline: FP8 fixed path, all 8 experts, A 0.7 | 20.8 | 89/100 (Good) |
- Decode +35% on the same benchmark; tool calling unchanged (1 point, within run-to-run noise at parallel 4).
- The bench prompts (thinking on) give lower absolute tok/s than our 15-request pool (36.2 / 26.4 for the same configs).
- 16K / 32K decode cells are empty: the prefill outran the 60 s cell window; rerun with a longer duration pending.

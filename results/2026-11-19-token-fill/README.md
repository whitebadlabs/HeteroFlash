# Token-keyed and input-based fills for skipped expert picks (Qwen3-30B-A3B, Qwen3.6-35B-A3B)

Setup: drop each token's routed picks at rank >= K (K = 5, 4) and fill them with w * fill. Fits on calibration text
(~104K tokens), evaluation on the balanced eval text, KL vs the all-experts model (teacher-forced, deterministic
kernels; all-8-experts check gives KL 0 exactly). Pre-set go/no-go for the token-ID table on Qwen3-30B at K=5:
remove >= 25% more skip error than mean-fill AND cut KL >= 15% vs mean-fill. Result: no-go.

KL at K=5 / K=4:

| fill | Qwen3-30B-A3B | Qwen3.6-35B-A3B |
|---|---|---|
| drop | 0.0707 / 0.1260 | 0.0670 / 0.1136 |
| mean, all picks | 0.0766 / 0.1440 | 0.0664-0.0672 / 0.1133-0.1137 |
| mean, weak (tail) picks only | 0.0678-0.0686 / 0.1286-0.1291 | 0.0654-0.0656 / 0.1090-0.1095 |
| rank-bucketed mean | 0.0679 / 0.1270-0.1273 | |
| token-ID table, >= 3 samples | 0.0897 / 0.1668 | 0.0663 / 0.1135 |
| token-ID table fit on tail picks only | 0.0684 / 0.1285 | |
| linear correction from expert input (rank 32) | 0.0828 / 0.1666 | 0.0619-0.0624 / 0.1046-0.1053 |
| token-ID table + linear | 0.1117 / 0.1970 | |

Share of skip error any constant fill can remove (in-sample ceiling, outside the last layer): Qwen3-30B 8.4%,
Qwen3.6-35B 6.7%. Token coverage (share of K=5 tail-pick weight in each token's top-16/32/64 most frequent experts):
Qwen3-30B 38/52/62%, Qwen3.6-35B 24/33/44%.

Files: `q3/`, `q36/` (main runs), `q3_sel/`, `q36_sel/` (layer-selected linear, exploratory), `q3_tt/` (tail-only token
table, exploratory). Each has `kl.json` and `errors.json` (per-layer error split and coverage).

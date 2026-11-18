# Expert co-selection study: progress

Working copy (NAS): NAS_PATH (no git on the NAS host).
Committed copy: Spark worktree /srv/vllm-research/work/HeteroFlash-ecorr, branch expert-correlation,
dir studies/expert-correlation/. Push: NAS_PATH

## Done
- 09:4x PT GLM-5.3-Flash route3 traces reduced -> data/reduced/m1_top8_v2 (288 experts, top-8, MoE layers 3..44,
  7 task types, 364 requests).
- 10:07 PT Qwen3.6-35B-A3B-FP8 trace on MaxQ GPU 0 done: data/qwen36/routes/qwen36_v1 (256 experts, top-8,
  40 MoE layers, chat/code/math from lmsys-chat-1m/starcoderdata/gsm8k, 1200 requests, EXIT 0).
- 10:24 PT first analyze.py runs died with the session (no partial output was saved).
- 11:00 PT analyze.py now checkpoints to <out>.ckpt/ (stage1.json = step 4 statistics; sim_G<G>_<fit>.json per
  simulated GPU count and fit set). Rerunning the same command resumes.

- 11:08 PT full runs launched (scripts/run_all.sh; logs out/glm_run2.log, out/qwen_run2.log). Smoke test of
  resume passed (identical output). Placement search now records swaps per layer (cap 5000).
- Verified: vendored default.py sha256 == vLLM main 6092bb9; vLLM always calls the hierarchical function with
  (groups, nodes) = (1, 1) when num_groups = 1; num_redundant_experts defaults to 0 when E % GPUs == 0.
- GLM input: m1_top8.npz == m1_top8_v2.npz (v2 only fixed the JSON metadata write).
- Qwen prompts used the ungated fallbacks: WildChat-1M (chat), codeparrot/github-code-clean (code), gsm8k (math).

- 11:14 PT the 11:08 runs were killed by another session restart (setsid/nohup is not enough). Relaunched as
  systemd user units ecorr-glm / ecorr-qwen (scripts/run_all.sh); logs out/glm_run3.log, out/qwen_run3.log.
  Check: systemctl --user status ecorr-glm ecorr-qwen. If dead, rerun scripts/run_all.sh (it resumes).

## Next
- Full runs on the NAS CPU (nohup, logs in out/):
  GLM:  analyze.py --reduced data/reduced/m1_top8_v2 --out out/glm.json  (train-cap 12000, eval-cap 30000)
  Qwen: analyze.py --reduced data/qwen36/routes/qwen36_v1 --out out/qwen.json --train-cap 20000 --eval-cap 20000
- plots.py for both, RESULTS.md, commit + push.

- 13:09 PT GLM run done (out/glm.json); Qwen done 12:15 PT. plots.py -> results/plots/, summarize.py -> results/RESULTS.md
  (+ raw results/glm.json, results/qwen.json). Lane "Topic: Expert Correlation Research" owns the study from 12:15 PT.
- Next: copy-budget experiment (paired + R redundant experts vs Default + R, p50/p95 batch imbalance, MegaFlux kappa);
  prefill-only / task-label fit; HF blog draft (notes/hf-blog-outline-2026-11-18.md on the NAS).

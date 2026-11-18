# Frontier run: progress (resume notes)

Carmen, task 35 (2026-11-17 07:06): "Try to decrease some accuracy to increase tok/s as much as possible."
Control bar (dual Max-Q EXL3 control vs Q8 proxy, balanced eval): **dNLL +0.0148, KLD 0.042, top-1 94.0%**.
A point is inside the bar only if all three metrics pass; within 2 SE of dNLL (SE ~0.002) is "at the bar".

## Done (measured, committed)
- Speed, boot f1 (spec mode, MTP-3, NIC defaults), all-15-request tok/s: base 20.8, KDA off 23.2 (lossless),
  tau0.08 26.4, top-6 25.8, top-5 27.1, top-4 27.6, top-6+tau0.08 27.6, top-4+Q3_K15 28.0. (`f1_frontier_speed.json`)
- Lossless: MaxQ i40e coalescing off (temporary, restored): KDA off 24.7, top-4 30.4.
- Quality, boot fq1 (agree mode, 41 chunks vs Q8 proxy): base +0.0044/0.027/95.0%; top-6 +0.0181/0.059/92.4%;
  top-4 +0.0886/0.156/86.9%; top-4 renorm +0.1234/0.174/86.6%.
- Quality, boot fq2 (same method): tau0.08 +0.0313 (SE 0.0022)/0.077/91.4% (fq2.log, e98224b);
  k7 +0.0099 (SE 0.0016)/0.037/93.9% (fq2b.log, 19:39 resume): dNLL and KLD inside the bar, top-1 0.1 pt under it.
  tau0.04 +0.0055 (SE 0.0014)/0.030/94.9%: inside the bar on all three, close to base (+0.0044/0.027/95.0%).
  k7tau0.04 +0.0112 (SE 0.0017)/0.040/93.8%: dNLL and KLD inside the bar, top-1 0.2 pt under. fq2 DONE 22:03 PT.

## Running
- **Speed boot f3** (= f2 config + `--kv-cache-memory-bytes 8181829386`, i.e. f1's 7.62 GiB / 302,193-token KV cache),
  `tmux -L hfrt` session `f3`, boot log `/srv/vllm-research/work/cache/r17/f3.log`; then the same 5 points x 3 rounds
  (session `f3pts`, log `f3_points.log`). Stop with `touch results/2026-11-17-dualrt/f3.stop`.
- MaxQ expert server `es` still up (needed by f3).

## f2 speed boot (22:19-23:22 PT): ANCHOR FAILED, points not trusted
- f2 aggregate (all 15 req, measured): kdaoff 21.33 (f1 23.21, miss 1.88), tau0.08_kdaoff 26.26 (f1 26.39, ok),
  k7_kdaoff 23.11, tau0.04_kdaoff 21.05, k7tau0.04_kdaoff 23.96. `f2_frontier_speed.json`, `f2_points.log`.
- Cause found: f2's KV cache was 230,589 tokens vs f1's 302,193 (non-torch memory +1.8 GiB at the same 0.84 util:
  consumed 94.67 vs 92.85 GiB). Hermes long prompts then miss the prefix cache and re-prefill: short Hermes items take
  ~5 s instead of ~0.9 s, while 300-token decode items match f1 (exchange per-request times are equal or lower than f1).
  Prefix-cache hit rate on f2: 6377/19549 = 33%. The penalty depends on eviction order, so it hits points unequally.
- f2 boot 1 (22:04 PT) failed at FULL CUDA-graph capture: the hook's 30 s stats-dump thread ran CUDA ops on a side
  stream during capture. Race, same code as f1. Logs in `f2_bootfail_2217/`. Retry unchanged came up 22:36 PT.
- 19:19 outage handling: k7 partial in `partial_fq2_pre1919/` (unused); fq2 resumed 19:39 with the unscored passes only (fq2b.log).

## Next
1. (running as f3, see above) Speed boot f2 (`boot_frontier.sh`) for k7 / tau0.04 / k7tau0.04, interleaved with the anchors kdaoff and tau0.08_kdaoff
   (3 rounds). If the anchors miss f1 by more than 0.3 tok/s, do not trust the new points.
2. Final frontier summary (README in this dir) + task 35 post.
3. Stop `es` on the MaxQ. Then the PR #64825 slowed-CPU rerun (task 6) on the MaxQ. Then SendMessage "MAXQ RELEASED".
   Then FlashInfer #9464 rerun on the Spark (task 2).

## How to relaunch after an outage
MaxQ expert server (wait for "listening on 47201", ~2 min):
```
cd /srv/cache/heteroflash/runtime-v1 && tmux -L hfrt new -d -s es "HF_ALT_GGUF=/srv/models/gguf/GLM-5.3-Flash-Q8attn-q3_Kimexp.gguf HF_ALT_IDX=/srv/cache/heteroflash/dualrt/q3_Kimexp.idx HF_ALT_LAYERS=26,27,28,29,30,31,33,34,36,38,40,41,42,43,44 HF_DENSE_PACK=/srv/cache/heteroflash/dualrt/dense_lmh_kda32.pack HF_BIND=WS_LINK_IP OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OMP_WAIT_POLICY=active /srv/cache/heteroflash/dualrt/expert_server_gguf_v5 serve 47201 /srv/models/gguf/GLM-5.3-Flash-Q8attn-G0_q4kimexp.gguf q4k_experts.idx 2>&1 | tee /srv/cache/heteroflash/dualrt/es_f1.log"
```
Quality boot (Spark; reusing the tag wipes the old capture dir). Relaunch ONLY the passes that have no
q_fq2_<pass>_vs_q8_bal.json yet, and write a NEW log file (fq2.log holds tau0.08, fq2b.log the 19:39 resume).
First move any partial q_fq2_<pass>_* files + results/2026-11-17-dualrt/fq2_{stats.json,serve.log} into partial_fq2_pre<HHMM>/.
As launched 19:39 PT:
```
H=/srv/vllm-research/work/HeteroFlash; tmux -L hfrt new -d -s fq2 "QTAG=fq2 bash $H/bench/dualrt/run_frontier_quality.sh \"k7|7 0 0|none|1 0 0\" \"tau0.04|8 0.04 0|none|1 0 0\" \"k7tau0.04|7 0.04 0|none|1 0 0\" > $H/results/2026-11-17-frontier/fq2b.log 2>&1"
```
A pass counts only with "logits files: 41" and chunks_scored == chunks_total == 41.
Speed boot: `tmux -L hfrt new -d -s f2 "bash $H/bench/dualrt/boot_frontier.sh f2 > /srv/vllm-research/work/cache/r17/f2.log 2>&1"`,
then `TAG=f2 POINTS_ONLY="..." ROUNDS=3 bash bench/dualrt/run_f1_points.sh`, stop with `touch results/2026-11-17-dualrt/f2.stop`.
Push to Forgejo from the NAS: `NAS_PATH`.
Rules: no MaxQ GPU; Qwen qfn stays down; tmux only on `-L hfrt`; NIC toggles are temporary and the restore is verified.

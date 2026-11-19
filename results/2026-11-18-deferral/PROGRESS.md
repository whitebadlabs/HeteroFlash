# Expert deferral study (expert-cache-project idea): progress (resume notes)

Carmen, 2026-11-18 07:02 PT (task 36): fold "expert deferral + expert correlation" into HeteroFlash and find out
whether it improves interactivity (decode tok/s). Operating point to beat: **k7 + KDA split off, 25.7 tok/s (f3, warm)**,
quality +0.0099 / KLD 0.037 / top-1 93.9% vs the control bar **+0.0148 / 0.042 / 94.0%**.

## Plan (cheapest first)
1. GPU wait per layer at k7 (one spec boot) -> deferral ceiling. Same boot: deferral speed points (defer depth 1 is
   just the reply wait moved one layer later; the mailbox is already asynchronous).
2. Quality replay in agree mode: whole remote partial added one layer late. 7-chunk subset screen (one chunk per
   domain: 0,6,12,18,24,30,36), KLD vs a same-boot undeferred capture; then defer-all / greedy layer sets; full 41-chunk
   passes vs Q8 only for the 2-3 final candidates. Includes "add expert 8 late" vs "drop expert 8" (k7).
3. Only with headroom from 1+2: E2E speed + quality for the chosen set.
4. Design note only: affinity-aware / per-request hot set.

## Code (2026-11-18, unit-tested on the Spark with a fake server: `bench/dualrt/test_defer.py`, all_ok)
- `hook/hf_drt.cu`: GPU wait timers (globaltimer; per slot, decode T<=4 vs other: n, blocked ns, pack->wait GPU work,
  zero-wait count; dense lm_head waits too), host-mapped (read with no CUDA op). Gated pack and wait kernels.
- `hook/sitecustomize.py`: `HF_SPLIT_DEFER=1` captures both waits per layer (own layer gated defer[L]==0, previous
  layer gated defer[L-1]==1), mask at run time via `drt_defer.req` ("none" | "all" | "l1,l2,.." [LATE]); layer 44 never.
  `HF_SPLIT_LATE=1` (needs DROP + DEFER): experts removed by the drop kernel go to the workstation in slot 1 and are
  added one layer late. Stats dump now request-only (the 30 s dump raced graph capture on f2) and includes "wait".
- Subset tokens: data/q_eval_balanced_v1.sub7.tokens.json (data/ is gitignored; chunks 0,6,12,18,24,30,36 of q_eval_balanced_v1.tokens.json, same keys + chunk_ids).
- `waits.py` (per-layer blocked ms, ceiling), `kld_pair.py` (same-boot KLD), `compare_q8.py HF_CHUNKS` (subset),
  `run_quality_multi.sh` (FO, TOKENS, 5th field DEFER, KEEPCAP same-boot reference), `frontier_point.sh` (FO, DEFER).
- Deferral is decode-size windows only (T <= 16); prefill chunks are unchanged. In agree mode every position is a
  16-token window, so the quality replay defers everywhere (slightly pessimistic vs spec-mode serving).

## Status
- 07:32 PT MaxQ free (diffusion run_seeds end, no GPU processes). es started 07:33 PT (es_d1.log on the MaxQ).
- d1 speed boot (spec, HF_SPLIT_DEFER=1, KV pinned 302,193 tokens): READY 07:51 PT. Points: run_defer_points.sh
  (warm-up, then kdaoff / k7 / k7_defall / kdaoff_defall x 3 rounds) -> d1_points.log; then run_defer_points2.sh
  (defer-all x balance rule A 0.5 / 0.35 / 0.2, plus k7 A 0.5 without deferral, 2 rounds) -> d1_points2.log (tmux d1q).
- Round 1 (measured, single round): kdaoff 24.37 tok/s, 117.1 ms/cycle, MoE blocked 5.71 ms/window (0.08-0.24 ms/layer,
  53-75% zero-wait; GPU work send->wait 1.0-1.4 ms/layer > 1.05 ms round trip). k7 26.29, 110.5 ms, blocked 5.67.
  k7_defall 26.77 (104.5 ms, blocked 0.18, committed/cycle 2.905 -> 2.798). kdaoff_defall 24.48 (111.7 ms, 2.853 -> 2.735).
  => exchange is already hidden behind GPU work; deferral removes ~5.5 ms/cycle but costs ~4% acceptance.

## How to relaunch
MaxQ expert server: same command as results/2026-11-17-frontier/PROGRESS.md ("es", tmux -L hfrt), log es_d1.log.
Speed boot d1 (Spark): `tmux -L hfrt new -d -s d1 "HF_SPLIT_DEFER=1 EXTRA_ARGS=\"--kv-cache-memory-bytes 8181829386\" bash $H/bench/dualrt/boot_frontier.sh d1 > /srv/vllm-research/work/cache/r18/d1.log 2>&1"`
Points: `FO=$H/results/2026-11-18-deferral DEFER="none" bash bench/dualrt/frontier_point.sh d1 NAME "1 0 0" "0.7 -1" "7 0 0" none`
Stop: `touch results/2026-11-17-dualrt/d1.stop`.
Rules: no MaxQ GPU; Qwen qfn stays down; tmux only on `-L hfrt`; MaxQ NIC coalescing OFF is the default (never restore 50 us).
- 08:48 PT all d1 rounds done (defer_agg.py, rounds 2-3 pooled; points2 rounds 1-2 => only round 2 in r23 file):
  kdaoff 24.77 (115.8 ms, 2.868/cycle, blocked 5.9 ms/window); k7 26.78 (108.8, 2.913, 5.8);
  k7_defall 27.12 (102.7, 2.784, 0.19); kdaoff_defall 25.42 (108.6, 2.762, 0.17);
  balance sweep (rounds 1-2 pooled): k7_defall A0.5 29.06; **A0.35 32.39 (85.4 ms, 2.766, blocked 0.78, rt 1.35 ms)**;
  A0.2 32.13 (CPU-bound: blocked 12.1, rt 1.57); k7 A0.5 without deferral 27.65 (blocked 9.5).
  Reading: deferral alone ~+1% (acceptance loss eats the 6 ms); deferral lets the CPU take more experts: +21% at A0.35.
- 08:50 PT queued d1_points3.log (tmux d1r): k7 A0.35 / A0.25 without deferral, defer-all A0.28, defer-all A0.35 again.
- 09:10 PT attribution points (d1_points3.log, rounds 3-4): without deferral, lower A is SLOWER: k7 A0.35 25.19
  (blocked 30.0 ms/window), A0.25 24.25 (blocked 44.0). With deferral: A0.35 32.44 (again), **A0.28 33.26** (80.0 ms/cycle,
  2.662/cycle, blocked 1.7). So deferral is what lets the CPU take more experts. Acceptance falls as more is deferred
  (2.91 k7 -> 2.78 defall A0.7 -> 2.76 A0.35 -> 2.66 A0.28): the quality cost is the open question.
  All points pooled: d1_speed_all.jsonl. d1 stopped 09:11 PT (DONE d1).
- 09:12 PT quality screen dq1 launched (tmux -L hfrt dq1, agree mode with MNBT=4 so windows match the spec verify
  window T=4 and the remote share matches serving; 7-chunk subset; ref_k8 capture kept as the same-boot reference):
  ref_k8, k7, k7_defall_a07, k7_defall_a035, k7_defall_a028, k7_late8, k7_a035, k8_defall_a035 -> dq1.log.
  Note: the deferred share depends on A (A0.35 defers ~2/3 of the routed output), so quality is measured at each A.
- 10:27 PT dq1 COMPLETE (all 8 passes, 7/7 chunks each; no partials). The driving Claude session died ~10:56 PT after it
  finished; resumed 11:05 PT. Measured, 7-chunk subset, agree mode MNBT=4 (vs Q8 proxy: dNLL / KLD / top-1; paired vs same-boot ref_k8 KLD):
  | point | dNLL vs Q8 | KLD vs Q8 | top-1 | KLD vs ref_k8 |
  |---|---|---|---|---|
  | ref_k8 | +0.0056 | 0.018 | 95.8% | - |
  | k7 | +0.0071 | 0.027 | 95.2% | 0.0160 +-0.0006 |
  | k7_a035 (no deferral) | +0.0061 | 0.026 | 95.2% | 0.0161 |
  | k7_late8 | +0.0075 | 0.022 | 95.5% | **0.0131 +-0.0005** |
  | k7_defall_a07 | +0.228 | 0.309 | 81.8% | 0.304 |
  | k7_defall_a035 | +0.323 | 0.407 | 77.9% | 0.402 |
  | k7_defall_a028 | +0.366 | 0.451 | 76.7% | 0.447 |
  | k8_defall_a035 | +0.311 | 0.394 | 78.2% | 0.390 |
  Reading: deferring the whole remote partial one layer (untrained) is ~10x past the bar (0.042) at every A; A alone is
  free (k7_a035 = k7) and the drop is not the cause (k8_defall = k7_defall). KLD is flat along chunk position (0.29-0.35
  per 170-token bin at A0.7), so it is a per-token effect, not KV-cache contamination. The 32-33 tok/s defer-all points
  are therefore out. Only "late" (dropped low-weight experts computed on the workstation and added one layer late) keeps
  quality: k7_late8 is closer to k8 than k7 is. Partial-layer deferral has no speed case (at low A the undeferred layers
  block again: 30 ms/window measured at k7 A0.35), so it is not screened.
- Next: d2 speed boot (DEFER+LATE, KV pinned): k7, k7_late8, k6, k6_late interleaved x3 (no warm-up kept). If k6_late
  beats same-boot k7 by > 0.5 tok/s it joins the full-length quality boot (dq2: FULL=1 MNBT=4 KEEPCAP=k7; k7, k7_late8,
  [k6_late], k7_defall_a035 last).
- 11:05 PT d2 speed boot started (tmux -L hfrt d2, HF_SPLIT_DEFER=1 HF_SPLIT_LATE=1, KV pinned). Driver died ~11:13 PT
  (Manager restart); resumed 11:17 PT. MaxQ GPUs idle (no compute apps), coalescing off (rx-usecs 0, adaptive off).
- 11:17 PT chain launched so a driver death no longer stalls the study: `tmux -L hfrt d2c` runs
  `bench/dualrt/run_d2_chain.sh` (log cache/r18/d2c.log): waits for READY d2 -> run_defer_points2.sh (now takes a 5th
  field LATE and WARMUP=1) warm-up + 3 interleaved rounds of k7 / k7_late8 / k6 / k6_late (A 0.7) -> d2_points.log ->
  stop d2 -> defer_agg.py rounds 1-3 -> d2_speed.jsonl -> dq2 (FULL=1 MNBT=4 KEEPCAP=k7: k7, k7_late8, [k6_late if
  > same-boot k7 + 0.5 tok/s], k7_defall_a035 last) -> dq2.log, then CHAIN_DONE.
- 11:39 PT Carmen (Model Fundamentals room): go/no-go is now **KLD < 0.1** vs Q8 (replaces the 0.042 control bar). Still
  report dNLL, top-1 and per-domain (code especially), and where the old bar falls.
- 11:48 PT researcher session took over. d2 rounds 1-2 (same boot, pooled later): k7 26.4, k7_late8 24.2, k6 27.8,
  k6_late 23.5 tok/s. The late slot doubles decode requests (42 -> 83/window) and blocked time (5.8 -> 16.7 ms/window).
  Not sweeping A below 0.35 for defer-all: dq1 KLD rises as A falls (0.31/0.41/0.45), so lower A cant reach 0.1.
  Under 0.1 the drop-K axis reopens (frontier full-eval top-6 KLD 0.059; top-5 never measured).
  Per-layer GPU blocking at k7 A0.35 without deferral is uniform (0.66-0.74 ms/layer, d1_k7_a035_r4_waits.json), so
  any layer subset needs a per-layer A to turn its deferral into speed (global A makes the other layers block).
- 11:48 PT queued `bench/dualrt/run_dq3_chain.sh` (tmux -L hfrt dq3c, log cache/r18/dq3c.log): waits for CHAIN_DONE,
  then dq3 = 7-chunk subset, agree MNBT=4, KEEPCAP=k7: k7, k6, k5 (A 0.7), k7 + deferral of 3-12 / 13-22 / 23-32 /
  33-43 / even / odd (A 0.35) -> dq3.log. Stop rule (set before results): if fewer than ~10 deferred layers fit the
  0.1 budget after calibrating subset -> full with dq2 k7, deferral is a no-go (best case ~+1 tok/s) and per-layer A
  is not built.
- 11:58 PT d2 pooled (rounds 1-3, d2_speed.jsonl): k7 26.44, k6 27.70, k7_late8 24.20, k6_late 23.57 tok/s; k6_late not added to dq2. dq2 booting.
- 12:52 PT dq2 k7 full (41/41, MNBT=4): KLD 0.0378, dNLL +0.0096 (frontier fq2 k7 MNBT16: 0.037/+0.0099), so MNBT does not change non-deferred quality and frontier full-eval drop-K numbers are reusable. Subset/full calibration for k7: 0.027 -> 0.038.

## HANDOVER 2026-11-18 12:56 PT (researcher session -> "Topic: Expert Correlation Research" lane)
Running now, unattended (Spark, tmux -L hfrt; never touch tmux main):
- `d2c` = bench/dualrt/run_d2_chain.sh (log cache/r18/d2c.log). d2 speed done (d2_speed.jsonl). Now in dq2 (full 41
  chunks, agree MNBT=4, KEEPCAP=k7, ~42 min/pass, log dq2.log): k7 DONE (KLD 0.0378, dNLL +0.0096, 41/41);
  k7_late8 started 20:52Z (ends ~13:35 PT); then k7_defall_a035 (~14:20 PT); then `CHAIN_DONE` in d2c.log.
  The chain's MaxQ GPU check from the Spark fails (no ssh Spark->MaxQ); check from the NAS: `ssh maxq nvidia-smi`
  (12:50 PT: both MaxQ GPUs idle; es up in tmux -L hfrt on the MaxQ, 239 GB RAM free).
- `dq3c` = bench/dualrt/run_dq3_chain.sh (log cache/r18/dq3c.log): waits for CHAIN_DONE, then dq3 = 7-chunk subset,
  agree MNBT=4, KEEPCAP=k7 -> dq3.log, ends `DQ3_DONE` (~9 passes x ~7.5 min + boot, ~15:40 PT):
  k7, k6, k5 (A 0.7); k7 + deferral at A 0.35 of layers 3-12 (dQ1) / 13-22 (dQ2) / 23-32 (dQ3) / 33-43 (dQ4) /
  even (deven) / odd (dodd). even + odd vs defer-all (subset 0.402 paired) = additivity check.
Tools: `python3 bench/dualrt/defer_table.py results/2026-11-18-deferral dq3 k7 k7 k6 k5 k7_dQ1 ...` prints
dNLL/KLD/top-1, worst domain, code, paired KLD (same for dq2 with ref k7).

Findings so far (all measured):
- Defer-all fails KLD < 0.1 at every A (subset 0.31/0.41/0.45 at A 0.7/0.35/0.28); lower A defers more, so an A sweep
  below 0.35 cannot help. Not run, on purpose.
- Late experts keep quality but lose speed (k7_late8 24.2, k6_late 23.6 vs k7 26.4): out.
- Under KLD < 0.1 the drop-K axis reopens: frontier full-eval k6 KLD 0.059 / dNLL +0.0181 / top-1 92.4%, but
  instruction_following 0.103 (> 0.1); speed k6 27.70 vs k7 26.44 same boot (d2). k5 quality never measured (dq3).
- Free lever: balance rule A 0.5 without deferral, k7 27.65 vs 26.61 at A 0.7 (d1 same boot); A alone does not change
  quality (dq1 paired k7_a035 0.0161 vs k7 0.0160). Not yet combined with k6/k5.
- No tool-calling domain in q_eval_balanced_v1; code (MBPP) is the closest. Flag in the README.
- Old 0.042 bar: inside it only k7 (0.038 full) and tau0.04; k7 at A 0.5 should be the fastest point inside 0.042
  (needs the d3 confirmation below).

Stop rule (fixed before dq3 results): calibrate subset -> full with k7 (0.027 -> 0.038, x1.4). If fewer than ~10
deferred layers fit under 0.1 (full), deferral is a no-go (best case ~+1 tok/s, ~0.4% per deferred layer) and per-layer
A is not built. If a layer subset fits: per-layer A is needed for speed (blocking is uniform per layer at global A 0.35);
a ready, compiled (not yet unit-tested) hook patch is at /srv/vllm-research/work/cache/r18/perlayer/sitecustomize.py (copy committed at bench/dualrt/hook_perlayer/)
(`ab` becomes [nl,2]; drt_ab.req "A B [json {layer: A}]"; assign gets b["ab"][li]). Test with bench/dualrt/test_defer.py
before copying into bench/dualrt/hook/, and only when no boot is running.

Exact next steps:
1. After DQ3_DONE: defer_table.py on dq2 and dq3; post results to task 36.
2. d3 speed boot (spec, KV pinned; see "How to relaunch", tag d3, HF_SPLIT_DEFER=1): WARMUP=1 + 3 interleaved rounds
   via run_defer_points2.sh: k7 A0.7 (ref), k7 A0.5, k6 A0.7, k6 A0.5, k5 A0.7, k5 A0.5 (+ any deferral candidate
   with per-layer A). Entries "name|K 0 0|DEFER|A -1|0".
3. dq4 full 41-chunk quality for finalists (k5 if subset k5 x1.4 < 0.1; any deferral candidate; optionally k7 A0.5).
4. README.md in the frontier format: fastest point with KLD < 0.1 (with the per-domain caveat: instruction_following
   for k6), fastest inside 0.042, dNLL/top-1, per-domain, where 0.042 falls. Commit as Carmen Lok <lok@whitebad.com>, no
   co-author; push with _admin/research/heteroflash/r17/push_forgejo.sh (Forgejo only). Final post on task 36, status done.

- 14:20 PT (lane session) dq2 DONE, full 41 chunks, agree MNBT=4 (vs Q8: dNLL / KLD / top-1; worst domain; code; paired vs k7):
  k7 +0.0096 / 0.038 / 94.0%, instr 0.060, code 0.034 / 97.2%.
  k7_late8 +0.0087 / 0.030 / 94.5%, instr 0.039, code 0.028 / 97.0%, paired 0.0268 (quality ok, but 24.2 tok/s: out on speed).
  k7_defall_a035 +0.3575 / 0.444 / 77.2%, long_context 0.629, code 0.451 / 84.3%, paired 0.432. Fails KLD < 0.1.
  Subset->full calibration differs: k7 x1.4 (0.027 -> 0.038), defall x1.09 (0.407 -> 0.444); use ~1.1-1.4 when reading dq3.
- 14:20 PT dq3 started (dq3c). Queued d3 speed chain (tmux -L hfrt d3c, bench/dualrt/run_d3_chain.sh, log cache/r18/d3c.log):
  waits DQ3_DONE, boots d3 (HF_SPLIT_DEFER=1, KV pinned), WARMUP + 3 rounds of k7/k7_a05/k6/k6_a05/k5/k5_a05 -> d3_speed.jsonl, D3_DONE.

- 15:41 PT dq3 DONE (7-chunk subset, agree MNBT=4; vs Q8 dNLL / KLD / top-1; worst domain; paired vs k7):
  k7 +0.0071 / 0.027 / 95.2%; k6 +0.0093 / 0.042 / 93.7% (knowledge_qa 0.062); k5 +0.0282 / 0.079 / 90.8% (instr 0.150, paired 0.056).
  Deferral of 10-11 layers at A 0.35: dQ1 (3-12) 0.159, dQ2 (13-22) 0.132, dQ3 (23-32) 0.085, dQ4 (33-43) 0.128; even 0.191, odd 0.264
  (even+odd 0.455 vs defer-all 0.402: roughly additive, slightly super-additive per half).
  STOP RULE APPLIED: only one 10-layer group (dQ3) is under 0.1 on the subset, and at x1.1-1.4 subset->full it lands at
  ~0.09-0.12, i.e. on the bar with nothing left for anything else. => Deferral is a NO-GO for HeteroFlash; per-layer A is not built.
  k5: subset 0.079 x1.4 ~ 0.11 full and instruction_following 0.150 => likely fails 0.1; full eval only if d3 shows k5 well ahead of k6.

- 16:47 PT d3 DONE (spec, same boot, 3 rounds pooled; d3_speed.jsonl): k7 26.37 (110.4 ms/cycle), k7_a05 27.33, k6 27.56,
  k6_a05 27.85, k5 29.12 (98.2 ms), k5_a05 30.08 (96.4 ms). A 0.5 = +0.3..+1.0 tok/s (blocked 5.8 -> 9.4-10.2 ms/window).
  Anatomy (derived): each routed pick removed saves ~5-6 ms/cycle (k7 -> k5: -12 ms), so routed experts cost ~45-50 ms of the
  ~110 ms cycle and the fixed path + drafter + overheads ~60 ms. Even perfectly hidden experts cap at ~45-47 tok/s at
  2.9 tokens/cycle; 80 tok/s needs the fixed/draft ~60 ms cut AND more tokens per cycle.

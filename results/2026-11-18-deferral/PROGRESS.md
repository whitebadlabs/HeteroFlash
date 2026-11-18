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

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
- 07:07 PT: MaxQ busy with diffusion seeds 2-3 (dgrpo); poll until "run_seeds end" + no GPU compute processes.

## How to relaunch
MaxQ expert server: same command as results/2026-11-17-frontier/PROGRESS.md ("es", tmux -L hfrt), log es_d1.log.
Speed boot d1 (Spark): `tmux -L hfrt new -d -s d1 "HF_SPLIT_DEFER=1 EXTRA_ARGS=\"--kv-cache-memory-bytes 8181829386\" bash $H/bench/dualrt/boot_frontier.sh d1 > /srv/vllm-research/work/cache/r18/d1.log 2>&1"`
Points: `FO=$H/results/2026-11-18-deferral DEFER="none" bash bench/dualrt/frontier_point.sh d1 NAME "1 0 0" "0.7 -1" "7 0 0" none`
Stop: `touch results/2026-11-17-dualrt/d1.stop`.
Rules: no MaxQ GPU; Qwen qfn stays down; tmux only on `-L hfrt`; MaxQ NIC coalescing OFF is the default (never restore 50 us).

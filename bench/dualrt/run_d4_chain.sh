#!/bin/bash
# 2026-11-18 NVFP4-attention fixed path (our own RTN W4A16 checkpoint mixed_full_iq2xxs_nvfp4attn, built 11-09 from BF16)
# in the dual-memory runtime (hook_union; KDA split needs MXFP8 bytes -> KDA_HC=0; DEFER+LATE compiled in). After DQ6_DONE:
#   d4 speed (spec, KV pinned, warm-up + 3 interleaved rounds) -> d4_speed.jsonl
#   dq7 full quality (41 chunks, agree MNBT=4): k7, k5_late, k4_late -> dq7.log
# (rev 17:5x PT after dq5: weak picks added late keep quality on the subset: k5_late 0.036, k4_late 0.054)
H=/srv/vllm-research/work/HeteroFlash; cd $H; FO=$H/results/2026-11-18-deferral; OUT=$H/results/2026-11-17-dualrt
C=/srv/vllm-research/work/cache/r18; L=$C/d4.log
export M=/srv/vllm-research/models/mixed/full_iq2xxs_nvfp4attn KDA_HC=0 HF_HOOK_DIR=hook_union
until grep -q DQ6_DONE $C/dq56c.log; do sleep 30; done
while docker ps --format "{{.Names}}" | grep -q .; do sleep 30; done
echo "$(date -u +%H:%M:%SZ) boot d4 ($M)"
tmux -L hfrt new -d -s d4 "M=$M KDA_HC=0 HF_HOOK_DIR=hook_union HF_SPLIT_DEFER=1 HF_SPLIT_LATE=1 EXTRA_ARGS=\"--kv-cache-memory-bytes 8181829386\" bash $H/bench/dualrt/boot_frontier.sh d4 > $L 2>&1"
until grep -q "READY d4" $L; do grep -q "DONE d4" $L && { echo "d4 died before READY"; tail -30 $L; echo "D4_FAILED"; break; }; sleep 20; done
if grep -q "READY d4" $L; then
  echo "$(date -u +%H:%M:%SZ) d4 ready; points"
  TAG=d4 ROUNDS=3 WARMUP=1 bash bench/dualrt/run_defer_points2.sh \
    "k7_a05|7 0 0 0|none|0.5 -1|0" "k6_a05|6 0 0 0|none|0.5 -1|0" "k5_a05|5 0 0 0|none|0.5 -1|0" "k5u_a05|5 0 0 1|none|0.5 -1|0" \
    "k5_late_a05|5 0 0 0|none|0.5 -1|1" "k4_late_a05|4 0 0 0|none|0.5 -1|1" "k4_late_a10|4 0 0 0|none|1.0 0|1" > $FO/d4_points.log 2>&1
  touch $OUT/d4.stop; until grep -q "DONE d4" $L; do sleep 10; done
  python3 bench/dualrt/defer_agg.py $FO d4 1,2,3 > $FO/d4_speed.jsonl; cat $FO/d4_speed.jsonl
fi
echo "$(date -u +%H:%M:%SZ) D4_DONE"; sleep 30
while docker ps --format "{{.Names}}" | grep -q .; do sleep 30; done
echo "$(date -u +%H:%M:%SZ) dq7 start"
FULL=1 MNBT=4 KEEPCAP=k7 QTAG=dq7 bash bench/dualrt/run_defer_quality.sh \
  "k7|7 0 0 0|none|1 0 0|none 0|0.7 -1" "k5_late|5 0 0 0|none|1 0 0|none 1|0.7 -1" "k4_late|4 0 0 0|none|1 0 0|none 1|0.7 -1" > $FO/dq7.log 2>&1
echo "$(date -u +%H:%M:%SZ) DQ7_DONE"

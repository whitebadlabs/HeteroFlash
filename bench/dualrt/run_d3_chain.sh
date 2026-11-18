#!/bin/bash
# 2026-11-18 d3 speed boot (after the dq3 subset screen): drop-K x balance rule without deferral.
# Waits for DQ3_DONE, boots d3 (spec, HF_SPLIT_DEFER=1, KV pinned, same as d1/d2), warm-up + 3 interleaved rounds:
# k7/k6/k5 at A 0.7 and 0.5. Stops d3, pools rounds 1-3 -> d3_speed.jsonl, prints D3_DONE.
H=/srv/vllm-research/work/HeteroFlash; cd $H; FO=$H/results/2026-11-18-deferral; OUT=$H/results/2026-11-17-dualrt
C=/srv/vllm-research/work/cache/r18; L=$C/d3.log
until grep -q DQ3_DONE $C/dq3c.log; do sleep 30; done
sleep 30; docker ps --format "{{.Names}}" | grep -q "^hf-drt$" && { echo "hf-drt still up; abort"; exit 1; }
echo "$(date -u +%H:%M:%SZ) boot d3"
tmux -L hfrt new -d -s d3 "HF_SPLIT_DEFER=1 EXTRA_ARGS=\"--kv-cache-memory-bytes 8181829386\" bash $H/bench/dualrt/boot_frontier.sh d3 > $L 2>&1"
until grep -q "READY d3" $L; do grep -q "DONE d3" $L && { echo "d3 died before READY"; exit 1; }; sleep 20; done
echo "$(date -u +%H:%M:%SZ) d3 ready; points"
TAG=d3 ROUNDS=3 WARMUP=1 bash bench/dualrt/run_defer_points2.sh \
  "k7|7 0 0|none|0.7 -1|0" "k7_a05|7 0 0|none|0.5 -1|0" "k6|6 0 0|none|0.7 -1|0" "k6_a05|6 0 0|none|0.5 -1|0" \
  "k5|5 0 0|none|0.7 -1|0" "k5_a05|5 0 0|none|0.5 -1|0" > $FO/d3_points.log 2>&1
touch $OUT/d3.stop; until grep -q "DONE d3" $L; do sleep 10; done
python3 bench/dualrt/defer_agg.py $FO d3 1,2,3 > $FO/d3_speed.jsonl
cat $FO/d3_speed.jsonl
echo "$(date -u +%H:%M:%SZ) D3_DONE"

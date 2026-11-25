#!/bin/bash
# 2026-11-18 per-layer deferral sensitivity map (the project lead's go-ahead 17:1x PT): after DQ7_DONE, one 7-chunk subset boot
# (agree MNBT=4, FP8 checkpoint, default hook, KEEPCAP=k7): k7 reference, then k7 + deferral of ONE layer L at A 0.35
# for L = 3..43 (41 passes, ~7.5 min each). Paired KLD vs k7 per layer = the price of deferring that layer. -> dq8.log
H=/srv/vllm-research/work/HeteroFlash; cd $H; FO=$H/results/2026-11-18-deferral; C=/srv/vllm-research/work/cache/r18
until grep -q DQ7_DONE $C/d4c.log 2>/dev/null; do grep -q "died\|abort" $C/d4c.log 2>/dev/null && break; sleep 60; done
while docker ps --format "{{.Names}}" | grep -q .; do sleep 30; done
echo "$(date -u +%H:%M:%SZ) dq8 map start"
P=("k7|7 0 0|none|1 0 0|none 0|0.7 -1")
for L in $(seq 3 43); do P+=("d$L|7 0 0|none|1 0 0|$L 0|0.35 -1"); done
MNBT=4 KEEPCAP=k7 QTAG=dq8 bash bench/dualrt/run_defer_quality.sh "${P[@]}" > $FO/dq8.log 2>&1
echo "$(date -u +%H:%M:%SZ) DQ8_DONE"

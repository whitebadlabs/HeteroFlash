#!/bin/bash
# 2026-11-18 union-aware drop quality screen (hook_union): after DQ5_DONE, 7-chunk subset, agree MNBT=4 (window = verify T=4),
# KEEPCAP=k7. kNu = core picks top-N; tail picks kept when their expert is already in the window union of core picks
# (experts read = plain top-N union, so speed ~ kN). Plain kN in the same boot for paired comparison.
H=/srv/vllm-research/work/HeteroFlash; cd $H; FO=$H/results/2026-11-18-deferral; C=/srv/vllm-research/work/cache/r18
until grep -q DQ5_DONE $C/dq5c.log; do grep -q "abort" $C/dq5c.log && { echo "dq5 failed; abort"; exit 1; }; sleep 30; done
sleep 30; docker ps --format "{{.Names}}" | grep -q "^hf-drt$" && { echo "hf-drt still up; abort"; exit 1; }
echo "$(date -u +%H:%M:%SZ) dq6 start"
G="1 0 0"; A7="0.7 -1"
HF_HOOK_DIR=hook_union MNBT=4 KEEPCAP=k7 QTAG=dq6 bash bench/dualrt/run_defer_quality.sh \
  "k7|7 0 0 0|none|$G|none 0|$A7" "k6u|6 0 0 1|none|$G|none 0|$A7" "k5u|5 0 0 1|none|$G|none 0|$A7" \
  "k4u|4 0 0 1|none|$G|none 0|$A7" "k4|4 0 0 0|none|$G|none 0|$A7" "k3u|3 0 0 1|none|$G|none 0|$A7" > $FO/dq6.log 2>&1
echo "$(date -u +%H:%M:%SZ) DQ6_DONE"

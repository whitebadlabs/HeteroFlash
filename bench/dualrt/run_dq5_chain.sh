#!/bin/bash
# 2026-11-18 gate-weight deferral quality screen (the rank rule: defer only low-gate-weight picks): after D3_DONE,
# 7-chunk subset, agree MNBT=4, KEEPCAP=k7. kN_late = top-N picks by gate weight on time, the rest (low weight)
# computed on the workstation and added one layer late. Compare with plain drop kN (dq3: k6 0.042, k5 0.079).
H=/srv/vllm-research/work/HeteroFlash; cd $H; FO=$H/results/2026-11-18-deferral; C=/srv/vllm-research/work/cache/r18
until grep -q D3_DONE $C/d3c.log; do grep -q "abort\|died" $C/d3c.log && { echo "d3 failed; abort"; exit 1; }; sleep 30; done
sleep 30; docker ps --format "{{.Names}}" | grep -q "^hf-drt$" && { echo "hf-drt still up; abort"; exit 1; }
echo "$(date -u +%H:%M:%SZ) dq5 start"
G="1 0 0"; A7="0.7 -1"
MNBT=4 KEEPCAP=k7 QTAG=dq5 bash bench/dualrt/run_defer_quality.sh \
  "k7|7 0 0|none|$G|none 0|$A7" "k6_late|6 0 0|none|$G|none 1|$A7" "k5_late|5 0 0|none|$G|none 1|$A7" \
  "k4_late|4 0 0|none|$G|none 1|$A7" "k3_late|3 0 0|none|$G|none 1|$A7" > $FO/dq5.log 2>&1
echo "$(date -u +%H:%M:%SZ) DQ5_DONE"

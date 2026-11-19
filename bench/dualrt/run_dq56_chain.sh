#!/bin/bash
# 2026-11-18 rerun of dq5 + dq6 (the first dq5 start was refused: another container, wb-convo-build, was running on the Spark).
# Waits until no other container runs (never touches them), then dq5 (gate-weight deferral kN_late) and dq6 (union-aware drop).
H=/srv/vllm-research/work/HeteroFlash; cd $H; FO=$H/results/2026-11-18-deferral
waitfree() { while docker ps --format "{{.Names}}" | grep -q .; do sleep 30; done; }
G="1 0 0"; A7="0.7 -1"
waitfree; echo "$(date -u +%H:%M:%SZ) dq5 start"
MNBT=4 KEEPCAP=k7 QTAG=dq5 bash bench/dualrt/run_defer_quality.sh \
  "k7|7 0 0|none|$G|none 0|$A7" "k6_late|6 0 0|none|$G|none 1|$A7" "k5_late|5 0 0|none|$G|none 1|$A7" \
  "k4_late|4 0 0|none|$G|none 1|$A7" "k3_late|3 0 0|none|$G|none 1|$A7" > $FO/dq5.log 2>&1
echo "$(date -u +%H:%M:%SZ) DQ5_DONE"; sleep 30
waitfree; echo "$(date -u +%H:%M:%SZ) dq6 start"
HF_HOOK_DIR=hook_union MNBT=4 KEEPCAP=k7 QTAG=dq6 bash bench/dualrt/run_defer_quality.sh \
  "k7|7 0 0 0|none|$G|none 0|$A7" "k6u|6 0 0 1|none|$G|none 0|$A7" "k5u|5 0 0 1|none|$G|none 0|$A7" \
  "k4u|4 0 0 1|none|$G|none 0|$A7" "k4|4 0 0 0|none|$G|none 0|$A7" "k3u|3 0 0 1|none|$G|none 0|$A7" > $FO/dq6.log 2>&1
echo "$(date -u +%H:%M:%SZ) DQ6_DONE"

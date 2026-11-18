#!/bin/bash
# 2026-11-18 deferral follow-up (KLD < 0.1 rule): waits for the d2 -> dq2 chain (CHAIN_DONE), then one 7-chunk subset quality
# boot (agree, MNBT=4, same-boot k7 reference): drop-K baselines k6 / k5 (no deferral) and deferral of layer groups at A 0.35
# (quarters 3-12 / 13-22 / 23-32 / 33-43, even, odd; even + odd = additivity check). AB is spelled out on every pass.
H=/srv/vllm-research/work/HeteroFlash; cd $H; FO=$H/results/2026-11-18-deferral
until grep -q CHAIN_DONE /srv/vllm-research/work/cache/r18/d2c.log; do sleep 30; done
sleep 20; docker ps --format "{{.Names}}" | grep -q "^hf-drt$" && { echo "hf-drt still up; abort"; exit 1; }
r() { python3 -c "import sys; a,b,s=map(int,sys.argv[1:]); print(\",\".join(map(str,range(a,b+1,s))))" "$@"; }
K="7 0 0"; G="1 0 0"; A7="0.7 -1"; A3="0.35 -1"
echo "$(date -u +%H:%M:%SZ) dq3 start"
MNBT=4 KEEPCAP=k7 QTAG=dq3 bash bench/dualrt/run_defer_quality.sh \
  "k7|$K|none|$G|none 0|$A7" "k6|6 0 0|none|$G|none 0|$A7" "k5|5 0 0|none|$G|none 0|$A7" \
  "k7_dQ1|$K|none|$G|$(r 3 12 1) 0|$A3" "k7_dQ2|$K|none|$G|$(r 13 22 1) 0|$A3" \
  "k7_dQ3|$K|none|$G|$(r 23 32 1) 0|$A3" "k7_dQ4|$K|none|$G|$(r 33 43 1) 0|$A3" \
  "k7_deven|$K|none|$G|$(r 4 42 2) 0|$A3" "k7_dodd|$K|none|$G|$(r 3 43 2) 0|$A3" > $FO/dq3.log 2>&1
echo "$(date -u +%H:%M:%SZ) DQ3_DONE"

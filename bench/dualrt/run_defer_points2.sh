#!/bin/bash
# 2026-11-18 deferral x balance-rule points (held boot, HF_SPLIT_DEFER=1). Entries "name|DROP|DEFER|A B[|LATE]"; ROUNDS interleaved.
# WARMUP=1: one throwaway warm-up point first (k8, no deferral).
TAG=${TAG:-d1}; H=/srv/vllm-research/work/HeteroFlash; FP=$H/bench/dualrt/frontier_point.sh
export FO=$H/results/2026-11-18-deferral
P=("$@")
if [ "${WARMUP:-0}" = 1 ]; then echo "$(date -u +%H:%M:%SZ) warm-up"; DEFER="none 0" bash $FP $TAG warmup "1 0 0" "0.7 -1" "8 0 0" none essay,chat,hermes 1; fi
for r in $(seq ${R0:-1} $(( ${R0:-1} + ${ROUNDS:-2} - 1 ))); do
  for p in "${P[@]}"; do
    IFS='|' read N D DF AB LT <<< "$p"
    echo "$(date -u +%H:%M:%SZ) round $r point $N"
    DEFER="$DF ${LT:-0}" bash $FP $TAG ${N}_r$r "1 0 0" "$AB" "$D" none essay,chat,hermes 1
    python3 $H/bench/dualrt/waits.py $FO/${TAG}_${N}_r${r}_stats0.json $FO/${TAG}_${N}_r${r}_stats1.json $FO/${TAG}_${N}_r${r}_client.jsonl > $FO/${TAG}_${N}_r${r}_waits.json 2>&1
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: d[k] for k in d if k not in ('per_layer','dense')})" $FO/${TAG}_${N}_r${r}_waits.json 2>&1 | tail -1
  done
done
echo "$(date -u +%H:%M:%SZ) POINTS2_DONE"

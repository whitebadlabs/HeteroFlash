#!/bin/bash
# 2026-11-18 deferral speed points on a held boot (boot_frontier.sh with HF_SPLIT_DEFER=1): one throwaway warm-up point,
# then ROUNDS interleaved rounds. Each point: essay + 4 chat + 10 Hermes, wait timers in the stats dumps (waits.py).
#   TAG=d1 ROUNDS=3 [POINTS_ONLY="a b"] run_defer_points.sh
TAG=${TAG:-d1}; H=/srv/vllm-research/work/HeteroFlash; FP=$H/bench/dualrt/frontier_point.sh
export FO=$H/results/2026-11-18-deferral
P=(
 "kdaoff|8 0 0|none"
 "k7|7 0 0|none"
 "k7_defall|7 0 0|all"
 "kdaoff_defall|8 0 0|all"
)
if [ -n "${POINTS_ONLY:-}" ]; then Q=(); for p in "${P[@]}"; do [[ " $POINTS_ONLY " == *" ${p%%|*} "* ]] && Q+=("$p"); done; P=("${Q[@]}"); fi
if [ "${WARMUP:-1}" = 1 ]; then echo "$(date -u +%H:%M:%SZ) warm-up"; DEFER="none 0" bash $FP $TAG warmup "1 0 0" "0.7 -1" "8 0 0" none essay,chat,hermes 1; fi
for r in $(seq ${R0:-1} $(( ${R0:-1} + ${ROUNDS:-3} - 1 ))); do
  for p in "${P[@]}"; do
    IFS='|' read N D DF <<< "$p"
    echo "$(date -u +%H:%M:%SZ) round $r point $N"
    DEFER="$DF 0" bash $FP $TAG ${N}_r$r "1 0 0" "0.7 -1" "$D" none essay,chat,hermes 1
    python3 $H/bench/dualrt/waits.py $FO/${TAG}_${N}_r${r}_stats0.json $FO/${TAG}_${N}_r${r}_stats1.json $FO/${TAG}_${N}_r${r}_client.jsonl > $FO/${TAG}_${N}_r${r}_waits.json 2>&1
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: d[k] for k in d if k not in ('per_layer','dense')})" $FO/${TAG}_${N}_r${r}_waits.json 2>&1 | tail -1
  done
done
echo "$(date -u +%H:%M:%SZ) POINTS_DONE"

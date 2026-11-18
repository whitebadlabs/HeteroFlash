#!/bin/bash
# 2026-11-17 frontier speed points on held boot $TAG, ROUNDS rounds, points interleaved (ABC... ABC...) so boot drift and
# trajectory noise spread over all points. Each point: essay + 4 chat + 10 Hermes items, 1 rep per round.
TAG=${TAG:-f1}; H=/srv/vllm-research/work/HeteroFlash; FP=$H/bench/dualrt/frontier_point.sh
ALT15=26,27,28,29,30,31,33,34,36,38,40,41,42,43,44
P=(
 "base|1 1 0|8 0 0|none"
 "kdaoff|1 0 0|8 0 0|none"
 "frs|1 1 1|8 0 0|none"
 "frs_kdaoff|1 0 1|8 0 0|none"
 "alt15q3k|1 1 0|8 0 0|$ALT15"
 "tau0.08|1 1 0|8 0.08 0|none"
 "k6|1 1 0|6 0 0|none"
 "k5|1 1 0|5 0 0|none"
 "k4|1 1 0|4 0 0|none"
 "combo_frs_kdaoff_alt15|1 0 1|8 0 0|$ALT15"
 "combo_frs_kdaoff_alt15_k6|1 0 1|6 0 0|$ALT15"
 "combo_frs_kdaoff_alt15_k4|1 0 1|4 0 0|$ALT15"
 "k6_kdaoff|1 0 0|6 0 0|none"
 "k5_kdaoff|1 0 0|5 0 0|none"
 "k4_kdaoff|1 0 0|4 0 0|none"
 "tau0.08_kdaoff|1 0 0|8 0.08 0|none"
 "k6tau0.08_kdaoff|1 0 0|6 0.08 0|none"
 "alt15_k4_kdaoff|1 0 0|4 0 0|$ALT15"
 "k7_kdaoff|1 0 0|7 0 0|none"
 "tau0.04_kdaoff|1 0 0|8 0.04 0|none"
 "k7tau0.04_kdaoff|1 0 0|7 0.04 0|none"
)
# POINTS_ONLY="name1 name2": run only these points (e.g. the non-dominated ones in later rounds); R0: first round number
if [ -n "${POINTS_ONLY:-}" ]; then Q=(); for p in "${P[@]}"; do [[ " $POINTS_ONLY " == *" ${p%%|*} "* ]] && Q+=("$p"); done; P=("${Q[@]}"); fi
for r in $(seq ${R0:-1} $(( ${R0:-1} + ${ROUNDS:-3} - 1 ))); do
  for p in "${P[@]}"; do
    IFS='|' read N G D A <<< "$p"
    echo "$(date -u +%H:%M:%SZ) round $r point $N"
    bash $FP $TAG ${N}_r$r "$G" "0.7 -1" "$D" "$A" essay,chat,hermes 1
  done
done
echo "$(date -u +%H:%M:%SZ) POINTS_DONE"

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
 "combo_frs_kdaoff_alt15|1 0 1|8 0 0|$ALT15"
)
[ -n "${POINTS:-}" ] && P=("${POINTS[@]}")
for r in $(seq ${ROUNDS:-3}); do
  for p in "${P[@]}"; do
    IFS='|' read N G D A <<< "$p"
    echo "$(date -u +%H:%M:%SZ) round $r point $N"
    bash $FP $TAG ${N}_r$r "$G" "0.7 -1" "$D" "$A" essay,chat,hermes 1
  done
done
echo "$(date -u +%H:%M:%SZ) POINTS_DONE"

#!/bin/bash
# 2026-11-17 frontier, lossless lever: MaxQ i40e interrupt coalescing (default adaptive rx/tx 50 us) vs off (0 us),
# temporary (run 1 21:xx: the one-command restore failed with EINVAL, so its 'cD' arms after the first were really c0; fixed), restored to the recorded defaults at the end (and on exit). Alternating ABBA on held boot f1.
IF=enp209s0f1np1; FP=/srv/vllm-research/work/HeteroFlash/bench/dualrt/frontier_point.sh
restore() { ssh maxq "sudo -n ethtool -C $IF rx-usecs 50 tx-usecs 50; sudo -n ethtool -C $IF adaptive-rx on adaptive-tx on; ethtool -c $IF | grep -E 'Adaptive|^rx-usecs:|^tx-usecs:'"; }
trap restore EXIT
setc() { if [ $1 = c0 ]; then ssh maxq "sudo -n ethtool -C $IF adaptive-rx off adaptive-tx off rx-usecs 0 tx-usecs 0"; else restore >/dev/null; fi
         ssh maxq "ethtool -c $IF | grep -E 'Adaptive|^rx-usecs:|^tx-usecs:' | tr '\n' ' '"; echo; }
declare -A cnt=([cD]=${CD0:-0} [c0]=${C00:-0})
for r in $(seq ${ROUNDS:-2}); do
  for st in cD c0 c0 cD; do
    setc $st; cnt[$st]=$(( ${cnt[$st]} + 1 )); k=${cnt[$st]}
    for p in "kdaoff|1 0 0|8 0 0" "k4_kdaoff|1 0 0|4 0 0"; do
      IFS='|' read N G D <<< "$p"; echo "$(date -u +%H:%M:%SZ) round $r $N $st"
      ssh spark "bash $FP f1 coal_${N}_${st}_r$k '$G' '0.7 -1' '$D' none essay,chat,hermes 1" | tail -1
    done
  done
done
echo COAL_DONE

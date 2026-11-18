#!/bin/bash
# 2026-11-18 deferral d2 -> dq2 chain (survives the driving session): wait for the d2 speed boot, run the late-expert
# speed points (warm-up + 3 interleaved rounds), stop d2, pool, then the full-length quality boot dq2 (k6_late only if
# it beats same-boot k7 by > 0.5 tok/s).
H=/srv/vllm-research/work/HeteroFlash; cd $H; FO=$H/results/2026-11-18-deferral; OUT=$H/results/2026-11-17-dualrt
L=/srv/vllm-research/work/cache/r18/d2.log
until grep -q "READY d2" $L; do grep -q "DONE d2" $L && { echo "d2 died before READY"; exit 1; }; sleep 20; done
echo "$(date -u +%H:%M:%SZ) d2 ready; points"
TAG=d2 ROUNDS=3 WARMUP=1 bash bench/dualrt/run_defer_points2.sh \
  "k7|7 0 0|none|0.7 -1|0" "k7_late8|7 0 0|none|0.7 -1|1" "k6|6 0 0|none|0.7 -1|0" "k6_late|6 0 0|none|0.7 -1|1" > $FO/d2_points.log 2>&1
touch $OUT/d2.stop; until grep -q "DONE d2" $L; do sleep 10; done
python3 bench/dualrt/defer_agg.py $FO d2 1,2,3 > $FO/d2_speed.jsonl
cat $FO/d2_speed.jsonl
INC=$(python3 - $FO/d2_speed.jsonl <<'PY'
import json, sys
d = {r["point"]: r["tok_s"] for r in map(json.loads, open(sys.argv[1]))}
print(1 if d.get("k6_late", 0) - d.get("k7", 1e9) > 0.5 else 0)
PY
)
echo "$(date -u +%H:%M:%SZ) k6_late joins dq2: $INC"
ssh -o BatchMode=yes maxq nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | tail -3
P=("k7|7 0 0|none|1 0 0|none 0" "k7_late8|7 0 0|none|1 0 0|none 1")
[ "$INC" = 1 ] && P+=("k6_late|6 0 0|none|1 0 0|none 1")
P+=("k7_defall_a035|7 0 0|none|1 0 0|all 0|0.35 -1")
FULL=1 MNBT=4 KEEPCAP=k7 QTAG=dq2 bash bench/dualrt/run_defer_quality.sh "${P[@]}" > $FO/dq2.log 2>&1
echo "$(date -u +%H:%M:%SZ) CHAIN_DONE"

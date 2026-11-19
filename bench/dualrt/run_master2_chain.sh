#!/bin/bash
# 2026-11-19 ~03:45 PT: replaces run_master_chain.sh from its step 3 (launched by the lane session when step 3 starts).
# Order: (A) NVFP4 shared+dense checkpoint speed (d5) -> (B) subset quality of it (dq11) -> (C) mean-output test (fixed hook_mean)
# -> (D) per-layer map remainder (layers 5..43) -> (E) Local Inference Lab bench.
H=/srv/vllm-research/work/HeteroFlash; cd $H; C=/srv/vllm-research/work/cache/r18; FO=$H/results/2026-11-18-deferral
OUT=$H/results/2026-11-17-dualrt; SD=/srv/vllm-research/models/mixed/full_iq2xxs_nvfp4attn_sd
free() { while docker ps --format "{{.Names}}" | grep -q .; do sleep 20; done; }
free; echo "$(date -u +%H:%M:%SZ) A: d5 speed (nvfp4attn_sd)"
L=$C/d5.log
tmux -L hfrt new -d -s d5 "M=$SD KDA_HC=0 HF_HOOK_DIR=hook_union HF_SPLIT_DEFER=1 EXTRA_ARGS=\"--kv-cache-memory-bytes 8181829386\" bash $H/bench/dualrt/boot_frontier.sh d5 > $L 2>&1"
until grep -q "READY d5" $L; do grep -q "DONE d5" $L && { echo "d5 died before READY"; tail -40 $L; break; }; sleep 20; done
if grep -q "READY d5" $L; then
  TAG=d5 ROUNDS=3 WARMUP=1 bash bench/dualrt/run_defer_points2.sh \
    "k7_a05|7 0 0 0|none|0.5 -1|0" "k6_a05|6 0 0 0|none|0.5 -1|0" "k5_a05|5 0 0 0|none|0.5 -1|0" "k6u_a05|6 0 0 1|none|0.5 -1|0" > $FO/d5_points.log 2>&1
  touch $OUT/d5.stop; until grep -q "DONE d5" $L; do sleep 10; done
  python3 bench/dualrt/defer_agg.py $FO d5 1,2,3 > $FO/d5_speed.jsonl; cat $FO/d5_speed.jsonl
fi
free; echo "$(date -u +%H:%M:%SZ) B: dq11 subset quality (nvfp4attn_sd)"
M=$SD KDA_HC=0 HF_HOOK_DIR=hook_union MNBT=4 KEEPCAP=k7 QTAG=dq11 bash bench/dualrt/run_defer_quality.sh \
  "k7|7 0 0 0|none|1 0 0|none 0|0.7 -1" "k6u|6 0 0 1|none|1 0 0|none 0|0.7 -1" > $FO/dq11.log 2>&1
tail -3 $FO/dq11.log
free; echo "$(date -u +%H:%M:%SZ) C: mean chain"
sed -e 's/^until grep -q DQ9_DONE.*$/true/' bench/dualrt/run_mean_chain.sh > $C/mean_now2.sh
bash $C/mean_now2.sh > $C/meanc.log 2>&1; tail -2 $C/meanc.log
free; echo "$(date -u +%H:%M:%SZ) D: map remainder"
P=("k7|7 0 0|none|1 0 0|none 0|0.7 -1")
for Lr in $(seq 5 43); do P+=("d$Lr|7 0 0|none|1 0 0|$Lr 0|0.35 -1"); done
MNBT=4 KEEPCAP=k7 QTAG=dq8b bash bench/dualrt/run_defer_quality.sh "${P[@]}" > $FO/dq8b.log 2>&1
echo "$(date -u +%H:%M:%SZ) DQ8_DONE"
free; echo "$(date -u +%H:%M:%SZ) E: bench"
sed -e 's/^until grep -q "DQ10_DONE.*$/true/' bench/dualrt/run_bench_chain.sh > $C/bench_now.sh
bash $C/bench_now.sh > $C/benchc.log 2>&1; tail -1 $C/benchc.log
echo "$(date -u +%H:%M:%SZ) MASTER2_DONE"

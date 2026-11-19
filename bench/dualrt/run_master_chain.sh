#!/bin/bash
# 2026-11-18 23:15 PT reorder (Carmen: "is there a way to speed things up"): highest-payoff test first.
# Replaces the dq8c -> dq9c -> meanc -> benchc chain. Order: mean-output test (meanc steps) -> dq9 (NVFP4 k6u/k6/k5u full)
# -> per-layer map remainder (layers 5..43; d3, d4 done in dq8) -> Local Inference Lab bench. Each step waits for a free GPU.
H=/srv/vllm-research/work/HeteroFlash; cd $H; C=/srv/vllm-research/work/cache/r18; FO=$H/results/2026-11-18-deferral
free() { while docker ps --format "{{.Names}}" | grep -q .; do sleep 20; done; }
free; echo "$(date -u +%H:%M:%SZ) step 1: mean chain"
sed -e 's/^until grep -q DQ9_DONE.*$/true/' bench/dualrt/run_mean_chain.sh > $C/mean_now.sh
bash $C/mean_now.sh > $C/meanc.log 2>&1; tail -2 $C/meanc.log
free; echo "$(date -u +%H:%M:%SZ) step 2: dq9"
sed -e 's/^until grep -q DQ8_DONE.*$/true/' bench/dualrt/run_dq9_chain.sh > $C/dq9_now.sh
bash $C/dq9_now.sh > $C/dq9c.log 2>&1; tail -1 $C/dq9c.log
free; echo "$(date -u +%H:%M:%SZ) step 3: map remainder"
P=("k7|7 0 0|none|1 0 0|none 0|0.7 -1")
for L in $(seq 5 43); do P+=("d$L|7 0 0|none|1 0 0|$L 0|0.35 -1"); done
MNBT=4 KEEPCAP=k7 QTAG=dq8b bash bench/dualrt/run_defer_quality.sh "${P[@]}" > $FO/dq8b.log 2>&1
echo "$(date -u +%H:%M:%SZ) DQ8_DONE (dq8 d3-d4 + dq8b d5-d43)"
free; echo "$(date -u +%H:%M:%SZ) step 4: bench"
sed -e 's/^until grep -q "DQ10_DONE.*$/true/' bench/dualrt/run_bench_chain.sh > $C/bench_now.sh
bash $C/bench_now.sh > $C/benchc.log 2>&1; tail -1 $C/benchc.log
echo "$(date -u +%H:%M:%SZ) MASTER_DONE"

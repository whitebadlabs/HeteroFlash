#!/bin/bash
# 2026-11-19 04:45 PT: after the mean-output result (subset: k5 0.085 -> k5+union+mean 0.045, k4+union+mean 0.063) and the
# NVFP4 shared+dense speed (k5 35.5 tok/s), confirm the new stack first. Replaces master2 from its map step.
# (A) d6 speed: SD checkpoint + hook_mean + mean table; (B) dq12 full eval of the same; (C) map remainder d6..d43;
# (D) bench with the new best config.
H=/srv/vllm-research/work/HeteroFlash; cd $H; C=/srv/vllm-research/work/cache/r18; FO=$H/results/2026-11-18-deferral
OUT=$H/results/2026-11-17-dualrt; SD=/srv/vllm-research/models/mixed/full_iq2xxs_nvfp4attn_sd; MT=/out/expert_mean_imx5.npy
free() { while docker ps --format "{{.Names}}" | grep -q .; do sleep 20; done; }
free; echo "$(date -u +%H:%M:%SZ) A: d6 speed (SD + mean)"
L=$C/d6.log
tmux -L hfrt new -d -s d6 "M=$SD KDA_HC=0 HF_HOOK_DIR=hook_mean HF_SPLIT_MEAN=$MT HF_SPLIT_DEFER=1 EXTRA_ARGS=\"--kv-cache-memory-bytes 8181829386\" bash $H/bench/dualrt/boot_frontier.sh d6 > $L 2>&1"
until grep -q "READY d6" $L; do grep -q "DONE d6" $L && { echo "d6 died before READY"; tail -40 $L; break; }; sleep 20; done
if grep -q "READY d6" $L; then
  TAG=d6 ROUNDS=3 WARMUP=1 bash bench/dualrt/run_defer_points2.sh \
    "k6us_a05|6 0 0 1 1|none|0.5 -1|0" "k5us_a05|5 0 0 1 1|none|0.5 -1|0" "k4us_a05|4 0 0 1 1|none|0.5 -1|0" "k4_a05|4 0 0 0 0|none|0.5 -1|0" "k5_a05|5 0 0 0 0|none|0.5 -1|0" > $FO/d6_points.log 2>&1
  touch $OUT/d6.stop; until grep -q "DONE d6" $L; do sleep 10; done
  python3 bench/dualrt/defer_agg.py $FO d6 1,2,3 > $FO/d6_speed.jsonl; cat $FO/d6_speed.jsonl
fi
free; echo "$(date -u +%H:%M:%SZ) B: dq12 full eval (SD + mean)"
M=$SD KDA_HC=0 HF_HOOK_DIR=hook_mean HF_SPLIT_MEAN=$MT FULL=1 MNBT=4 KEEPCAP=k6us QTAG=dq12 bash bench/dualrt/run_defer_quality.sh \
  "k6us|6 0 0 1 1|none|1 0 0|none 0|0.7 -1" "k5us|5 0 0 1 1|none|1 0 0|none 0|0.7 -1" "k4us|4 0 0 1 1|none|1 0 0|none 0|0.7 -1" > $FO/dq12.log 2>&1
tail -2 $FO/dq12.log
free; echo "$(date -u +%H:%M:%SZ) C: map remainder (d6..d43; d3-d5 done)"
P=("k7|7 0 0|none|1 0 0|none 0|0.7 -1")
for Lr in $(seq 6 43); do P+=("d$Lr|7 0 0|none|1 0 0|$Lr 0|0.35 -1"); done
MNBT=4 KEEPCAP=k7 QTAG=dq8c bash bench/dualrt/run_defer_quality.sh "${P[@]}" > $FO/dq8c.log 2>&1
echo "$(date -u +%H:%M:%SZ) DQ8_DONE"
free; echo "$(date -u +%H:%M:%SZ) D: bench"
sed -e 's/^until grep -q "DQ10_DONE.*$/true/' \
    -e "s#^run_cfg lb_best .*#HF_SPLIT_MEAN=$MT run_cfg lb_best $SD hook_mean 0 \"5 0 0 1 1\" \"0.5 -1\"#" bench/dualrt/run_bench_chain.sh > $C/bench_now3.sh
grep -n "run_cfg lb_" $C/bench_now3.sh
bash $C/bench_now3.sh > $C/benchc.log 2>&1; tail -1 $C/benchc.log
echo "$(date -u +%H:%M:%SZ) MASTER3_DONE"

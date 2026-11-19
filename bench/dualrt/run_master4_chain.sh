#!/bin/bash
# 2026-11-19 08:00 PT: Carmen needs the Local Inference Lab bench now -> bench first, then the per-layer map remainder.
H=/srv/vllm-research/work/HeteroFlash; cd $H; C=/srv/vllm-research/work/cache/r18; FO=$H/results/2026-11-18-deferral
SD=/srv/vllm-research/models/mixed/full_iq2xxs_nvfp4attn_sd; MT=/out/expert_mean_imx5.npy
free() { while docker ps --format "{{.Names}}" | grep -q .; do sleep 20; done; }
free; echo "$(date -u +%H:%M:%SZ) bench"
sed -e 's/^until grep -q "DQ10_DONE.*$/true/' \
    -e "s#^run_cfg lb_best .*#HF_SPLIT_MEAN=$MT run_cfg lb_best $SD hook_mean 0 \"5 0 0 1 1\" \"0.5 -1\"#" bench/dualrt/run_bench_chain.sh > $C/bench_now4.sh
grep -n "run_cfg lb_\|HF_SPLIT_MEAN" $C/bench_now4.sh
bash $C/bench_now4.sh > $C/benchc.log 2>&1; tail -1 $C/benchc.log
free; echo "$(date -u +%H:%M:%SZ) map remainder (d6..d43)"
P=("k7|7 0 0|none|1 0 0|none 0|0.7 -1")
for Lr in $(seq 6 43); do P+=("d$Lr|7 0 0|none|1 0 0|$Lr 0|0.35 -1"); done
MNBT=4 KEEPCAP=k7 QTAG=dq8c bash bench/dualrt/run_defer_quality.sh "${P[@]}" > $FO/dq8c.log 2>&1
echo "$(date -u +%H:%M:%SZ) MASTER4_DONE"

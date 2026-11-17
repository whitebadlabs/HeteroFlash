#!/bin/bash
# Three balanced-eval runs at 16-token batches, then paired comparisons:
#   q_unsplit  = config B on this mailbox (every route on the workstation, no fixed-path split), bf16 partials
#   q_split_bf16 / q_split_fp32 = experts by expert (112 resident/layer, A 0.7) + lm_head + KDA q/k/v split
H=/srv/vllm-research/work/HeteroFlash; OUT=$H/results/2026-11-17-dualrt; C=/srv/vllm-research/work/cache/dualrt
cd $H
GPU_UTIL=0.84 HF_SPLIT_MODE=0 bash bench/dualrt/run_quality.sh q_unsplit
COMMON="GPU_UTIL=0.84 HF_SPLIT_MODE=1 HF_SPLIT_NRES=112 HF_SPLIT_HOT=/out/hotset_calib.json HF_SPLIT_A=0.7 HF_SPLIT_B=-1 HF_SPLIT_LMHEAD_VG=84992 HF_SPLIT_KDA_HC=32 HF_SPLIT_KEEPALIVE_US=50"
env $COMMON HF_SPLIT_F32=0 bash bench/dualrt/run_quality.sh q_split_bf16
env $COMMON HF_SPLIT_F32=1 bash bench/dualrt/run_quality.sh q_split_fp32
P() { docker run --rm -v $H:/h -v $C:/c --entrypoint python3 local/hf-spark-native:1 -I /h/bench/dualrt/paired_quality.py /h/data/q_eval_balanced_v1.tokens.json /c/cap_$1 /c/cap_$2 /h/results/2026-11-17-dualrt/paired_$1_vs_$2.json; }
P q_unsplit q_split_bf16; P q_unsplit q_split_fp32; P q_split_fp32 q_split_bf16
echo "$(date -u +%H:%M:%SZ) CHAIN DONE"

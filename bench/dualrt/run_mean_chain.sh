#!/bin/bash
# 2026-11-18/19 "never skip into nothing" (the K3.1 sensitivity study): weight x expert mean output for dropped weak picks.
# After DQ9_DONE: (1) calibration tokens from data/q_imatrix_calib.txt (5 x 2048), (2) capture MoE inputs (hook_mean, NVFP4
# attention checkpoint, agree MNBT=4, k8), (3) mean table with the runtime's own Q4_K kernel, (4) 7-chunk subset quality:
# k7, k5, k5 + mean, k5 union + mean, k4 + mean, k4 union + mean, k6 union + mean -> dq10.log
H=/srv/vllm-research/work/HeteroFlash; cd $H; FO=$H/results/2026-11-18-deferral; C=/srv/vllm-research/work/cache/r18
OUT=$H/results/2026-11-17-dualrt; IMG=local/hf-spark-native:1
export M=/srv/vllm-research/models/mixed/full_iq2xxs_nvfp4attn KDA_HC=0 HF_HOOK_DIR=hook_mean
until grep -q DQ9_DONE $C/dq9c.log; do sleep 60; done
while docker ps --format "{{.Names}}" | grep -q .; do sleep 30; done
echo "$(date -u +%H:%M:%SZ) calib tokens"
docker run --rm -v $H:/h --entrypoint python3 $IMG /h/bench/dualrt/make_calib_tokens.py /h 5 2>&1 | tail -1
echo "$(date -u +%H:%M:%SZ) capture"
rm -rf $OUT/xcap_imx5
HF_SPLIT_XCAP_DIR=/out/xcap_imx5 FULL=1 MNBT=4 TOKENS=$H/data/calib_imx5.tokens.json QTAG=mcap bash bench/dualrt/run_defer_quality.sh \
  "cap|8 0 0 0 0|none|1 0 0|none 0|0.7 -1" > $FO/mcap.log 2>&1
ls -la $OUT/xcap_imx5 | tail -8
echo "$(date -u +%H:%M:%SZ) means"
docker run --rm --gpus all --ipc=host -v $H/bench/dualrt:/drt:ro -v $OUT:/out -v /srv/vllm-research/models/gguf:/gguf:ro \
  -v /srv/vllm-research/work/cache/hf_ext:/ext -e TORCH_EXTENSIONS_DIR=/ext --entrypoint python3 $IMG \
  /drt/expert_means.py /out/xcap_imx5 /out/expert_mean_imx5.npy > $FO/mean_table.log 2>&1
tail -3 $FO/mean_table.log
[ -s $OUT/expert_mean_imx5.npy ] || { echo "no mean table; abort"; echo MEAN_FAILED; exit 1; }
echo "$(date -u +%H:%M:%SZ) dq10 start"
G="1 0 0"; A7="0.7 -1"
HF_SPLIT_MEAN=/out/expert_mean_imx5.npy MNBT=4 KEEPCAP=k7 QTAG=dq10 TOKENS=$H/data/q_eval_balanced_v1.sub7.tokens.json \
  bash bench/dualrt/run_defer_quality.sh \
  "k7|7 0 0 0 0|none|$G|none 0|$A7" "k5|5 0 0 0 0|none|$G|none 0|$A7" "k5s|5 0 0 0 1|none|$G|none 0|$A7" \
  "k5us|5 0 0 1 1|none|$G|none 0|$A7" "k4s|4 0 0 0 1|none|$G|none 0|$A7" "k4us|4 0 0 1 1|none|$G|none 0|$A7" \
  "k6us|6 0 0 1 1|none|$G|none 0|$A7" > $FO/dq10.log 2>&1
echo "$(date -u +%H:%M:%SZ) DQ10_DONE"

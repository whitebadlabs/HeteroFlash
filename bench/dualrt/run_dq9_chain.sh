#!/bin/bash
# 2026-11-18 full-eval quality of the fastest NVFP4 candidates (after the dq8 map): NVFP4 attention, hook_union, KDA_HC=0,
# agree MNBT=4, 41 chunks: k6u, k6, k5u -> dq9.log
H=/srv/vllm-research/work/HeteroFlash; cd $H; FO=$H/results/2026-11-18-deferral; C=/srv/vllm-research/work/cache/r18
export M=/srv/vllm-research/models/mixed/full_iq2xxs_nvfp4attn KDA_HC=0 HF_HOOK_DIR=hook_union
until grep -q DQ8_DONE $C/dq8c.log; do sleep 60; done
while docker ps --format "{{.Names}}" | grep -q .; do sleep 30; done
echo "$(date -u +%H:%M:%SZ) dq9 start"
FULL=1 MNBT=4 KEEPCAP=k6u QTAG=dq9 bash bench/dualrt/run_defer_quality.sh \
  "k6u|6 0 0 1|none|1 0 0|none 0|0.7 -1" "k6|6 0 0 0|none|1 0 0|none 0|0.7 -1" "k5u|5 0 0 1|none|1 0 0|none 0|0.7 -1" > $FO/dq9.log 2>&1
echo "$(date -u +%H:%M:%SZ) DQ9_DONE"

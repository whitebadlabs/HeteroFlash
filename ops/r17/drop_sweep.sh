#!/bin/bash
# 2026-11-17 frontier, Spark-only: quality of training-free expert dropping on the GPU-only FP8 build (diagnostic,
# relative to its own no-drop run, paired per token). Each point: serve_eval quality (eager, 2048-token batches),
# per-token arrays saved, then the 25 GB capture dir is deleted.   CONFIGS="tag:K:TAU:RENORM ..."
H=/srv/vllm-research/work/HeteroFlash; M=/srv/vllm-research/models/mixed/full_iq2xxs_fp8
OUT=$H/results/2026-11-17-frontier/gpuonly; mkdir -p $OUT
for C in ${CONFIGS:-base:8:0:0 tau0.04:8:0.04:0 tau0.08:8:0.08:0 k6:6:0:0}; do
  IFS=: read N K TAU RN <<< "$C"; TAG=r17_drop_$N
  echo "$(date -u +%H:%M:%SZ) start $TAG (K=$K TAU=$TAU renorm=$RN)"
  HF_OUT=$OUT HF_WINDOW_OK=1 RESTART_QFN=0 DROP_K=$K DROP_TAU=$TAU DROP_RENORM=$RN PERTOKEN=/h/results/2026-11-17-frontier/gpuonly/${TAG}_pertoken.npz \
    timeout 5400 bash $H/ops/route3/serve_eval.sh $M $TAG quality > $OUT/$TAG.log 2>&1
  mv -f $H/results/2026-11-08-route3/${TAG}_vs_q8_bal.json $OUT/ 2>/dev/null
  rm -rf /srv/vllm-research/work/cache/route3_traces/$TAG
  echo "$(date -u +%H:%M:%SZ) done $TAG: $(tail -3 $OUT/$TAG.log | tr '\n' ' ')"
done
echo "$(date -u +%H:%M:%SZ) DROP_SWEEP_DONE"

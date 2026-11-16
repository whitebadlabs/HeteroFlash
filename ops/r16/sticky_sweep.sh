#!/bin/bash
# Bet 4 step 1: beta sweep of causal window-sticky routing on the GPU-only FP8 build (diagnostic; relative to its own beta=0).
# Waits for the cycle-anatomy container to exit. Spark only.
H=/srv/vllm-research/work/HeteroFlash; M=/srv/vllm-research/models/mixed/full_iq2xxs_fp8
OUT=$H/results/2026-11-16-cycle/sticky; mkdir -p $OUT
while docker ps -q --filter name=hf-r16 | grep -q .; do sleep 30; done
for B in ${BETAS:-0 0.05 0.02 0.1}; do
  TAG=r16_sticky_b${B}_w${STICKY_W:-3}
  echo "$(date -u +%H:%M:%SZ) start $TAG"
  HF_OUT=$OUT HF_WINDOW_OK=1 RESTART_QFN=0 STICKY_BETA=$B STICKY_W=${STICKY_W:-3} timeout 5400 bash $H/ops/route3/serve_eval.sh $M $TAG quality > $OUT/$TAG.log 2>&1
  cp -f $H/results/2026-11-08-route3/${TAG}_vs_q8_bal.json $OUT/ 2>/dev/null && \rm -f $H/results/2026-11-08-route3/${TAG}_vs_q8_bal.json
  echo "$(date -u +%H:%M:%SZ) done $TAG: $(tail -4 $OUT/$TAG.log | tr '\n' ' ')"
done
echo "$(date -u +%H:%M:%SZ) SWEEP_DONE"

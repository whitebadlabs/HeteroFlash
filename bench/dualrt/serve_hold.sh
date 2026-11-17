#!/bin/bash
# Serve a split configuration (spec mode) and keep it up until $OUT/${TAG}.stop exists (clients are run separately).
set -u
TAG=$1; H=/srv/vllm-research/work/HeteroFlash; OUT=$H/results/2026-11-17-dualrt
export HF_SPLIT=1 HF_SPLIT_STATS=/out/${TAG}_stats.json
rm -f $OUT/drt_load.done $OUT/${TAG}.stop
bash $H/bench/dualrt/serve.sh $TAG spec || { docker rm -f hf-drt >/dev/null 2>&1; exit 1; }
if [ "${HF_SPLIT_NRES:-0}" -gt 0 ]; then
  touch $OUT/drt_load.req
  for i in $(seq 120); do
    curl -s -m 60 http://127.0.0.1:8095/v1/completions -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash","prompt":"Hello","max_tokens":4}' >/dev/null
    [ -s $OUT/drt_load.done ] && break; sleep 2
  done
  echo "$(date -u +%H:%M:%SZ) $(cat $OUT/drt_load.done 2>/dev/null || echo 'pool load did not report')"
fi
echo "$(date -u +%H:%M:%SZ) READY $TAG"
while [ ! -e $OUT/${TAG}.stop ]; do sleep 5; docker ps -q --filter name=^hf-drt$ | grep -q . || break; done
touch $OUT/${TAG}_stats.json.req; sleep 5
docker logs hf-drt > $OUT/${TAG}_serve.log 2>&1; docker rm -f hf-drt >/dev/null 2>&1
echo "$(date -u +%H:%M:%SZ) DONE $TAG"

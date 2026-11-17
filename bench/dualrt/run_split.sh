#!/bin/bash
# Split runtime speed run: serve (spec mode, HF_SPLIT=1), load the resident Q4_K pool after startup, client, stop.
#   run_split.sh TAG     env: HF_SPLIT_MODE/A/B/NRES/HOT/F32/LMHEAD_VG..., GPU_UTIL, REPS, ONLY
set -u
TAG=$1; H=/srv/vllm-research/work/HeteroFlash; OUT=$H/results/2026-11-17-dualrt
export HF_SPLIT=1 HF_SPLIT_STATS=/out/${TAG}_stats.json
rm -f $OUT/drt_load.done
bash $H/bench/dualrt/serve.sh $TAG spec || { docker rm -f hf-drt >/dev/null 2>&1; exit 1; }
if [ "${HF_SPLIT_NRES:-0}" -gt 0 ]; then
  touch $OUT/drt_load.req
  for i in $(seq 120); do
    curl -s -m 60 http://127.0.0.1:8095/v1/completions -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash","prompt":"Hello","max_tokens":4}' >/dev/null
    [ -s $OUT/drt_load.done ] && break; sleep 2
  done
  echo "$(date -u +%H:%M:%SZ) $(cat $OUT/drt_load.done 2>/dev/null || echo 'pool load did not report')"
fi
python3 -I $H/bench/dualrt/client.py > $OUT/${TAG}_client.jsonl 2> $OUT/${TAG}_client.err
touch $OUT/${TAG}_stats.json.req; sleep 5
docker logs hf-drt > $OUT/${TAG}_serve.log 2>&1; docker rm -f hf-drt >/dev/null 2>&1
python3 $H/bench/dualrt/summarize.py $OUT/${TAG}_client.jsonl
echo "$(date -u +%H:%M:%SZ) DONE $TAG"

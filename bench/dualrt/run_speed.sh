#!/bin/bash
# run_speed.sh TAG : serve (spec mode) + client, then stop the container. env passes through to serve.sh / client.py
set -u
TAG=$1; H=/srv/vllm-research/work/HeteroFlash; OUT=$H/results/2026-11-17-dualrt
bash $H/bench/dualrt/serve.sh $TAG spec || { docker rm -f hf-drt >/dev/null 2>&1; exit 1; }
python3 -I $H/bench/dualrt/client.py > $OUT/${TAG}_client.jsonl 2> $OUT/${TAG}_client.err
docker logs hf-drt > $OUT/${TAG}_serve.log 2>&1
docker rm -f hf-drt >/dev/null 2>&1
echo "$(date -u +%H:%M:%SZ) DONE $TAG"

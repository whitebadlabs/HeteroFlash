#!/bin/bash
# M1 step: split hook in mode 0 (every route to the CPU = config B on the new mailbox) with decode-window route counting.
# 1) calibration workload (Hermes-suite items 2-7 of 5 categories, disjoint from the measured items 0-1) -> hot-set counts
# 2) the standard measured workloads, 1 rep (mode-0 speed vs config B)
set -u
H=/srv/vllm-research/work/HeteroFlash; OUT=$H/results/2026-11-17-dualrt; TAG=${TAG:-m1_mode0_calib}
export HF_SPLIT=1 HF_SPLIT_MODE=0 HF_SPLIT_COUNT=/out/${TAG}_counts.json HF_SPLIT_STATS=/out/${TAG}_stats.json
bash $H/bench/dualrt/serve.sh $TAG spec || { docker rm -f hf-drt >/dev/null 2>&1; exit 1; }
HERMES_SLICE=2:8 ONLY=hermes REPS=1 python3 -I $H/bench/dualrt/client.py > $OUT/${TAG}_calibclient.jsonl 2>&1
touch $OUT/${TAG}_stats.json.req; sleep 5; cp $OUT/${TAG}_counts.json $OUT/${TAG}_counts_calib.json
python3 -I $H/bench/dualrt/hotset.py $OUT/${TAG}_counts_calib.json $OUT/hotset_calib.json
REPS=${REPS:-1} python3 -I $H/bench/dualrt/client.py > $OUT/${TAG}_client.jsonl 2> $OUT/${TAG}_client.err
touch $OUT/${TAG}_stats.json.req; sleep 5
docker logs hf-drt > $OUT/${TAG}_serve.log 2>&1; docker rm -f hf-drt >/dev/null 2>&1
echo "$(date -u +%H:%M:%SZ) DONE $TAG"

#!/bin/bash
# Balanced-eval quality of a split configuration at decode-size batches (max-num-batched-tokens 16, eager), so every
# position runs through the split path; logits captured by the split hook; scored against the Q8_0 BF16-proxy reference
# with src/control_kld.py compare (same tokens / KLD definition as every earlier quality number).
#   run_quality.sh TAG    env: HF_SPLIT_* (HF_SPLIT=1 always; MODE=0 = unsplit config B on this mailbox)
set -u
TAG=$1; H=/srv/vllm-research/work/HeteroFlash; OUT=$H/results/2026-11-17-dualrt; IMG=local/hf-spark-native:1
CAP=/srv/vllm-research/work/cache/dualrt/cap_$TAG; REF=/srv/vllm-research/models/gguf/kld/base_q8_0_bal.kld
rm -rf $CAP; mkdir -p $CAP; chmod 777 $CAP
export HF_SPLIT=1 HF_SPLIT_CAPTURE=/cap HF_SPLIT_STATS=/out/${TAG}_stats.json MNBT=16
EXTRA_DOCKER="-v $CAP:/cap" bash $H/bench/dualrt/serve.sh $TAG agree || { docker rm -f hf-drt >/dev/null 2>&1; exit 1; }
rm -f $OUT/drt_load.done
if [ "${HF_SPLIT_NRES:-0}" -gt 0 ]; then
  touch $OUT/drt_load.req
  for i in $(seq 120); do
    curl -s -m 300 http://127.0.0.1:8095/v1/completions -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash","prompt":"Hello","max_tokens":4}' >/dev/null
    [ -s $OUT/drt_load.done ] && break; sleep 2
  done
  echo "$(date -u +%H:%M:%SZ) $(cat $OUT/drt_load.done 2>/dev/null || echo 'pool load did not report')"
fi
python3 -I $H/bench/dualrt/quality_client.py $H/data/q_eval_balanced_v1.tokens.json http://127.0.0.1:8095/v1/completions $CAP > $OUT/${TAG}_quality_client.log 2>&1
touch $OUT/${TAG}_stats.json.req; sleep 5
docker logs hf-drt > $OUT/${TAG}_serve.log 2>&1; docker rm -f hf-drt >/dev/null 2>&1
cp $CAP/capture_rows.json $OUT/${TAG}_capture_rows.json 2>/dev/null
echo "$(date -u +%H:%M:%SZ) logits files: $(ls $CAP/logits_*.npy 2>/dev/null | wc -l)"
docker run --rm -v $H:/h -v $CAP:/cap -v $(dirname $REF):/ref:ro --entrypoint python3 $IMG /h/bench/dualrt/compare_q8.py \
  /ref/$(basename $REF) /cap /h/data/q_eval_balanced_v1.manifest.json /h/results/2026-11-17-dualrt/${TAG}_vs_q8_bal.json 2>&1 | tail -3
echo "$(date -u +%H:%M:%SZ) DONE $TAG"

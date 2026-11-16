#!/bin/bash
# Cycle anatomy of a GPU-only decode build on the Spark (one boot): acceptance, speed, torch-profiler traces.
#   cycle_anatomy.sh MODEL_DIR TAG MTP_N V2RUNNER(0|1) [PROFILE=1]
# Spark only. Does not touch qfn or any other container. Container name hf-r16.
set -u
M=$1; TAG=$2; N=$3; V2=$4
H=/srv/vllm-research/work/HeteroFlash; IMG=${IMG:-local/hf-spark-native:1}
OUT=$H/results/2026-11-16-cycle/$TAG; mkdir -p $OUT
PROF=/srv/vllm-research/work/cache/prof/r16_$TAG; rm -rf $PROF; mkdir -p $PROF
t() { date -u +%H:%M:%SZ; }
docker rm -f hf-r16 >/dev/null 2>&1
docker run -d --name hf-r16 --gpus all --ipc=host -p 127.0.0.1:8095:8095 -v $M:/m:ro -v $PROF:/prof \
  -v /srv/vllm-research/work/cache/control_chat_template.jinja:/tmpl.jinja:ro \
  -v /srv/vllm-research/work/cache/b12x:/cache/b12x -e B12X_COMPILE_CACHE_DIR=/cache/b12x \
  -e VLLM_KV_CACHE_LAYOUT=${KV_LAYOUT:-BLHNC} -e B12X_LUT_E4M3_SMEM=0 -e HF_HUB_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_USE_V2_MODEL_RUNNER=$V2 \
  --entrypoint vllm $IMG serve /m --port 8095 --host 0.0.0.0 --served-model-name GLM-5.3-Flash \
  --quantization modelopt_mixed --dtype bfloat16 --max-num-seqs 1 --gpu-memory-utilization 0.90 --trust-remote-code \
  --block-size 256 --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 --attention-backend B12X --linear-backend b12x --moe-backend b12x \
  --max-model-len 65536 --max-num-batched-tokens 8192 \
  --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $N, \"attention_backend\": \"B12X\"}" \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/prof"}' \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice --chat-template /tmpl.jinja \
  --default-chat-template-kwargs.reasoning_effort=high >/dev/null
echo "$(t) started $TAG"
for i in $(seq 240); do sleep 15; docker ps -q --filter name=hf-r16 | grep -q . || break; curl -fs http://127.0.0.1:8095/health >/dev/null && break; done
docker logs hf-r16 > $OUT/serve.log 2>&1
if ! curl -fs http://127.0.0.1:8095/health >/dev/null; then echo "$(t) NOT HEALTHY"; grep -iE "Error" $OUT/serve.log | tail -5; docker rm -f hf-r16 >/dev/null; exit 1; fi
echo "$(t) healthy; mem used $(free -g | awk '/^Mem:/{print $3}') GiB"
MTP_N=$N PROFILE=${PROFILE:-1} python3 -I $H/ops/r16/cycle_client.py | tee $OUT/client.jsonl
sleep 10; docker logs hf-r16 > $OUT/serve.log 2>&1
docker rm -f hf-r16 >/dev/null 2>&1
echo "$(t) traces: $(find $PROF -type f | wc -l) files, $(du -sh $PROF | cut -f1)"; echo "$(t) DONE $TAG"

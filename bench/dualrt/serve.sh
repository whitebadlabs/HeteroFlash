#!/bin/bash
# Bet 1 runtime integration (2026-11-17): serve the FP8 fixed-path build with routed experts as Q4_K on the workstation
# CPU (config B, mailbox, CUDA graphs) and, optionally, the dual-memory split hook (HF_SPLIT=1: GPU Q4_K hot experts +
# per-step expert rebalancing, fixed-path splits as they land). Spark only; container hf-drt; leaves other containers alone.
#   serve.sh TAG [spec|agree]    spec: MTP-3 + CUDA graphs + chat template (speed); agree: no MTP, eager,
#                                max-num-batched-tokens $MNBT (default 16) so every token runs the decode-size path
# env: HF_SPLIT=0|1, HF_SPLIT_* knobs (see split_hook.py), MTP_N (3), EXTRA_ARGS, EXTRA_DOCKER, PORT (8095)
set -u
TAG=$1; MODE=${2:-spec}
H=/srv/vllm-research/work/HeteroFlash; IMG=${IMG:-local/hf-spark-native:1}; M=${M:-/srv/vllm-research/models/mixed/full_iq2xxs_fp8}
OUT=$H/results/2026-11-17-dualrt; mkdir -p $OUT /srv/vllm-research/work/cache/hf_ext
. $H/ops/local.env
PORT=${PORT:-8095}
t() { date -u +%H:%M:%SZ; }
if docker ps --format '{{.Names}}' | grep -v '^hf-drt$' | grep -q .; then echo "refusing: other containers are running"; docker ps --format '{{.Names}}'; exit 3; fi
docker rm -f hf-drt >/dev/null 2>&1
DARGS=(-e PYTHONPATH=$( [ "${HF_SPLIT:-0}" = 1 ] && echo /drt/${HF_HOOK_DIR:-hook} || echo /rhook ) -v $H/ops/route3/remote_hook:/rhook:ro -v $H/bench/dualrt:/drt:ro
       -e HF_REMOTE_EXPERTS=/rplan.json -v $H/results/2026-11-08-route3/remote_plan_all.json:/rplan.json:ro
       -e HF_REMOTE_ADDR=$WS_IP:47201 -e HF_REMOTE_MODE=mailbox -e TORCH_EXTENSIONS_DIR=/ext -v /srv/vllm-research/work/cache/hf_ext:/ext
       -e HF_REMOTE_EXPERT_MAP=0 -e HF_REMOTE_MAX_T=0 -e HF_REMOTE_FLAGS=${HF_REMOTE_FLAGS:-3}
       -e HF_SPLIT=${HF_SPLIT:-0} -e HF_CHAIN_HOOK=${HF_CHAIN_HOOK:-} -v /srv/vllm-research/models/gguf:/gguf:ro -v $OUT:/out)
for v in $(env | grep -o '^HF_SPLIT_[A-Z0-9_]*'); do DARGS+=(-e $v=${!v}); done
if [ $MODE = spec ]; then
  VARGS=(--max-model-len 65536 --max-num-batched-tokens ${MNBT:-8192}
         --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${MTP_N:-3}, \"attention_backend\": \"B12X\"${SPEC_EXTRA:-}}"
         --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice --chat-template /tmpl.jinja
         --default-chat-template-kwargs.reasoning_effort=high)
else
  VARGS=(--enforce-eager --max-model-len 8192 --max-num-batched-tokens ${MNBT:-16} --no-enable-prefix-caching)
fi
docker run -d --name hf-drt --gpus all --ipc=host -p 127.0.0.1:$PORT:8095 -v $M:/m:ro \
  -v /srv/vllm-research/work/cache/control_chat_template.jinja:/tmpl.jinja:ro \
  -v /srv/vllm-research/work/cache/b12x:/cache/b12x -e B12X_COMPILE_CACHE_DIR=/cache/b12x \
  -e VLLM_KV_CACHE_LAYOUT=BLHNC -e B12X_LUT_E4M3_SMEM=0 -e HF_HUB_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_USE_V2_MODEL_RUNNER=${V2RUNNER:-1} "${DARGS[@]}" ${EXTRA_DOCKER:-} \
  --entrypoint vllm $IMG serve /m --port 8095 --host 0.0.0.0 --served-model-name GLM-5.3-Flash \
  --quantization modelopt_mixed --dtype bfloat16 --max-num-seqs 1 --gpu-memory-utilization ${GPU_UTIL:-0.90} --trust-remote-code \
  --block-size 256 --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 --attention-backend B12X --linear-backend b12x --moe-backend b12x \
  "${VARGS[@]}" ${EXTRA_ARGS:-} >/dev/null
echo "$(t) started $TAG ($MODE)"
for i in $(seq 240); do sleep 15; docker ps -q --filter name=^hf-drt$ | grep -q . || break; curl -fs http://127.0.0.1:$PORT/health >/dev/null && break; done
docker logs hf-drt > $OUT/${TAG}_serve.log 2>&1
if ! curl -fs http://127.0.0.1:$PORT/health >/dev/null; then echo "$(t) NOT HEALTHY"; grep -iE "Error|Traceback" $OUT/${TAG}_serve.log | tail -8; exit 1; fi
echo "$(t) healthy; mem used $(free -g | awk '/^Mem:/{print $3}') GiB"

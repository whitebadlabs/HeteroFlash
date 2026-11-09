#!/bin/bash
# Serve a Spark candidate with the control's chat template and parsers, then run the Hermes suite against it.
#   suite_candidate.sh MODEL_DIR TAG "<suite args>"   env: REMOTE_PLAN/REMOTE_ADDR (+MB=1) for CPU experts, MTP_N
set -u
M=$1; TAG=$2; SARGS=${3:-}
H=/srv/vllm-research/work/HeteroFlash; IMG=local/hf-spark-native:1; S=/srv/vllm-research/work/cache/hermes_suite
OUT=$H/results/2026-11-09-audit/suite_$TAG; mkdir -p $OUT
t() { date -u +%H:%M:%SZ; }
RARGS=()
if [ -n "${REMOTE_PLAN:-}" ]; then
  mkdir -p /srv/vllm-research/work/cache/hf_ext
  RARGS=(-e PYTHONPATH=/rhook -v $H/ops/route3/remote_hook:/rhook:ro -e HF_REMOTE_EXPERTS=/rplan.json -v $REMOTE_PLAN:/rplan.json:ro
         -e HF_REMOTE_ADDR=$REMOTE_ADDR -e HF_REMOTE_MODE=mailbox -e TORCH_EXTENSIONS_DIR=/ext -v /srv/vllm-research/work/cache/hf_ext:/ext
         -e HF_REMOTE_MAX_T=${MAX_T:-0})
fi
docker stop qfn >/dev/null 2>&1
for i in $(seq 40); do [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 100 ] && break; sleep 3; done
docker rm -f hf-r3 >/dev/null 2>&1
docker run -d --name hf-r3 --gpus all --ipc=host -p 127.0.0.1:8095:8095 -v $M:/m:ro -v /srv/vllm-research/work/cache/control_chat_template.jinja:/tmpl.jinja:ro \
  -v /srv/vllm-research/work/cache/b12x:/cache/b12x -e B12X_COMPILE_CACHE_DIR=/cache/b12x -e VLLM_KV_CACHE_LAYOUT=BLHNC -e B12X_LUT_E4M3_SMEM=0 \
  -e HF_HUB_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 "${RARGS[@]}" \
  --entrypoint vllm $IMG serve /m --port 8095 --host 0.0.0.0 --served-model-name GLM-5.3-Flash \
  --quantization modelopt_mixed --dtype bfloat16 --max-num-seqs 1 --gpu-memory-utilization 0.90 --trust-remote-code \
  --block-size 256 --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 --attention-backend B12X --linear-backend b12x --moe-backend b12x \
  --max-model-len 65536 --max-num-batched-tokens 8192 \
  --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${MTP_N:-3}, \"attention_backend\": \"B12X\"}" \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice --chat-template /tmpl.jinja \
  --default-chat-template-kwargs.reasoning_effort=high >/dev/null
for i in $(seq 240); do sleep 15; docker ps -q --filter name=hf-r3 | grep -q . || break; curl -fs http://127.0.0.1:8095/health >/dev/null && break; done
docker logs hf-r3 > $OUT/serve.log 2>&1
if ! curl -fs http://127.0.0.1:8095/health >/dev/null; then echo "$(t) NOT HEALTHY"; grep -iE "Error" $OUT/serve.log | tail -5; docker rm -f hf-r3 >/dev/null; docker start qfn >/dev/null; exit 1; fi
echo "$(t) healthy; running suite $SARGS"
PY="docker run --rm --network host -e PYTHONPATH=/s/pylib -e NLTK_DATA=/s/pylib/nltk_data -v $S:/s -w /s --entrypoint python3 $IMG"
$PY run_suite.py --endpoint http://127.0.0.1:8095 --model GLM-5.3-Flash --out answers_$TAG.jsonl --thinking off $SARGS 2>&1 | tail -3
$PY score.py answers_$TAG.jsonl --ref answers_control_thinkoff.jsonl 2>&1 | tail -30 | tee $OUT/score.txt
cp $S/answers_$TAG.jsonl $OUT/ 2>/dev/null
docker logs hf-r3 > $OUT/serve.log 2>&1; docker rm -f hf-r3 >/dev/null 2>&1; [ "${RESTART_QFN:-0}" = 1 ] && docker start qfn >/dev/null 2>&1
echo "$(t) SUITE_DONE $TAG; qfn $(docker inspect -f '{{.State.Status}}' qfn)"

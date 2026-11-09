#!/bin/bash
# Feasibility: the control's EXL3 3.5 bpw weights on the Spark GPU (sm_121) via vLLM + b12x, truncated to N layers so the
# weights fit. Output text is meaningless with truncation; this checks kernels run, memory per layer, and step time.
# Runs ON the Spark. Stops Qwen (qfn) for the run and restarts it.
set -u
N=${1:-20}
EXTRA=${2:-}
M=${MODEL_DIR:-/srv/vllm-research/models/exl3/GLM-5.3-Flash-EXL3-TR3-3.5bpw}
IMG=${IMG:-local/hf-spark-exl3:1}
QUANT=${QUANT:-exl3}
COMPAT=${COMPAT:-1}
OUT=/srv/vllm-research/work/HeteroFlash/results/2026-11-08-spark-exl3; mkdir -p $OUT
LOG=$OUT/truncated_L${N}_${QUANT}.log
t() { date -u +%H:%M:%SZ; }
# qfn (Qwen) backs the Hermes agent: only stop it inside a benchmark window agreed with Carmen.
if [ "${HF_WINDOW_OK:-}" != "1" ]; then echo "refusing: stopping qfn needs an agreed benchmark window (set HF_WINDOW_OK=1 only inside one)"; exit 3; fi
docker stop qfn >/dev/null 2>&1
for i in $(seq 40); do [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 100 ] && break; sleep 3; done   # wait until memory is really released
echo "$(t) mem before: $(free -g | awk '/^Mem:/{print $3" GiB used"}')"
docker rm -f hf-exl3 >/dev/null 2>&1
docker run -d --name hf-exl3 --gpus all --ipc=host -p 127.0.0.1:8095:8095 -v $M:/m:ro -v /srv/vllm-research/work/cache/b12x:/cache/b12x \
  -e B12X_COMPILE_CACHE_DIR=/cache/b12x -e HF_TRUNCATE_LAYERS=1 -e HF_B12X_COMPAT=$COMPAT -e VLLM_KV_CACHE_LAYOUT=${KV_LAYOUT:-LBNHC} -e PYTHONPATH=/trunc -v /srv/vllm-research/work/HeteroFlash/ops/spark-exl3/trunc_hook:/trunc:ro -e HF_HUB_OFFLINE=1 -e B12X_LUT_E4M3_SMEM=${LUT_SMEM:-1} -e VLLM_ENGINE_READY_TIMEOUT_S=1800 ${DOCKER_EXTRA:-} \
  --entrypoint vllm $IMG serve /m --port 8095 --host 0.0.0.0 --served-model-name t \
  --quantization $QUANT --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.85 --enforce-eager --trust-remote-code \
  --hf-overrides "{\"text_config\": {\"num_hidden_layers\": $N, \"num_nextn_predict_layers\": 0}}" $EXTRA >/dev/null
for i in $(seq 120); do
  sleep 15
  docker ps -q --filter name=hf-exl3 | grep -q . || break
  curl -fs http://127.0.0.1:8095/health >/dev/null && break
done
docker logs hf-exl3 > $LOG 2>&1
if ! curl -fs http://127.0.0.1:8095/health >/dev/null; then
  echo "$(t) NOT HEALTHY"; grep -vE "^\s*$" $LOG | grep -iE "error|exception|raise|Traceback|not supported|unsupported" | tail -15
else
  echo "$(t) healthy; mem after load: $(free -g | awk '/^Mem:/{print $3" GiB used"}')"
  grep -iE "EXL3 layer|Model loading took|model weights|KV cache|memory" $LOG | tail -6
  LP_OUT=${LP_OUT:-} python3 -I - <<'PY'
import json, time, urllib.request
def req(body):
    r = urllib.request.Request("http://127.0.0.1:8095/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); o = json.load(urllib.request.urlopen(r, timeout=600)); return o, time.perf_counter() - t0
o, dt = req({"model": "t", "prompt": "The capital of France is", "max_tokens": 4, "temperature": 0})
print("smoke ok", round(dt, 2), "s", repr(o["choices"][0]["text"])[:60])
for n in (1, 129):
    o, dt = req({"model": "t", "prompt": "Write about rivers.", "max_tokens": n, "temperature": 0, "ignore_eos": True})
    print("max_tokens", n, "wall", round(dt, 3), "s")
o, dt = req({"model": "t", "prompt": [1000 + (i % 5000) for i in range(2000)], "max_tokens": 1, "temperature": 0})
print("prefill 2000 tokens wall", round(dt, 3), "s")
import os
if os.environ.get("LP_OUT"):    # prompt logprobs of a fixed prompt, for exact A/B comparisons between hook modes
    ids = [1000 + (i * 7919) % 50000 for i in range(512)]
    o, _ = req({"model": "t", "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 5})
    json.dump(o["choices"][0]["prompt_logprobs"], open(os.environ["LP_OUT"], "w"))
    print("prompt logprobs saved", os.environ["LP_OUT"])
PY
fi
docker rm -f hf-exl3 >/dev/null 2>&1; docker start qfn >/dev/null 2>&1
echo "$(t) done; qfn: $(docker inspect -f '{{.State.Status}}' qfn)"

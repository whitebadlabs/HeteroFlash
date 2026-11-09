#!/bin/bash
# Route 3: serve a full BF16-derived MIXED_PRECISION checkpoint on the Spark (Spark-native vLLM + b12x) and measure it.
#   serve_eval.sh MODEL_DIR TAG [speed|quality|both|mtp]   (mtp: speed only, MTP speculative decoding, MTP_N tokens)
# speed:   CUDA graphs on, no hook; decode tok/s (256 tokens), cold 45K TTFT, cached 45K+32 turn (prefix cache).
# quality: eager + logits hook; balanced eval (42 x 2048 tokens) scored against the Q8_0 BF16-proxy reference,
#          same tokens and KLD definition as the EXL3 control measurement (src/control_kld.py).
# Runs ON the Spark. Stops Qwen (qfn) for the run and restarts it.
set -u
M=$1; TAG=$2; MODE=${3:-both}
H=/srv/vllm-research/work/HeteroFlash; IMG=${IMG:-local/hf-spark-native:1}
OUT=$H/results/2026-11-08-route3; mkdir -p $OUT
TR=/srv/vllm-research/work/cache/route3_traces/$TAG
REF=/srv/vllm-research/models/gguf/kld/base_q8_0_bal.kld
t() { date -u +%H:%M:%SZ; }
FLAGS="--block-size 256 --kv-cache-dtype fp8 --mamba-ssm-cache-dtype ${MAMBA_DTYPE:-bfloat16} --attention-backend B12X --linear-backend b12x --moe-backend b12x"
start() {   # start <extra docker args> -- <extra vllm args>
  docker rm -f hf-r3 >/dev/null 2>&1
  local dargs=() vargs=() sep=0
  for x in "$@"; do if [ "$x" = "--" ]; then sep=1; elif [ $sep = 0 ]; then dargs+=("$x"); else vargs+=("$x"); fi; done
  docker run -d --name hf-r3 --gpus all --ipc=host -p 127.0.0.1:8095:8095 -v $M:/m:ro \
    -v /srv/vllm-research/work/cache/b12x:/cache/b12x -e B12X_COMPILE_CACHE_DIR=/cache/b12x \
    -e VLLM_KV_CACHE_LAYOUT=${KV_LAYOUT:-BLHNC} -e B12X_LUT_E4M3_SMEM=0 -e HF_HUB_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 ${V2RUNNER:+-e VLLM_USE_V2_MODEL_RUNNER=$V2RUNNER} "${dargs[@]}" \
    --entrypoint vllm $IMG serve /m --port 8095 --host 0.0.0.0 --served-model-name GLM-5.3-Flash \
    --quantization modelopt_mixed --dtype bfloat16 --max-num-seqs 1 --gpu-memory-utilization ${GPU_UTIL:-0.90} \
    --trust-remote-code "${HFO[@]}" $FLAGS "${vargs[@]}" >/dev/null
  for i in $(seq 240); do
    sleep 15
    docker ps -q --filter name=hf-r3 | grep -q . || break
    curl -fs http://127.0.0.1:8095/health >/dev/null && return 0
  done
  return 1
}
# Checkpoints without the MTP layer (built with --max-layers 45) need NO_MTP=1 so the model does not expect it.
HFO=()
[ -n "${NO_MTP:-}" ] && HFO=(--hf-overrides '{"text_config": {"num_nextn_predict_layers": 0}}')
SPEC=()
if [ $MODE = mtp ]; then
  HFO=(); SPEC=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${MTP_N:-3}, \"attention_backend\": \"B12X\"}")
  TAG=${TAG}_mtp${MTP_N:-3}
fi
# qfn (Qwen) backs the Hermes agent: only stop it inside a benchmark window agreed with Carmen.
if [ "${HF_WINDOW_OK:-}" != "1" ]; then echo "refusing: stopping qfn needs an agreed benchmark window (set HF_WINDOW_OK=1 only inside one)"; exit 3; fi
docker stop qfn >/dev/null 2>&1
for i in $(seq 40); do [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 100 ] && break; sleep 3; done
echo "$(t) $TAG: mem before $(free -g | awk '/^Mem:/{print $3}') GiB used"
if [ $MODE != quality ]; then
  SARGS=(); SV=()
  if [ -n "${EAGER:-}" ]; then SV=(--enforce-eager); fi
  if [ -n "${REMOTE_PLAN:-}" ]; then   # runtime-v2 decode speed: remote-expert hook, short prompts only
    SARGS=(-e PYTHONPATH=/rhook -v $H/ops/route3/remote_hook:/rhook:ro -e HF_REMOTE_EXPERTS=/rplan.json
           -v $REMOTE_PLAN:/rplan.json:ro -e HF_REMOTE_ADDR=$REMOTE_ADDR)
    if [ -n "${MB:-}" ]; then           # mailbox handoff: graph-capturable, CUDA graphs stay on
      mkdir -p /srv/vllm-research/work/cache/hf_ext
      SARGS+=(-e HF_REMOTE_MODE=mailbox -e TORCH_EXTENSIONS_DIR=/ext -v /srv/vllm-research/work/cache/hf_ext:/ext -e HF_REMOTE_EXPERT_MAP=${EMAP:-0} -e HF_REMOTE_MAX_T=${MAX_T:-0})
    else
      SV=(--enforce-eager)              # python socket path needs eager mode
    fi
  fi
  if start "${SARGS[@]}" -- --max-model-len 65536 --max-num-batched-tokens 8192 "${SPEC[@]}" "${SV[@]}" ${SPEED_EXTRA:-}; then
    docker logs hf-r3 > $OUT/${TAG}_speed.log 2>&1
    echo "$(t) speed server healthy; mem $(free -g | awk '/^Mem:/{print $3}') GiB used"
    SKIP_LONG=${SKIP_LONG:-} python3 -I - <<'PY' | tee $OUT/${TAG}_speed.txt
import json, time, urllib.request
U = "http://127.0.0.1:8095/v1/completions"
def req(body, stream=False):
    r = urllib.request.Request(U, data=json.dumps(dict(body, model="GLM-5.3-Flash", temperature=0)).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); o = json.load(urllib.request.urlopen(r, timeout=1800)); return o, time.perf_counter() - t0
o, _ = req({"prompt": "Question: What is the capital of France? Answer in one sentence.\nAnswer:", "max_tokens": 40})
print("sample:", repr(o["choices"][0]["text"]))
o, _ = req({"prompt": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n", "max_tokens": 60})
print("sample:", repr(o["choices"][0]["text"]))
for _ in range(2):
    _, d1 = req({"prompt": "Write a long essay about rivers.", "max_tokens": 1, "ignore_eos": True})
    _, d2 = req({"prompt": "Write a long essay about rivers.", "max_tokens": 257, "ignore_eos": True})
print(json.dumps({"decode_tok_s": round(256 / (d2 - d1), 2), "ttft_short_s": round(d1, 3)}))
import os, random
if os.environ.get("SKIP_LONG"):
    raise SystemExit(0)
random.seed(7); ids = [random.randrange(1000, 60000) for _ in range(45000)]
_, cold = req({"prompt": ids, "max_tokens": 1})
_, cached = req({"prompt": ids + [random.randrange(1000, 60000) for _ in range(32)], "max_tokens": 1})
print(json.dumps({"cold_45k_ttft_s": round(cold, 2), "cached_45k_plus32_s": round(cached, 3)}))
PY
  else
    docker logs hf-r3 > $OUT/${TAG}_speed.log 2>&1; echo "$(t) speed server NOT healthy"; grep -iE "error|Traceback" $OUT/${TAG}_speed.log | tail -8
  fi
fi
if [ $MODE != speed ] && [ $MODE != mtp ]; then
  mkdir -p $TR/raw
  echo "{\"capture\": false, \"capture_inputs\": null, \"k_active\": 8, \"k_ref\": null, \"capture_logits\": {\"min_rows\": 1000, \"from\": 0, \"to\": 1024}, \"fixed_quant\": ${FQ_SPEC:-null}}" > $TR/control.json
  RARGS=(-e PYTHONPATH=/hook -e HF_FQ_NO_RESTORE=1)
  if [ -n "${REMOTE_PLAN:-}" ]; then   # runtime-v2 prototype: planned experts computed on the workstation CPU
    RARGS=(-e PYTHONPATH=/rhook -e HF_CHAIN_HOOK=/hook/sitecustomize.py -v $H/ops/route3/remote_hook:/rhook:ro
           -e HF_REMOTE_EXPERTS=/rplan.json -v $REMOTE_PLAN:/rplan.json:ro -e HF_REMOTE_ADDR=$REMOTE_ADDR
           -e HF_REMOTE_STATS=/tr/remote_stats.jsonl -e HF_FQ_NO_RESTORE=1)
    [ -n "${MB:-}" ] && RARGS+=(-e HF_REMOTE_MODE=mailbox -e TORCH_EXTENSIONS_DIR=/ext -v /srv/vllm-research/work/cache/hf_ext:/ext -e HF_REMOTE_EXPERT_MAP=${EMAP:-0} -e HF_REMOTE_MAX_T=${MAX_T:-0} -e HF_REMOTE_SPLIT_POS=${SPLIT_POS:-0})
    rm -f $TR/remote_stats.jsonl
  fi
  if start -e HF_TRACE_DIR=/tr/raw -e HF_TRACE_CONTROL=/tr/control.json "${RARGS[@]}" -v $TR:/tr -v $H/src/trace_hook:/hook:ro \
       -- --enforce-eager --max-model-len 4096 --max-num-batched-tokens 2048; then
    docker logs hf-r3 > $OUT/${TAG}_quality_serve.log 2>&1
    echo "$(t) quality server healthy; scoring"
    python3 -I $H/src/control_kld.py client $H/data/q_eval_balanced_v1.tokens.json http://127.0.0.1:8095/v1/completions | tail -2
    sleep 5
    docker logs hf-r3 > $OUT/${TAG}_quality_serve.log 2>&1
    docker rm -f hf-r3 >/dev/null 2>&1
    echo "$(t) logits files: $(ls $TR/raw/logits_*.npy 2>/dev/null | wc -l)"
    [ -s $TR/remote_stats.jsonl ] && { tail -1 $TR/remote_stats.jsonl; cp $TR/remote_stats.jsonl $OUT/${TAG}_remote_stats.jsonl; }
    docker run --rm -v $H:/h -v $TR:/tr -v $(dirname $REF):/ref:ro --entrypoint python3 $IMG -I /h/src/control_kld.py compare \
      /ref/$(basename $REF) /tr/raw /h/data/q_eval_balanced_v1.manifest.json /h/results/2026-11-08-route3/${TAG}_vs_q8_bal.json 2>&1 | tail -3
  else
    docker logs hf-r3 > $OUT/${TAG}_quality_serve.log 2>&1; echo "$(t) quality server NOT healthy"; grep -iE "error|Traceback" $OUT/${TAG}_quality_serve.log | tail -8
  fi
fi
docker rm -f hf-r3 >/dev/null 2>&1; docker start qfn >/dev/null 2>&1
echo "$(t) done; qfn: $(docker inspect -f '{{.State.Status}}' qfn)"

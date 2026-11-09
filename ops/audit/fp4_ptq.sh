#!/bin/bash
# FP4 (NVFP4) fixed-weight PTQ sensitivity on our stack, one boot: the trace hook fake-quantizes BF16 fixed weights in
# place (live, per control-file spec); each pass scores the balanced eval against the Q8_0 BF16 proxy.
# Model: full_iq2xxs (BF16 fixed weights, IQ2_XXS experts on GPU). Increments are vs pass "none".
set -u
H=/srv/vllm-research/work/HeteroFlash; IMG=local/hf-spark-native:1; M=/srv/vllm-research/models/mixed/full_iq2xxs
OUT=$H/results/2026-11-09-audit/fp4_ptq; mkdir -p $OUT
TR=/srv/vllm-research/work/cache/route3_traces/fp4_ptq; rm -rf $TR; mkdir -p $TR/raw
REF=/srv/vllm-research/models/gguf/kld/base_q8_0_bal.kld
t() { date -u +%H:%M:%SZ; }
ctl() { python3 -I -c "import json,sys; c={'capture': False, 'capture_inputs': None, 'k_active': 8, 'k_ref': None, 'capture_logits': {'min_rows': 1000, 'from': 0, 'to': 1024}, 'fixed_quant': json.loads(sys.argv[1])}; json.dump(c, open('$TR/control.json.tmp','w'))" "$1" && mv $TR/control.json.tmp $TR/control.json; }
ctl null
docker stop qfn >/dev/null 2>&1
for i in $(seq 40); do [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 100 ] && break; sleep 3; done
docker rm -f hf-r3 >/dev/null 2>&1
docker run -d --name hf-r3 --gpus all --ipc=host -p 127.0.0.1:8095:8095 -v $M:/m:ro -v $TR:/tr -v $H/src/trace_hook:/hook:ro \
  -e PYTHONPATH=/hook -e HF_TRACE_DIR=/tr/raw -e HF_TRACE_CONTROL=/tr/control.json \
  -v /srv/vllm-research/work/cache/b12x:/cache/b12x -e B12X_COMPILE_CACHE_DIR=/cache/b12x -e VLLM_KV_CACHE_LAYOUT=BLHNC -e B12X_LUT_E4M3_SMEM=0 \
  -e HF_HUB_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 --entrypoint vllm $IMG serve /m --port 8095 --host 0.0.0.0 --served-model-name GLM-5.3-Flash \
  --quantization modelopt_mixed --dtype bfloat16 --max-num-seqs 1 --gpu-memory-utilization 0.90 --trust-remote-code \
  --hf-overrides '{"text_config": {"num_nextn_predict_layers": 0}}' --block-size 256 --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 \
  --attention-backend B12X --linear-backend b12x --moe-backend b12x --enforce-eager --max-model-len 4096 --max-num-batched-tokens 2048 >/dev/null
for i in $(seq 240); do sleep 15; docker ps -q --filter name=hf-r3 | grep -q . || break; curl -fs http://127.0.0.1:8095/health >/dev/null && break; done
docker logs hf-r3 > $OUT/serve.log 2>&1
curl -fs http://127.0.0.1:8095/health >/dev/null || { echo "$(t) NOT HEALTHY"; grep -iE "Error" $OUT/serve.log | tail -5; docker rm -f hf-r3 >/dev/null; docker start qfn >/dev/null; exit 1; }
echo "$(t) healthy"
pass() {  # pass NAME SPEC
  ctl "$2"; sleep 2
  python3 -I -c "import json,urllib.request; urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8095/v1/completions', data=json.dumps({'model':'GLM-5.3-Flash','prompt':'warm','max_tokens':1}).encode(), headers={'Content-Type':'application/json'}))"
  sleep 5; cat $TR/raw/fixed_quant_*.json 2>/dev/null | tail -c 400; echo
  rm -f $TR/raw/logits_*.npy
  python3 -I $H/src/control_kld.py client $H/data/q_eval_balanced_v1.tokens.json http://127.0.0.1:8095/v1/completions > /dev/null
  sleep 3
  docker run --rm -v $H:/h -v $TR:/tr -v $(dirname $REF):/ref:ro --entrypoint python3 $IMG -I /h/src/control_kld.py compare \
    /ref/$(basename $REF) /tr/raw /h/data/q_eval_balanced_v1.manifest.json /h/results/2026-11-09-audit/fp4_ptq/$1.json 2>&1 | tail -1 | sed "s/^/$1 /"
}
pass none 'null'
pass nvfp4_attn '{"fmt": "nvfp4", "include": "attn"}'
pass nvfp4_shared_dense_lmhead '{"fmt": "nvfp4", "include": "(shared_expert|lm_head|mlp\\.(gate_up_proj|down_proj))"}'
pass nvfp4_all '{"fmt": "nvfp4"}'
pass fp8_all_sim '{"fmt": "fp8"}'
docker logs hf-r3 > $OUT/serve.log 2>&1; docker rm -f hf-r3 >/dev/null 2>&1; docker start qfn >/dev/null 2>&1
echo "$(t) FP4_PTQ_DONE; qfn $(docker inspect -f '{{.State.Status}}' qfn)"

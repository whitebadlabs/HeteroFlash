#!/bin/bash
# 2026-11-19 Local Inference Lab llm-inference-bench (Carmen 23:00 PT) against the HeteroFlash spec server, single stream:
# decode matrix concurrency 1 x contexts 0/16K/32K + tool-eval, for (a) best config: NVFP4 attention, k7, A 0.5 and
# (b) baseline: FP8, k8, A 0.7. After DQ10_DONE (mean chain). Bench runs in a throwaway container (pip deps there).
H=/srv/vllm-research/work/HeteroFlash; cd $H; C=/srv/vllm-research/work/cache/r18; OUT=$H/results/2026-11-17-dualrt
B=/srv/vllm-research/work/cache/llmbench/src; R=$H/results/2026-11-19-llmbench; mkdir -p $R; IMG=local/hf-spark-native:1
until grep -q "DQ10_DONE\|MEAN_FAILED" $C/meanc.log 2>/dev/null; do sleep 60; done
hi() { curl -s -m 300 http://127.0.0.1:8095/v1/completions -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash","prompt":"Hello","max_tokens":4}' >/dev/null; }
run_cfg() {  # tag ckpt hookdir kda_hc drop ab
  local tag=$1 m=$2 hd=$3 kda=$4 drop=$5 ab=$6
  while docker ps --format "{{.Names}}" | grep -q .; do sleep 30; done
  echo "$(date -u +%H:%M:%SZ) boot $tag"
  tmux -L hfrt new -d -s $tag "HF_SPLIT_MEAN=${HF_SPLIT_MEAN:-} M=$m KDA_HC=$kda HF_HOOK_DIR=$hd EXTRA_ARGS=\"--kv-cache-memory-bytes 8181829386\" bash $H/bench/dualrt/boot_frontier.sh $tag > $C/$tag.log 2>&1"
  until grep -q "READY $tag" $C/$tag.log; do grep -q "DONE $tag" $C/$tag.log && { echo "$tag died"; return 1; }; sleep 20; done
  echo "$drop" > $OUT/drt_drop.req; echo "$ab" > $OUT/drt_ab.req; echo "1 0 0" > $OUT/drt_gates.req
  for i in $(seq 60); do hi; [ -e $OUT/drt_drop.req ] || [ -e $OUT/drt_ab.req ] || [ -e $OUT/drt_gates.req ] || break; done
  echo "$(date -u +%H:%M:%SZ) $tag ready (drop [$drop] ab [$ab]); bench"
  mkdir -p $R/$tag
  timeout 7200 docker run --rm --network host -v $B:/b:ro -v $R/$tag:/r -w /r -e LLM_BENCH_NO_UPDATE_CHECK=1 --entrypoint bash $IMG -c \
    "pip install -q httpx rich psutil >/dev/null 2>&1; python3 /b/llm_decode_bench.py --host 127.0.0.1 --port 8095 --model GLM-5.3-Flash \
     --concurrency 1 --contexts 0,16384,32768 --duration 60 --display-mode plain" > $R/$tag/decode.log 2>&1
  echo "$(date -u +%H:%M:%SZ) $tag decode bench exit $?"
  timeout 7200 docker run --rm --network host -v $B:/b:ro -v $R/$tag:/r -w /r -e LLM_BENCH_NO_UPDATE_CHECK=1 --entrypoint bash $IMG -c \
    "pip install -q httpx rich psutil >/dev/null 2>&1; python3 /b/llm_decode_bench.py --host 127.0.0.1 --port 8095 --model GLM-5.3-Flash \
     --tool-eval --display-mode plain" > $R/$tag/tooleval.log 2>&1
  echo "$(date -u +%H:%M:%SZ) $tag tool-eval exit $?"
  touch $OUT/$tag.stop; until grep -q "DONE $tag" $C/$tag.log; do sleep 10; done
}
run_cfg lb_best /srv/vllm-research/models/mixed/full_iq2xxs_nvfp4attn hook_union 0 "7 0 0 0" "0.5 -1"
run_cfg lb_base /srv/vllm-research/models/mixed/full_iq2xxs_fp8 hook 32 "8 0 0" "0.7 -1"
echo "$(date -u +%H:%M:%SZ) BENCH_DONE"

#!/bin/bash
# E2E-v0: GLM-5.3-Flash (owner's BF16 -> GGUF, Q8_0 fixed weights, Q4_K routed experts) on Spark GPU + workstation CPU/RAM via
# llama.cpp RPC. Runs on the NAS (drives both machines). Stops Qwen (qfn) on the Spark for the run and restarts it afterwards.
set -u
# Machine-specific address of the workstation (git-ignored): ops/local.env with WS_IP=<workstation LAN address>
. "$(dirname "$0")/local.env"
: "${WS_IP:?set WS_IP in ops/local.env}"
F=GLM-5.3-Flash-E2E-v0-Q8attn-Q4Kexp.gguf
SP=/srv/vllm-research/models/gguf/$F
LB=/srv/vllm-research/work/llama.cpp/build/bin
OUT=/srv/vllm-research/work/HeteroFlash/results/e2e-v0-raw
REMOTE_LAYERS="blk\\.(2[4-9]|3[0-9]|4[0-4])\\.ffn_.*_exps\\.weight=RPC0[$WS_IP:50052]"
t() { date -u +%H:%M:%SZ; }
cleanup() {
  echo "$(t) cleanup: stop llama-server + rpc-server, restart Qwen"
  ssh spark 'P=$(pgrep -x llama-server); [ -n "$P" ] && kill $P; sleep 5; docker start qfn >/dev/null 2>&1; echo qfn: $(docker inspect -f "{{.State.Status}}" qfn)'
  ssh maxq 'P=$(cat /srv/models/gguf/rpc-server.pid 2>/dev/null); [ -n "$P" ] && kill $P 2>/dev/null; echo rpc stopped'
}
trap cleanup EXIT
until ssh -o ConnectTimeout=10 spark "test -s $SP"; do sleep 60; done
echo "$(t) GGUF present on Spark: $(ssh spark "stat -c %s $SP") bytes"
ssh spark "mkdir -p $OUT /srv/vllm-research/work/HeteroFlash/data"
ssh maxq 'cat /srv/cache/heteroflash/prompts/m1_calib_v1.jsonl' | ssh spark 'cat > /srv/vllm-research/work/HeteroFlash/data/m1_calib_v1.jsonl'
ssh maxq 'cat /srv/cache/heteroflash/prompts/longctx_corpus.txt' | ssh spark 'cat > /srv/vllm-research/work/HeteroFlash/data/longctx_corpus.txt'
echo "$(t) stop the workstation experiment vLLM server (its workers hold ~130 GB host RAM; rpc-server needs it)"
ssh maxq '/srv/tools/heteroflash/ops/lok/hf-run stop >/dev/null 2>&1; sleep 5; free -g | head -2'
echo "$(t) start rpc-server on the workstation (LAN-only bind, cache under /srv/cache)"
ssh maxq "mkdir -p /srv/cache/heteroflash/llama-rpc-cache; LLAMA_CACHE=/srv/cache/heteroflash/llama-rpc-cache nohup /srv/tools/llama.cpp/build/bin/ggml-rpc-server -H $WS_IP -p 50052 -t 32 -c > /srv/models/gguf/rpc-server.log 2>&1 < /dev/null & echo \$! > /srv/models/gguf/rpc-server.pid; sleep 3; tail -3 /srv/models/gguf/rpc-server.log"
echo "$(t) stop Qwen (qfn) on the Spark"
ssh spark 'docker stop qfn >/dev/null && echo qfn stopped; sleep 10; free -g | head -2'
echo "$(t) start llama-server (layers 24-44 experts -> RPC0)"
ssh spark "cd $LB && LD_LIBRARY_PATH=$LB nohup ./llama-server -m $SP --rpc $WS_IP:50052 -ngl 999 -ot '$REMOTE_LAYERS' -c 49152 --host 127.0.0.1 --port 8090 --jinja -t 20 --metrics --cache-reuse 256 > $OUT/llama-server.log 2>&1 < /dev/null & echo \$! > $OUT/llama-server.pid"
for i in $(seq 120); do
  sleep 30
  ssh spark 'pgrep -x llama-server >/dev/null' || { echo "llama-server exited"; ssh spark "tail -40 $OUT/llama-server.log"; exit 1; }
  ssh spark 'curl -fs http://127.0.0.1:8090/health >/dev/null' && break
done
ssh spark 'curl -fs http://127.0.0.1:8090/health' || { echo "not healthy after 60 min"; ssh spark "tail -40 $OUT/llama-server.log"; exit 1; }
echo "$(t) healthy; load log:"; ssh spark "grep -iE 'model size|buffer size|RPC0|CUDA0|offload' $OUT/llama-server.log | tail -12"
B="cd /srv/vllm-research/work/HeteroFlash && python3 -I src/ttft_bench.py"
echo "$(t) decode (prose holdout, 256 tokens, temperature 1)"
ssh spark "$B decode --engine llamacpp --base http://127.0.0.1:8090 --model x --prompts data/m1_calib_v1.jsonl --domains writing_chat --split holdout --limit 10 --max-tokens 256 --temperature 1.0 --out $OUT/decode_prose.jsonl 2>&1 | tail -12"
echo "$(t) TTFT cold 45K / cached 45K + 32, 128"
ssh spark "$B --engine llamacpp --base http://127.0.0.1:8090 --model x --corpus data/longctx_corpus.txt --n 45000 --suffixes 32 128 --repeats 3 --block-size 256 --out $OUT/ttft.jsonl 2>&1 | tail -30"
echo "$(t) E2E_V0_DONE"

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
OUT=${E2E_OUT:-/srv/vllm-research/work/HeteroFlash/results/e2e-v0-raw}
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
ssh -n maxq "mkdir -p /srv/cache/heteroflash/llama-rpc-cache; LLAMA_CACHE=/srv/cache/heteroflash/llama-rpc-cache setsid nohup /srv/tools/llama.cpp/build/bin/ggml-rpc-server -H $WS_IP -p 50052 -t 32 -c > /srv/models/gguf/rpc-server.log 2>&1 < /dev/null & echo \$! > /srv/models/gguf/rpc-server.pid; sleep 3; tail -3 /srv/models/gguf/rpc-server.log"
echo "$(t) stop Qwen (qfn) on the Spark"
ssh spark 'docker stop qfn >/dev/null && echo qfn stopped; sleep 10; free -g | head -2'
echo "$(t) start llama-server (all layers on CUDA0 via --tensor-split 0,1; routed experts of layers 24-44 -> RPC0)"
ssh -n spark "cd $LB; LD_LIBRARY_PATH=$LB setsid nohup ./llama-server -m $SP --rpc $WS_IP:50052 -ngl 999 --tensor-split 0,1 -ot '$REMOTE_LAYERS' -c 49152 --host 127.0.0.1 --port 8090 --jinja -t 20 --metrics --cache-reuse 256 ${LLAMA_EXTRA:-} > $OUT/llama-server.log 2>&1 < /dev/null & echo \$! > $OUT/llama-server.pid"
for i in $(seq 120); do
  sleep 30
  ssh spark 'pgrep -x llama-server >/dev/null' || { echo "llama-server exited"; ssh spark "tail -40 $OUT/llama-server.log"; exit 1; }
  ssh spark 'curl -fs http://127.0.0.1:8090/health >/dev/null' && break
done
ssh spark 'curl -fs http://127.0.0.1:8090/health' || { echo "not healthy after 60 min"; ssh spark "tail -40 $OUT/llama-server.log"; exit 1; }
ssh spark 'echo "Spark memory after load:"; free -g | head -2'
echo "$(t) healthy; load log:"; ssh spark "grep -iE 'model size|buffer size|RPC0|CUDA0|offload' $OUT/llama-server.log | tail -12"
SPARK_USED_GB=$(ssh spark "free -g | awk '/^Mem:/{print \$3}'")
echo "$(t) Spark memory in use after load: ${SPARK_USED_GB:-?} GB"
awk -v m="${SPARK_USED_GB:-0}" 'BEGIN{exit !(m>60)}' || { echo "$(t) PLACEMENT_FAIL: fixed layers not on the Spark GPU"; exit 3; }
B="cd /srv/vllm-research/work/HeteroFlash && python3 -I src/ttft_bench.py"
prof() {  # prof <delay_s> <window_s> <tag>: sample GPU util, link MB/s, rpc-server CPU in the background
( sleep $1; W=$2
  ssh spark "timeout $W nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 200" > /tmp/hf_prof_gpu.$$ 2>/dev/null &
  ssh spark "a=\$(cat /sys/class/net/enP7s7/statistics/rx_bytes); b=\$(cat /sys/class/net/enP7s7/statistics/tx_bytes); sleep $W; c=\$(cat /sys/class/net/enP7s7/statistics/rx_bytes); d=\$(cat /sys/class/net/enP7s7/statistics/tx_bytes); echo \"link rx \$(( (c-a)/($W*1000000) )) MB/s tx \$(( (d-b)/($W*1000000) )) MB/s\"" > /tmp/hf_prof_net.$$ &
  ssh maxq "pidstat -u -p \$(pgrep -x ggml-rpc-server) $W 1 | tail -1" > /tmp/hf_prof_cpu.$$ &
  wait
  { echo "PROFILE[$3]: GPU util mean $(awk '{s+=$1;n++} END {if(n) printf "%.0f%%", s/n}' /tmp/hf_prof_gpu.$$) over $(wc -l < /tmp/hf_prof_gpu.$$) samples"
    echo "PROFILE[$3]: $(head -1 /tmp/hf_prof_net.$$)"
    echo "PROFILE[$3]: rpc-server CPU $(awk '{print $8}' /tmp/hf_prof_cpu.$$)%"; } > /tmp/hf_prof_summary_$3.$$
  rm -f /tmp/hf_prof_gpu.$$ /tmp/hf_prof_net.$$ /tmp/hf_prof_cpu.$$ ) &
PROF_PID=$!
}
if [ -z "${E2E_SKIP_DECODE:-}" ]; then
echo "$(t) decode (prose holdout, 256 tokens, temperature 1) + live profile"
prof 40 20 decode
ssh spark "$B decode --engine llamacpp --base http://127.0.0.1:8090 --model x --prompts data/m1_calib_v1.jsonl --domains writing_chat --split holdout --limit 10 --max-tokens 256 --temperature 1.0 --out $OUT/decode_prose.jsonl 2>&1 | tail -12"
wait $PROF_PID 2>/dev/null; cat /tmp/hf_prof_summary_decode.$$ 2>/dev/null; rm -f /tmp/hf_prof_summary_decode.$$
fi
prof 60 60 prefill
echo "$(t) TTFT cold 45K / cached 45K + 32, 128"
ssh spark "$B --engine llamacpp --base http://127.0.0.1:8090 --model x --corpus data/longctx_corpus.txt --n 45000 --suffixes 32 128 --repeats ${E2E_REPEATS:-3} --block-size 256 --out $OUT/ttft.jsonl 2>&1 | tail -30"
wait $PROF_PID 2>/dev/null; cat /tmp/hf_prof_summary_prefill.$$ 2>/dev/null; rm -f /tmp/hf_prof_summary_prefill.$$
echo "$(t) E2E_V0_DONE"

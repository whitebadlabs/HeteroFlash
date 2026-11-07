#!/bin/bash
# Expert-precision quality gate (Q1): KL divergence of lower-bit routed experts against the current Q4_K-expert GGUF.
# Runs on the NAS and drives both machines (same topology as ops/e2e_v0.sh: Spark GPU + workstation CPU experts via RPC).
#   bash ops/quant_quality.sh base                 # imatrix + KLD reference logits from the Q4_K GGUF
#   bash ops/quant_quality.sh quant <type>         # quantize routed experts from BF16 on the workstation (e.g. iq3_xxs)
#   bash ops/quant_quality.sh eval <type>          # copy the variant to the Spark and measure KLD vs the reference
set -u
. "$(dirname "$0")/local.env"
: "${WS_IP:?set WS_IP in ops/local.env}"
MODE=$1; QT=${2:-}
LB=/srv/vllm-research/work/llama.cpp/build/bin
SG=/srv/vllm-research/models/gguf
WG=/srv/models/gguf
HF=/srv/vllm-research/work/HeteroFlash
Q=$SG/kld; OUT=$HF/results/q1-expert-bits-raw
REF=GLM-5.3-Flash-E2E-v0-Q8attn-Q4Kexp.gguf
REMOTE="blk\\.(2[4-9]|3[0-9]|4[0-4])\\.ffn_.*_exps\\.weight=RPC0[$WS_IP:50052]"
COMMON="--rpc $WS_IP:50052 -ngl 999 --tensor-split 0,1 -ot '$REMOTE' -t 20"
t() { date -u +%H:%M:%SZ; }
up() {
  ssh maxq '/srv/tools/heteroflash/ops/lok/hf-run stop >/dev/null 2>&1; true'
  ssh -n maxq "LLAMA_CACHE=/srv/cache/heteroflash/llama-rpc-cache setsid nohup /srv/tools/llama.cpp/build/bin/ggml-rpc-server -H $WS_IP -p 50052 -t 32 -c > $WG/rpc-server.log 2>&1 < /dev/null & echo \$! > $WG/rpc-server.pid; sleep 3"
  ssh spark 'docker stop qfn >/dev/null 2>&1; sleep 5'
  TRAP=1
}
down() {
  ssh spark 'pkill -x llama-imatrix; pkill -x llama-perplexity; sleep 3; docker start qfn >/dev/null 2>&1; echo qfn: $(docker inspect -f "{{.State.Status}}" qfn)'
  ssh maxq "P=\$(cat $WG/rpc-server.pid 2>/dev/null); [ -n \"\$P\" ] && kill \$P 2>/dev/null; echo rpc stopped"
}
TRAP=0; trap '[ $TRAP = 1 ] && down' EXIT
ssh spark "mkdir -p $Q $OUT"
case $MODE in
base)
  up
  echo "$(t) imatrix on the calibration split (Q4_K model as the activation source)"
  ssh spark "cd $LB && LD_LIBRARY_PATH=$LB ./llama-imatrix -m $SG/$REF $COMMON -f $HF/data/q_imatrix_calib.txt -c 512 --chunks 160 -o $Q/imatrix_q4k.gguf > $OUT/imatrix.log 2>&1; tail -3 $OUT/imatrix.log; ls -la $Q/imatrix_q4k.gguf"
  echo "$(t) KLD reference logits on the holdout split"
  ssh spark "cd $LB && LD_LIBRARY_PATH=$LB ./llama-perplexity -m $SG/$REF $COMMON -f $HF/data/q_kld_holdout.txt -c 2048 --chunks 20 --kl-divergence-base $Q/base_q4k.kld > $OUT/base_ppl.log 2>&1; grep -E 'Final estimate|PPL' $OUT/base_ppl.log | tail -2; ls -la $Q/base_q4k.kld"
  ;;
quant)
  [ -n "$QT" ] || { echo "usage: quant <type>"; exit 2; }
  ssh spark "cat $Q/imatrix_q4k.gguf" | ssh maxq "cat > $WG/imatrix_q4k.gguf"
  echo "$(t) quantize BF16 -> routed experts $QT (fixed weights Q8_0, as in the reference)"
  ssh maxq "/srv/tools/llama.cpp/build/bin/llama-quantize --imatrix $WG/imatrix_q4k.gguf --tensor-type 'ffn_.*_exps=$QT' --output-tensor-type q8_0 --token-embedding-type q8_0 $WG/GLM-5.3-Flash-BF16.gguf $WG/GLM-5.3-Flash-Q8attn-${QT}exp.gguf Q8_0 32 > $WG/quantize_$QT.log 2>&1; tail -2 $WG/quantize_$QT.log; ls -la $WG/GLM-5.3-Flash-Q8attn-${QT}exp.gguf"
  ;;
eval)
  [ -n "$QT" ] || { echo "usage: eval <type>"; exit 2; }
  V=GLM-5.3-Flash-Q8attn-${QT}exp.gguf
  if ! ssh spark "test -s $SG/$V"; then
    echo "$(t) copy $V to the Spark over the direct link (temporary HTTP server bound to the link address only)"
    ssh -n maxq "cd $WG && setsid nohup python3 -m http.server 8765 --bind $WS_IP > $WG/http_copy.log 2>&1 < /dev/null & echo \$! > $WG/http_copy.pid; sleep 2"
    ssh spark "curl -fsS -o $SG/$V.part http://$WS_IP:8765/$V && mv $SG/$V.part $SG/$V; ls -la $SG/$V"
    ssh maxq "kill \$(cat $WG/http_copy.pid) 2>/dev/null; rm -f $WG/http_copy.pid"
  fi
  up
  echo "$(t) KLD of $QT experts vs the Q4_K reference"
  ssh spark "cd $LB && LD_LIBRARY_PATH=$LB ./llama-perplexity -m $SG/$V $COMMON -f $HF/data/q_kld_holdout.txt -c 2048 --chunks 20 --kl-divergence-base $Q/base_q4k.kld --kl-divergence > $OUT/kld_$QT.log 2>&1; sed -n '/KL divergence statistics/,\$p' $OUT/kld_$QT.log | head -30"
  ;;
esac
echo "$(t) Q1_${MODE}_DONE"

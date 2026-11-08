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
RL=${Q1_REMOTE:-2[4-9]|3[0-9]|4[0-5]}   # remote MoE layers (regex); Q6_K needs 20-45
REMOTE="blk\\.(${RL})\\.ffn_.*_exps\\.weight=RPC0[$WS_IP:50052]"
KB=${Q1_BASE:-base_q4k.kld}
COMMON="--rpc $WS_IP:50052 -ngl 999 --tensor-split 0,1 -ot '$REMOTE' -t 20"
t() { date -u +%H:%M:%SZ; }
up() {
  ssh maxq '/srv/tools/heteroflash/ops/lok/hf-run stop >/dev/null 2>&1; true'
  ssh -n maxq "LLAMA_CACHE=/srv/cache/heteroflash/llama-rpc-cache setsid nohup /srv/tools/llama.cpp/build/bin/ggml-rpc-server -H $WS_IP -p 50052 -t 32 -c > $WG/rpc-server.log 2>&1 < /dev/null & echo \$! > $WG/rpc-server.pid; sleep 3"
  ssh spark 'docker stop qfn >/dev/null 2>&1; sleep 5'
  TRAP=1
}
down() {
  ssh spark 'pkill -f "[l]lama-imatrix"; pkill -f "[l]lama-perplexity"; sleep 3; docker start qfn >/dev/null 2>&1; echo qfn: $(docker inspect -f "{{.State.Status}}" qfn)'
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
  ssh spark "cd $LB && LD_LIBRARY_PATH=$LB ./llama-perplexity -m $SG/$REF $COMMON -f $HF/${Q1_TEXT:-data/q_kld_holdout.txt} -c 2048 --chunks ${Q1_CHUNKS:-20} --kl-divergence-base $Q/base_q4k.kld > $OUT/base_ppl.log 2>&1; grep -E 'Final estimate|PPL' $OUT/base_ppl.log | tail -2; ls -la $Q/base_q4k.kld"
  ;;
quant)
  [ -n "$QT" ] || { echo "usage: quant <type>"; exit 2; }
  ssh spark "cat $Q/imatrix_q4k.gguf" | ssh maxq "cat > $WG/imatrix_q4k.gguf"
  echo "$(t) quantize BF16 -> routed experts $QT (fixed weights Q8_0, as in the reference)"
  ssh maxq "/srv/tools/llama.cpp/build/bin/llama-quantize --imatrix $WG/imatrix_q4k.gguf --tensor-type 'blk\.([0-9]|[1-3][0-9]|4[0-4])\.ffn_.*_exps=$QT' --tensor-type 'blk\.45\.ffn_.*_exps=q4_K' --output-tensor-type q8_0 --token-embedding-type q8_0 $WG/GLM-5.3-Flash-BF16.gguf $WG/GLM-5.3-Flash-Q8attn-${QT}exp.gguf Q8_0 32 > $WG/quantize_$QT.log 2>&1; tail -2 $WG/quantize_$QT.log; ls -la $WG/GLM-5.3-Flash-Q8attn-${QT}exp.gguf" ; ssh maxq "grep -q 'failed to quantize' $WG/quantize_$QT.log" && { echo "$(t) QUANT_FAILED"; ssh maxq "rm -f $WG/GLM-5.3-Flash-Q8attn-${QT}exp.gguf"; exit 4; }
  ;;
eval)
  [ -n "$QT" ] || { echo "usage: eval <type>"; exit 2; }
  V=GLM-5.3-Flash-Q8attn-${QT}exp.gguf; [ "$QT" = q4k ] && V=$REF
  if ! ssh spark "test -s $SG/$V"; then
    echo "$(t) copy $V to the Spark over the direct link (temporary HTTP server bound to the link address only)"
    ssh -n maxq "cd $WG; setsid nohup python3 -m http.server 8765 --bind $WS_IP > $WG/http_copy.log 2>&1 < /dev/null & echo \$! > $WG/http_copy.pid; sleep 2"
    ssh spark "curl -fsS -o $SG/$V.part http://$WS_IP:8765/$V && mv $SG/$V.part $SG/$V; ls -la $SG/$V"
    ssh maxq "kill \$(cat $WG/http_copy.pid) 2>/dev/null; rm -f $WG/http_copy.pid"
  fi
  up
  echo "$(t) KLD of $QT experts vs the Q4_K reference"
  ssh spark "cd $LB && LD_LIBRARY_PATH=$LB ./llama-perplexity -m $SG/$V $COMMON -f $HF/${Q1_TEXT:-data/q_kld_holdout.txt} -c 2048 --chunks ${Q1_CHUNKS:-20} --kl-divergence-base $Q/$KB --kl-divergence > $OUT/kld_${QT}_vs_${KB%.kld}.log 2>&1; sed -n '/KL divergence statistics/,\$p' $OUT/kld_${QT}_vs_${KB%.kld}.log | head -30"
  ;;
refbase)
  [ -n "$QT" ] || { echo "usage: refbase <type>"; exit 2; }
  V=GLM-5.3-Flash-Q8attn-${QT}exp.gguf
  if ! ssh spark "test -s $SG/$V"; then
    echo "$(t) copy $V to the Spark over the direct link"
    ssh -n maxq "cd $WG; setsid nohup python3 -m http.server 8765 --bind $WS_IP > $WG/http_copy.log 2>&1 < /dev/null & echo \$! > $WG/http_copy.pid; sleep 2"
    ssh spark "curl -fsS -o $SG/$V.part http://$WS_IP:8765/$V && mv $SG/$V.part $SG/$V; ls -la $SG/$V"
    ssh maxq "kill \$(cat $WG/http_copy.pid) 2>/dev/null; rm -f $WG/http_copy.pid"
  fi
  up
  echo "$(t) KLD reference logits from $QT experts (remote layers $RL)"
  ssh spark "cd $LB && LD_LIBRARY_PATH=$LB ./llama-perplexity -m $SG/$V $COMMON -f $HF/${Q1_TEXT:-data/q_kld_holdout.txt} -c 2048 --chunks ${Q1_CHUNKS:-20} --kl-divergence-base $Q/base_${QT}${Q1_SET:+_$Q1_SET}.kld > $OUT/base_${QT}${Q1_SET:+_$Q1_SET}_ppl.log 2>&1; grep -E 'Final estimate' $OUT/base_${QT}${Q1_SET:+_$Q1_SET}_ppl.log | tail -1; ls -la $Q/base_${QT}${Q1_SET:+_$Q1_SET}.kld"
  ;;
esac
echo "$(t) Q1_${MODE}_DONE"

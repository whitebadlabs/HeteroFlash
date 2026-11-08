#!/bin/bash
# Q1 group attribution: IQ3_XXS on one disjoint layer group, Q4_K (same imatrix) elsewhere; KLD vs the Q8_0 BF16 proxy.
# Runs on the NAS; quantizes on the workstation CPU, evaluates on the Spark GPU + workstation RPC (ops/quant_quality.sh).
# No Max-Q GPU use.
set -u
D=$(dirname "$0"); T=${Q1G_LOG:-/tmp}
W=/srv/models/gguf; IM=$W/imatrix_q4k.gguf; BF=$W/GLM-5.3-Flash-BF16.gguf
declare -A G
G[G0_q4kim]=""                                              # control: all Q4_K with the same imatrix
G[GS_iq3small]="3|4|5|6|7|8|9|11|12|13|15|32|35|37|39"      # 14 low-output layers
G[GM_iq3mid]="10|14|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31"   # 18 layers
G[GL_iq3late]="33|34|36|38|40|41|42|43|44"                  # 9 late layers (+ see note)
# note: 3..44 = 42 layers; GS 15 + GM 18 + GL 9 = 42
for name in G0_q4kim GS_iq3small GM_iq3mid GL_iq3late; do
  L=${G[$name]}
  OUTF=$W/GLM-5.3-Flash-Q8attn-${name}exp.gguf
  if [ -n "$L" ]; then TT="--tensor-type 'blk\\.(${L})\\.ffn_.*_exps=iq3_xxs' --tensor-type 'ffn_.*_exps=q4_K'"; else TT="--tensor-type 'ffn_.*_exps=q4_K'"; fi
  echo "$(date -u +%H:%M:%SZ) quantize $name (iq3 layers: ${L:-none})"
  ssh maxq "test -s $OUTF || { /srv/tools/llama.cpp/build/bin/llama-quantize --imatrix $IM $TT --output-tensor-type q8_0 --token-embedding-type q8_0 $BF $OUTF.part Q8_0 32 > $W/quantize_$name.log 2>&1 && mv $OUTF.part $OUTF; }; grep -c 'failed to quantize' $W/quantize_$name.log; ls -la $OUTF"
  ssh maxq "test -s $OUTF" || { echo "QUANT_FAILED $name"; continue; }
  [ -n "${Q1G_SKIP_EVAL:-}" ] && continue
  Q1_BASE=base_q8_0.kld bash $D/quant_quality.sh eval $name > $T/q1g_eval_$name.log 2>&1
  echo "== $name vs Q8_0 exit $?"
  ssh spark "grep -E 'Mean ln\(PPL|Mean    KLD|99.0%   KLD|Same top p' /srv/vllm-research/work/HeteroFlash/results/q1-expert-bits-raw/kld_${name}_vs_base_q8_0.log"
done
echo "Q1_GROUPS_DONE"

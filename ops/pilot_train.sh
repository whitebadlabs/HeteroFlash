#!/bin/bash
# QAT pilot step 2 (docs/QAT_PILOT_PROTOCOL.md): IQ3_XXS QAT on two representative layers x two held-out domains.
# Layers: 43 (high 3-bit sensitivity, late/large-output, group GL) and 32 (low, small-output, group GS).
# Folds: hold out knowledge_qa (most 3-bit-sensitive domain) and code. The capture covers code, instruction_following,
# knowledge_qa, math (+ partial multilingual); writing_chat is missing (capture hit its 55-min cap).
# Budget: 3000 s training per run, 2 runs in parallel (one per Max-Q GPU) -> ~3.6 GPU-h incl. setup.
set -u
T=${PT_LOG:-/tmp}; C=/srv/cache/heteroflash
BUDGET=${PT_BUDGET:-3000}
t() { date -u +%H:%M:%SZ; }
ssh maxq "touch $C/GPU_RUN.lock"
trap 'ssh maxq "rm -f $C/GPU_RUN.lock"; echo "$(t) lock released"' EXIT
while ssh maxq 'pgrep -x ggml-rpc-server >/dev/null || docker -H unix:///run/user/1000/docker.sock ps -q --filter name=glm53-trace | grep -q .'; do sleep 30; done
run() {   # run <gpu> <layer> <holdout>
  ssh maxq "export DOCKER_HOST=unix:///run/user/1000/docker.sock; docker run --rm --device nvidia.com/gpu=$1 --ipc=host -v $C:/c -v /srv/models:/m:ro -v /srv/tools/llama.cpp/gguf-py:/g:ro --entrypoint python3 localhost/vllm-glm53f-exl3-sm120-turbo:r6 /c/analysis/qat_iq3xxs.py train --gguf-py /g --gguf-iq3 /m/gguf/GLM-5.3-Flash-Q8attn-iq3_xxsexp.gguf --gguf-q4k /m/gguf/GLM-5.3-Flash-Q8attn-G0_q4kimexp.gguf --model /m/GLM-5.3-Flash-BF16 --captures '/c/traces/raw/inputs_[a-z]*_*t*_*.npz' --layer $2 --holdout $3 --budget-s $BUDGET --passes 5 --lr-w 1e-4 --lr-s 1e-2 --batch 512 --eval-tokens 2048 --init ptq --out /c/qat_pilot 2>&1 | grep -E '^\{' " > $T/pilot_L$2_$3.log 2>&1
}
for fold in knowledge_qa code; do
  echo "$(t) fold holdout=$fold: layer 43 on GPU0, layer 32 on GPU1"
  run 0 43 $fold & P1=$!
  run 1 32 $fold & P2=$!
  wait $P1 $P2
  for L in 43 32; do echo "== L$L hold-$fold"; cut -c1-400 $T/pilot_L${L}_$fold.log | tail -7; done
done
echo "$(t) PILOT_TRAIN_DONE"

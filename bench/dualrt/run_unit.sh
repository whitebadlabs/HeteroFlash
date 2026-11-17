#!/bin/bash
# GPU Q4_K kernel unit test + bandwidth (Spark GPU, container), against the workstation server if reachable.
H=/srv/vllm-research/work/HeteroFlash; OUT=$H/results/2026-11-17-dualrt; . $H/ops/local.env
docker run --rm --gpus all --ipc=host -v $H/bench/dualrt:/drt:ro -v /srv/vllm-research/models/gguf:/gguf:ro -v $OUT:/out \
  -v /srv/vllm-research/work/cache/hf_ext:/ext -e TORCH_EXTENSIONS_DIR=/ext -e OUT=/out/unit_q4k_L${1:-3}.json \
  --entrypoint python3 local/hf-spark-native:1 /drt/test_q4k.py ${1:-3} $WS_IP:47201

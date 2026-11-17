#!/bin/bash
H=/srv/vllm-research/work/HeteroFlash; OUT=$H/results/2026-11-17-dualrt; . $H/ops/local.env
docker run --rm --gpus all --ipc=host -v $H/bench/dualrt:/drt:ro -v /srv/vllm-research/models/mixed/full_iq2xxs_fp8:/m:ro -v $OUT:/out \
  -v /srv/vllm-research/work/cache/hf_ext:/ext -e TORCH_EXTENSIONS_DIR=/ext -e OUT=/out/unit_dense_lmhead.json \
  --entrypoint python3 local/hf-spark-native:1 /drt/test_dense.py $WS_IP:47201

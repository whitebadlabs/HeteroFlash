#!/bin/bash
# Bet 5 step 1: KDA-state checkpoint granularity vs the cached 45K+32 turn (config-only), GPU-only FP8 build, MTP-3, V2 runner.
H=/srv/vllm-research/work/HeteroFlash; M=/srv/vllm-research/models/mixed/full_iq2xxs_fp8
run() { echo "$(date -u +%H:%M:%SZ) start $1"; CACHED_ONLY=1 CACHED_REPS=6 PROFILE=0 "${@:2}" bash $H/ops/r16/cycle_anatomy.sh $M $1 3 1; }
run cached_mbs256 env EXTRA_ARGS="--mamba-block-size 256"
run cached_mnbt9728 env MNBT=9728
echo "$(date -u +%H:%M:%SZ) CACHED_SWEEP_DONE"

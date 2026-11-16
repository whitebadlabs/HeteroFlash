#!/bin/bash
# Bet 2 step 1: FR-Spec draft-only head (all 20,548 calibration-seen tokens), GPU-only FP8 build, MTP-3, V2 runner. Waits for the cached sweep.
H=/srv/vllm-research/work/HeteroFlash; M=/srv/vllm-research/models/mixed/full_iq2xxs_fp8
until grep -q CACHED_SWEEP_DONE /srv/vllm-research/work/cache/r16_cached.log 2>/dev/null; do sleep 30; done
EXTRA_DOCKER="-v $H/ops/r16/frspec:/frs:ro -v $H/results/2026-11-16-cycle/frspec_ids_32768.json:/frs_ids.json:ro -e PYTHONPATH=/frs -e HF_FRSPEC_IDS=/frs_ids.json -e HF_FRSPEC_CKPT=/m" \
  CACHED_REPS=2 bash $H/ops/r16/cycle_anatomy.sh $M frspec20k_fp8_mtp3 3 1
grep -h "frspec" $H/results/2026-11-16-cycle/frspec20k_fp8_mtp3/serve.log | head -3
echo "$(date -u +%H:%M:%SZ) FRSPEC_DONE"

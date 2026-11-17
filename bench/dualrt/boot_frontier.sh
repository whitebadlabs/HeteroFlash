#!/bin/bash
# 2026-11-17 frontier speed boot: the m4 final configuration + compiled-in run-time knobs (expert drop no-op, FR-Spec
# off, alt table off) so every frontier point is switched on ONE boot.   boot_frontier.sh TAG   (env MTP_N, NRES)
H=/srv/vllm-research/work/HeteroFlash; cd $H
export GPU_UTIL=0.84 HF_SPLIT_MODE=1 HF_SPLIT_NRES=${NRES:-112} HF_SPLIT_HOT=/out/hotset_calib.json HF_SPLIT_A=0.7 HF_SPLIT_B=-1 \
  HF_SPLIT_LMHEAD_VG=84992 HF_SPLIT_KDA_HC=32 HF_SPLIT_KEEPALIVE_US=50 HF_SPLIT_DROP=1 HF_SPLIT_FRSPEC=/frs_ids.json HF_SPLIT_FRS_ON=0
export SPEC_EXTRA=', "use_local_argmax_reduction": true'
export EXTRA_DOCKER="--cap-add NET_ADMIN -v $H/results/2026-11-16-cycle/frspec_ids_${FRS_K:-32768}.json:/frs_ids.json:ro"
exec bash bench/dualrt/serve_hold.sh $1

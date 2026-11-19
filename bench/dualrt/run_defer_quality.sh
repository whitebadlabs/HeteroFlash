#!/bin/bash
# 2026-11-18 deferral quality screen: agree mode (eager, 16-token windows), boot_frontier knobs + HF_SPLIT_DEFER/LATE,
# 7-chunk subset (one per domain), first pass kept as the same-boot reference (kld_pair). Args: "name|DROP|ALT|GATES|DEFER" ...
H=/srv/vllm-research/work/HeteroFlash; cd $H
export GPU_UTIL=0.84 HF_SPLIT_MODE=1 HF_SPLIT_NRES=${NRES:-112} HF_SPLIT_HOT=/out/hotset_calib.json HF_SPLIT_A=0.7 HF_SPLIT_B=-1 \
  HF_SPLIT_LMHEAD_VG=84992 HF_SPLIT_KDA_HC=${KDA_HC:-32} HF_SPLIT_KEEPALIVE_US=50 HF_SPLIT_DROP=1 HF_SPLIT_F32=0 HF_SPLIT_DEFER=1 HF_SPLIT_LATE=1
export EXTRA_DOCKER="--cap-add NET_ADMIN" FO=$H/results/2026-11-18-deferral
if [ "${FULL:-0}" != 1 ]; then export TOKENS=$H/data/q_eval_balanced_v1.sub7.tokens.json HF_CHUNKS=0,6,12,18,24,30,36; fi
exec bash bench/dualrt/run_quality_multi.sh ${QTAG:-dq1} "$@"

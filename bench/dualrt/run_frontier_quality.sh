#!/bin/bash
# 2026-11-17 frontier quality: the f1 speed boot configuration (boot_frontier.sh knobs, expert drop compiled in) in agree
# mode (eager, 16-token batches), several passes on one boot via run_quality_multi.sh.  Args: "name|DROP|ALT|GATES" ...
H=/srv/vllm-research/work/HeteroFlash; cd $H
export GPU_UTIL=0.84 HF_SPLIT_MODE=1 HF_SPLIT_NRES=${NRES:-112} HF_SPLIT_HOT=/out/hotset_calib.json HF_SPLIT_A=0.7 HF_SPLIT_B=-1 \
  HF_SPLIT_LMHEAD_VG=84992 HF_SPLIT_KDA_HC=32 HF_SPLIT_KEEPALIVE_US=50 HF_SPLIT_DROP=1 HF_SPLIT_F32=0
export EXTRA_DOCKER="--cap-add NET_ADMIN"
exec bash bench/dualrt/run_quality_multi.sh ${QTAG:-fq1} "$@"

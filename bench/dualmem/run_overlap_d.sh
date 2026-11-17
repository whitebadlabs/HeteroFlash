#!/bin/bash
# Leg D (control for leg B): Spark Arm busy threads WITHOUT DRAM traffic (512 KB cache-resident buffer), 20 threads, while
# the GPU streams [10,30]; then leg E: 16 Arm threads streaming 8 GB (4 cores left free) with the GPU.
R=/srv/vllm-research/work/HeteroFlash
GPU="docker run --rm --name hfdual-gpu --user \$(id -u):\$(id -g) -e HOME=/tmp --device nvidia.com/gpu=all -v $R/bench/dualmem:/b:ro --entrypoint python3 vllm/vllm-openai:v0.32.0 /b/stream_gpu.py"
S=$(( $(date +%s) + 40 ))
ssh spark "OMP_NUM_THREADS=20 OMP_PLACES=cores OMP_PROC_BIND=close OMP_WAIT_POLICY=active /srv/vllm-research/runs/hfdual/stream_cpu 0.0005 $S 40 1 spark_arm_cacheonly" > legD_arm.txt 2>/dev/null &
ssh spark "$GPU $((S+10)) 20 1 spark_gpu" > legD_gpu.txt 2>/dev/null &
wait; cat legD_*.txt | sort -k2 -n > overlap_legD.txt
S=$(( $(date +%s) + 40 ))
ssh spark "OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close /srv/vllm-research/runs/hfdual/stream_cpu 8 $S 40 1 spark_arm16" > legE_arm.txt 2>/dev/null &
ssh spark "$GPU $((S+10)) 20 1 spark_gpu" > legE_gpu.txt 2>/dev/null &
wait; cat legE_*.txt | sort -k2 -n > overlap_legE.txt

#!/bin/bash
# Coordinated overlap legs (run from the NAS). Output: overlap_legA.txt, overlap_legB.txt (tag epoch GB/s).
R=/srv/vllm-research/work/HeteroFlash
GPU="docker run --rm --name hfdual-gpu --user \$(id -u):\$(id -g) -e HOME=/tmp --device nvidia.com/gpu=all -v $R/bench/dualmem:/b:ro --entrypoint python3 vllm/vllm-openai:v0.32.0 /b/stream_gpu.py"
# Leg A: Spark GPU [0,60], MaxQ CPU 32 cores [15,45]
S=$(( $(date +%s) + 40 ))
ssh spark "$GPU $S 60 1 spark_gpu" > legA_gpu.txt 2>/dev/null &
ssh maxq "cd /srv/cache/heteroflash/dualmem && OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OMP_WAIT_POLICY=active ./stream_cpu 8 $((S+15)) 30 1 maxq_cpu" > legA_maxq.txt 2>/dev/null &
wait; cat legA_gpu.txt legA_maxq.txt | sort -k2 -n > overlap_legA.txt
# Leg B: Spark Arm CPU 20 threads [0,40], Spark GPU [10,30]
S=$(( $(date +%s) + 40 ))
ssh spark "OMP_NUM_THREADS=20 OMP_PLACES=cores OMP_PROC_BIND=close /srv/vllm-research/runs/hfdual/stream_cpu 8 $S 40 1 spark_arm" > legB_arm.txt 2>/dev/null &
ssh spark "$GPU $((S+10)) 20 1 spark_gpu" > legB_gpu.txt 2>/dev/null &
wait; cat legB_arm.txt legB_gpu.txt | sort -k2 -n > overlap_legB.txt

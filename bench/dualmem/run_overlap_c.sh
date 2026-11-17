#!/bin/bash
# Leg C: MaxQ CPU 32 cores [0,40] alone then with Spark GPU + Spark Arm both streaming [10,30].
R=/srv/vllm-research/work/HeteroFlash
GPU="docker run --rm --name hfdual-gpu --user \$(id -u):\$(id -g) -e HOME=/tmp --device nvidia.com/gpu=all -v $R/bench/dualmem:/b:ro --entrypoint python3 vllm/vllm-openai:v0.32.0 /b/stream_gpu.py"
S=$(( $(date +%s) + 40 ))
ssh maxq "cd /srv/cache/heteroflash/dualmem && OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OMP_WAIT_POLICY=active ./stream_cpu 8 $S 40 1 maxq_cpu" > legC_maxq.txt 2>/dev/null &
ssh spark "$GPU $((S+10)) 20 1 spark_gpu" > legC_gpu.txt 2>/dev/null &
ssh spark "OMP_NUM_THREADS=20 OMP_PLACES=cores OMP_PROC_BIND=close /srv/vllm-research/runs/hfdual/stream_cpu 8 $((S+10)) 20 1 spark_arm" > legC_arm.txt 2>/dev/null &
wait; cat legC_*.txt | sort -k2 -n > overlap_legC.txt

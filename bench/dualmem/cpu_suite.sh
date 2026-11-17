#!/bin/bash
# MaxQ CPU suite for Bet 1: GEMV at split shapes (f = CPU share) + per-CCD read bandwidth. Run on the MaxQ.
cd /srv/cache/heteroflash/dualmem; export OMP_WAIT_POLICY=active OMP_PROC_BIND=close
out=cpu_suite_$(date -u +%Y%m%dT%H%M%SZ).txt; exec > >(tee $out) 2>&1
echo "# $(date -u) $(uname -n) gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) epp=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)"
G() { echo -n "$1 "; shift; ./gemv_cpu "$@"; }
# --- per-CCD read bandwidth (Q2). CCD k = cores 8k..8k+7, SMT siblings +32.
ccd_places() { local n=$1 smt=$2 p=""; for k in $(seq 0 $((n-1))); do for c in $(seq $((8*k)) $((8*k+7))); do p+="{$c},"; [ $smt = 1 ] && p+="{$((c+32))},"; done; done; echo ${p%,}; }
for smt in 0 1; do for n in 1 2 3 4; do P=$(ccd_places $n $smt); NT=$(( 8*n*(1+smt) ))
  echo -n "membw ccds=$n smt=$smt threads=$NT "; OMP_NUM_THREADS=$NT OMP_PLACES="$P" ./stream_cpu 8 $(date +%s) 5 1 bw 2>/dev/null | awk '{print $3}' | sort -n | tail -1; done; done
echo -n "membw spread2perCCD threads=8 "; OMP_NUM_THREADS=8 OMP_PLACES="{0},{1},{8},{9},{16},{17},{24},{25}" ./stream_cpu 8 $(date +%s) 5 1 bw 2>/dev/null | awk '{print $3}' | sort -n | tail -1
echo -n "membw spread4perCCD threads=16 "; OMP_NUM_THREADS=16 OMP_PLACES="{0},{1},{2},{3},{8},{9},{10},{11},{16},{17},{18},{19},{24},{25},{26},{27}" ./stream_cpu 8 $(date +%s) 5 1 bw 2>/dev/null | awk '{print $3}' | sort -n | tail -1
echo -n "membw 1ccd4cores threads=4 "; OMP_NUM_THREADS=4 OMP_PLACES="{0},{1},{2},{3}" ./stream_cpu 8 $(date +%s) 5 1 bw 2>/dev/null | awk '{print $3}' | sort -n | tail -1
# --- fp8 KDA-half GEMV vs CCD count
for n in 1 2 3 4; do P=$(ccd_places $n 0); OMP_NUM_THREADS=$((8*n)) OMP_PLACES="$P" G "kda_f0.5_ccds=$n" fp8 4 100 34 4096x4096 4096x4096 4096x4096 4096x4096; done
# --- split shapes at 32 cores (f = CPU share of heads / columns / vocab)
export OMP_NUM_THREADS=32 OMP_PLACES=cores
for T in 1 4; do
 G kda_f0.5 fp8 $T 200 34 4096x4096 4096x4096 4096x4096 4096x4096
 G kda_f0.25 fp8 $T 200 34 2048x4096 2048x4096 2048x4096 4096x2048
 G kda_small_bf16_f0.5 bf16 $T 200 34 128x4096 128x4096 64x4096 4096x128 4096x128
 G mla_f0.5_fp8 fp8 $T 200 11 1536x4096 512x4096 8192x1536 4096x8192
 G mla_f0.5_bf16 bf16 $T 200 11 16384x512 4096x1536 128x4096 32x4096
 G dense_mlp_f0.5 fp8 $T 200 3 6144x4096 6144x4096 4096x6144
 G shared_exp_f0.5 fp8 $T 400 42 1024x4096 1024x4096 4096x1024
 G shared_exp_full fp8 $T 400 42 2048x4096 2048x4096 4096x2048
 G lm_head_f0.5 fp8 $T 40 6 77440x4096
done
# --- SMT 64 threads and 16 cores on the KDA shape
OMP_NUM_THREADS=64 OMP_PLACES=threads G kda_f0.5_64thr fp8 4 200 34 4096x4096 4096x4096 4096x4096 4096x4096
OMP_NUM_THREADS=16 OMP_PLACES="$(ccd_places 2 0)" G kda_f0.5_16cores_2ccd fp8 4 200 34 4096x4096 4096x4096 4096x4096 4096x4096
echo "# done $(date -u)"

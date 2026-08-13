#!/bin/bash
# Runs N_RESAMPLES subsample fits (fit_single_resample.py) with bounded parallelism.
# Each resample is single-threaded (BLAS threads capped at 1, pure-Python heat loop anyway),
# so parallelism here comes from running multiple OS processes concurrently, not from
# multithreading a single fit.
set -e
cd "$(dirname "$0")"

N_RESAMPLES=20
MAX_PARALLEL=4
PY=/opt/anaconda3/envs/bobaT_env_py3.13/bin/python3

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

for seed in $(seq 1 $N_RESAMPLES); do
    while [ "$(jobs -r -p | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 5
    done
    nice -n 10 "$PY" fit_single_resample.py --seed "$seed" > "bootstrap_runs/log_seed${seed}.log" 2>&1 &
done
wait
echo "=== all $N_RESAMPLES resamples done ==="

#!/bin/bash
# Launch decode.py as a transient systemd user unit. Usage: run.sh UNIT LOG decode.py-args...
DATA=NAS_PATH
HERE=/tmp/ecorr-decode-20261118
u=$1 log=$2; shift 2
systemd-run --user --unit="$u" --collect -p WorkingDirectory="$DATA" -p Nice=15 \
  -E PYTHONPATH="$DATA/pylib" -E ECORR_SRC="$DATA/repo/src" -E OPENBLAS_NUM_THREADS=2 -E OMP_NUM_THREADS=2 \
  -E PYTHONDONTWRITEBYTECODE=1 \
  /bin/bash -c "/usr/bin/python3 -u $HERE/decode.py $* >> $log 2>&1"

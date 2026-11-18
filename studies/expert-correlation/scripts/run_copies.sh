#!/bin/bash
# Launch copies.py as a transient systemd user unit (survives session restarts). Usage: run.sh UNIT LOG -- copies.py args...
# Multiple invocations can be chained by passing ';;' between arg groups (each resumes from the checkpoint).
DATA=NAS_PATH
HERE=/tmp/ecorr-copies-20261118
u=$1 log=$2; shift 2
cmd=""
grp=""
for x in "$@"; do
  if [ "$x" = ";;" ]; then cmd="$cmd /usr/bin/python3 -u $HERE/copies.py $grp >> $log 2>&1 &&"; grp=""; else grp="$grp $x"; fi
done
cmd="$cmd /usr/bin/python3 -u $HERE/copies.py $grp >> $log 2>&1"
systemd-run --user --unit="$u" --collect -p WorkingDirectory="$DATA" -p Nice=15 \
  -E PYTHONPATH="$DATA/pylib" -E ECORR_SRC="$DATA/repo/src" -E OPENBLAS_NUM_THREADS=2 -E OMP_NUM_THREADS=2 \
  /bin/bash -c "$cmd"

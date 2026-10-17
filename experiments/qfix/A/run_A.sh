#!/usr/bin/env bash
: "${CCS_WORK:=$(cd "$(dirname "$0")/../.." && pwd)}"  # path configuration of the released copy (README)
set -u
cd ${CCS_WORK}/qfix/A
exec 9>${CCS_WORK}/gpu1.extra.lock
flock -n 9 || { echo LOCK_BUSY; exit 3; }
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=${CCS_PYTHON:-python}
echo "sha $(sha256sum qfix_a.py)"; date +%s > t_start
[ -f out/synthetic.json ] || $PY -B qfix_a.py --synthetic
$PY -B qfix_a.py --run
date +%s > t_end
$PY -B qfix_a.py --analyze && touch ALL_DONE || touch ANALYZE_FAIL

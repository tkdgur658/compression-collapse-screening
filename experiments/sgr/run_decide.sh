#!/bin/bash
: "${CCS_WORK:=$(cd "$(dirname "$0")/.." && pwd)}"  # path configuration of the released copy (README)
# PREREG_SGR decision step: runs ONCE, only after out/ALL_DONE. Pure python (CPU, 1 thread).
set -u
BASE=${CCS_WORK}/sgr
PY=${CCS_PYTHON:-python}
[ -f $BASE/out/ALL_DONE ] || { echo "refuse: $BASE/out/ALL_DONE missing (run not finished)"; exit 6; }
[ -e $BASE/RESULT_SGR.json ] && { echo "refuse: RESULT_SGR.json exists (decide runs once)"; exit 7; }
[ -e $BASE/DECIDE_STARTED ] && { echo "refuse: DECIDE_STARTED exists (decide runs once)"; exit 7; }
exec 9>"$BASE/decide.lock"; flock -n 9 || { echo "decide already running"; exit 4; }
( cd $BASE && sha256sum -c --quiet FROZEN_SHA256SUMS ) || { echo "frozen sha mismatch; refusing"; exit 3; }
date '+%F %T %Z' > $BASE/DECIDE_STARTED
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
cd $BASE
nice -n 10 $PY -B $BASE/sgr_decide.py --decide 2>&1 | tee $BASE/decide.log
( cd $BASE && sha256sum RESULT_SGR.json RESULT_SGR.md ) >> $BASE/decide.log 2>&1

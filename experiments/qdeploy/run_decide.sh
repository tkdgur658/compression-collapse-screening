#!/bin/bash
: "${CCS_WORK:=$(cd "$(dirname "$0")/.." && pwd)}"  # path configuration of the released copy (README)
# F3 QDEPLOY decision step: runs ONCE, only after out/ALL_DONE. Pure python (CPU, 1 thread).
# Usage: cd ${CCS_WORK}/qdeploy && bash run_decide.sh
set -u
BASE=${CCS_WORK}/qdeploy
PY=${CCS_PYTHON:-python}
[ -f $BASE/out/ALL_DONE ] || { echo "refuse: $BASE/out/ALL_DONE missing (run not finished)"; exit 6; }
ls $BASE/out/STOPPED_* >/dev/null 2>&1 && { echo "refuse: STOPPED marker present"; exit 6; }
[ -e $BASE/RESULT_QDEPLOY.json ] && { echo "refuse: RESULT_QDEPLOY.json exists (decide runs once)"; exit 7; }
[ -e $BASE/DECIDE_STARTED ] && { echo "refuse: DECIDE_STARTED exists (decide runs once)"; exit 7; }
exec 9>"$BASE/decide.lock"; flock -n 9 || { echo "decide already running"; exit 4; }
( cd $BASE && sha256sum -c --quiet FROZEN_SHA256SUMS ) || { echo "frozen sha mismatch; refusing"; exit 3; }
date '+%F %T %Z' > $BASE/DECIDE_STARTED
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd $BASE
nice -n 10 $PY -B $BASE/qdeploy_analyze.py --decide 2>&1 | tee $BASE/decide.log
( cd $BASE && sha256sum RESULT_QDEPLOY.json RESULT_QDEPLOY.md ) >> $BASE/decide.log 2>&1

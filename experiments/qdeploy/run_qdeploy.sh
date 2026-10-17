#!/bin/bash
: "${CCS_WORK:=$(cd "$(dirname "$0")/.." && pwd)}"  # path configuration of the released copy (README)
# F3 QDEPLOY measurement run (PREREG_QDEPLOY.md section 8-9). gpu-server GPU 0 only.
# Usage: cd ${CCS_WORK}/qdeploy && nohup bash run_qdeploy.sh [--only TAG,...] > out/nohup_run.out 2>&1 < /dev/null &
# Restart-safe: per-item DONE/FAIL markers in out/items are skipped. ALL_DONE is written only when all 330 items are DONE or FAIL
# and no STOPPED_* marker exists. Stop rules live in qdeploy.py (6 GPU-h compute cap, 2nd OOM, PBE sha/E1, foreign process -> wait).
set -u
BASE=${CCS_WORK}/qdeploy
PY=${CCS_PYTHON:-python}
LOG=$BASE/out/run.log
mkdir -p $BASE/out/items
exec 9>"$BASE/qdeploy.lock"; flock -n 9 || { echo "qdeploy already running"; exit 4; }
exec 8>"$BASE/gpu0.lock"; flock -n 8 || { echo "gpu0.lock held"; exit 4; }
( cd $BASE && sha256sum -c --quiet FROZEN_SHA256SUMS ) || { echo "$(date '+%F %T %Z') frozen sha mismatch; refusing" | tee -a "$LOG"; exit 3; }
ls $BASE/out/STOPPED_* >/dev/null 2>&1 && { echo "$(date '+%F %T %Z') STOPPED marker present; refusing (study lead decision needed)" | tee -a "$LOG"; exit 5; }
export CUDA_VISIBLE_DEVICES=0 QD_PHYS_GPU=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
echo "$(date '+%F %T %Z') run start pid=$$ args=$* sha $(sha256sum $BASE/qdeploy.py | cut -c1-16)" >> "$LOG"
cd $BASE
$PY -B $BASE/qdeploy.py --run "$@" >> "$LOG" 2>&1
rc=$?
n=$(ls $BASE/out/items/*.DONE $BASE/out/items/*.FAIL 2>/dev/null | wc -l)
echo "$(date '+%F %T %Z') run end rc=$rc markers=$n/330" >> "$LOG"
if [ "$#" -eq 0 ] && [ "$rc" -eq 0 ] && [ "$n" -eq 330 ] && ! ls $BASE/out/STOPPED_* >/dev/null 2>&1; then
  date '+%F %T %Z' > $BASE/out/ALL_DONE
  echo "$(date '+%F %T %Z') ALL_DONE written" >> "$LOG"
fi

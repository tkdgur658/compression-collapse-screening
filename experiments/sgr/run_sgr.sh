#!/bin/bash
: "${CCS_WORK:=$(cd "$(dirname "$0")/.." && pwd)}"  # path configuration of the released copy (README)
# PREREG_SGR GPU worker. Usage: bash run_sgr.sh gpu0   |   bash run_sgr.sh gpu1  (gpu1 waits for abl/ALL_DONE inside)
set -u
W=${1:?worker gpu0|gpu1}
BASE=${CCS_WORK}/sgr
PY=${CCS_PYTHON:-python}
case $W in gpu0) G=0;; gpu1) G=1;; *) echo "bad worker"; exit 2;; esac
( cd $BASE && sha256sum -c --quiet FROZEN_SHA256SUMS ) || { echo "frozen sha mismatch; refusing"; exit 3; }
exec 9>"$BASE/$W.lock"; flock -n 9 || { echo "$W already running"; exit 4; }
export CUDA_VISIBLE_DEVICES=$G SGR_PHYS_GPU=$G OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p $BASE/out/items $BASE/logs
echo "$(date '+%F %T %Z') $W start pid=$$ sha $(sha256sum $BASE/sgr_run.py | cut -c1-16)" >> $BASE/logs/run_$W.log
cd $BASE
$PY -B $BASE/sgr_run.py --run --worker $W >> $BASE/logs/run_$W.log 2>&1
rc=$?
echo "$(date '+%F %T %Z') $W end rc=$rc" >> $BASE/logs/run_$W.log

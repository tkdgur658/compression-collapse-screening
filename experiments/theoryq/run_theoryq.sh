#!/bin/bash
: "${CCS_WORK:=$(cd "$(dirname "$0")/.." && pwd)}"  # path configuration of the released copy (README)
# THEORY_Q (PREREG_THEORYQ.md section 7). gpu-server CPU only, 2 threads, nice 10. Never touches a GPU.
# Usage: cd ${CCS_WORK}/theoryq && nohup bash run_theoryq.sh {synth|run|decide} > out_<mode>.log 2>&1 < /dev/null &
set -u
BASE=${CCS_WORK}/theoryq
PY=${CCS_PYTHON:-python}
MODE=${1:-}
exec 9>"$BASE/theoryq.lock"; flock -n 9 || { echo "theoryq already running"; exit 4; }
( cd $BASE && sha256sum -c --quiet FROZEN_SHA256SUMS ) || { echo "$(date '+%F %T %Z') frozen sha mismatch; refusing"; exit 3; }
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
cd $BASE
mkdir -p $BASE/out
case "$MODE" in
  synth)
    nice -n 10 $PY -B $BASE/theoryq_synth.py ;;
  run)
    ls $BASE/out/STOPPED_* >/dev/null 2>&1 && { echo "STOPPED marker present; refusing"; exit 5; }
    echo "$(date '+%F %T %Z') run start pid=$$"
    nice -n 10 $PY -B $BASE/theoryq_run.py --run
    rc=$?
    n=$(ls $BASE/out/*.DONE $BASE/out/*.FAIL 2>/dev/null | wc -l)
    echo "$(date '+%F %T %Z') run end rc=$rc markers=$n/330"
    if [ "$rc" -eq 0 ] && [ "$n" -eq 330 ] && ! ls $BASE/out/STOPPED_* >/dev/null 2>&1; then
      date '+%F %T %Z' > $BASE/out/ALL_DONE; echo "ALL_DONE written"
    fi ;;
  decide)
    nice -n 10 $PY -B $BASE/theoryq_decide.py ;;
  *) echo "mode synth|run|decide"; exit 2 ;;
esac

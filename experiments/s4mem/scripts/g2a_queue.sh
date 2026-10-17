#!/bin/bash
: "${CCS_WORK:=$(cd "$(dirname "$0")/../.." && pwd)}"  # path configuration of the released copy (README)
# G2 (FIVES) sequential queue. Usage: g2a_queue.sh GPU_ID      (e.g. 1). Launch by the user only.
# Order: prep (CPU cache) -> for model in ucmnet utanet maresunet: native, rs256, rs512, rs1024.
# Markers in g2a/markers/: PREP_DONE|PREP_FAIL, <model>__<variant>.DONE|.FAIL ; g2a/ALL_DONE at the end (lists DONE/FAIL counts).
# Restart-safe: a job with a DONE marker is skipped; a job with ckpt_last.pth resumes from its last 1000-step checkpoint.
GPU="${1:?usage: g2a_queue.sh GPU_ID}"
ROOT=${CCS_WORK}/s4mem
OUT=$ROOT/g2a
PY=${CCS_PYTHON:-python}
SC=$ROOT/scripts
mkdir -p "$OUT/logs" "$OUT/markers"
if ! mkdir "$OUT/QUEUE.lock" 2>/dev/null; then echo "queue already running or stale lock: $OUT/QUEUE.lock"; exit 1; fi
trap 'rmdir "$OUT/QUEUE.lock"' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="$GPU"
CPUS=0-7,12-31
echo "queue start $(date -u +%F\ %T) GPU=$GPU" >> "$OUT/logs/queue.log"

if [ ! -f "$OUT/markers/PREP_DONE" ]; then
  rm -f "$OUT/markers/PREP_FAIL"
  if taskset -c $CPUS $PY -I $SC/g2a_train.py prep --out "$OUT" >> "$OUT/logs/prep.log" 2>&1; then
    date -u +%F\ %T > "$OUT/markers/PREP_DONE"
  else
    date -u +%F\ %T > "$OUT/markers/PREP_FAIL"; echo "prep failed" >> "$OUT/logs/queue.log"
    echo "PREP_FAIL $(date -u)" > "$OUT/ALL_DONE"; exit 1
  fi
fi

nd=0; nf=0
for m in ucmnet utanet maresunet; do
  for v in native rs256 rs512 rs1024; do
    key=${m}__${v}
    if [ -f "$OUT/markers/$key.DONE" ]; then nd=$((nd+1)); continue; fi
    rm -f "$OUT/markers/$key.FAIL"
    echo "start $key $(date -u +%T)" >> "$OUT/logs/queue.log"
    if taskset -c $CPUS $PY -I $SC/g2a_train.py run --model $m --variant $v --out "$OUT" --device cuda:0 --workers 4 \
         >> "$OUT/logs/$key.log" 2>&1 && [ -f "$OUT/$m/$v/results.json" ]; then
      date -u +%F\ %T > "$OUT/markers/$key.DONE"; nd=$((nd+1))
    else
      date -u +%F\ %T > "$OUT/markers/$key.FAIL"; nf=$((nf+1))
    fi
    echo "end $key $(date -u +%T) done=$nd fail=$nf" >> "$OUT/logs/queue.log"
  done
done
echo "ALL_DONE $(date -u +%F\ %T) done=$nd fail=$nf" > "$OUT/ALL_DONE"
$PY -I $SC/g2a_summary.py "$OUT" >> "$OUT/logs/summary.log" 2>&1

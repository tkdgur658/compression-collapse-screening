#!/bin/bash
: "${CCS_WORK:=$(cd "$(dirname "$0")/../.." && pwd)}"  # path configuration of the released copy (README)
# F5 (PREREG_F5.md, frozen sha 42dc2398...) sequential queue for PBE-Net on FIVES. Usage: f5_queue.sh GPU_ID   (GPU 0 only)
# f5_train.py = byte-identical copy of g2a_train.py. Cache = symlink to ../g2a/cache (read-only use).
# Order: smoke (S1, out f5_smoke, 50 steps, 32 train images) -> pbe native, rs256, rs512, rs1024.
# Markers in f5/markers/: SMOKE.DONE|SMOKE.FAIL, pbe__<variant>.DONE|.FAIL ; f5/ALL_DONE at the end.
# No file is deleted: a stale FAIL marker is renamed with a timestamp suffix.
GPU="${1:?usage: f5_queue.sh GPU_ID}"
[ "$GPU" = "0" ] || { echo "F5 runs on GPU 0 only"; exit 1; }
ROOT=${CCS_WORK}/s4mem
OUT=$ROOT/f5
SMK=$ROOT/f5_smoke
PY=${CCS_PYTHON:-python}
SC=$ROOT/scripts
mkdir -p "$OUT/logs" "$OUT/markers"
if ! mkdir "$OUT/QUEUE.lock" 2>/dev/null; then echo "queue already running or stale lock: $OUT/QUEUE.lock"; exit 1; fi
trap 'rmdir "$OUT/QUEUE.lock"' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1   # no __pycache__ in the read-only archive
export CUDA_VISIBLE_DEVICES="$GPU"
CPUS=0-7,12-31
echo "queue start $(date -u +%F\ %T) GPU=$GPU" >> "$OUT/logs/queue.log"
[ -f "$OUT/cache/PREP_DONE" ] || { echo "cache missing" >> "$OUT/logs/queue.log"; echo "NO_CACHE $(date -u)" > "$OUT/ALL_DONE"; exit 1; }
keep() { [ -f "$1" ] && mv "$1" "$1.old_$(date -u +%Y%m%dT%H%M%S)"; return 0; }

# S1 smoke
if [ ! -f "$OUT/markers/SMOKE.DONE" ]; then
  keep "$OUT/markers/SMOKE.FAIL"
  echo "start smoke $(date -u +%T)" >> "$OUT/logs/queue.log"
  if taskset -c $CPUS $PY -I $SC/f5_train.py run --model pbe --variant native --out "$SMK" --device cuda:0 --workers 4 \
       --steps 50 --n-train-limit 32 --smoke >> "$OUT/logs/smoke.log" 2>&1 && [ -f "$SMK/pbe/native/results.json" ] && [ -f "$SMK/pbe/native/best.pth" ]; then
    date -u +%F\ %T > "$OUT/markers/SMOKE.DONE"
  else
    date -u +%F\ %T > "$OUT/markers/SMOKE.FAIL"; echo "SMOKE_FAIL $(date -u +%F\ %T)" > "$OUT/ALL_DONE"; exit 1
  fi
  echo "end smoke $(date -u +%T)" >> "$OUT/logs/queue.log"
fi

nd=0; nf=0
for v in native rs256 rs512 rs1024; do
  key=pbe__${v}
  if [ -f "$OUT/markers/$key.DONE" ]; then nd=$((nd+1)); continue; fi
  if [ -f "$OUT/markers/$key.DROP" ]; then echo "dropped $key (S2/O5)" >> "$OUT/logs/queue.log"; continue; fi
  keep "$OUT/markers/$key.FAIL"
  echo "start $key $(date -u +%T)" >> "$OUT/logs/queue.log"
  if taskset -c $CPUS $PY -I $SC/f5_train.py run --model pbe --variant $v --out "$OUT" --device cuda:0 --workers 4 \
       >> "$OUT/logs/$key.log" 2>&1 && [ -f "$OUT/pbe/$v/results.json" ]; then
    date -u +%F\ %T > "$OUT/markers/$key.DONE"; nd=$((nd+1))
  else
    date -u +%F\ %T > "$OUT/markers/$key.FAIL"; nf=$((nf+1))
  fi
  echo "end $key $(date -u +%T) done=$nd fail=$nf" >> "$OUT/logs/queue.log"
done
echo "ALL_DONE $(date -u +%F\ %T) done=$nd fail=$nf" > "$OUT/ALL_DONE"

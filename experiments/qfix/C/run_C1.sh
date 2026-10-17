#!/bin/bash
: "${CCS_WORK:=$(cd "$(dirname "$0")/../.." && pwd)}"  # path configuration of the released copy (README)
# QFIX stage C1 training queue (PBE-Net compression-aware training), one GPU, sequential, restart-safe.
# Usage: nohup bash run_C1.sh <0|1> > logs/nohup_gpu<G>.out 2>&1 < /dev/null &
# GPU 0: isic18 x5, cvc x5, PH2 x5 ; GPU 1: isic17 x5, kvasir x5, etis x5  (B_fix_design.md section 5, LPT split)
# Test split sealed: trainer runs with skip_test=True (no test loader, no test metric).  Nothing is deleted; a partial
# or failed previous attempt is moved to <run>_attempt<n>.  Nothing is written under <ARCHIVE_ROOT>/.
set -u
GPU="${1:-}"
case "$GPU" in 0|1) ;; *) echo "usage: bash run_C1.sh <GPU 0|1>"; exit 2;; esac
BASE=${CCS_WORK}/qfix/C
case "$BASE" in "${CCS_ARCHIVE:-/nonexistent}"/*) echo "BASE inside protected archive"; exit 3;; esac
PY=${CCS_PYTHON:-python}
mkdir -p "$BASE/runs" "$BASE/logs"
exec 9>"$BASE/gpu$GPU.lock"; flock -n 9 || { echo "LOCK_BUSY gpu$GPU"; exit 4; }
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
QL=$BASE/logs/queue_gpu$GPU.log
say () { echo "$(date '+%F %T %Z') $*" | tee -a "$QL"; }
(cd "$BASE" && sha256sum -c --quiet SHA256SUMS) || { say "ABORT frozen file sha mismatch"; exit 5; }
if [ "$GPU" = 0 ]; then
  LIST="isic18:1 isic18:2 isic18:3 isic18:4 isic18:5 cvc:1 cvc:2 cvc:3 cvc:4 cvc:5 PH2:1 PH2:2 PH2:3 PH2:4 PH2:5"
else
  LIST="isic17:1 isic17:2 isic17:3 isic17:4 isic17:5 kvasir:1 kvasir:2 kvasir:3 kvasir:4 kvasir:5 etis:1 etis:2 etis:3 etis:4 etis:5"
fi
say "queue start GPU=$GPU pid=$$ list=[$LIST]"
for item in $LIST; do
  ds=${item%%:*}; k=${item##*:}
  out=$BASE/runs/$ds/k$k; log=$BASE/logs/cat_${ds}_k${k}.log
  if [ -f "$out/DONE" ]; then say "skip $ds k$k (DONE)"; continue; fi
  for attempt in 1 2; do
    if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
      n=1; while [ -e "${out}_attempt$n" ]; do n=$((n+1)); done
      mv "$out" "${out}_attempt$n"; say "kept previous attempt $out -> ${out}_attempt$n"
    fi
    mkdir -p "$out"
    say "start $ds k$k attempt $attempt"
    ( cd "$BASE/code" && taskset -c 0-7,12-31 $PY -B "$BASE/qfix_c_cat.py" --ds "$ds" --k "$k" ) >> "$log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then date '+%F %T %Z' > "$out/DONE"; say "DONE $ds k$k"; break; fi
    if [ $rc -eq 3 ]; then echo "precheck failed (cache) rc=3" > "$out/FAIL"; say "FAIL(pre) $ds k$k"; break; fi
    if [ $rc -eq 4 ]; then echo "verify failed rc=4 (see log)" > "$out/MISMATCH"; say "MISMATCH $ds k$k"; break; fi
    say "FAIL $ds k$k rc=$rc attempt $attempt"
    if [ $attempt -eq 2 ]; then echo "rc=$rc after 2 attempts" > "$out/FAIL"; fi
  done
done
date '+%F %T %Z' > "$BASE/ALL_DONE_gpu$GPU"
say "QUEUE END GPU=$GPU DONE=$(ls $BASE/runs/*/k*/DONE 2>/dev/null | wc -l) FAIL=$(ls $BASE/runs/*/k*/FAIL 2>/dev/null | wc -l) MISMATCH=$(ls $BASE/runs/*/k*/MISMATCH 2>/dev/null | wc -l) (of 30, both queues)"

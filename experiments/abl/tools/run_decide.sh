#!/bin/bash
: "${CCS_WORK:=$(cd "$(dirname "$0")/../.." && pwd)}"  # path configuration of the released copy (README)
: "${CCS_ARCHIVE:=${CCS_WORK}/../archive}"
# F4 decision: evaluate the sealed official test ONCE per DONE run, then abl_analyze.py real (once). GPU 1 only.
# Refuses before ALL_DONE, while run_abl.sh holds the lock, or if abl_result.json already exists.
set -u
BASE=${CCS_WORK}/abl
PY=${CCS_PYTHON:-python}
CODE=$BASE/code; TOOLS=$BASE/tools; RUNS=$BASE/rr/abl4; LOGS=$BASE/logs
CACHE=${CCS_ARCHIVE}/cache
[ -f "$BASE/ALL_DONE" ] || { echo "refuse: $BASE/ALL_DONE missing (queue not finished)"; exit 2; }
[ -e "$BASE/abl_result.json" ] && { echo "refuse: abl_result.json exists (decision runs once)"; exit 3; }
exec 9>"$BASE/gpu1.lock"; flock -n 9 || { echo "refuse: queue lock held"; exit 4; }
( cd "$BASE" && sha256sum -c --quiet SHA256SUMS ) || { echo "ABORT SHA256SUMS check failed"; exit 5; }
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1
say () { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOGS/decide.log"; }
while [ -n "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader | tr -d ' ')" ]; do say "WAIT foreign process on GPU 1"; sleep 60; done
t0=$(date +%s); s0=$(date -u +%FT%TZ)
for d in $RUNS/*/*/k*; do
  [ -f "$d/DONE" ] || continue
  [ -f "$d/test_eval.json" ] && continue
  row=$(basename $(dirname $(dirname $d))); ds=$(basename $(dirname $d)); k=$(basename $d); k=${k#k}
  (cd "$CODE" && $PY "$TOOLS/eval_test.py" "$row" "$ds" "$k" "$CACHE") >> "$LOGS/decide.log" 2>&1 || say "eval_test FAILED $row $ds k$k"
done
printf '%s\t%s\t%s\t%s\t%s\n' "decide_test_eval" 1 "$s0" "$(date -u +%FT%TZ)" "$(awk -v a=$t0 -v b=$(date +%s) 'BEGIN{printf "%.4f",(b-a)/3600}')" >> "$BASE/gpuh_ledger.tsv"
(cd "$CODE" && $PY "$TOOLS/abl_analyze.py" real) 2>&1 | tee -a "$LOGS/decide.log"
say "decision written: $BASE/abl_result.json $BASE/abl_result.md"

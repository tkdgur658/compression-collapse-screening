#!/bin/bash
: "${CCS_WORK:=$(cd "$(dirname "$0")/../.." && pwd)}"  # path configuration of the released copy (README)
: "${CCS_ARCHIVE:=${CCS_WORK}/../archive}"
: "${CCS_DATA:=${CCS_WORK}/../datasets}"
# F4 ablation queue (PREREG_ABL): 60 runs = 3 datasets x 4 variants x 5 splits, protocol v1, GPU 1 ONLY.
# Usage: nohup setsid bash run_abl.sh 1 > logs/queue.nohup 2>&1 &        (argument must be 1)
# Restart-safe (DONE skipped, attempts kept), single instance (flock), official test SEALED (ABL_SEAL_TEST=1).
# Stop rules: GPU-h cap guard (refuses a run whose estimate would exceed the cap), OOM twice -> stop,
# foreign process on GPU 1 -> wait. Never deletes: failed attempts are moved to <run>_attempt<n>.
set -u
[ "${1:-}" = "1" ] || { echo "usage: bash run_abl.sh 1   (GPU 1 only)"; exit 2; }
GPU=1
BASE=${CCS_WORK}/abl
case "$BASE" in "${CCS_ARCHIVE:-/nonexistent}"/*) echo "BASE inside protected archive"; exit 3;; esac
PY=${CCS_PYTHON:-python}
CODE=$BASE/code; TOOLS=$BASE/tools; RR=$BASE/rr; RUNS=$RR/abl4; LOGS=$BASE/logs
DATA=${CCS_DATA}
CACHE=${CCS_ARCHIVE}/cache
EXPECT_SRC=ee357c57035bad31fffbfb18832e2f9f235fac1234616804fef1719719dd022f
CAP_H=60.0          # total GPU-h cap for F4 (study lead)
RESERVE_H=1.0       # kept for run_decide.sh (60 sealed test evaluations)
LEDGER=$BASE/gpuh_ledger.tsv
mkdir -p "$LOGS" "$RUNS"
exec 9>"$BASE/gpu1.lock"; flock -n 9 || { echo "another run_abl.sh holds $BASE/gpu1.lock"; exit 4; }
say () { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOGS/queue.log"; }
for f in "$BASE/STOPPED_OOM"; do [ -e "$f" ] && { say "refuse: $f present (needs a decision)"; exit 6; }; done

# integrity: frozen tool/code sha256 and snapshot digest
( cd "$BASE" && sha256sum -c --quiet SHA256SUMS ) || { say "ABORT SHA256SUMS check failed"; exit 5; }
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU ABL_SEAL_TEST=1
GOT=$(cd "$CODE" && $PY -c "from common.provenance import source_digest as s; print(s())")
[ "$GOT" = "$EXPECT_SRC" ] || { say "ABORT snapshot digest $GOT != $EXPECT_SRC"; exit 5; }
[ -f "$LEDGER" ] || printf 'item\tattempt\tstart_utc\tend_utc\tgpu_h\n' > "$LEDGER"
say "queue start GPU=$GPU src=${EXPECT_SRC:0:8} pid=$$ cap=${CAP_H}h"

used_h () { awk -F'\t' 'NR>1{s+=$5} END{printf "%.4f", s+0}' "$LEDGER"; }
# estimate (h) = mean v1 train_minutes(ds) x measured ms/step ratio(variant) x 1.25 safety + 3 min start-up
est_h () {
  local tm r
  case $1 in cvc) tm=13.6;; isic17) tm=43.4;; isic18) tm=57.1;; esac
  case $2 in pbe_idskip) r=0.962;; pbe_lam0) r=0.994;; pbe_mbv2) r=1.058;; pbe_se) r=1.015;; esac
  awk -v a=$tm -v b=$r 'BEGIN{printf "%.4f", (a*b*1.25+3)/60}'
}
wait_gpu_free () {  # foreign compute process on GPU 1 -> wait (poll 60 s, log every 10 min)
  local n=0 p
  while true; do
    p=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' | tr '\n' ' ')
    [ -z "$p" ] && return 0
    [ $((n % 10)) -eq 0 ] && say "WAIT foreign process on GPU $GPU: $p"
    n=$((n+1)); sleep 60
  done
}

ROWS="pbe_idskip pbe_lam0 pbe_mbv2 pbe_se"
stopped=""
for ds in cvc isic17 isic18; do
 for row in $ROWS; do
  for k in 1 2 3 4 5; do
   out=$RUNS/$row/$ds/k$k; item="$row/$ds/k$k"
   [ -f "$out/DONE" ] && { say "skip $item (DONE)"; continue; }
   [ -f "$out/MISMATCH" ] && { say "skip $item (MISMATCH kept)"; continue; }
   [ -f "$out/FAIL_FINAL" ] && { say "skip $item (FAIL_FINAL)"; continue; }
   while true; do
    nprev=$(ls -d "${out}"_attempt* 2>/dev/null | wc -l)
    if [ -e "$out" ] && [ ! -f "$out/DONE" ]; then   # leftover partial/failed attempt: keep it, move aside
      mv "$out" "${out}_attempt$((nprev+1))"; nprev=$((nprev+1)); say "moved previous attempt of $item -> _attempt$nprev"
    fi
    log=$LOGS/${row}_${ds}_k${k}_a$((nprev+1)).log   # one log per attempt (OOM count is per attempt)
    if [ "$nprev" -ge 2 ]; then mkdir -p "$out"; echo "2 attempts failed" > "$out/FAIL_FINAL"; say "FAIL_FINAL $item"; break; fi
    U=$(used_h); E=$(est_h $ds $row)
    if awk -v u=$U -v e=$E -v c=$CAP_H -v r=$RESERVE_H 'BEGIN{exit !(u+e > c-r)}'; then
      say "CAP guard: used ${U}h + est ${E}h > ${CAP_H}h - ${RESERVE_H}h reserve -> stop before $item"
      echo "used=$U est_next=$E item=$item" > "$BASE/STOPPED_CAP"; stopped=cap; break 4
    fi
    wait_gpu_free
    mkdir -p "$out"
    if ! (cd "$CODE" && $PY "$TOOLS/verify_abl.py" pre "$ds" "$k" "$row" "$CODE" "$DATA" "$CACHE") >> "$log" 2>&1; then
      echo "precheck failed" > "$out/FAIL_FINAL"; say "FAIL(pre) $item"; break
    fi
    t0=$(date +%s); s0=$(date -u +%FT%TZ)
    say "start $item attempt $((nprev+1)) (used ${U}h, est ${E}h)"
    ( cd "$CODE" && taskset -c 0-7,12-31 $PY main.py --suite abl4 --rows "$row" --datasets "$ds" --splits "$k" \
        --stage train --execute --device cuda:0 --data-root "$DATA" --cache-root "$CACHE" --run-root "$RR" ) >> "$log" 2>&1
    rc=$?
    t1=$(date +%s); gh=$(awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.4f", (b-a)/3600}')
    printf '%s\t%s\t%s\t%s\t%s\n' "$item" "$((nprev+1))" "$s0" "$(date -u +%FT%TZ)" "$gh" >> "$LEDGER"
    if [ $rc -ne 0 ] || [ -f "$out/FAILED.txt" ] || [ ! -f "$out/results.json" ]; then
      echo "rc=$rc FAILED.txt=$([ -f "$out/FAILED.txt" ] && echo yes || echo no)" > "$out/FAIL"; say "FAIL $item rc=$rc (${gh}h)"
      if grep -qi "out of memory" "$log"; then
        n=$(( $(cat "$BASE/oom_count" 2>/dev/null || echo 0) + 1 )); echo $n > "$BASE/oom_count"; say "OOM count $n"
        if [ $n -ge 2 ]; then echo "OOM twice, last $item" > "$BASE/STOPPED_OOM"; say "STOP: OOM twice"; stopped=oom; break 4; fi
      fi
      continue   # retry once (the failed attempt is moved aside at the top of the loop)
    fi
    if (cd "$CODE" && $PY "$TOOLS/verify_abl.py" post "$ds" "$k" "$row" "$CODE" "$DATA" "$CACHE" "$EXPECT_SRC") >> "$log" 2>&1; then
      date -u +%FT%TZ > "$out/DONE"; say "DONE $item (${gh}h)"
    else
      cp "$out/verify_report.json" "$out/MISMATCH" 2>/dev/null || echo "verify error" > "$out/MISMATCH"; say "MISMATCH $item"
    fi
    break
   done
  done
 done
done
cnt () { ls $RUNS/*/*/k*/$1 2>/dev/null | wc -l; }
summary="DONE=$(cnt DONE) FAIL_FINAL=$(cnt FAIL_FINAL) MISMATCH=$(cnt MISMATCH) of 60, used=$(used_h)h, stopped=${stopped:-no}"
if [ "$stopped" = "oom" ]; then say "queue halted (OOM): $summary; ALL_DONE not written"; exit 7; fi
echo "$(date -u +%FT%TZ) $summary" > "$BASE/ALL_DONE"
say "ALL_DONE: $summary"

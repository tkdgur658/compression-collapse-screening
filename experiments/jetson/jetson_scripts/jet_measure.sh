#!/bin/bash
# PREREG_JETSON timing driver, identical for every model.
# Per model block: pre-conditions (who, 5 s tegrastats with GR3D 0% and 1-min load average < 0.5, nvpmodel -q),
# clock snapshot immediately before, trtexec 60 iterations (first 10 = warm-up, dropped in the analysis; 50 timed),
# clock snapshot immediately after. Clock change between the two snapshots -> block discarded and re-measured
# (at most 2 re-measurements, then "unstable"). Another login session -> POSTPONED marker, exit.
# No sudo, no jetson_clocks, no nvpmodel change, writes only under $D.
set -u
D=$HOME/lsh_latency_20261016
TRT=/usr/src/tensorrt/bin/trtexec
G=/sys/devices/57000000.gpu/devfreq/57000000.gpu
MODELS=${MODELS:-"pbe utanet mk_unet tinyunet ucmnet lbunet cmunext egeunet malunet"}
RUN=${RUN:-r1}
R=$D/runs/$RUN; mkdir -p $R
LOG=$R/driver.log
say() { echo "$(date '+%F %T.%N %Z') $*" >> $LOG; }

# clock values compared before/after (frequencies and the clock configuration)
clk() {
  local c=""
  for i in 0 1 2 3; do c="$c cpu${i}_cur=$(cat /sys/devices/system/cpu/cpu$i/cpufreq/scaling_cur_freq)"; done
  for i in 0 1 2 3; do c="$c cpu${i}_min=$(cat /sys/devices/system/cpu/cpu$i/cpufreq/scaling_min_freq) cpu${i}_max=$(cat /sys/devices/system/cpu/cpu$i/cpufreq/scaling_max_freq) cpu${i}_gov=$(cat /sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor)"; done
  echo "gpu_cur=$(cat $G/cur_freq) gpu_min=$(cat $G/min_freq) gpu_max=$(cat $G/max_freq) gpu_gov=$(cat $G/governor) cpu_online=$(cat /sys/devices/system/cpu/online)$c"
}
# recorded only (not compared): GPU target_freq, EMC (not readable without root), temperatures
extra() {
  echo "gpu_target=$(cat $G/target_freq) emc_freq=not_readable_without_root emc_force_max_rate=$(cat /sys/module/t210_emc/parameters/emc_force_max_rate) temps_mC=$(cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | tr '\n' ,)"
}

precond() {  # $1 = file prefix; returns 0 ok, 1 wait, 2 postpone
  local p=$1
  who > $p.who.txt 2>&1
  # allowed: only the persistent console (GDM) session ':0' of the device account; our own ssh commands have no tty
  if grep -v -E "^${JETSON_USER:-jetson} +:0 " $p.who.txt | grep -q .; then return 2; fi
  timeout 6 tegrastats --interval 1000 > $p.tegrastats.txt 2>&1
  cat /proc/loadavg > $p.loadavg.txt
  nvpmodel -q > $p.nvpmodel.txt 2>&1
  ps -eo user,pid,pcpu,etime,args --sort=-pcpu | head -15 > $p.ps.txt
  local l1; l1=$(cut -d' ' -f1 /proc/loadavg)
  local n; n=$(grep -c 'GR3D_FREQ' $p.tegrastats.txt)
  local busy; busy=$(grep -o 'GR3D_FREQ [0-9]*%' $p.tegrastats.txt | grep -v -c 'GR3D_FREQ 0%')
  echo "load1=$l1 tegrastats_samples=$n gr3d_nonzero=$busy" > $p.precond.txt
  if [ "$n" -lt 4 ] || [ "$busy" -ne 0 ]; then return 1; fi
  awk -v l="$l1" 'BEGIN{exit !(l < 0.5)}' || return 1
  return 0
}

say "driver start RUN=$RUN models=$MODELS pid=$$"
for m in $MODELS; do
  E=$D/engines/${m}_fp16.engine
  [ -f $E ] || { say "$m no engine -> skip"; continue; }
  att=0; status=""
  while [ $att -lt 3 ]; do
    att=$((att+1)); P=$R/${m}_a$att
    w=0
    while true; do
      precond $P; pc=$?
      if [ $pc -eq 2 ]; then say "$m a$att POSTPONED: other session in who"; cp $P.who.txt $R/POSTPONED_who.txt; date > $R/POSTPONED; exit 3; fi
      [ $pc -eq 0 ] && break
      w=$((w+1)); say "$m a$att precondition not met ($(cat $P.precond.txt)), wait 30 s (#$w)"
      cp $P.precond.txt $P.precond_fail$w.txt; cp $P.tegrastats.txt $P.tegrastats_fail$w.txt
      if [ $w -ge 40 ]; then say "$m a$att precondition not met after 40 waits -> STOP"; date > $R/STOPPED_precond; exit 4; fi
      sleep 30
    done
    say "$m a$att precondition ok ($(cat $P.precond.txt))"
    B=$(clk); BX=$(extra); TB=$(date +%s.%N)
    $TRT --loadEngine=$E --noDataTransfers --warmUp=0 --duration=0 --iterations=60 --exportTimes=$P.times.json > $P.trtexec.log 2>&1
    rc=$?
    TA=$(date +%s.%N); A=$(clk); AX=$(extra)
    { echo "before_t=$TB $B"; echo "before_extra $BX"; echo "after_t=$TA $A"; echo "after_extra $AX"; echo "trtexec_rc=$rc"; } > $P.clocks.txt
    if [ $rc -ne 0 ]; then say "$m a$att trtexec rc=$rc"; status="trtexec_failed"; break; fi
    if [ "$B" == "$A" ]; then say "$m a$att clocks unchanged -> KEEP"; echo "KEEP a$att" > $R/${m}.status; status=keep; break
    else say "$m a$att clocks changed -> DISCARD"; echo "a$att" >> $R/${m}.discarded; fi
  done
  [ -z "$status" ] && { say "$m unstable after 3 attempts"; echo "UNSTABLE" > $R/${m}.status; }
  [ "$status" == "trtexec_failed" ] && echo "TRTEXEC_FAILED" > $R/${m}.status
done
say "driver end"
date > $R/DONE

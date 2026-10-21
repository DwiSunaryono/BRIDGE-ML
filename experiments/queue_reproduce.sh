#!/bin/bash
# Queue for the reproduction runs on one machine: wait until the running reproduction ends, then start the next steps.
# Usage: queue_reproduce.sh <python> <jobs> <shard i/n> <steps: comma list of "compare" or run lists separated by ';'>
PY=$1; J=$2; SH=$3; STEPS=$4
cd "$(dirname "$0")/.."
log=experiments/out/queue_$(echo $SH | sed "s,/,of,").log
echo "$(date '+%F %T') queue start: $STEPS" >> $log
while pgrep -f "^[^ ]*python[^ ]* .*reproduce_(runs|paper)[.]py" > /dev/null; do sleep 60; done   # python processes only
IFS=';' read -ra parts <<< "$STEPS"
for s in "${parts[@]}"; do
  if [ "$s" = "compare" ]; then
    for mode in "" "--via-program"; do
      t0=$(date +%s)
      echo "$(date '+%F %T') compare start ${mode:-library}" >> $log
      OMP_NUM_THREADS=1 $PY experiments/reproduce_paper.py --part real --reps 20 --jobs $J --fresh $mode >> $log 2>&1
      echo "$(date '+%F %T') compare end ${mode:-library}: $(( $(date +%s) - t0 )) s" >> $log
    done
  else
    echo "$(date '+%F %T') runs start: $s" >> $log
    OMP_NUM_THREADS=1 $PY experiments/reproduce_runs.py --runs $s --jobs $J --shard $SH >> $log 2>&1 || { echo "$(date '+%F %T') STOPPED in $s" >> $log; exit 1; }
    echo "$(date '+%F %T') runs end: $s" >> $log
  fi
done
echo "$(date '+%F %T') queue done" >> $log

#!/bin/bash
# Waits until this machine has no reproduction work left (workers, queues, waiting loops), then checks the bridge-gate
# procedure against every MINI-21 and MINI-22 trial (its shard). Usage: chain_verify.sh <python> <n_proc> <i/n>
PY=$1; NP=$2; SH=$3
cd "$(dirname "$0")"
PAT='^[^ ]*python[^ ]* .*reproduce_(runs|paper)[.]py|queue_reproduce[.]sh|laptop_switch[.]sh|kill -[0] '
idle=0
while [ $idle -lt 2 ]; do
  if [ "$(pgrep -fc "$PAT")" = "0" ]; then idle=$((idle+1)); else idle=0; fi
  sleep 60
done
echo "$(date '+%F %T') start verify $SH"
export OMP_NUM_THREADS=1
$PY verify_protocol.py binary 100 $NP all --shard=$SH > out/verify_protocol_binary_${SH/\//of}.out 2>&1
$PY verify_protocol.py multiclass 100 $NP all --shard=$SH > out/verify_protocol_multiclass_${SH/\//of}.out 2>&1
echo "$(date '+%F %T') verify done $SH"

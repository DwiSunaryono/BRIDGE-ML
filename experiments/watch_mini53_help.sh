#!/bin/bash
# watcher for the MINI-53 reverse-order helper on Dept; appends to the main watcher log
bad0=0
while true; do
  log=$(ssh -o BatchMode=yes -o ConnectTimeout=10 user@machine3 "cat ~/BRIDGE_m53/experiments/out/mini53_dept_help.log" 2>/dev/null)
  run=$(ssh -o BatchMode=yes -o ConnectTimeout=10 user@machine3 "pgrep -fc '^[^ ]*python[^ ]* mini53_helper[.]py'" 2>/dev/null || echo 0)
  bad=$(echo "$log" | grep -cE 'id=0|ERROR|Traceback')
  [ "$bad" != "$bad0" ] && echo "dept-helper PROBLEM: $(echo "$log" | grep -E 'id=0|ERROR|Traceback' | tail -2 | tr '\n' ' ')" && bad0=$bad
  n=$(echo "$log" | grep -c ': ok\|: SKIP')
  if echo "$log" | grep -q '^done'; then echo "dept-helper DONE ($n)"; exit 0; fi
  if [ "${run:-0}" = 0 ]; then echo "dept-helper IDLE/STOPPED without done ($n)"; exit 0; fi
  sleep 120
done

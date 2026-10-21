#!/bin/bash
# watcher for the local reverse-order helper on Equnix's jobs; appends to the main watcher log
L=$HOME/BRIDGE_repo/experiments/out/mini53_local_help.log; bad0=0
while true; do
  bad=$(grep -cE 'id=0|ERROR|Traceback' $L)
  [ "$bad" != "$bad0" ] && echo "local-helper PROBLEM: $(grep -E 'id=0|ERROR|Traceback' $L | tail -2 | tr '\n' ' ')" && bad0=$bad
  if grep -q '^done' $L; then echo "local-helper DONE ($(grep -c ': ok' $L))"; exit 0; fi
  if ! pgrep -f '^[^ ]*python[^ ]* mini53_helper[.]py out/equnix' >/dev/null; then echo "local-helper IDLE/STOPPED without done ($(grep -c ': ok' $L))"; exit 0; fi
  sleep 120
done

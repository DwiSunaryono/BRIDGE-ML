#!/bin/bash
# emits: problem lines (id=0, ERROR), idle machine, per-machine done, and a progress line every 30 min
L=$HOME/BRIDGE_repo/experiments/out/mini53_local.log
declare -A seen; last=0
while true; do
  st=""
  for m in local equnix dept; do
    if [ $m = local ]; then log=$(cat $L 2>/dev/null); run=$(pgrep -fc '^[^ ]*python[^ ]* mini53_anggota[_]learner_BCE[.]py' || true)
    else h=$([ $m = equnix ] && echo user@machine2 || echo user@machine3)
      log=$(ssh -o BatchMode=yes -o ConnectTimeout=10 $h "cat ~/BRIDGE_m53/experiments/out/mini53_$m.log" 2>/dev/null)
      run=$(ssh -o BatchMode=yes -o ConnectTimeout=10 $h "pgrep -fc '^[^ ]*python[^ ]* mini53_anggota[_]learner_BCE[.]py'" 2>/dev/null || echo 0); fi
    n=$(echo "$log" | grep -c ': ok\|: SKIP\|: ERROR'); tot=$(echo "$log" | head -1 | awk '{print $1}')
    bad=$(echo "$log" | grep -E 'id=0|ERROR|Traceback' | wc -l)
    if [ "${seen[$m.bad]:-0}" != "$bad" ]; then echo "$m PROBLEM: $(echo "$log" | grep -E 'id=0|ERROR|Traceback' | tail -2 | tr '\n' ' ')"; seen[$m.bad]=$bad; fi
    if echo "$log" | grep -q '^done' ; then [ -z "${seen[$m.done]}" ] && echo "$m DONE ($n/$tot)" && seen[$m.done]=1
    elif [ "${run:-0}" = 0 ]; then [ -z "${seen[$m.idle]}" ] && echo "$m IDLE/STOPPED without done ($n/$tot)" && seen[$m.idle]=1; fi
    st="$st $m $n/$tot;"
  done
  now=$(date +%s); if [ $((now-last)) -ge 1800 ]; then echo "progress$st"; last=$now; fi
  [ -n "${seen[local.done]}" ] && [ -n "${seen[equnix.done]}" ] && [ -n "${seen[dept.done]}" ] && exit 0
  sleep 120
done

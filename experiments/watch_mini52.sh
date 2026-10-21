#!/usr/bin/env bash
set -u
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/out"
log="$out/mini52_watcher.log"

check_local() {
  [[ -f "$out/mini52_m1.exit" ]] && return 0
  if ! pgrep -f '[m]ini52_ruang_kandidat_E.py.*1/3' >/dev/null; then
    printf '%s ERROR local idle without exit marker\n' "$(date '+%F %T %Z')" >> "$log"
    return 90
  fi
  return 1
}

check_remote() {
  local host="$1" shard="$2"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "test -f ~/BRIDGE_m52/experiments/out/mini52_m${shard}.exit" && return 0
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "pgrep -f '[m]ini52_ruang_kandidat_E.py.*${shard}/3' >/dev/null"; then
    printf '%s ERROR shard %s idle without exit marker\n' "$(date '+%F %T %Z')" "$shard" >> "$log"
    return 90
  fi
  return 1
}

while true; do
  check_local; a=$?
  check_remote user@machine2 2; b=$?
  check_remote user@machine3 3; c=$?
  if (( a == 90 || b == 90 || c == 90 )); then
    printf '90\n' > "$out/mini52_watcher.exit"
    exit 90
  fi
  (( a == 0 && b == 0 && c == 0 )) && break
  printf '%s active local=%s remote2=%s remote3=%s\n' "$(date '+%F %T %Z')" "$a" "$b" "$c" >> "$log"
  sleep 300
done

e1="$(cat "$out/mini52_m1.exit")"
e2="$(ssh user@machine2 'cat ~/BRIDGE_m52/experiments/out/mini52_m2.exit')"
e3="$(ssh user@machine3 'cat ~/BRIDGE_m52/experiments/out/mini52_m3.exit')"
printf '%s exits local=%s remote2=%s remote3=%s\n' "$(date '+%F %T %Z')" "$e1" "$e2" "$e3" >> "$log"
if [[ "$e1" != 0 || "$e2" != 0 || "$e3" != 0 ]]; then
  printf '91\n' > "$out/mini52_watcher.exit"
  exit 91
fi

rsync -a user@machine2:BRIDGE_m52/experiments/out/mini52_m2.csv "$out/"
rsync -a user@machine2:BRIDGE_m52/experiments/out/mini52_m2.log "$out/"
rsync -a user@machine3:BRIDGE_m52/experiments/out/mini52_m3.csv "$out/"
rsync -a user@machine3:BRIDGE_m52/experiments/out/mini52_m3.log "$out/"

for f in "$out"/mini52_m{1,2,3}.log; do
  grep -qx 'done' "$f" || { printf '92\n' > "$out/mini52_watcher.exit"; exit 92; }
done
python "$here/audit_mini52.py" >> "$log" 2>&1 || {
  printf '93\n' > "$out/mini52_watcher.exit"; exit 93;
}
python "$here/analisis_mini52.py" >> "$log" 2>&1 || {
  printf '94\n' > "$out/mini52_watcher.exit"; exit 94;
}
printf '%s audit-and-analysis-done\n' "$(date '+%F %T %Z')" >> "$log"
printf '0\n' > "$out/mini52_watcher.exit"

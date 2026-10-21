"""Consolidate and audit the three frozen amendment shards."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFFSET = 1_046_000_000
parts = sorted(HERE.glob("settingE2_amendment1_shard*.csv"))
rows = []
for part in parts:
    rows.extend(csv.DictReader(part.open()))

by_rep = defaultdict(list)
for row in rows:
    by_rep[int(row["rep"])].append(row)
expected = set(range(60))
missing = sorted(expected - set(by_rep))
unexpected = sorted(set(by_rep) - expected)
duplicates = {str(rep): len(items) for rep, items in by_rep.items() if len(items) != 1}
bad_seed = [
    {"rep": rep, "seed": int(items[0]["seed"]), "expected": OFFSET + rep}
    for rep, items in sorted(by_rep.items())
    if len(items) == 1 and int(items[0]["seed"]) != OFFSET + rep
]
statuses = Counter(row["status"] for row in rows)
ready = not missing and not unexpected and not duplicates and not bad_seed and statuses == {"ok": 60}
audit = {
    "parts": [p.name for p in parts], "rows": len(rows), "unique_reps": len(by_rep),
    "statuses": dict(statuses), "missing": missing, "unexpected": unexpected,
    "duplicates": duplicates, "bad_seed": bad_seed, "analysis_ready": ready,
}
(HERE / "settingE2_amendment1_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")

if ready:
    fieldnames = list(rows[0])
    with (HERE / "settingE2_amendment1_hasil.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rep in sorted(by_rep):
            writer.writerow(by_rep[rep][0])
print(json.dumps(audit, indent=2))
raise SystemExit(0 if ready else 1)

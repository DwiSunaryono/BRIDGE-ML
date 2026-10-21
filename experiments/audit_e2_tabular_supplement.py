"""Consolidate and audit the three tabular-supplement shards."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
parts = [HERE / f"settingE2_tabular_shard{i}.csv" for i in (1, 2, 3)]
rows = []
for path in parts:
    rows += list(csv.DictReader(path.open()))
by_rep = defaultdict(list)
for row in rows:
    by_rep[int(row["rep"])].append(row)
expected = set(range(60))
statuses = Counter(row["status"] for row in rows)
missing = sorted(expected - set(by_rep))
extra = sorted(set(by_rep) - expected)
duplicates = {str(k): len(v) for k, v in by_rep.items() if len(v) != 1}
bad_seed = [rep for rep, vals in by_rep.items() if len(vals) == 1 and int(vals[0]["seed"]) != 1_047_000_000 + rep]
ready = not missing and not extra and not duplicates and not bad_seed and statuses == {"ok": 60}
audit = {"rows": len(rows), "unique_reps": len(by_rep), "statuses": dict(statuses), "missing": missing,
         "extra": extra, "duplicates": duplicates, "bad_seed": bad_seed, "analysis_ready": ready}
(HERE / "settingE2_tabular_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
if ready:
    fields = list(rows[0])
    with (HERE / "settingE2_tabular_hasil.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for rep in sorted(by_rep): writer.writerow(by_rep[rep][0])
print(json.dumps(audit, indent=2))
raise SystemExit(0 if ready else 1)

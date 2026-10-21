"""Mechanical completeness and identity audit for MINI-51.

Reads the shard plan and the three shard CSVs, then checks row counts, key coverage, duplicates, shard assignment,
status, and K173 identity. Writes out/mini51_AUDIT.json and exits 0 only when every check passes.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
plan = pd.read_csv(HERE / "mini51_shards.csv")
expected = {}
for row in plan.itertuples():
    trials = 200 if row.setting == "C" else 100
    for rep in range(trials):
        for space in ("K173", "Ksedang", "Kbesar"):
            expected[(row.setting, row.dataset, rep, space)] = row.shard

rows, shard_rows = [], {}
for shard in (1, 2, 3):
    path = HERE / "out" / f"mini51_m{shard}.csv"
    part = pd.read_csv(path, float_precision="round_trip")
    part["shard_file"] = shard
    shard_rows[str(shard)] = len(part)
    rows.append(part)
d = pd.concat(rows, ignore_index=True)
keys = list(zip(d.setting, d.dataset, d.rep.astype(int), d.space))
counts = pd.Series(keys).value_counts()
observed = set(keys)
missing = sorted(set(expected) - observed)
extra = sorted(observed - set(expected))
duplicates = {str(k): int(v) for k, v in counts.items() if v != 1}
wrong_shard = [str(k) for k, shard in zip(keys, d.shard_file) if k in expected and expected[k] != shard]
statuses = d.status.value_counts(dropna=False).to_dict()
k173 = d[d.space == "K173"]
ready = (len(d) == 17_400 and shard_rows == {"1": 6600, "2": 6600, "3": 4200}
         and not missing and not extra and not duplicates and not wrong_shard
         and statuses == {"ok": 17_400} and len(k173) == 5_800 and k173.identical.eq(1).all())
audit = dict(rows=len(d), unique_keys=len(observed), unique_trials=d[["setting", "dataset", "rep"]].drop_duplicates().shape[0],
             shard_rows=shard_rows, statuses=statuses, missing=missing[:20], missing_count=len(missing),
             extra=extra[:20], extra_count=len(extra), duplicates=duplicates, wrong_shard=wrong_shard[:20],
             wrong_shard_count=len(wrong_shard), k173_identity_ok=int(k173.identical.eq(1).sum()),
             k173_identity_total=len(k173), analysis_ready=bool(ready))
(HERE / "out" / "mini51_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if ready else 1)

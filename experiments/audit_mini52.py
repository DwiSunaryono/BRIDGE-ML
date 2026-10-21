"""Mechanical completeness and identity audit for MINI-52 (PRAREG_MINI52.md).

Checks row counts, key coverage, duplicates, shard assignment, status, and K173 identity with the stored (E) results
(lower_bound, gain_ref and gain_pool). Writes out/mini52_AUDIT.json and exits 0 only when every check passes.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SPACES = ("K173", "Ksedang", "Kbesar")
plan = pd.read_csv(HERE / "mini52_shards.csv")
expected = {(int(r.data_id), rep, s): int(r.shard) for r in plan.itertuples() for rep in range(100) for s in SPACES}

rows, shard_rows = [], {}
for shard in (1, 2, 3):
    part = pd.read_csv(HERE / "out" / f"mini52_m{shard}.csv", float_precision="round_trip")
    part["shard_file"] = shard
    shard_rows[str(shard)] = len(part)
    rows.append(part)
d = pd.concat(rows, ignore_index=True)
keys = list(zip(d.data_id.astype(int), d.rep.astype(int), d.space))
counts = pd.Series(keys).value_counts()
observed = set(keys)
missing = sorted(set(expected) - observed)
extra = sorted(observed - set(expected))
duplicates = {str(k): int(v) for k, v in counts.items() if v != 1}
wrong_shard = [str(k) for k, s in zip(keys, d.shard_file) if k in expected and expected[k] != s]
statuses = d.status.value_counts(dropna=False).to_dict()

files = sorted((HERE / "out").glob("settingE_*laptop.csv"))
e = pd.concat([pd.read_csv(f, float_precision="round_trip") for f in files], ignore_index=True)
k173 = d[d.space == "K173"].merge(e[["data_id", "rep", "lower_bound", "gain_ref", "gain_pool"]],
                                  on=["data_id", "rep"], how="left", validate="one_to_one")
same = ((k173.lo_cf_boot == k173.lower_bound) & (k173.ref == k173.gain_ref) & (k173.pt_pool == k173.gain_pool))
ready = (len(d) == 3_300 and shard_rows == {"1": 900, "2": 1200, "3": 1200}
         and not missing and not extra and not duplicates and not wrong_shard
         and statuses == {"ok": 3_300} and len(k173) == 1_100 and k173.identical.eq(1).all() and same.all())
audit = dict(rows=len(d), unique_keys=len(observed), shard_rows=shard_rows, statuses=statuses,
             missing=[str(m) for m in missing[:20]], missing_count=len(missing), extra=[str(x) for x in extra[:20]],
             extra_count=len(extra), duplicates=duplicates, wrong_shard=wrong_shard[:20], wrong_shard_count=len(wrong_shard),
             k173_identity_ok=int(k173.identical.eq(1).sum()), k173_same_as_stored=int(same.sum()),
             k173_total=len(k173), analysis_ready=bool(ready))
(HERE / "out" / "mini52_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if ready else 1)

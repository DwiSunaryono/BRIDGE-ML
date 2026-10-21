"""Count the trials behind the paper from the stored per-trial outputs (no new runs).

A trial is one execution of the procedure on one dataset or task with one seed. Files are grouped by run
(MINI-n or setting); analysis and audit files are skipped, rows with an error status are dropped, and
duplicates (for example trials that a helper process repeated) are removed on the key columns that are
present: dataset or task, repetition, and the variant columns such as metric, learner, arm, size, or margin.
Runs whose files have no repetition column are not counted, so the total is a lower bound.
Usage: python count_trials.py"""
import collections
import re
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
KEYS = ["setting", "dataset", "data_id", "source_id", "task", "sumber", "rep", "metric", "learner", "arm", "param", "variant", "policy",
        "K", "TOTAL", "size", "margin", "q", "eps", "mode", "fold", "batch", "cell", "rule"]


def main():
    fam = collections.defaultdict(list)
    for f in sorted(OUT.glob("*.csv")):
        if re.match(r"(analysis|analisis|audit)", f.name) or "diag" in f.name or "snapshot" in f.name:
            continue
        m = re.match(r"(mini\d+[a-z]?|setting[A-Za-z0-9]*?)(?=_|\.|$)", f.name)
        if m:
            fam[m.group(1)].append(f)
    total, per = 0, {}
    for run, files in fam.items():
        d = pd.concat([pd.read_csv(f, low_memory=False) for f in files])
        if "rep" not in d.columns:
            continue
        if "status" in d.columns:
            d = d[d.status.astype(str).str.startswith("ok") | d.status.isna()]
        per[run] = d.drop_duplicates([c for c in KEYS if c in d.columns]).shape[0]
        total += per[run]
    for run, n in sorted(per.items()):
        print(f"{run:>14}: {n:>7,}")
    print(f"{len(per)} runs, {total:,} trials (lower bound)")


if __name__ == "__main__":
    main()

"""Frozen runner for PRAREG_SETTING_E2_TABULAR_SUPPLEMENT.md.

Usage: python setting_e2_tabular_supplement.py OUT.csv NPROC I/3
"""
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import setting_e2 as E2

SOURCE_ID = "OpenML-41166"
NAME = "volkert"
OFFSET = 1_047_000_000

if __name__ == "__main__":
    out, nproc = Path(sys.argv[1]), int(sys.argv[2])
    shard, count = map(int, sys.argv[3].split("/"))
    if count != 3 or shard not in (1, 2, 3):
        raise SystemExit("use shard 1/3, 2/3, or 3/3")
    E2.OFF = OFFSET
    E2.split_paths(SOURCE_ID)
    jobs = [(0, SOURCE_ID, NAME, rep) for rep in range(60) if rep % 3 == shard - 1]
    done = set()
    if out.exists():
        done = {int(row["rep"]) for row in csv.DictReader(out.open()) if row["status"] == "ok"}
    todo = [job for job in jobs if job[3] not in done]
    print(f"{len(jobs)} trials, {len(jobs)-len(todo)} already done", flush=True)
    new = not out.exists()
    with out.open("a", newline="") as handle, Pool(nproc) as pool:
        writer = csv.DictWriter(handle, fieldnames=E2.COLS, restval="")
        if new:
            writer.writeheader()
        for index, result in enumerate(pool.imap_unordered(E2.run_one, todo, chunksize=1), 1):
            writer.writerow(result); handle.flush()
            print(f"{index}/{len(todo)} rep {result['rep']}: {result['status']} {result['seconds']} s", flush=True)

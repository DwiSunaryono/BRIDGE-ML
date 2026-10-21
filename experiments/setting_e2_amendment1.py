"""Frozen Amendment-1 runner for the technical replacement of shuttle.

Usage: python setting_e2_amendment1.py OUT.csv NPROC [I/N]
"""
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import setting_e2 as E2

SOURCE_ID = "OpenML-32"
NAME = "pendigits"
OFFSET = 1_046_000_000


if __name__ == "__main__":
    out = Path(sys.argv[1])
    nproc = int(sys.argv[2])
    E2.OFF = OFFSET

    # Create and validate the locked split serially before workers start.
    E2.split_paths(SOURCE_ID)
    jobs = [(0, SOURCE_ID, NAME, rep) for rep in range(60)]
    if len(sys.argv) > 3:
        shard, count = map(int, sys.argv[3].split("/"))
        jobs = [job for job in jobs if job[3] % count == shard - 1]

    done = set()
    if out.exists():
        done = {
            int(row["rep"])
            for row in csv.DictReader(out.open())
            if row["status"] == "ok"
        }
    todo = [job for job in jobs if job[3] not in done]
    print(f"{len(jobs)} trials in this shard, {len(jobs) - len(todo)} already done", flush=True)

    new = not out.exists()
    with out.open("a", newline="") as handle, Pool(nproc) as pool:
        writer = csv.DictWriter(handle, fieldnames=E2.COLS, restval="")
        if new:
            writer.writeheader()
        for index, result in enumerate(pool.imap_unordered(E2.run_one, todo, chunksize=1), 1):
            writer.writerow(result)
            handle.flush()
            print(f"{index}/{len(todo)} rep {result['rep']}: {result['status']} {result['seconds']} s", flush=True)

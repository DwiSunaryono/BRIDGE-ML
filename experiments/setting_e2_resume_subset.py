"""Run an explicit frozen subset of the E2 dataset/rep jobs listed in a CSV,
without changing the protocol.
Usage: python setting_e2_resume_subset.py <jobs.csv> <out.csv> <nproc>"""
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import setting_e2 as E2


if __name__ == "__main__":
    jobs_file, out, nproc = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    rows = list(csv.DictReader(jobs_file.open()))
    jobs = [(int(r["pos"]), r["source_id"], r["name"], int(r["rep"])) for r in rows]
    for source_id in sorted({job[1] for job in jobs}):
        E2.split_paths(source_id)
    done = set()
    if out.exists():
        done = {
            (row["source_id"], int(row["rep"]))
            for row in csv.DictReader(out.open())
            if row["status"] == "ok"
        }
    todo = [job for job in jobs if (job[1], job[3]) not in done]
    print(f"{len(jobs)} frozen jobs, {len(jobs)-len(todo)} already done", flush=True)
    new = not out.exists()
    with out.open("a", newline="") as handle, Pool(nproc) as pool:
        writer = csv.DictWriter(handle, fieldnames=E2.COLS, restval="")
        if new:
            writer.writeheader()
        for index, result in enumerate(pool.imap_unordered(E2.run_one, todo, chunksize=1), 1):
            writer.writerow(result)
            handle.flush()
            print(f"{index}/{len(todo)} {result['source_id']} rep {result['rep']}: {result['status']} {result['seconds']} s", flush=True)

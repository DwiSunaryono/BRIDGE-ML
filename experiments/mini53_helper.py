"""MINI-53 helper: runs the not-yet-done jobs of another machine's part in reverse order (that machine runs them
forward) and writes to its own CSV; results are identical per trial given the fixed seeds, and the analysis keeps one
row per (setting, dataset, rep).
Usage: python mini53_helper.py <done.csv of the other machine, snapshot> <plan.csv> <machine> <out.csv> <n_proc> [i/n] [--forward]"""
import csv, sys
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini53_anggota_learner_BCE as M
sys.argv = _argv

if __name__ == "__main__":
    done_csv, plan, machine, out, n_proc = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), int(sys.argv[5])
    pl = pd.read_csv(plan); mine = set(zip(pl[pl.machine == machine].setting, pl[pl.machine == machine].dataset))
    jobs = [j for j in M.all_jobs() if (j[0], j[1]) in mine]
    done = {(r["setting"], r["dataset"], int(r["rep"])) for r in csv.DictReader(open(done_csv)) if r["status"] in ("ok", "SKIP")}
    if out.exists():
        done |= {(r["setting"], r["dataset"], int(r["rep"])) for r in csv.DictReader(open(out)) if r["status"] in ("ok", "SKIP")}
    todo = [j for j in jobs if j not in done]
    if "--forward" not in sys.argv:   # default: reverse order (the other machine runs forward)
        todo = todo[::-1]
    part = [a for a in sys.argv[6:] if "/" in a]
    if part:   # optional i/n: take every n-th job of the reversed list, so that several helpers do not overlap
        i, n = map(int, part[0].split("/")); todo = todo[i - 1::n]
    print(f"{len(todo)} jobs of {machine} left; running in {'forward' if '--forward' in sys.argv else 'reverse'} order", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as f, Pool(n_proc, maxtasksperchild=20) as pool:
        w = csv.DictWriter(f, fieldnames=M.COLS, restval="")
        if new:
            w.writeheader()
        for k, r in enumerate(pool.imap_unordered(M.run_one, todo, chunksize=1)):
            w.writerow(r); f.flush()
            print(f"{k + 1}/{len(todo)} {r['setting']} {r['dataset']} rep {r['rep']}: {r['status']} id={r.get('identical', '')} {r['seconds']} s", flush=True)
    print("done", flush=True)

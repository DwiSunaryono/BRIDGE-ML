"""Helper for MINI-50 shard 2/3: the laptop runs the not-yet-done binary jobs of induk's shard in reverse order (induk
runs them forward) and writes to its own CSV. The two files are merged by (dataset, rep), keeping the first finished row.
run_one is the same as in mini50_all_families.py, so results are identical per trial given the fixed seeds.
Usage: python mini50_helper.py <done_csv_of_induk> <out.csv> <n_proc> [bin|mc]  (mc: shard-2 multiclass jobs, for komp-3)
"""
import csv, sys
from multiprocessing import Pool
from pathlib import Path
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini50_all_families as M
import mini44_tuning_tests as M44
sys.argv = _argv

if __name__ == "__main__":
    done_csv, out, n_proc = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    jobs = [(kind, name, rep, None) for kind, name in M44.ORDER for rep in range(M44.TRIALS[kind])]
    kind = sys.argv[4] if len(sys.argv) > 4 else "bin"
    jb = ([j for j in jobs if j[0] != "mc"][1::3] if kind == "bin"     # shard 2/3 binary jobs, same rule as the main script
          else [j for j in jobs if j[0] == "mc"][0::2])                  # shard 2/3 multiclass jobs (--mc-shards=2,3)
    done = {(r["dataset"], int(r["rep"])) for r in csv.DictReader(open(done_csv)) if r["status"] == "ok"}
    if out.exists():
        done |= {(r["dataset"], int(r["rep"])) for r in csv.DictReader(open(out)) if r["status"] == "ok"}
    todo = [j for j in jb if (j[1], j[2]) not in done][::-1]
    print(f"{len(todo)} {kind} jobs of shard 2/3 left; running in reverse order", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as f, Pool(n_proc) as pool:
        w = csv.DictWriter(f, fieldnames=M.COLS, restval="")
        if new:
            w.writeheader()
        for k, r in enumerate(pool.imap_unordered(M.run_one, todo, chunksize=1)):
            w.writerow(r); f.flush()
            if k % 25 == 0 or r["status"] != "ok":
                print(f"{k + 1}/{len(todo)} {r['dataset']} rep {r['rep']}: {r['status']} {r['seconds']} s", flush=True)
    print("done", flush=True)

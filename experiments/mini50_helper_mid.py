"""Second MINI-50 helper: the laptop (CPU-only XGBoost 3.4.1 on PYTHONPATH; identical results to
komp-3, checked on DryBean reps 39 and 51) runs the not-yet-done multiclass jobs of shard 2/3 from
the middle outward, while induk runs them forward and komp-3 in reverse. Merge later by (dataset,
rep), keeping the first ok row.
Usage: PYTHONPATH=~/xgbcpu python mini50_helper_mid.py <done1.csv> <done2.csv> <out.csv> <n_proc>"""
import csv, sys
from multiprocessing import Pool
from pathlib import Path
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini50_all_families as M
import mini44_tuning_tests as M44
sys.argv = _argv

if __name__ == "__main__":
    d1, d2, out, n_proc = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    jobs = [(kind, name, rep, None) for kind, name in M44.ORDER for rep in range(M44.TRIALS[kind])]
    jm = [j for j in jobs if j[0] == "mc"][0::2]
    done = set()
    for p in (d1, d2, out):
        if p.exists():
            done |= {(r["dataset"], int(r["rep"])) for r in csv.DictReader(open(p)) if r["status"] == "ok"}
    left = [j for j in jm if (j[1], j[2]) not in done]
    mid = len(left) // 2; order = []
    for k in range(len(left)):
        i = mid + (k + 1) // 2 * (1 if k % 2 else -1)
        if 0 <= i < len(left):
            order.append(left[i])
    order += [j for j in left if j not in order]
    print(f"{len(left)} multiclass jobs left; running from the middle outward", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as f, Pool(n_proc) as pool:
        w = csv.DictWriter(f, fieldnames=M.COLS, restval="")
        if new:
            w.writeheader()
        for k, r in enumerate(pool.imap_unordered(M.run_one, order, chunksize=1)):
            w.writerow(r); f.flush()
            print(f"{k + 1}/{len(order)} {r['dataset']} rep {r['rep']}: {r['status']} {r['seconds']} s", flush=True)
    print("done", flush=True)

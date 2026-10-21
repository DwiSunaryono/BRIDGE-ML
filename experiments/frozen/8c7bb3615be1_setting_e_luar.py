"""Setting (E-luar) (PRAREG_SETTING_E_LUAR.md): exploratory run of the same program and protocol as setting (E) on the safe
multiclass datasets OUTSIDE the validated 7-10 classes (3, 4, 6, 11 and 26 classes), 60 trials each. Identical to
setting_e.py except the seed offset (1_044_000_000 + 1000 p + r), so no seed is shared with (E).
Usage: python setting_e_luar.py <frozen_list.csv> <out.csv> <n_proc> [i/n]"""
import csv, sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tabel_c as TC
import setting_e as E
TC.OFF = 1_044_000_000

if __name__ == "__main__":
    lst = list(csv.DictReader(open(sys.argv[1])))
    out, n_proc = Path(sys.argv[2]), int(sys.argv[3])
    jobs = [(p, int(r["data_id"]), rep) for p, r in enumerate(lst) for rep in range(100 if int(float(r["K"])) == 2 else 60)]
    if len(sys.argv) > 4:
        i, n = map(int, sys.argv[4].split("/")); jobs = [j for j in jobs if j[0] % n == i - 1]
    done = {(int(r["data_id"]), int(r["rep"])) for r in csv.DictReader(open(out)) if r["status"] == "ok"} if out.exists() else set()
    todo = [j for j in jobs if (j[1], j[2]) not in done]
    print(f"{len(jobs)} trials in this part, {len(jobs) - len(todo)} already done", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as f, Pool(n_proc) as pool:
        w = csv.DictWriter(f, fieldnames=E.COLS, restval="")
        if new:
            w.writeheader()
        for k, r in enumerate(pool.imap_unordered(E.run_one, todo, chunksize=1)):
            w.writerow(r); f.flush()
            if k % 100 == 0 or r["status"] != "ok":
                print(f"{k + 1}/{len(todo)} {r['data_id']} rep {r['rep']}: {r['status']} {r['seconds']} s", flush=True)
    print("done", flush=True)

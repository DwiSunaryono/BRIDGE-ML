"""Setting (E) (PRAREG_SETTING_E.md): the bridge-gate program in its default procedure (protocol.run, identical to every
decision of MINI-21 and MINI-22), on new OpenML datasets that pass the paper's inclusion rules, with the same protocol as
(A)-(D): locked reference by the paper's rule and seed (LOCKED_REF_SEED), weighted-F1, 2,430 training rows (1,000 on small
datasets, chosen by the program), the frozen gate, and 100 trials per binary dataset or 60 per multiclass dataset.
Seed of trial r of the dataset at position p of the frozen list: 1_043_000_000 + 1000 * p + r (trial 0 is the one-run
seed of PRAREG_TABEL_C.md). The dev/reference split is made once per dataset and cached. One row per trial is appended to
the CSV as soon as it is known, and a rerun skips finished trials.
Usage: python setting_e.py <frozen_list.csv> <out.csv> <n_proc> [i/n]"""
import csv, json, os, subprocess, sys, tempfile, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tabel_c as TC

SPLIT = Path.home() / "settingE_split"
PKG = HERE.parent / "pkg" / "src"            # the verified program (bridge-gate 0.8.0), not an older installed copy
ENV = {**os.environ, "PYTHONPATH": str(PKG), "OMP_NUM_THREADS": "1"}
COLS = ["pos", "data_id", "rep", "seed", "status", "K", "rows", "total", "bags", "validated", "chosen", "accept", "naive_accept",
        "lower_bound", "gain_pool", "gain_ref", "ref_lo", "ref_hi", "ref_class", "ref_rows", "seconds"]


def split_paths(did):
    SPLIT.mkdir(exist_ok=True); d, r = SPLIT / f"{did}_dev.pkl", SPLIT / f"{did}_ref.pkl"
    if not (d.exists() and r.exists()):
        from sklearn.model_selection import train_test_split
        X, y, _ = TC.load(did)
        Xd, Xr, yd, yr = train_test_split(X, y, test_size=TC.ref_size(len(y)), stratify=y, random_state=TC.LOCKED_REF_SEED)
        for p, A, b in ((d, Xd, yd), (r, Xr, yr)):
            df = A.reset_index(drop=True).copy(); df["__target__"] = np.asarray(b); tmp = p.with_suffix(".tmp"); df.to_pickle(tmp); tmp.rename(p)
    return d, r


def run_one(job):
    pos, did, rep = job; seed = TC.OFF + 1000 * pos + rep; t0 = time.time()
    try:
        d, r = split_paths(did)
        with tempfile.TemporaryDirectory() as tmp:
            p = subprocess.run([sys.executable, "-m", "bridge_gate", "--dataset", str(d), "--reference", str(r), "--target", "__target__",
                                "--seed", str(seed), "--metric", "wF1", "--no-progress", "--out", f"{tmp}/res"], capture_output=True, text=True, env=ENV)
            if p.returncode != 0:
                raise RuntimeError((p.stderr or p.stdout).strip().splitlines()[-1][:200])
            j = json.load(open(f"{tmp}/res.json"))
        return dict(pos=pos, data_id=did, rep=rep, seed=seed, status="ok", K=j["K"], rows=j["rows"], total=j["total"], bags=j["bags"],
                    validated=j["validated"], chosen=j["chosen_config"], accept=j["accept"], naive_accept=j["naive_accept"],
                    lower_bound=repr(j["lower_bound"]), gain_pool=repr(j["gain_pool_chosen"]), gain_ref=repr(j["gain_ref"]),
                    ref_lo=repr(j["ref_interval"][0]), ref_hi=repr(j["ref_interval"][1]), ref_class=j["ref_class"],
                    ref_rows=j["reference_rows"], seconds=round(time.time() - t0, 1))
    except Exception as e:
        return dict(pos=pos, data_id=did, rep=rep, seed=seed, status="ERROR " + str(e)[:200], seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    lst = list(csv.DictReader(open(sys.argv[1])))
    out, n_proc = Path(sys.argv[2]), int(sys.argv[3])
    jobs = [(p, int(r["data_id"]), rep) for p, r in enumerate(lst) for rep in range(100 if int(float(r["K"])) == 2 else 60)]
    if len(sys.argv) > 4:   # shard by dataset, so a dataset's split is made on one machine
        i, n = map(int, sys.argv[4].split("/")); jobs = [j for j in jobs if j[0] % n == i - 1]
    done = {(int(r["data_id"]), int(r["rep"])) for r in csv.DictReader(open(out)) if r["status"] == "ok"} if out.exists() else set()
    todo = [j for j in jobs if (j[1], j[2]) not in done]
    print(f"{len(jobs)} trials in this part, {len(jobs) - len(todo)} already done", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as f, Pool(n_proc) as pool:
        w = csv.DictWriter(f, fieldnames=COLS, restval="")
        if new:
            w.writeheader()
        for k, r in enumerate(pool.imap_unordered(run_one, todo, chunksize=1)):
            w.writerow(r); f.flush()
            if k % 100 == 0 or r["status"] != "ok":
                print(f"{k + 1}/{len(todo)} {r['data_id']} rep {r['rep']}: {r['status']} {r['seconds']} s", flush=True)
    print("done", flush=True)

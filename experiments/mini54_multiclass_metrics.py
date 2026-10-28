"""MINI-54 (PRAREG_MINI54.md): class-balanced metrics on the multiclass datasets of the paper.

The eight multiclass datasets of collections (D) and (E) are rerun with the frozen gate and the reference procedure
(protocol.run_metrics: the models are fitted once per trial and judged under weighted F1, balanced accuracy, and
macro-F1). Data, locked references, seeds, training draw, bags, learners, folds and candidates are those of the stored
runs: (D) mini43_supporting_D.py, seed 1_044_000_000 + 1000 p + r; (E) setting_e.py, seed 1_043_000_000 + 1000 p + r.
Identity check: under weighted F1 the lower quantile and the reference gain must equal the stored values exactly.
Every finished trial is appended to the output CSV at once; a rerun skips finished trials.
Usage: python mini54_multiclass_metrics.py <out.csv> <n_proc> [dataset,dataset,...]"""
import csv, os, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pkg" / "src")); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini43_supporting_D as D43
sys.argv = _argv
from bridge_gate import core, protocol as P_

METRICS = core.METRICS_MULTI                       # wF1, bacc, macroF1
TRIALS = 60
SPLIT = Path.home() / "settingE_split"             # dev/reference splits cached by setting_e.py
OFF_E = 1_043_000_000
E_LIST = list(csv.DictReader(open(HERE / "settingE_daftar_beku.csv")))
JOBS_D = [(name, "D", pos) for pos, (kind, name) in enumerate(D43.ORDER) if kind == "mc"]
JOBS_E = [(str(int(r["data_id"])), "E", pos) for pos, r in enumerate(E_LIST) if int(float(r["K"])) > 2]
COLS = (["collection", "dataset", "rep", "seed", "K", "identical"]
        + [f"{k}_{m}" for m in METRICS for k in ("accept", "naive", "lo", "gain_pool", "ref", "ref_class")]
        + ["status", "seconds"])
_D = {}


def frame(path):
    """Same steps as the program reads a table: target column, rows with a label."""
    df = pd.read_pickle(path)
    X, y = df.drop(columns=["__target__"]), pd.Series(df["__target__"]).astype(str)
    keep = (y.notna() & (y != "nan")).to_numpy()
    return X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)


def data(coll, name):
    if (coll, name) not in _D:
        _D.clear()
        if coll == "D":
            Xd, yd, Xr, yr, K = D43.data("mc", name)
        else:
            X, ys = frame(SPLIT / f"{name}_dev.pkl"); Xr, yrs = frame(SPLIT / f"{name}_ref.pkl"); n = len(X)
            both = P_.encode(pd.concat([X, Xr], ignore_index=True)); Xd, Xr = both.iloc[:n], both.iloc[n:].reset_index(drop=True)
            names = sorted(set(ys) | set(yrs)); code = {c: i for i, c in enumerate(names)}
            yd, yr, K = ys.map(code).to_numpy(), yrs.map(code).to_numpy(), len(names)
        _D[(coll, name)] = (Xd, np.asarray(yd, int), Xr, np.asarray(yr, int), K)
    return _D[(coll, name)]


def stored():
    d = pd.concat([pd.read_csv(HERE / "out" / f"mini43_m{i}.csv", float_precision="round_trip") for i in (1, 2, 3)])
    s = {("D", r.dataset, int(r.rep)): (float(r.lower_bound), float(r.gain_ref)) for r in d.itertuples() if r.group == "D-mc"}
    e = pd.concat([pd.read_csv(HERE / "out" / f, float_precision="round_trip") for f in ("settingE_laptop.csv", "settingE_part2_laptop.csv")])
    s.update({("E", str(int(r.data_id)), int(r.rep)): (float(r.lower_bound), float(r.gain_ref)) for r in e.itertuples() if r.status == "ok"})
    return s


STORED = stored()


def run_one(job):
    name, coll, pos, rep = job; seed = (D43.OFF if coll == "D" else OFF_E) + 1000 * pos + rep; t0 = time.time()
    try:
        Xd, yd, Xr, yr, K = data(coll, name)
        R = P_.run_metrics(Xd, yd, seed, METRICS, X_ref=Xr, y_ref=yr, K=K)
        lo, ref = STORED[(coll, name, rep)]
        row = dict(collection=coll, dataset=name, rep=rep, seed=seed, K=K,
                   identical=int(R["wF1"].lower_bound == lo and R["wF1"].gain_ref == ref), status="ok")
        for m, r in R.items():
            row.update({f"accept_{m}": int(r.accept), f"naive_{m}": int(r.naive_accept), f"lo_{m}": repr(r.lower_bound),
                        f"gain_pool_{m}": repr(r.gain_pool), f"ref_{m}": repr(r.gain_ref), f"ref_class_{m}": r.ref_class})
        row["seconds"] = round(time.time() - t0, 1)
        return row
    except Exception as e:
        return dict(collection=coll, dataset=name, rep=rep, seed=seed, status="ERROR " + str(e)[:150], seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    out, n_proc = Path(sys.argv[1]), int(sys.argv[2])
    pick = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    jobs = [(n, c, p, r) for n, c, p in JOBS_D + JOBS_E if pick is None or n in pick for r in range(TRIALS)]
    done = set()
    if out.exists():
        done = {(r["dataset"], int(r["rep"])) for r in csv.DictReader(open(out)) if r["status"] == "ok"}
    todo = [j for j in jobs if (j[0], j[3]) not in done]
    print(f"{len(jobs)} trials in this part, {len(jobs) - len(todo)} already done", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as f, Pool(n_proc) as pool:
        w = csv.DictWriter(f, fieldnames=COLS, restval="")
        if new:
            w.writeheader()
        for k, r in enumerate(pool.imap_unordered(run_one, todo, chunksize=1)):
            w.writerow(r); f.flush()
            print(f"{k + 1}/{len(todo)} {r['dataset']} rep {r['rep']}: {r['status']} id={r.get('identical', '')} {r['seconds']} s", flush=True)
    print("done", flush=True)

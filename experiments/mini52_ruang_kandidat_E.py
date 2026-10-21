"""MINI-52: candidate-space density and full ablation on the binary datasets of setting (E).

The three binary HGB/DT/SVC spaces match MINI-35's K173, Ksedang and Kbesar.
Data, cached dev/reference splits, seeds, encoding, training draw, bags, learners, folds
and bootstrap seeds follow setting_e.py, which runs bridge_gate.protocol.run (bridge-gate 0.8.0).
K173 must reproduce the stored (E) lower_bound and gain_ref exactly.

Usage: python mini52_ruang_kandidat_E.py OUT.csv NPROC [i/3] [--smoke]
Shard i/3 assigns whole datasets (mini52_shards.csv), not individual repetitions.
"""
import csv
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pkg" / "src"))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

_argv = sys.argv
sys.argv = sys.argv[:1]
from mini35_ruang_kandidat import RUANG
sys.argv = _argv
from bridge_gate import core, protocol as P_

OFF = 1_043_000_000                      # tabel_c.OFF, the seed offset of setting_e.py
SPLIT = Path.home() / "settingE_split"   # cached dev/reference splits made by setting_e.py
METRIC = "wF1"
COLS = ["setting", "dataset", "data_id", "pos", "rep", "seed", "space", "nK", "cfg", "ref",
        "pt_pool", "lo_pool_boot", "pt_cf", "lo_cf_boot", "identical", "status", "seconds"]


def frozen():
    lst = list(csv.DictReader(open(HERE / "settingE_daftar_beku.csv")))
    return [(pos, int(r["data_id"]), r["name"]) for pos, r in enumerate(lst) if int(float(r["K"])) == 2]


def stored():
    files = sorted((HERE / "out").glob("settingE_*laptop.csv"))
    d = pd.concat([pd.read_csv(f, float_precision="round_trip") for f in files], ignore_index=True)
    return {(int(r.data_id), int(r.rep)): (float(r.lower_bound), float(r.gain_ref)) for r in d.itertuples()}


def frame(path):
    """Same steps as cli._frame and cli.run_reference."""
    df = pd.read_pickle(path)
    X, y = df.drop(columns=["__target__"]), df["__target__"]
    y = pd.Series(y).astype(str); keep = (y.notna() & (y != "nan")).to_numpy()
    return X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)


def setup(did, seed):
    X, ys = frame(SPLIT / f"{did}_dev.pkl")
    Xr, yrs = frame(SPLIT / f"{did}_ref.pkl")
    n = len(X)
    both = P_.encode(pd.concat([X, Xr], ignore_index=True)); X, Xr = both.iloc[:n], both.iloc[n:]
    names = sorted(set(ys) | set(yrs)); code = {c: i for i, c in enumerate(names)}
    assert len(names) == 2
    y, yr = ys.map(code).to_numpy().astype(int), yrs.map(code).to_numpy().astype(int)
    # protocol.run, binary branch
    if len(y) >= P_.TOTAL_TRAIN + 50:
        tot, ss = P_.TOTAL_TRAIN, P_.SEARCH_SIZE
    elif len(y) >= P_.SMALL_TOTAL + 50:
        tot, ss = P_.SMALL_TOTAL, P_.SMALL_SEARCH
    else:
        tot, ss = len(y), int(round(P_.SEARCH_SIZE / P_.TOTAL_TRAIN * len(y)))
    X = X.reset_index(drop=True)
    if tot < len(y):
        Xtr, _, ytr, _ = train_test_split(X, pd.Series(y), train_size=tot, stratify=y, random_state=seed)
    else:
        Xtr, ytr = X, pd.Series(y)
    Xs, Xp, ys_, yp = train_test_split(Xtr, ytr, train_size=ss, stratify=ytr, random_state=seed)
    Xs, ys_, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys_), Xp.reset_index(drop=True), np.asarray(yp)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(P_.BAG_FRACTION * len(ys_)); bags = []
    for _ in range(P_.BAGS):
        i = rng.choice(len(ys_), size=sub, replace=False)
        if np.bincount(ys_[i], minlength=2).min() >= 10:
            bags.append(i)
    if not bags:
        raise ValueError("SKIP: no usable bag")
    fitted = {m: [P_._Fitted(m, Xs.iloc[i].reset_index(drop=True), ys_[i], 2) for i in bags]
              for m in ("HGB", "DT", "SVC")}

    def avg(Xe):
        return {m: np.column_stack([f.proba(Xe) for f in fs]).mean(1) for m, fs in fitted.items()}
    P, Pr = avg(Xp), avg(Xr.reset_index(drop=True))
    return dict(yp=yp, yr=yr, P=P, Pr=Pr, base=(P["HGB"] >= .5).astype(int), baseref=(Pr["HGB"] >= .5).astype(int))


def evaluate(T, seed, space, candidates):
    yp, P, base = T["yp"], T["P"], T["base"]
    cnt = lambda y, b, p: core.counts(y, b, p, 2)
    delta = lambda c: float(core.delta(c, 2, METRIC))
    cf = np.zeros(8, int)
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp):
        c = P_._pick(yp, base, P, a, candidates, 2, METRIC)
        cf += cnt(yp[b], base[b], base[b] if c is None else P_.labels(P, c, 2, b))
    cfg = P_._pick(yp, base, P, np.arange(len(yp)), candidates, 2, METRIC)
    pool = cnt(yp, base, base if cfg is None else P_.labels(P, cfg, 2))
    refcnt = cnt(T["yr"], T["baseref"], T["baseref"] if cfg is None else P_.labels(T["Pr"], cfg, 2))
    boot_cf = seed + P_.BOOT_SEED_OFFSET_BIN + P_.METRICS_BINARY.index(METRIC)
    lo_pool = float("nan") if cfg is None else core.lower_bound(pool, 2, METRIC, seed + 62)
    return dict(space=space, nK=len(candidates), cfg=str(cfg), ref=delta(refcnt),
                pt_pool=0.0 if cfg is None else delta(pool), lo_pool_boot=lo_pool,
                pt_cf=delta(cf), lo_cf_boot=core.lower_bound(cf, 2, METRIC, boot_cf))


def run_one(job):
    pos, did, name, rep, want = job
    seed, start = OFF + 1000 * pos + rep, time.time()
    base = dict(setting="E", dataset=name, data_id=did, pos=pos, rep=rep, seed=seed)
    try:
        T = setup(did, seed)
        rows = []
        for space, candidates in RUANG.items():
            row = evaluate(T, seed, space, candidates)
            identical = int(space == "K173" and want is not None
                            and row["lo_cf_boot"] == want[0] and row["ref"] == want[1])
            rows.append(dict(**base, identical=identical, status="ok",
                             seconds=round(time.time() - start, 1), **row))
        return rows
    except Exception as exc:
        status = "SKIP" if str(exc).startswith("SKIP") else "ERROR " + str(exc)[:180]
        return [dict(**base, status=status, seconds=round(time.time() - start, 1))]


if __name__ == "__main__":
    out, nproc = Path(sys.argv[1]), int(sys.argv[2])
    saved = stored()
    selected = frozen()
    shards = [x for x in sys.argv[3:] if "/" in x]
    if shards:
        shard, count = map(int, shards[0].split("/"))
        if count != 3 or not 1 <= shard <= 3:
            raise SystemExit("shard must be i/3")
        plan = pd.read_csv(HERE / "mini52_shards.csv")
        keep = set(plan[plan.shard == shard].data_id.astype(int))
        selected = [s for s in selected if s[1] in keep]
    jobs = [(pos, did, name, rep, saved.get((did, rep))) for pos, did, name in selected for rep in range(100)]
    if "--smoke" in sys.argv:
        jobs = [j for j in jobs if j[3] == 0 and j[1] in (44, 42178)]
    done = set()
    if out.exists():
        counts = {}
        for row in csv.DictReader(out.open()):
            if row["status"] == "ok":
                key = (int(row["data_id"]), int(row["rep"]))
                counts[key] = counts.get(key, 0) + 1
        done = {key for key, c in counts.items() if c == 3}
    todo = [j for j in jobs if (j[1], j[3]) not in done]
    print(f"{len(jobs)} trials in this part, {len(jobs) - len(todo)} already done", flush=True)
    new = not out.exists()
    with out.open("a", newline="") as handle, Pool(nproc) as pool:
        writer = csv.DictWriter(handle, fieldnames=COLS, restval="")
        if new:
            writer.writeheader()
        for idx, rows in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            writer.writerows(rows)
            handle.flush()
            r = rows[0]
            print(f"{idx}/{len(todo)} {r['dataset']} rep {r['rep']}: {r['status']} {r['seconds']} s", flush=True)
    print("done", flush=True)

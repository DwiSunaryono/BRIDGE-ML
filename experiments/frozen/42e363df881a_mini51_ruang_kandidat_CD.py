"""MINI-51: candidate-space density on settings (C) and binary (D).

The three binary HGB/DT/SVC spaces are exactly MINI-35's K173, Ksedang,
and Kbesar.  Data, splits, bags, seeds, metrics, and locked references are
those of MINI-30 (C, net benefit at p_t=0.2) and MINI-43 (D, weighted-F1).

Usage: python mini51_ruang_kandidat_CD.py OUT.csv NPROC [i/n] [--smoke]
Shard i/n assigns whole datasets, not individual repetitions.
"""
import csv
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pkg" / "src"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

_argv = sys.argv
sys.argv = sys.argv[:1]
import exp41_gate_confirm as G
import bridge_metrics_klinis as KM
import mini30_klinis as M30
import mini43_supporting_D as D43
import mini44_tuning_tests as M44
from mini35_ruang_kandidat import RUANG
sys.argv = _argv
from bridge_gate import core, protocol as P_

ORDER = [("C", n) for n in M30.DS] + [("bin", n) for n in D43.BIN]
TRIALS = {"C": 200, "bin": 100}
K_NB20 = M30.METRIK.index("nb20")
COLS = ["setting", "dataset", "source", "rep", "seed", "space", "nK", "cfg", "ref",
        "pt_pool", "lo_pool_boot", "pt_cf", "lo_cf_boot", "identical", "status", "seconds"]


def stored():
    c = pd.concat([pd.read_csv(HERE / "out" / f"mini30_m{i}.csv", float_precision="round_trip",
                               keep_default_na=False) for i in (1, 2, 3)])
    c = c[c.metric == "nb20"]
    out = {(r.dataset, int(r.rep)): (float(r.lo95), float(r.ref)) for r in c.itertuples()}
    d = pd.concat([pd.read_csv(HERE / "out" / f"mini43_m{i}.csv", float_precision="round_trip")
                   for i in (1, 2, 3)])
    d = d[d.group == "D"]
    out.update({(r.dataset, int(r.rep)): (float(r.lower_bound), float(r.gain_ref)) for r in d.itertuples()})
    return out


def seed_of(kind, name, rep):
    if kind == "C":
        return M30.OFF + 1000 * M30.DS.index(name) + rep
    return D43.OFF + 1000 * [n for k, n in D43.ORDER].index(name) + rep


def setup_c(name, seed):
    Xd, yd, Xr, yr = M30.data(name)
    tot, ss = (G.TOTAL_TRAIN, G.SEARCH_SIZE) if len(Xd) >= G.TOTAL_TRAIN + 50 else (1000, 259)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=tot, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=ss, stratify=ytr, random_state=seed)
    Xs, ys = Xs.reset_index(drop=True), np.asarray(ys)
    Xp, yp = Xp.reset_index(drop=True), np.asarray(yp)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1))
    sub = int(G.SUBSAMPLE_FRAC * len(ys))
    bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS))
            if np.bincount(ys[i], minlength=2).min() >= 10]
    if not bags:
        raise ValueError("SKIP: no usable bag")
    cols, probs, pref = list(Xs.columns), {}, {}
    for model in ("HGB", "DT", "SVC"):
        fitted = [G.fit_proba_m(model, {}, Xs.iloc[i].reset_index(drop=True), ys[i], cols) for i in bags]
        probs[model] = np.column_stack([f(Xp) for f in fitted]).mean(1)
        pref[model] = np.column_stack([f(Xr) for f in fitted]).mean(1)
    base = (probs["HGB"] >= .5).astype(int)
    baseref = (pref["HGB"] >= .5).astype(int)
    return dict(yp=yp, yr=np.asarray(yr), P=probs, Pr=pref, base=base, baseref=baseref,
                count=G.counts8, delta=lambda c: KM.delta(c, "nb20"),
                boot=lambda c, s: KM.delta(np.random.default_rng(s).multinomial(
                    int(np.asarray(c).sum()), np.asarray(c) / np.asarray(c).sum(), size=20000), "nb20"),
                boot_cf=seed + 61 + K_NB20)


def setup_d(name, seed):
    Xd, yd, Xr, yr, K = M44.data("bin", name)
    assert K == 2
    tot, ss = ((P_.TOTAL_TRAIN, P_.SEARCH_SIZE) if len(yd) >= P_.TOTAL_TRAIN + 50
               else (P_.SMALL_TOTAL, P_.SMALL_SEARCH))
    Xtr, _, ytr, _ = train_test_split(Xd.reset_index(drop=True), pd.Series(yd), train_size=tot,
                                      stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=ss, stratify=ytr, random_state=seed)
    Xs, ys = Xs.reset_index(drop=True), np.asarray(ys)
    Xp, yp = Xp.reset_index(drop=True), np.asarray(yp)
    fitted = M44.fit_all(Xs, ys, 2, seed, None)
    probs = M44.avg(fitted, Xp, 2)
    pref = M44.avg(fitted, Xr.reset_index(drop=True), 2)
    base = (probs["HGB"] >= .5).astype(int)
    baseref = (pref["HGB"] >= .5).astype(int)
    bs = seed + P_.BOOT_SEED_OFFSET_BIN + P_.METRICS_BINARY.index("wF1")
    return dict(yp=yp, yr=np.asarray(yr, int), P=probs, Pr=pref, base=base, baseref=baseref,
                count=lambda y, b, c: core.counts(y, b, c, 2),
                delta=lambda c: core.delta(c, 2, "wF1"),
                boot=lambda c, s: M44.boot(c, 2, "wF1", s), boot_cf=bs)


def evaluate(T, seed, space, candidates):
    yp, P, base = T["yp"], T["P"], T["base"]

    def labels(cfg, idx=None, probs=P):
        return P_.labels(probs, cfg, 2, idx)

    def pick(idx):
        return P_._pick(yp, base, P, idx, candidates, 2,
                        "wF1") if T["setting"] == "D" else pick_c(idx)

    def pick_c(idx):
        best, gain0 = None, 0.0
        for cfg in candidates:
            gain = float(T["delta"](T["count"](yp[idx], base[idx], labels(cfg, idx))))
            if gain > gain0 + 1e-12:
                best, gain0 = cfg, gain
        return best

    cf = np.zeros(8, int)
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp):
        cfg_fold = pick(a)
        pred = base[b] if cfg_fold is None else labels(cfg_fold, b)
        cf += T["count"](yp[b], base[b], pred)
    cfg = pick(np.arange(len(yp)))
    pred_pool = base if cfg is None else labels(cfg)
    pool = T["count"](yp, base, pred_pool)
    pred_ref = T["baseref"] if cfg is None else P_.labels(T["Pr"], cfg, 2)
    refcnt = T["count"](T["yr"], T["baseref"], pred_ref)
    lo_pool = float("nan") if cfg is None else float(np.quantile(T["boot"](pool, seed + 62), .05))
    return dict(space=space, nK=len(candidates), cfg=str(cfg), ref=float(T["delta"](refcnt)),
                pt_pool=float(T["delta"](pool)), lo_pool_boot=lo_pool,
                pt_cf=float(T["delta"](cf)),
                lo_cf_boot=float(np.quantile(T["boot"](cf, T["boot_cf"]), .05)))


def run_one(job):
    kind, name, rep, want = job
    seed, start = seed_of(kind, name, rep), time.time()
    setting = "C" if kind == "C" else "D"
    source = M30.KLASTER[name] if kind == "C" else name
    try:
        T = setup_c(name, seed) if kind == "C" else setup_d(name, seed)
        T["setting"] = setting
        rows = []
        for space, candidates in RUANG.items():
            row = evaluate(T, seed, space, candidates)
            identical = int(space == "K173" and want is not None
                            and row["lo_cf_boot"] == want[0] and row["ref"] == want[1])
            rows.append(dict(setting=setting, dataset=name, source=source, rep=rep, seed=seed,
                             identical=identical, status="ok", seconds=round(time.time() - start, 1), **row))
        return rows
    except Exception as exc:
        status = "SKIP" if str(exc).startswith("SKIP") else "ERROR " + str(exc)[:180]
        return [dict(setting=setting, dataset=name, source=source, rep=rep, seed=seed,
                     status=status, seconds=round(time.time() - start, 1))]


if __name__ == "__main__":
    out, nproc = Path(sys.argv[1]), int(sys.argv[2])
    saved = stored()
    selected = ORDER
    shards = [x for x in sys.argv[3:] if "/" in x]
    if shards:
        shard, count = map(int, shards[0].split("/"))
        if not (1 <= shard <= count):
            raise SystemExit("shard must be i/n with 1 <= i <= n")
        if count == 3:
            plan = pd.read_csv(HERE / "mini51_shards.csv")
            selected = [("bin" if r.setting == "D" else r.setting, r.dataset)
                        for r in plan[plan.shard == shard].itertuples()]
        else:
            selected = ORDER[shard - 1::count]
    jobs = [(kind, name, rep, saved.get((name, rep))) for kind, name in selected
            for rep in range(TRIALS[kind])]
    if "--smoke" in sys.argv:
        jobs = [j for j in jobs if j[2] == 0 and ((j[0] == "C" and j[1] == "mammography")
                                                  or (j[0] == "bin" and j[1] == "compas"))]
    done = set()
    if out.exists():
        prior = list(csv.DictReader(out.open()))
        counts = {}
        for row in prior:
            if row["status"] == "ok":
                key = (row["dataset"], int(row["rep"]))
                counts[key] = counts.get(key, 0) + 1
        done = {key for key, count in counts.items() if count == 3}
    todo = [job for job in jobs if (job[1], job[2]) not in done]
    print(f"{len(jobs)} trials in this part, {len(jobs)-len(todo)} already done", flush=True)
    new = not out.exists()
    with out.open("a", newline="") as handle, Pool(nproc) as pool:
        writer = csv.DictWriter(handle, fieldnames=COLS, restval="")
        if new:
            writer.writeheader()
        for idx, rows in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            writer.writerows(rows)
            handle.flush()
            status = rows[0]["status"]
            print(f"{idx}/{len(todo)} {rows[0]['dataset']} rep {rows[0]['rep']}: {status} {rows[0]['seconds']} s", flush=True)
    print("done", flush=True)

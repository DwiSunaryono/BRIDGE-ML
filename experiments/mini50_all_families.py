"""MINI-50 (PRAREG_MINI50.md): does the frozen gate meet the criterion when all tested model families
enter the candidate space at once, instead of being swapped in pairs (MINI-45)?
The current model (HGB at threshold 0.5 or most probable class), the bags, the gate and every seed are
the paper's. The candidate space applies the paper's structure to every member family tested so far:
  binary: HGB thresholds (18); HGB + X two-model mixtures (weights .75/.5/.25 x 19 thresholds) for every
          X in {DT, SVC, LR, NB, MLP, KNN, XGB}; three-model simplex mixtures (14 weights x 7 thresholds)
          for every tested pair (DT-SVC, LR-NB, MLP-KNN): 18 + 7*57 + 3*98 = 711 configurations
          (paper: 173);
  multiclass: simplex mixtures for the three pairs (3*14); HGB class scaling (4K); HGB + X (.5/.5) with
          class scaling for every X (7*4K): 42 + 32K configurations (paper: 14 + 8K).
Members: DT and SVC as in the paper; LR, NB, MLP, KNN as in MINI-45; XGB = XGBoost with early stopping
as in MINI-46 (XGBES). No member is tuned. Same trials as MINI-24 (A) and MINI-43 (D), so the stored
DT-SVC decisions are the paired default arm.
Usage: python mini50_all_families.py <out.csv> <n_proc> [i/n] [--mc-shards=2,3] [--smoke]"""
import csv, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pkg" / "src")); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini44_tuning_tests as M44
import mini45_member_families as M45
import mini46_base_learners_real as M46
sys.argv = _argv
from bridge_gate import protocol as P_

EXTRA = ("LR", "NB", "MLP", "KNN", "XGB")
MEMBERS = ("DT", "SVC") + EXTRA
PAIRS = (("DT", "SVC"), ("LR", "NB"), ("MLP", "KNN"))
COLS = ["setting", "dataset", "rep", "seed", "K", "total", "n_cand", "lb05", "cfg", "family", "gain_pool", "gain_ref",
        "accept", "naive", "status", "seconds"]


def space(K):
    """All-families candidate space. Order: thresholds, two-model mixtures by member, simplex by pair."""
    if K == 2:
        thr = [({"HGB": 1}, t) for t in P_.THRESHOLDS if t != .5]
        e2 = [({"HGB": w, x: 1 - w}, t) for x in MEMBERS for w in (.75, .5, .25) for t in P_.THRESHOLDS]
        e3 = [({"HGB": a, p: b, q: c}, t) for p, q in PAIRS for a, b, c in P_.SIMPLEX for t in (.3, .35, .4, .45, .5, .55, .6)]
        return thr + e2 + e3
    e = [({"HGB": a, p: b, q: c}, None) for p, q in PAIRS for a, b, c in P_.SIMPLEX]
    s = [({"HGB": 1}, (k, v)) for k in range(K) for v in P_.SCALES]
    es = [({"HGB": .5, x: .5}, (k, v)) for x in MEMBERS for k in range(K) for v in P_.SCALES]
    return e + s + es


_ORIG = P_._learner


def learner(name, n):
    if name in ("LR", "NB", "MLP", "KNN"):
        return M45.member(name)
    if name == "XGB":
        return M46.XGB(True)
    return _ORIG(name, n)


def fit_all(Xs, ys, K, seed):
    """Fit bags exactly as protocol.run. Every member is fitted on the same kept bags."""
    P_._learner = learner
    try:
        rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(P_.BAG_FRACTION * len(ys)); bags = []
        for _ in range(P_.BAGS):
            i = rng.choice(len(ys), size=sub, replace=False); cc = np.bincount(ys[i], minlength=K)
            if K == 2:
                ok = cc.min() >= 10
            else:
                cc = cc[cc > 0]; ok = len(cc) >= 2 and cc.min() >= 2
            if ok:
                bags.append(i)
        return {m: [P_._Fitted(m, Xs.iloc[i].reset_index(drop=True), ys[i], K) for i in bags] for m in ("HGB",) + MEMBERS}
    finally:
        P_._learner = _ORIG


def family(cfg):
    if cfg is None:
        return ""
    return "+".join(m for m, wi in cfg[0].items() if wi)


def run_one(job):
    kind, name, rep, _ = job; seed = M44.seed_of(kind, name, rep); t0 = time.time(); metric = "wF1"
    setting = "A" if kind == "A" else ("D" if kind == "bin" else "D-mc")
    orig_cand = P_.candidates
    try:
        Xd, yd, Xr, yr, K = M44.data(kind, name)
        TOT, SS = (P_.TOTAL_TRAIN, P_.SEARCH_SIZE) if len(yd) >= P_.TOTAL_TRAIN + 50 else (P_.SMALL_TOTAL, P_.SMALL_SEARCH)
        X = Xd.reset_index(drop=True)
        Xtr, _, ytr, _ = train_test_split(X, pd.Series(yd), train_size=TOT, stratify=yd, random_state=seed)
        Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
        Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
        fz = fit_all(Xs, ys, K, seed)
        P_.candidates = space
        P = M44.avg(fz, Xp, K); draws, cfg, g, _ = M44.gate_parts(yp, P, K, seed, metric)
        lb = float(np.quantile(draws, .05)); gr, _, _ = M44.ref_eval(fz, cfg, Xr, yr, K, metric)
        row = dict(setting=setting, dataset=name, rep=rep, seed=seed, K=K, total=TOT, n_cand=len(space(K)), lb05=repr(lb),
                   cfg=str(cfg), family=family(cfg), gain_pool=repr(g), gain_ref=repr(gr),
                   accept=int(cfg is not None and lb > .005), naive=int(g > 0), status="ok", seconds=round(time.time() - t0, 1))
        return row
    except Exception as e:
        return dict(setting=setting, dataset=name, rep=rep, seed=seed, status="ERROR " + str(e)[:150], seconds=round(time.time() - t0, 1))
    finally:
        P_.candidates = orig_cand


if __name__ == "__main__":
    out, n_proc = Path(sys.argv[1]), int(sys.argv[2])
    jobs = [(kind, name, rep, None) for kind, name in M44.ORDER for rep in range(M44.TRIALS[kind])]
    if "--smoke" in sys.argv:
        jobs = [j for j in jobs if (j[1], j[2]) in (("LICD", 0), ("compas", 0), ("DryBean", 0))]
        if "--no-mc" in sys.argv:
            jobs = [j for j in jobs if j[0] != "mc"]
    pos = [a for a in sys.argv[3:] if "/" in a]
    if pos:   # Binary jobs: every n-th from the i-th. Multiclass jobs only on shards in --mc-shards
              # because XGBoost multiclass fails on the laptop's xgboost build during the CUDA driver
              # check; take every k-th of them
        i, n = map(int, pos[0].split("/"))
        mcs = [int(x) for x in next((a.split("=")[1] for a in sys.argv if a.startswith("--mc-shards=")), ",".join(map(str, range(1, n + 1)))).split(",")]
        jb, jm = [j for j in jobs if j[0] != "mc"], [j for j in jobs if j[0] == "mc"]
        jobs = jb[i - 1::n] + (jm[mcs.index(i)::len(mcs)] if i in mcs else [])
    done = set()
    if out.exists():
        done = {(r["dataset"], int(r["rep"])) for r in csv.DictReader(open(out)) if r["status"] == "ok"}
    todo = [j for j in jobs if (j[1], j[2]) not in done]
    print(f"{len(jobs)} trials in this part, {len(jobs) - len(todo)} already done", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as f, Pool(n_proc) as pool:
        w = csv.DictWriter(f, fieldnames=COLS, restval="")
        if new:
            w.writeheader()
        for k, r in enumerate(pool.imap_unordered(run_one, todo, chunksize=1)):
            w.writerow(r); f.flush()
            if k % 50 == 0 or r["status"] != "ok":
                print(f"{k + 1}/{len(todo)} {r['dataset']} rep {r['rep']}: {r['status']} {r['seconds']} s", flush=True)
    print("done", flush=True)

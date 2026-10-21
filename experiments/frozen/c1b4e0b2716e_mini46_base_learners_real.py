"""MINI-46 (PRAREG_MINI46.md): base learners other than HGB on the real data. The base learner (the slot 'HGB': current
model and member of every candidate) is replaced; DT, SVC, the candidate-space structure, the gate and every seed are the
paper's. Arms (settings taken unchanged from the synthetic-population runs MINI-34 and MINI-36):
  XGB0  : the project-default XGBoost (lr 0.1, 300 rounds, max_depth 6, min_child_weight 1, reg_lambda 1), no early stopping;
  XGBES : the same plus early stopping after 10 rounds on a stratified 10% validation split (MINI-34 X_ES);
  MLPES : MLP (64, 32), early stopping (validation 10%, 10 iterations), max_iter 300 (MINI-36 MLP_ES).
Same trials as MINI-24 (A) and MINI-43 (D): same seeds, outer split, SEARCH/POOL split and bags; the stored HGB decisions
are the paired default arm. No learner is tuned.
Usage: python mini46_base_learners_real.py <out.csv> <n_proc> [i/n] [--mc-shards=2,3] [--smoke]"""
import csv, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pkg" / "src")); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini44_tuning_tests as M44
sys.argv = _argv
from bridge_gate import protocol as P_

BAKU = dict(learning_rate=.1, n_estimators=300, max_depth=6, min_child_weight=1.0, subsample=1.0, colsample_bytree=1.0,
            reg_lambda=1.0, random_state=42, eval_metric="logloss", n_jobs=1)
ARMS = ("XGB0", "XGBES", "MLPES")
COLS = (["setting", "dataset", "rep", "seed", "K", "total"]
        + [f"{a}_{c}" for a in ARMS for c in ("lb05", "cfg", "gain_pool", "gain_ref", "accept", "naive")]
        + ["status", "seconds"])


class XGB:
    """XGBoost with the classes present in the bag re-coded 0..m-1 (classes_ keeps the original codes)."""

    def __init__(self, es):
        self.es = es

    def fit(self, Z, y):
        self.classes_ = np.unique(y); yy = np.searchsorted(self.classes_, y)
        self.est = XGBClassifier(**{**BAKU, "eval_metric": "logloss" if len(self.classes_) == 2 else "mlogloss",
                                 **({"early_stopping_rounds": 10} if self.es else {})})
        if self.es:
            Ztr, Zva, ytr, yva = train_test_split(Z, yy, test_size=.1, stratify=yy, random_state=42)
            self.est.fit(Ztr, ytr, eval_set=[(Zva, yva)], verbose=False)
        else:
            self.est.fit(Z, yy)
        return self

    def predict_proba(self, Z):
        p = self.est.predict_proba(Z)
        return p if p.ndim == 2 else np.column_stack([1 - p, p])


def base(arm):
    if arm == "XGB0":
        return XGB(False)
    if arm == "XGBES":
        return XGB(True)
    return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, validation_fraction=.1,
                         n_iter_no_change=10, random_state=42)


_ORIG = P_._learner


def fit_arm(Xs, ys, K, seed, arm, others=None):
    P_._learner = lambda name, n: base(arm) if name == "HGB" else _ORIG(name, n)
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
        fit = lambda m: [P_._Fitted(m, Xs.iloc[i].reset_index(drop=True), ys[i], K) for i in bags]
        out = {"HGB": fit("HGB")}
        out.update(others if others is not None else {"DT": fit("DT"), "SVC": fit("SVC")})
        return out
    finally:
        P_._learner = _ORIG


def run_one(job):
    kind, name, rep, _ = job; seed = M44.seed_of(kind, name, rep); t0 = time.time(); metric = "wF1"
    setting = "A" if kind == "A" else ("D" if kind == "bin" else "D-mc")
    try:
        Xd, yd, Xr, yr, K = M44.data(kind, name)
        TOT, SS = (P_.TOTAL_TRAIN, P_.SEARCH_SIZE) if len(yd) >= P_.TOTAL_TRAIN + 50 else (P_.SMALL_TOTAL, P_.SMALL_SEARCH)
        X = Xd.reset_index(drop=True)
        Xtr, _, ytr, _ = train_test_split(X, pd.Series(yd), train_size=TOT, stratify=yd, random_state=seed)
        Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
        Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
        row = dict(setting=setting, dataset=name, rep=rep, seed=seed, K=K, total=TOT); others = None
        for arm in ARMS:
            fz = fit_arm(Xs, ys, K, seed, arm, others); others = {"DT": fz["DT"], "SVC": fz["SVC"]}
            P = M44.avg(fz, Xp, K); draws, cfg, g, _ = M44.gate_parts(yp, P, K, seed, metric)
            lb = float(np.quantile(draws, .05)); gr, _, _ = M44.ref_eval(fz, cfg, Xr, yr, K, metric)
            row.update({f"{arm}_lb05": repr(lb), f"{arm}_cfg": str(cfg), f"{arm}_gain_pool": repr(g), f"{arm}_gain_ref": repr(gr),
                        f"{arm}_accept": int(cfg is not None and lb > .005), f"{arm}_naive": int(g > 0)})
        row.update(status="ok", seconds=round(time.time() - t0, 1)); return row
    except Exception as e:
        return dict(setting=setting, dataset=name, rep=rep, seed=seed, status="ERROR " + str(e)[:150], seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    out, n_proc = Path(sys.argv[1]), int(sys.argv[2])
    jobs = [(kind, name, rep, None) for kind, name in M44.ORDER for rep in range(M44.TRIALS[kind])]
    if "--smoke" in sys.argv:
        jobs = [j for j in jobs if (j[1], j[2]) in (("LICD", 0), ("compas", 0), ("DryBean", 0))]
    pos = [a for a in sys.argv[3:] if "/" in a]
    if pos:   # binary jobs: every n-th from the i-th; multiclass jobs only on the shards in --mc-shards (XGBoost multiclass
              # fails on the laptop's xgboost build, CUDA driver check), every k-th of them
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

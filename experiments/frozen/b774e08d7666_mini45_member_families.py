"""MINI-45 (PRAREG_MINI45.md): does the frozen gate stay safe when the candidate space is built from other model families?
The current model (HGB at threshold 0.5 / most probable class), the bags, the candidate-space structure (173 binary
configurations, 14 + 8K multiclass), the gate and every seed are the paper's; only the two extra members DT and SVC are
replaced by another pair:
  arm LR-NB  : logistic regression (C = 1, max_iter 1000) and Gaussian naive Bayes (library defaults);
  arm MLP-KNN: MLP (64, 32) with early stopping (validation 10%, 10 iterations without improvement, max_iter 300; the
               MINI-36 setting) and KNN (k = 5; the MINI-09 setting).
Both arms run on the same trials as MINI-24 (A) and MINI-43 (D) (same seeds, so the outer split, SEARCH/POOL split and
bags are identical and the stored DT-SVC decisions are the paired default arm). No member is tuned.
Usage: python mini45_member_families.py <out.csv> <n_proc> [i/n] [--smoke]"""
import csv, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pkg" / "src")); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini44_tuning_tests as M44
sys.argv = _argv
from bridge_gate import protocol as P_

ARMS = {"LR-NB": ("LR", "NB"), "MLP-KNN": ("MLP", "KNN")}
COLS = (["setting", "dataset", "rep", "seed", "K", "total"]
        + [f"{a}_{c}" for a in ARMS for c in ("lb05", "cfg", "gain_pool", "gain_ref", "accept", "naive")]
        + ["status", "seconds"])


def member(name):
    if name == "LR":
        return LogisticRegression(C=1.0, max_iter=1000, random_state=P_.MODEL_SEED)
    if name == "NB":
        return GaussianNB()
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, validation_fraction=0.1,
                             n_iter_no_change=10, random_state=P_.MODEL_SEED)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=5)
    raise ValueError(name)


_ORIG = P_._learner


def fit_arm(Xs, ys, K, seed, pair, hgb=None):
    """Bags exactly as protocol.run; the slots 'DT' and 'SVC' hold the arm's two members (so the paper's candidate
    space is reused unchanged). hgb: already fitted HGB list (identical in every arm) to avoid refitting."""
    P_._learner = lambda name, n: member(pair[0]) if name == "DT" else member(pair[1]) if name == "SVC" else _ORIG(name, n)
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
        return {"HGB": hgb if hgb is not None else fit("HGB"), "DT": fit("DT"), "SVC": fit("SVC")}
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
        row = dict(setting=setting, dataset=name, rep=rep, seed=seed, K=K, total=TOT); hgb = None
        for arm, pair in ARMS.items():
            fz = fit_arm(Xs, ys, K, seed, pair, hgb); hgb = fz["HGB"]
            P = M44.avg(fz, Xp, K); draws, cfg, g, _ = M44.gate_parts(yp, P, K, seed, metric)
            lb = float(np.quantile(draws, .05)); gr, _, _ = M44.ref_eval(fz, cfg, Xr, yr, K, metric)
            row.update({f"{arm}_lb05": repr(lb), f"{arm}_cfg": str(cfg), f"{arm}_gain_pool": repr(g), f"{arm}_gain_ref": repr(gr),
                        f"{arm}_accept": int(cfg is not None and lb > .005), f"{arm}_naive": int(g > 0)})
            del fz["DT"], fz["SVC"]
        row.update(status="ok", seconds=round(time.time() - t0, 1)); return row
    except Exception as e:
        return dict(setting=setting, dataset=name, rep=rep, seed=seed, status="ERROR " + str(e)[:150], seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    out, n_proc = Path(sys.argv[1]), int(sys.argv[2])
    jobs = [(kind, name, rep, None) for kind, name in M44.ORDER for rep in range(M44.TRIALS[kind])]
    if "--smoke" in sys.argv:
        jobs = [j for j in jobs if (j[1], j[2]) in (("LICD", 0), ("compas", 0), ("DryBean", 0))]
    pos = [a for a in sys.argv[3:] if "/" in a]
    if pos:
        i, n = map(int, pos[0].split("/")); jobs = jobs[i - 1::n]
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

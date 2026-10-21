"""MINI-33: XGBoost yang DISETARAKAN dgn HGB sbg model dasar (PRAREG_MINI33.md) — menindaklanjuti MINI-31 (XGB bawaan gagal).
Pertanyaan: apakah kegagalan MINI-31 sifat keluarga gradient boosting, atau akibat pengaturan XGB yang overfit/tak terkalibrasi?
XGB_eq (setara HGB proyek): early stopping 10 ronde pada 10% validasi internal bertingkat (spt HGB early_stopping=True,
n_iter_no_change=10, validation_fraction=0,1), maks 300 ronde, lr 0,1, grow_policy='lossguide', max_leaves=31 (spt
max_leaf_nodes=31), max_depth=0, min_child_weight=5 (setara ~20 sampel x p(1-p) <= 0,25), reg_lambda=0 (spt l2=0).
Tugas BARU populasi Q: T1500-T2899 (belum pernah dijalankan; pilot T1400-T1459 & MINI-25 T0000-T1399 tidak dipakai).
Pada tugas yang sama dijalankan DUA kebijakan: model dasar HGB (seperti MINI-25) dan model dasar XGB_eq; gate baku.
Pakai: python mini33_xgb_setara.py <awal> <akhir> <n_proses> <out>"""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini25_sintetis as Q
sys.argv = _argv
import exp41_gate_confirm as G, bridge_metrics as BM, data as Dd
from mini12_optuna import union, lab
from xgboost import XGBClassifier
F = Q.F
F.DS = {f"T{i:04d}": ("sintetis", i, ()) for i in range(2900)}  # urutan kunci = i -> seed = OFF + 1000*i


def fit_xgb_eq(Xf, yf):
    cols = list(Xf.columns); num_c, cat_c = Dd.column_kinds(Xf)
    pre = Dd.make_preprocessor(num_c, cat_c); Z = pre.fit_transform(Xf, yf)
    Ztr, Zva, ytr, yva = train_test_split(Z, yf, test_size=.1, stratify=yf, random_state=42)
    est = XGBClassifier(n_estimators=300, learning_rate=.1, grow_policy="lossguide", max_leaves=31, max_depth=0,
                        min_child_weight=5, reg_lambda=0.0, early_stopping_rounds=10, eval_metric="logloss",
                        tree_method="hist", random_state=42, n_jobs=1)
    est.fit(Ztr, ytr, eval_set=[(Zva, yva)], verbose=False)
    return lambda Xn: est.predict_proba(pre.transform(Xn[cols]))[:, 1]


def run_job(job):
    try:
        return _trial(job)
    except Exception as e:
        return [dict(dataset=job[0], base="", cfg="ERROR " + str(e)[:150])]


def _trial(job):
    ds, rep = job; seed = F.OFF + 1000 * list(F.DS).index(ds) + rep
    Xd, yd, Xr, yr = F.data(ds)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=G.TOTAL_TRAIN, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=G.SEARCH_SIZE, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS)) if np.bincount(ys[i], minlength=2).min() >= 10]
    cols = list(Xs.columns); P, Pr = {}, {}
    for m in ("HGB", "XGBeq", "DT", "SVC"):
        fs = [fit_xgb_eq(Xs.iloc[i].reset_index(drop=True), ys[i]) if m == "XGBeq" else
              G.fit_proba_m(m, {}, Xs.iloc[i].reset_index(drop=True), ys[i], cols) for i in bags]
        P[m] = np.column_stack([f(Xp) for f in fs]).mean(1); Pr[m] = np.column_stack([f(Xr) for f in fs]).mean(1)
    out = []
    for B0 in ("HGB", "XGBeq"):
        U = union((B0, "DT", "SVC")); base, baser = (P[B0] >= .5).astype(int), (Pr[B0] >= .5).astype(int)
        pk = lambda idx: F.pick(yp, P, idx, U, base, "wF1")
        cnt = np.zeros(8, int)
        for a, b in StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp):
            c = pk(a); cnt += G.counts8(yp[b], base[b], base[b] if c is None else lab(P, c, b))
        lo95 = float(np.quantile(BM.delta(np.random.default_rng(seed + 61).multinomial(len(yp), cnt / len(yp), size=20000), "wF1"), .05))
        cfg = pk(np.arange(len(yp))); cr = G.counts8(yr, baser, baser if cfg is None else lab(Pr, cfg))
        dr = BM.delta(np.random.default_rng(seed + 987163).dirichlet(cr + .5, size=20000), "wF1"); lo, hi = np.quantile(dr, [.025, .975])
        pr = np.clip(Pr[B0], 1e-6, 1 - 1e-6)
        out.append(dict(dataset=ds, base=B0, cfg=str(cfg), lo95=lo95, ref=float(BM.delta(cr, "wF1")),
                        kelas="membaik" if lo > 0 else "memburuk" if hi < 0 else "tidak-pasti",
                        logloss_base=float(-np.mean(yr * np.log(pr) + (1 - yr) * np.log(1 - pr)))))
    return out


if __name__ == "__main__":
    a, b, Pn, outn = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    jobs = [(f"T{i:04d}", 0) for i in range(a, b)]; t0 = time.time()
    with Pool(Pn) as pool:
        res = [x for rows in pool.imap(run_job, jobs, chunksize=2) for x in rows]
    keys = list(next(r for r in res if "ref" in r)); p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(res)
    print(f"tugas {a}-{b}, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

"""MINI-34: XGBoost SATU-FAKTOR + diagnostik model dasar yang dapat diukur praktisi (PRAREG_MINI34.md).
Dari XGB bawaan proyek (models.build_estimator('XGB'): lr 0,1, 300 ronde, max_depth 6, min_child_weight 1, reg_lambda 1,
tanpa early stopping) diubah SATU pengaturan per kondisi:
  X0 bawaan | X_ES +early stopping 10 ronde (validasi 10% bertingkat) | X_MCW5 min_child_weight 5 | X_LEAF31 lossguide
  max_leaves 31, max_depth 0 | X_L2_0 reg_lambda 0 | X_N30, X_N100 ronde tetap 30/100 tanpa early stopping
ditambah HGB dan XGBeq (MINI-33; harus mereplikasi lo95 MINI-33 persis = cek konsistensi).
Diagnostik di POOL (tersedia saat deploy, tanpa data referensi; model dasar dilatih di SEARCH sehingga POOL = data luar
sampel): log-loss POOL, log-loss nol (prevalensi SEARCH), skill = 1 - LL/LL_nol, % probabilitas ekstrem di POOL.
Tugas Q T1500-T2899 (pembagian identik MINI-33). Pakai: python mini34_xgb_faktor.py <awal> <akhir> <n_proses> <out>"""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini33_xgb_setara as M33
sys.argv = _argv
import exp41_gate_confirm as G, bridge_metrics as BM, data as Dd
from mini12_optuna import union, lab
from xgboost import XGBClassifier
F = M33.F
BAKU = dict(learning_rate=.1, n_estimators=300, max_depth=6, min_child_weight=1.0, subsample=1.0, colsample_bytree=1.0,
            reg_lambda=1.0, random_state=42, eval_metric="logloss", n_jobs=1)  # == models.build_estimator('XGB', {})
KOND = {"X0": ({}, False), "X_ES": ({}, True), "X_MCW5": (dict(min_child_weight=5.0), False),
        "X_LEAF31": (dict(grow_policy="lossguide", max_leaves=31, max_depth=0), False), "X_L2_0": (dict(reg_lambda=0.0), False),
        "X_N30": (dict(n_estimators=30), False), "X_N100": (dict(n_estimators=100), False)}


def fit_xgb(Xf, yf, ubah, es):
    cols = list(Xf.columns); num_c, cat_c = Dd.column_kinds(Xf)
    pre = Dd.make_preprocessor(num_c, cat_c); Z = pre.fit_transform(Xf, yf)
    est = XGBClassifier(**{**BAKU, **ubah, **({"early_stopping_rounds": 10} if es else {})})
    if es:
        Ztr, Zva, ytr, yva = train_test_split(Z, yf, test_size=.1, stratify=yf, random_state=42)
        est.fit(Ztr, ytr, eval_set=[(Zva, yva)], verbose=False)
    else:
        est.fit(Z, yf)
    return lambda Xn: est.predict_proba(pre.transform(Xn[cols]))[:, 1]


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


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
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp); yr = np.asarray(yr)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS)) if np.bincount(ys[i], minlength=2).min() >= 10]
    cols = list(Xs.columns); P, Pr = {}, {}
    for m in ("HGB", "XGBeq", "DT", "SVC", *KOND):
        sub_ = lambda i: (Xs.iloc[i].reset_index(drop=True), ys[i])
        fs = [M33.fit_xgb_eq(*sub_(i)) if m == "XGBeq" else fit_xgb(*sub_(i), *KOND[m]) if m in KOND else
              G.fit_proba_m(m, {}, *sub_(i), cols) for i in bags]
        P[m] = np.column_stack([f(Xp) for f in fs]).mean(1); Pr[m] = np.column_stack([f(Xr) for f in fs]).mean(1)
    pi = ys.mean(); ll0 = ll(yp, np.full(len(yp), pi)); out = []
    for B0 in ("HGB", "XGBeq", *KOND):
        U = union((B0, "DT", "SVC")); base, baser = (P[B0] >= .5).astype(int), (Pr[B0] >= .5).astype(int)
        pk = lambda idx: F.pick(yp, P, idx, U, base, "wF1")
        cnt = np.zeros(8, int)
        for a, b in StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp):
            c = pk(a); cnt += G.counts8(yp[b], base[b], base[b] if c is None else lab(P, c, b))
        lo95 = float(np.quantile(BM.delta(np.random.default_rng(seed + 61).multinomial(len(yp), cnt / len(yp), size=20000), "wF1"), .05))
        cfg = pk(np.arange(len(yp))); cpool = G.counts8(yp, base, base if cfg is None else lab(P, cfg))
        cr = G.counts8(yr, baser, baser if cfg is None else lab(Pr, cfg))
        dr = BM.delta(np.random.default_rng(seed + 987163).dirichlet(cr + .5, size=20000), "wF1"); lo, hi = np.quantile(dr, [.025, .975])
        llp = ll(yp, P[B0])
        out.append(dict(dataset=ds, base=B0, cfg=str(cfg), lo95=lo95, pt_pool=float(BM.delta(cpool, "wF1")), ref=float(BM.delta(cr, "wF1")),
                        kelas="membaik" if lo > 0 else "memburuk" if hi < 0 else "tidak-pasti",
                        ll_pool=llp, ll_nol=ll0, skill_pool=1 - llp / ll0, ekstrem_pool=100 * float(np.mean((P[B0] < .01) | (P[B0] > .99))),
                        ll_ref=ll(yr, Pr[B0])))
    return out


if __name__ == "__main__":
    a, b, Pn, outn = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    jobs = [(f"T{i:04d}", 0) for i in range(a, b)]; t0 = time.time()
    with Pool(Pn) as pool:
        res = [x for rows in pool.imap(run_job, jobs, chunksize=1) for x in rows]
    keys = list(next(r for r in res if "ref" in r)); p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(res)
    print(f"tugas {a}-{b}, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

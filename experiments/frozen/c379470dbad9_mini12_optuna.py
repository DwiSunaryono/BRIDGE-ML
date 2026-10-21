"""MINI-12 (PENGEMBANGAN): Optuna untuk HGB, gate BRIDGE BEKU (kuantil-5% > 0.005).

Per trial, HANYA di SEARCH (630 baris): Optuna TPE (seed = seed trial), 25 percobaan,
CV stratified 3-lipat, objektif weighted-F1 @0.5, ruang = config.SEARCH_SPACES["HGB"]
(models.suggest_params). Parameter terbaik -> HGB_T bagged 32x (bag sama dgn default).
Desain (masing2 dinilai terpisah):
  tune_HGB : baseline HGB default@0.5 vs kandidat HGB_T@0.5. Kandidat ditetapkan dari SEARCH
             saja -> bukti gate = seluruh POOL 1800 (tanpa pemilihan di POOL).
  union_T  : baseline = HGB_T@0.5; pencarian tunggal (cross-fit 3) atas union kandidat seperti
             MINI-11 tetapi anggota HGB = HGB_T: threshold, HGB_T+DT (x thr), HGB_T+DT+SVC (x thr).
  union_D  : pembanding = union MINI-11 persis (baseline HGB default), pada trial yang sama.
18 dataset DEV (N<40000), seed 1_005_000_000. Pakai: python mini12_optuna.py <R> <n_proses>"""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np, optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import models, data as D
import exp41_gate_confirm as G
from mini01_threshold_crossfit import gate
from mini05_ensemble import DS, data

OFF = 1_005_000_000
TH = [round(t, 2) for t in np.arange(.05, .96, .05)]
SIMPLEX = [w for w in __import__("itertools").product((0, .25, .5, .75, 1), repeat=3)
           if abs(sum(w) - 1) < 1e-9 and w != (1, 0, 0)]


def union(names):
    h, dt, sv = names
    thr = [({h: 1}, t) for t in TH if t != .5]
    e2 = [({h: w, dt: 1 - w}, t) for w in (.75, .5, .25) for t in TH]
    e3 = [({h: a, dt: b, sv: c}, t) for a, b, c in SIMPLEX for t in (.3, .35, .4, .45, .5, .55, .6)]
    return thr + e2 + e3


def lab(P, cfg, idx=None):
    w, t = cfg
    p = sum(wi * P[m] for m, wi in w.items() if wi)
    return ((p if idx is None else p[idx]) >= t).astype(int)


def wf(y, b, c):
    return float(G.delta_from_counts(G.counts8(y, b, c)))


def pick(y, P, idx, cands, base):
    best, g0 = None, 0.0
    for cfg in cands:
        g = wf(y[idx], base[idx], lab(P, cfg, idx))
        if g > g0 + 1e-12:
            best, g0 = cfg, g
    return best


def truth(yr, baser, candr, seed):
    cr = G.counts8(yr, baser, candr); d = float(G.delta_from_counts(cr))
    dr = G.delta_from_counts(np.random.default_rng(seed + 987163).dirichlet(cr + .5, size=20000))
    lo, hi = np.quantile(dr, [.025, .975])
    return d, ("membaik" if lo > 0 else "memburuk" if hi < 0 else "tidak-pasti")


def tune(Xs, ys, seed):
    cols = list(Xs.columns)
    cv = list(StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(Xs, ys))

    def obj(tr):
        prm = models.suggest_params(tr, "HGB", len(ys) * 2 // 3)
        sc = []
        for a, b in cv:
            f = G.fit_proba_m("HGB", prm, Xs.iloc[a].reset_index(drop=True), ys[a], cols)
            pr = f(Xs.iloc[b].reset_index(drop=True))
            sc.append(f1_score(ys[b], (pr >= .5).astype(int), average="weighted"))
        return float(np.mean(sc))
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed % (2**31 - 1)))
    st.optimize(obj, n_trials=25)
    return dict(st.best_params)


def trial(job):
    ds, did, rep = job; seed = OFF + 1000 * list(DS).index(ds) + rep
    Xd, yd, Xr, yr = data(did)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=G.TOTAL_TRAIN, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=G.SEARCH_SIZE, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    best = tune(Xs, ys, seed)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    cols = list(Xs.columns); spec = {"HGB": ("HGB", {}), "HGBT": ("HGB", best), "DT": ("DT", {}), "SVC": ("SVC", {})}
    fn = {m: [] for m in spec}
    for _ in range(G.B_BAGS):
        idx = rng.choice(len(ys), size=sub, replace=False)
        if np.bincount(ys[idx], minlength=2).min() < 10:
            continue
        Xb, yb = Xs.iloc[idx].reset_index(drop=True), ys[idx]
        for m, (mm, pp) in spec.items():
            fn[m].append(G.fit_proba_m(mm, pp, Xb, yb, cols))
    P = {m: np.column_stack([f(Xp) for f in fs]).mean(1) for m, fs in fn.items()}
    Pr = {m: np.column_stack([f(Xr) for f in fs]).mean(1) for m, fs in fn.items()}
    yr_ = np.asarray(yr); out = []
    bD, bDr = (P["HGB"] >= .5).astype(int), (Pr["HGB"] >= .5).astype(int)
    bT, bTr = (P["HGBT"] >= .5).astype(int), (Pr["HGBT"] >= .5).astype(int)
    # tune_HGB: kandidat tetap dari SEARCH -> bukti = seluruh POOL
    cnt = G.counts8(yp, bD, bT); l99, l95 = gate(cnt, len(yp), seed + 21); d, k = truth(yr_, bDr, bTr, seed)
    out.append(dict(dataset=ds, rep=rep, design="tune_HGB", cfg=str(best), D=int(cnt[[1, 2, 5, 6]].sum()),
                    lo99=l99, lo95=l95, ref=d, kelas=k))
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp))
    for name, base, baser, cands, s_ in (("union_T", bT, bTr, union(("HGBT", "DT", "SVC")), 22),
                                         ("union_D", bD, bDr, union(("HGB", "DT", "SVC")), 23)):
        cnt = np.zeros(8, int)
        for a, b in folds:
            cfg = pick(yp, P, a, cands, base)
            cnt += G.counts8(yp[b], base[b], base[b] if cfg is None else lab(P, cfg, b))
        l99, l95 = gate(cnt, len(yp), seed + s_)
        cfg = pick(yp, P, np.arange(len(yp)), cands, base)
        d, k = truth(yr_, baser, baser if cfg is None else lab(Pr, cfg), seed)
        out.append(dict(dataset=ds, rep=rep, design=name, cfg=str(cfg), D=int(cnt[[1, 2, 5, 6]].sum()),
                        lo99=l99, lo95=l95, ref=d, kelas=k))
    # skor absolut referensi (utk melihat apakah baseline T lebih kuat)
    out[0]["wf1_ref_default"] = float(f1_score(yr_, bDr, average="weighted"))
    out[0]["wf1_ref_tuned"] = float(f1_score(yr_, bTr, average="weighted"))
    return out


if __name__ == "__main__":
    R, Pn = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    jobs = [(ds, did, r) for ds, did in DS.items() for r in range(R)]
    with Pool(Pn) as pool:
        res = [x for rows in pool.map(trial, jobs, chunksize=5) for x in rows]
    keys = sorted({k for r in res for k in r}, key=lambda k: list(res[0]).index(k) if k in res[0] else 99)
    p = HERE / "out" / "mini12_optuna_dev.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(res)
    print(f"{len(DS)} dataset, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

"""MINI-11 (PRAREG_MINI11.md): konfirmasi P_union pada 34 dataset, seed 1_003_000_000, tanpa KNN.
MINI-10 (pengembangan): kandidat union, pencarian tunggal atas semua konfigurasi. Turunan MINI-09.
MINI-09 (pengembangan): kandidat lebih kaya, gate BRIDGE beku (xfit3 + boot20000, kuantil-5% > 0.005).
Baseline = HGB bagged @0.5. Kandidat per trial (dinilai terpisah, masing-masing vs baseline):
  ens_DT     : w*HGB + (1-w)*DT, w in {.25,.5,.75}, thr .5               (pembanding, sel MINI-08)
  ens3_KNN   : bobot simpleks langkah .25 atas (HGB, DT, KNN) kecuali (1,0,0), thr .5
  ens3_SVC   : idem dengan (HGB, DT, SVC)
  ensthr_DT  : w in {1,.75,.5,.25} x thr grid .05..95, kecuali (w=1, thr=.5)  (gabungan ensemble+threshold)
Konfigurasi dipilih di 2 lipat (wF1 strictly > baseline; default baseline), counts di lipat ke-3.
18 dataset DEV N<40000, seed 999_000_000. Pakai: python mini09_dev.py <R> <n_proses> [out]"""
import csv, sys, time, warnings, itertools
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import exp41_gate_confirm as G
from mini01_threshold_crossfit import gate
from mini05_ensemble import data
DS = {}
for _f in ("exp40_manifest/datasets.csv", "exp41_manifest/datasets.csv"):
    for _r in pd.read_csv(HERE / _f).itertuples():
        DS.setdefault(_r.dataset, int(_r.data_id))

OFF = int(sys.argv[4]) if len(sys.argv) > 4 else 1_001_000_000
TH = [round(t, 2) for t in np.arange(.05, .96, .05)]
SIMPLEX = [w for w in itertools.product((0, .25, .5, .75, 1), repeat=3) if abs(sum(w) - 1) < 1e-9 and w != (1, 0, 0)]
MODELS = ("HGB", "DT", "SVC")


def configs():
    thr = [((1, 0, 0), t) for t in TH if t != .5]
    e3s = [((a, b, c), t) for a, b, c in SIMPLEX for t in (.3, .35, .4, .45, .5, .55, .6)]
    union = thr + [((w, 1 - w, 0), t) for w in (.75, .5, .25) for t in TH] + e3s
    return {
        "thr_HGB": thr,
        "union": union,
        "ens_DT": [((w, 1 - w, 0), .5) for w in (.25, .5, .75)],
        "ens3_SVC": [((a, b, c), .5) for a, b, c in SIMPLEX],
        "ensthr_DT": [((w, 1 - w, 0), t) for w in (1, .75, .5, .25) for t in TH if not (w == 1 and t == .5)],
    }


def lab(P, cfg, idx=None):
    w, t = cfg
    p = sum(wi * P[m] for wi, m in zip(w, MODELS) if wi)
    p = p if idx is None else p[idx]
    return (p >= t).astype(int)


def wf(y, b, c):
    return float(G.delta_from_counts(G.counts8(y, b, c)))


def pick(y, P, idx, cands, base):
    best, g0 = None, 0.0
    for cfg in cands:
        g = wf(y[idx], base[idx], lab(P, cfg, idx))
        if g > g0 + 1e-12:
            best, g0 = cfg, g
    return best


def trial(job):
    ds, did, rep = job; seed = OFF + 1000 * list(DS).index(ds) + rep
    Xd, yd, Xr, yr = data(did)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=G.TOTAL_TRAIN, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=G.SEARCH_SIZE, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    cols = list(Xs.columns); fn = {m: [] for m in MODELS}
    for _ in range(G.B_BAGS):
        idx = rng.choice(len(ys), size=sub, replace=False)
        if np.bincount(ys[idx], minlength=2).min() < 10:
            continue
        Xb, yb = Xs.iloc[idx].reset_index(drop=True), ys[idx]
        for m in MODELS:
            fn[m].append(G.fit_proba_m(m, {"n_neighbors": 5} if m == "KNN" else {}, Xb, yb, cols))
    P = {m: np.column_stack([f(Xp) for f in fs]).mean(1) for m, fs in fn.items()}
    Pr = {m: np.column_stack([f(Xr) for f in fs]).mean(1) for m, fs in fn.items()}
    yr_ = np.asarray(yr); base = (P["HGB"] >= .5).astype(int); baser = (Pr["HGB"] >= .5).astype(int)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp))
    out = []
    for k, (name, cands) in enumerate(configs().items()):
        cnt = np.zeros(8, int)
        for a, b in folds:
            cfg = pick(yp, P, a, cands, base)
            cand = base[b] if cfg is None else lab(P, cfg, b)
            cnt += G.counts8(yp[b], base[b], cand)
        l99, l95 = gate(cnt, len(yp), seed + 11 + k)
        cfg = pick(yp, P, np.arange(len(yp)), cands, base)
        candr = baser if cfg is None else lab(Pr, cfg)
        cr = G.counts8(yr_, baser, candr); d = float(G.delta_from_counts(cr))
        dr = G.delta_from_counts(np.random.default_rng(seed + 987163).dirichlet(cr + .5, size=20000))
        lo, hi = np.quantile(dr, [.025, .975])
        out.append(dict(dataset=ds, rep=rep, design=name, cfg=str(cfg), D=int(cnt[[1, 2, 5, 6]].sum()), lo99=l99,
                        lo95=l95, ref=d, kelas="membaik" if lo > 0 else "memburuk" if hi < 0 else "tidak-pasti"))
    return out


if __name__ == "__main__":
    R, Pn = int(sys.argv[1]), int(sys.argv[2]); outn = sys.argv[3] if len(sys.argv) > 3 else "mini09_dev.csv"
    t0 = time.time(); jobs = [(ds, did, r) for ds, did in DS.items() for r in range(R)]
    with Pool(Pn) as pool:
        res = [x for rows in pool.map(trial, jobs, chunksize=10) for x in rows]
    p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print(f"{len(DS)} dataset, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

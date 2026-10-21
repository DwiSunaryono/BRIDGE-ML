"""MINI-05 (tahap pengembangan): kandidat ensemble (folder lama: SVC/DT berguna) dengan gate BRIDGE
beku.
Baseline = HGB bagged (thr 0.5). Kandidat c in {SVC, DT}: p = w*p_HGB + (1-w)*p_c, w dipilih dari
{0.25, 0.5, 0.75} (strictly-greater vs baseline; default = baseline). Gate sama dengan MINI-04:
cross-fit 3 atas POOL 1800 (w dipilih di 2 lipat, counts di lipat ke-3), bootstrap 20000, terima
jika kuantil-5% > 0.005. Kandidat final = w dipilih di seluruh POOL; kebenaran di referensi.
Dataset: dataset DEV dengan N < 40000 (referensi <= 8000) agar cepat. Seed 995M.
Pakai: python mini05_ensemble.py <R> <n_proses>"""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import exp41_gate_confirm as G
from mini01_threshold_crossfit import gate

import warnings; warnings.filterwarnings("ignore")
_N = {int(r.did): int(r.N) for r in pd.read_csv(HERE / "exp40_dataset_audit_final.csv").itertuples()}
_N.update({int(r.id): int(r.N) for r in pd.read_csv(HERE / "exp38_manifest/kandidat_audit.csv").itertuples()})
_N.update({int(r.data_id): int(r.N) for r in pd.read_csv(HERE / "exp41_manifest/datasets.csv").itertuples()})
DS = {}
for f in ("exp40_manifest/datasets.csv", "exp41_manifest/datasets.csv"):
    for r in pd.read_csv(HERE / f).itertuples():
        if _N[int(r.data_id)] < 40000:  # referensi <= 8000 -> cepat (tahap pengembangan)
            DS.setdefault(r.dataset, int(r.data_id))
OFF = 995_000_000; W = (0.25, 0.5, 0.75); _D = {}


def data(did):
    if did not in _D:
        _D.clear(); _D[did] = G.make_dev_and_locked_reference(did)
    return _D[did]


def wf(y, b, c):
    return float(G.delta_from_counts(G.counts8(y, b, c)))


def pick_w(y, ph, po):
    base = (ph >= .5).astype(int); best, g0 = None, 0.0
    for w in W:
        g = wf(y, base, ((w * ph + (1 - w) * po) >= .5).astype(int))
        if g > g0 + 1e-12:
            best, g0 = w, g
    return best


def trial(job):
    ds, did, rep = job; seed = OFF + 1000 * list(DS).index(ds) + rep
    Xd, yd, Xr, yr = data(did)
    if len(Xd) < G.TOTAL_TRAIN + 10:
        return []
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=G.TOTAL_TRAIN, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=G.SEARCH_SIZE, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    cols = list(Xs.columns); fn = {"HGB": [], "SVC": [], "DT": []}
    for _ in range(G.B_BAGS):
        idx = rng.choice(len(ys), size=sub, replace=False)
        if np.bincount(ys[idx], minlength=2).min() < 10:
            continue
        Xb, yb = Xs.iloc[idx].reset_index(drop=True), ys[idx]
        for mname in fn:
            fn[mname].append(G.fit_proba_m(mname, {}, Xb, yb, cols))
    P = {m: np.column_stack([f(Xp) for f in fs]).mean(1) for m, fs in fn.items()}
    Pr = {m: np.column_stack([f(Xr) for f in fs]).mean(1) for m, fs in fn.items()}
    yr_ = np.asarray(yr); out = []
    for c in ("SVC", "DT"):
        cnt = np.zeros(8, int)
        for a, b in StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp):
            w = pick_w(yp[a], P["HGB"][a], P[c][a])
            cand = P["HGB"][b] if w is None else w * P["HGB"][b] + (1 - w) * P[c][b]
            cnt += G.counts8(yp[b], (P["HGB"][b] >= .5).astype(int), (cand >= .5).astype(int))
        l99, l95 = gate(cnt, len(yp), seed + (7 if c == "SVC" else 8))
        w = pick_w(yp, P["HGB"], P[c])
        candr = Pr["HGB"] if w is None else w * Pr["HGB"] + (1 - w) * Pr[c]
        cr = G.counts8(yr_, (Pr["HGB"] >= .5).astype(int), (candr >= .5).astype(int))
        d = float(G.delta_from_counts(cr))
        dr = G.delta_from_counts(np.random.default_rng(seed + 987163).dirichlet(cr + .5, size=20000))
        lo, hi = np.quantile(dr, [.025, .975])
        out.append(dict(dataset=ds, rep=rep, design=f"ens_{c}", w=w, D=int(cnt[[1, 2, 5, 6]].sum()), lo99=l99,
                        lo95=l95, ref=d, kelas="membaik" if lo > 0 else "memburuk" if hi < 0 else "tidak-pasti"))
    return out


if __name__ == "__main__":
    R, Pn = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    jobs = [(ds, did, r) for ds, did in DS.items() for r in range(R)]
    with Pool(Pn) as pool:
        res = [x for rows in pool.map(trial, jobs, chunksize=10) for x in rows]
    p = HERE / "out" / "mini05_ensemble.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print(f"{len(DS)} dataset, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

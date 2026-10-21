"""MINI-04 (PRAREG_MINI04.md): konfirmasi kebijakan P_thr untuk model dari argv[3].

MINI-03 memindahkan aturan beku (xfit3 + gate95 + lo95 > 0.005, tidak diubah) ke
model KNN (k=5, default exp39-41) dengan seed baru 992M pada 34 dataset DEV.
Kolom kebenaran dan gate identik dengan MINI-02.

MINI-01 (alur cepat, DEV): threshold-HGB, split kaku TUNE600/CONFIRM1200
dibandingkan dengan cross-fit 3 lipat. Bukan konfirmasi, hanya penyaring ide
cepat pada dataset DEV kecil. Referensi hanya untuk diagnosis kebenaran
kandidat (membaik/memburuk/tidak pasti) dan tidak masuk gate.
- kaku: threshold dipilih di TUNE (600), gate pada counts CONFIRM (1200).
  Kandidat adalah threshold tersebut.
- xfit: POOL 1800 dibagi 3 lipat stratified. Per lipat threshold dipilih di
  2 lipat lain (1200), counts dihitung di lipat itu (600), lalu dijumlah
  (1800) untuk gate. Kandidat yang dipakai adalah threshold dari seluruh
  POOL (1800). Kebenaran dinilai untuk kandidat itu di referensi.
Gate: bootstrap multinomial 20000, kuantil 1% (99%) dan 5% (95%) dari
bootstrap yang sama.
Pakai: python mini01_threshold_crossfit.py <R> <n_proses>
"""
import csv, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp41_gate_confirm as G

import pandas as _pd
DS = {}
for _f in ("exp40_manifest/datasets.csv", "exp41_manifest/datasets.csv"):
    for _r in _pd.read_csv(Path(__file__).resolve().parent / _f).itertuples():
        DS.setdefault(_r.dataset, int(_r.data_id))
_M = sys.argv[3]  # HGB | KNN
MODEL = {"HGB": ("HGB", {}), "KNN": ("KNN", {"n_neighbors": 5})}[_M]
OFF = {"HGB": 993_000_000, "KNN": 994_000_000}[_M]
N_BOOT = 20000
_DATA = {}


def data(did):
    if did not in _DATA:
        _DATA.clear()
        _DATA[did] = G.make_dev_and_locked_reference(did)
    return _DATA[did]


def best_thr(y, pb):
    base = (pb >= 0.5).astype(int); thr, g0 = 0.5, 0.0
    for t in G.GRID:
        g = float(G.delta_from_counts(G.counts8(y, base, (pb >= float(t)).astype(int))))
        if g > g0 + 1e-12:
            thr, g0 = float(t), g
    return thr


def gate(c, n, seed):
    boot = G.delta_from_counts(np.random.default_rng(seed).multinomial(n, c / n, size=N_BOOT))
    return float(np.quantile(boot, .01)), float(np.quantile(boot, .05))


def truth(yr, pbr, thr, seed):
    cr = G.counts8(yr, (pbr >= 0.5).astype(int), (pbr >= thr).astype(int))
    d = float(G.delta_from_counts(cr))
    dr = G.delta_from_counts(np.random.default_rng(seed + 987163).dirichlet(cr + 0.5, size=N_BOOT))
    lo, hi = np.quantile(dr, [.025, .975])
    return d, ("membaik" if lo > 0 else "memburuk" if hi < 0 else "tidak-pasti")


def trial(job):
    ds, did, rep = job; seed = OFF + 1000 * list(DS).index(ds) + rep
    Xd, yd, Xr, yr = data(did)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=G.TOTAL_TRAIN, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=G.SEARCH_SIZE, stratify=ytr, random_state=seed)
    Xs, ys = Xs.reset_index(drop=True), np.asarray(ys); Xp = Xp.reset_index(drop=True); yp = np.asarray(yp)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1))
    sub = int(G.SUBSAMPLE_FRAC * len(ys)); cols = list(Xs.columns); fns = []
    for _ in range(G.B_BAGS):
        idx = rng.choice(len(ys), size=sub, replace=False)
        if np.bincount(ys[idx], minlength=2).min() >= 10:
            fns.append(G.fit_proba_m(MODEL[0], MODEL[1], Xs.iloc[idx].reset_index(drop=True), ys[idx], cols))
    pp = np.column_stack([f(Xp) for f in fns]).mean(1); pr = np.column_stack([f(Xr) for f in fns]).mean(1)
    yr_ = np.asarray(yr); out = []
    # kaku, skema exp41
    rs = np.random.default_rng(seed + 20261014)
    i0 = rs.permutation(np.flatnonzero(yp == 0)); i1 = rs.permutation(np.flatnonzero(yp == 1))
    n1 = int(round(G.TUNE_SIZE * yp.mean())); n0 = G.TUNE_SIZE - n1
    tu = np.r_[i0[:n0], i1[:n1]]; co = np.r_[i0[n0:], i1[n1:]]
    thr = best_thr(yp[tu], pp[tu])
    c = G.counts8(yp[co], (pp[co] >= .5).astype(int), (pp[co] >= thr).astype(int))
    l99, l95 = gate(c, len(co), seed + 1); d, cls = truth(yr_, pr, thr, seed)
    out.append(dict(dataset=ds, rep=rep, design="kaku", thr=thr, D=int(c[[1, 2, 5, 6]].sum()), lo99=l99, lo95=l95, ref=d, kelas=cls))
    for K in (3,):
        out.append(xfit(K, yp, pp, yr_, pr, seed, ds, rep))
    return out


def xfit(K, yp, pp, yr_, pr, seed, ds, rep):
    c = np.zeros(8, int)
    for a, b in StratifiedKFold(K, shuffle=True, random_state=seed % (2**32 - 1)).split(pp, yp):
        t = best_thr(yp[a], pp[a])
        c += G.counts8(yp[b], (pp[b] >= .5).astype(int), (pp[b] >= t).astype(int))
    thr = best_thr(yp, pp)
    l99, l95 = gate(c, len(yp), seed + 2 + K); d, cls = truth(yr_, pr, thr, seed)
    return dict(dataset=ds, rep=rep, design=f"xfit{K}", thr=thr, D=int(c[[1, 2, 5, 6]].sum()), lo99=l99, lo95=l95, ref=d, kelas=cls)


if __name__ == "__main__":
    R, P = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    jobs = [(ds, did, r) for ds, did in DS.items() for r in range(R)]
    with Pool(P) as pool:
        res = [x for rows in pool.map(trial, jobs, chunksize=25) for x in rows]  # Potongan 25 trial
    outp = Path(__file__).resolve().parent / "out" / f"mini04_{_M}.csv"
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print(f"{len(jobs)} trial, {time.time()-t0:.0f} dtk -> {outp}")

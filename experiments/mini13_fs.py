"""MINI-13 (tahap pengembangan): seleksi fitur versi baru, gate BRIDGE beku (kuantil-5% > 0.005).

Baseline: HGB default bagged (SEARCH) @0.5. Kandidat fs_perm:
  pada data pemilih (2 lipat POOL, atau seluruh POOL untuk kandidat final), hitung permutation
  importance tiap fitur untuk model baseline (delta weighted-F1 saat kolom diacak, rata-rata 3
  acakan, rng tetap); buang fitur dengan importance <= 0 (minimal 1 fitur tersisa); latih ulang
  HGB bagged 32x di SEARCH tanpa fitur itu (bag sama); bila tidak ada yang dibuang, kandidat =
  baseline. Cross-fit 3 lipat untuk bukti gate; final dipilih di seluruh POOL.
Juga dicatat: n_dibuang. 18 dataset DEV N<40000, seed 1_007_000_000.
Pakai: python mini13_fs.py <R> <n_proses>"""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import exp41_gate_confirm as G
from mini01_threshold_crossfit import gate
from mini05_ensemble import DS, data
from mini12_optuna import truth

OFF = 1_007_000_000


def bag_fit(Xs, ys, cols, bags):
    fs = [G.fit_proba_m("HGB", {}, Xs.iloc[i].reset_index(drop=True), ys[i], cols) for i in bags]
    return lambda X: np.column_stack([f(X) for f in fs]).mean(1)


def perm_drop(model, X, y, cols, seed):
    base = f1_score(y, (model(X) >= .5).astype(int), average="weighted")
    rng = np.random.default_rng(seed); drop = []
    for c in cols:
        d = []
        for _ in range(3):
            Xc = X.copy(); Xc[c] = rng.permutation(Xc[c].values)
            d.append(base - f1_score(y, (model(Xc) >= .5).astype(int), average="weighted"))
        if np.mean(d) <= 0:
            drop.append(c)
    if len(drop) >= len(cols):
        drop = drop[:-1]
    return drop


def trial(job):
    ds, did, rep = job; seed = OFF + 1000 * list(DS).index(ds) + rep
    Xd, yd, Xr, yr = data(did)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=G.TOTAL_TRAIN, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=G.SEARCH_SIZE, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    bags = []
    for _ in range(G.B_BAGS):
        idx = rng.choice(len(ys), size=sub, replace=False)
        if np.bincount(ys[idx], minlength=2).min() >= 10:
            bags.append(idx)
    cols = list(Xs.columns); base_m = bag_fit(Xs, ys, cols, bags)
    pb = base_m(Xp); base = (pb >= .5).astype(int)
    cnt = np.zeros(8, int); nd = []
    for k, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp)):
        drop = perm_drop(base_m, Xp.iloc[a].reset_index(drop=True), yp[a], cols, seed + k)
        nd.append(len(drop))
        if drop:
            keep = [c for c in cols if c not in drop]
            cand = (bag_fit(Xs, ys, keep, bags)(Xp.iloc[b].reset_index(drop=True)) >= .5).astype(int)
        else:
            cand = base[b]
        cnt += G.counts8(yp[b], base[b], cand)
    l99, l95 = gate(cnt, len(yp), seed + 31)
    drop = perm_drop(base_m, Xp, yp, cols, seed + 9)
    baser = (base_m(Xr) >= .5).astype(int)
    candr = (bag_fit(Xs, ys, [c for c in cols if c not in drop], bags)(Xr) >= .5).astype(int) if drop else baser
    d, kls = truth(np.asarray(yr), baser, candr, seed)
    return [dict(dataset=ds, rep=rep, design="fs_perm", n_fitur=len(cols), n_buang_final=len(drop),
                 n_buang_lipat=str(nd), D=int(cnt[[1, 2, 5, 6]].sum()), lo99=l99, lo95=l95, ref=d, kelas=kls)]


if __name__ == "__main__":
    R, Pn = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    jobs = [(ds, did, r) for ds, did in DS.items() for r in range(R)]
    with Pool(Pn) as pool:
        res = [x for rows in pool.map(trial, jobs, chunksize=5) for x in rows]
    p = HERE / "out" / "mini13_fs_dev.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print(f"{len(DS)} dataset, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

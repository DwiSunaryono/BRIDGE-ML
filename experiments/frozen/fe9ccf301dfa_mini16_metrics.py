"""MINI-16 (PENGEMBANGAN): kebijakan pencarian tunggal (union MINI-11) dgn METRIK berbeda.
Untuk tiap metrik m in {wF1, acc, bacc, macroF1}: pemilihan konfigurasi (cross-fit 3), gate
(bootstrap 20000, TERIMA iff kuantil-5% Δm > 0.005) DAN kebenaran referensi memakai metrik m.
Ruang kandidat & model (HGB, DT, SVC bagged) identik MINI-11. 18 dataset DEV N<40000,
seed 1_013_000_000. Pakai: python mini16_metrics.py <R> <n_proses> [out] [offset] [semua34]"""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import exp41_gate_confirm as G
import bridge_metrics as BM
from mini05_ensemble import DS as DS18, data
from mini12_optuna import union, lab

OFF = int(sys.argv[4]) if len(sys.argv) > 4 else 1_013_000_000
if len(sys.argv) > 5 and sys.argv[5] == "semua34":
    DS = {}
    for _f in ("exp40_manifest/datasets.csv", "exp41_manifest/datasets.csv"):
        for _r in pd.read_csv(HERE / _f).itertuples():
            DS.setdefault(_r.dataset, int(_r.data_id))
else:
    DS = DS18
CANDS = union(("HGB", "DT", "SVC"))


def pick(y, P, idx, base, m):
    best, g0 = None, 0.0
    for cfg in CANDS:
        g = float(BM.delta(G.counts8(y[idx], base[idx], lab(P, cfg, idx)), m))
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
    cols = list(Xs.columns); fn = {m: [] for m in ("HGB", "DT", "SVC")}
    for _ in range(G.B_BAGS):
        idx = rng.choice(len(ys), size=sub, replace=False)
        if np.bincount(ys[idx], minlength=2).min() < 10:
            continue
        for m in fn:
            fn[m].append(G.fit_proba_m(m, {}, Xs.iloc[idx].reset_index(drop=True), ys[idx], cols))
    P = {m: np.column_stack([f(Xp) for f in fs]).mean(1) for m, fs in fn.items()}
    Pr = {m: np.column_stack([f(Xr) for f in fs]).mean(1) for m, fs in fn.items()}
    base, baser, yr_ = (P["HGB"] >= .5).astype(int), (Pr["HGB"] >= .5).astype(int), np.asarray(yr)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp))
    out = []
    for k, m in enumerate(BM.METRICS):
        cnt = np.zeros(8, int)
        for a, b in folds:
            cfg = pick(yp, P, a, base, m)
            cnt += G.counts8(yp[b], base[b], base[b] if cfg is None else lab(P, cfg, b))
        boot = BM.delta(np.random.default_rng(seed + 61 + k).multinomial(len(yp), cnt / len(yp), size=20000), m)
        l95 = float(np.quantile(boot, .05))
        cfg = pick(yp, P, np.arange(len(yp)), base, m)
        cr = G.counts8(yr_, baser, baser if cfg is None else lab(Pr, cfg))
        d = float(BM.delta(cr, m))
        dr = BM.delta(np.random.default_rng(seed + 987163).dirichlet(cr + .5, size=20000), m)
        lo, hi = np.quantile(dr, [.025, .975])
        out.append(dict(dataset=ds, rep=rep, design=f"union_{m}", metric=m, cfg=str(cfg), D=int(cnt[[1, 2, 5, 6]].sum()),
                        lo95=l95, ref=d, kelas="membaik" if lo > 0 else "memburuk" if hi < 0 else "tidak-pasti"))
    return out


if __name__ == "__main__":
    R, Pn = int(sys.argv[1]), int(sys.argv[2]); outn = sys.argv[3] if len(sys.argv) > 3 else "mini16_metrics_dev.csv"
    t0 = time.time(); jobs = [(ds, did, r) for ds, did in DS.items() for r in range(R)]
    with Pool(Pn) as pool:
        res = [x for rows in pool.map(trial, jobs, chunksize=5) for x in rows]
    p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print(f"{len(DS)} dataset, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

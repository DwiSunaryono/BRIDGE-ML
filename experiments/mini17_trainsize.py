"""MINI-17 (pengembangan): kebijakan pencarian tunggal (union MINI-11, wF1, gate beku) pada
ukuran data latih berbeda: TOTAL in {1000, 2430, 6000}; SEARCH = round(0.259*TOTAL) (rasio 630/2430),
POOL = sisanya (bukti gate cross-fit 3). Dataset DEV dengan dev pool >= 6000+50. 40 trial/ds/ukuran.
Seed 1_015_000_000 + 1000*idx + 100*k + rep. Pakai: python mini17_trainsize.py <R> <n_proses> [out] [offset] [ukuran,dipisah,koma]
"""
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
from mini12_optuna import union, lab, pick, truth
from mini14_optuna_size import _N

SIZES = (1000, 2430, 6000)
OFF = int(sys.argv[4]) if len(sys.argv) > 4 else 1_015_000_000
DS = {}
for f in ("exp40_manifest/datasets.csv", "exp41_manifest/datasets.csv"):
    for r in pd.read_csv(HERE / f).itertuples():
        n = _N[int(r.data_id)]
        if n - G.ref_size(n) >= max(SIZES) + 50:
            DS.setdefault(r.dataset, int(r.data_id))
CANDS = union(("HGB", "DT", "SVC"))
_D = {}


def data(did):
    if did not in _D:
        _D.clear(); _D[did] = G.make_dev_and_locked_reference(did)
    return _D[did]


def trial(job):
    ds, did, k, rep = job; TOT = SIZES[k]; SS = int(round(0.259 * TOT))
    seed = OFF + 1000 * list(DS).index(ds) + 100 * k + rep
    Xd, yd, Xr, yr = data(did)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=TOT, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    cols = list(Xs.columns); fn = {m: [] for m in ("HGB", "DT", "SVC")}
    for _ in range(G.B_BAGS):
        idx = rng.choice(len(ys), size=sub, replace=False)
        if np.bincount(ys[idx], minlength=2).min() < 10:
            continue
        for m in fn:
            fn[m].append(G.fit_proba_m(m, {}, Xs.iloc[idx].reset_index(drop=True), ys[idx], cols))
    if not fn["HGB"]:
        return dict(dataset=ds, TOTAL=TOT, rep=rep, cfg="SKIP", D=0, lo95=np.nan, ref=np.nan, kelas="skip")
    P = {m: np.column_stack([f(Xp) for f in fs]).mean(1) for m, fs in fn.items()}
    Pr = {m: np.column_stack([f(Xr) for f in fs]).mean(1) for m, fs in fn.items()}
    base, baser, yr_ = (P["HGB"] >= .5).astype(int), (Pr["HGB"] >= .5).astype(int), np.asarray(yr)
    cnt = np.zeros(8, int)
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp):
        cfg = pick(yp, P, a, CANDS, base)
        cnt += G.counts8(yp[b], base[b], base[b] if cfg is None else lab(P, cfg, b))
    l99, l95 = gate(cnt, len(yp), seed + 71)
    cfg = pick(yp, P, np.arange(len(yp)), CANDS, base)
    d, kls = truth(yr_, baser, baser if cfg is None else lab(Pr, cfg), seed)
    return dict(dataset=ds, TOTAL=TOT, rep=rep, cfg=str(cfg), D=int(cnt[[1, 2, 5, 6]].sum()), lo95=l95, ref=d, kelas=kls)


if __name__ == "__main__":
    R, Pn = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    outn = sys.argv[3] if len(sys.argv) > 3 else "mini17_trainsize_dev.csv"
    KS = [SIZES.index(int(x)) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else range(len(SIZES))
    assert R <= 100  # seed unik: 100*k + rep
    jobs = [(ds, did, k, r) for ds, did in DS.items() for k in KS for r in range(R)]
    jobs.sort(key=lambda j: -SIZES[j[2]])
    with Pool(Pn) as pool:
        res = pool.map(trial, jobs, chunksize=2)
    p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print(f"{len(DS)} dataset, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

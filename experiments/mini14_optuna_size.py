"""MINI-14: pengaruh ukuran data tuning S terhadap keandalan Optuna (HGB); S di {630, 2000, 5000}.

Praregistrasi singkat (ditulis sebelum run): hipotesis H1 = seiring S naik, (a) rata-rata delta wF1
referensi (Optuna - default) naik, (b) persentase trial yang jelas memburuk (CI Dirichlet ref < 0) turun.
Gate BRIDGE beku dicatat (terima iff kuantil 5% > 0.005 pada POOL 1800 tetap).
Per trial: dari dev ambil S + 1800 baris (stratified); SEARCH = S, POOL = 1800.
Optuna TPE 25 percobaan, CV 3 lipat di SEARCH, objektif wF1@0.5, ruang config HGB (sama MINI-12).
Default dan Optuna masing-masing bagged 32x (bag 80% SEARCH, sama untuk keduanya).
Kandidat ditetapkan dari SEARCH saja, bukti gate = seluruh POOL.
Dataset: DEV dengan dev pool >= 5000+1800 (sama untuk semua S agar sebanding). 40 trial/dataset/S.
Seed 1_009_000_000 + 1000*idx + 100*k(S) + rep. Pakai: python mini14_optuna_size.py <R> <n_proses>"""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import pandas as pd
import exp41_gate_confirm as G
from mini01_threshold_crossfit import gate
from mini12_optuna import tune, truth

SIZES = (630, 2000, 5000)
POOLN = 1800
OFF = 1_009_000_000
_N = {int(r.did): int(r.N) for r in pd.read_csv(HERE / "exp40_dataset_audit_final.csv").itertuples()}
_N.update({int(r.id): int(r.N) for r in pd.read_csv(HERE / "exp38_manifest/kandidat_audit.csv").itertuples()})
_N.update({int(r.data_id): int(r.N) for r in pd.read_csv(HERE / "exp41_manifest/datasets.csv").itertuples()})
DS = {}
for f in ("exp40_manifest/datasets.csv", "exp41_manifest/datasets.csv"):
    for r in pd.read_csv(HERE / f).itertuples():
        n = _N[int(r.data_id)]; nref = G.ref_size(n)
        if n - nref >= max(SIZES) + POOLN + 50:
            DS.setdefault(r.dataset, int(r.data_id))
_D = {}


def data(did):
    if did not in _D:
        _D.clear(); _D[did] = G.make_dev_and_locked_reference(did)
    return _D[did]


def bagged(Xs, ys, params, seed):
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    fs = []
    for _ in range(G.B_BAGS):
        idx = rng.choice(len(ys), size=sub, replace=False)
        if np.bincount(ys[idx], minlength=2).min() >= 10:
            fs.append(G.fit_proba_m("HGB", params, Xs.iloc[idx].reset_index(drop=True), ys[idx], list(Xs.columns)))
    return lambda X: np.column_stack([f(X) for f in fs]).mean(1)


def trial(job):
    ds, did, k, rep = job; S = SIZES[k]
    seed = OFF + 1000 * list(DS).index(ds) + 100 * k + rep
    Xd, yd, Xr, yr = data(did)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=S + POOLN, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=S, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    best = tune(Xs, ys, seed)
    mD, mT = bagged(Xs, ys, {}, seed), bagged(Xs, ys, best, seed)
    bD, bT = (mD(Xp) >= .5).astype(int), (mT(Xp) >= .5).astype(int)
    cnt = G.counts8(yp, bD, bT); l99, l95 = gate(cnt, len(yp), seed + 41)
    yr_ = np.asarray(yr); rD, rT = (mD(Xr) >= .5).astype(int), (mT(Xr) >= .5).astype(int)
    d, kls = truth(yr_, rD, rT, seed)
    return dict(dataset=ds, S=S, rep=rep, D=int(cnt[[1, 2, 5, 6]].sum()), lo99=l99, lo95=l95, ref=d, kelas=kls,
                wf1_def=float(f1_score(yr_, rD, average="weighted")), wf1_opt=float(f1_score(yr_, rT, average="weighted")))


if __name__ == "__main__":
    R, Pn = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    jobs = [(ds, did, k, r) for ds, did in DS.items() for k in range(len(SIZES)) for r in range(R)]
    jobs.sort(key=lambda j: -SIZES[j[2]])  # yang berat dulu agar core terisi rata
    with Pool(Pn) as pool:
        res = pool.map(trial, jobs, chunksize=2)
    p = HERE / "out" / "mini14_optuna_size.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print(f"{len(DS)} dataset, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

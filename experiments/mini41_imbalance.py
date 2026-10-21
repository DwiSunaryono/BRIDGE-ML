"""MINI-41: BRIDGE pada kelas tidak seimbang (prevalensi minoritas dikendalikan) dan panduan label minimum.
Data: 5 dataset biner besar dari 11 dataset baru (LICD, KDD98, Homesite, Census-Income-KDD, Santander2016).
Per dataset dan level prevalensi pi in {0.20, 0.10, 0.05, 0.02}: baris diambil acak sekali (seed tetap
PREV_SEED + 100*i_level + i_ds) dari data penuh. Bila prevalensi asal > pi, minoritas dikurangi (semua mayoritas
dipakai); bila < pi, mayoritas dikurangi (semua minoritas dipakai). Pembagian luar terkunci seperti MINI-21
(ref_size, LOCKED_REF_SEED), sehingga referensi punya prevalensi pi.
Trial memakai kode trial MINI-32 (mode nyata) apa adanya: ruang union HGB/DT/SVC x threshold, 32 bag 80% dengan
>= 10 per kelas, baseline HGB@0.5, cross-fit 3, bootstrap multinomial 20.000. Trial diulang untuk tiga metrik m
(pemilihan, gate, dan kebenaran memakai m, seperti union_* MINI-21): wF1 (utama, k=0: seed bootstrap seed+61,
sama dengan MINI-32), bacc (k=1), macroF1 (k=2).
# Aturan:
#   naif     : terima bila Δ_m konfigurasi terbaik di seluruh POOL > 0 (pt_pool > 0)
#   BRIDGE   : terima iff kuantil-5% bootstrap counts cross-fit > 0,005 (lo_cf_boot)
#   kebenaran: Δ_m di referensi terkunci; harmful = Δ <= 0; kelas Dirichlet(cr + 0,5) 95% (cara MINI-21)
# Tanpa bag yang memenuhi syarat, trial dicatat berstatus 'abstain' (tidak ada update untuk dinilai; kedua aturan
# tidak menerima). Seed trial: 1_041_000_000 + 10000*i_level + 1000*i_ds + rep, rep 0-99.
# Mode identitas: level 'asli' (tanpa manipulasi) dan seed MINI-21/32, wF1 harus identik dengan out/mini32_nyata_m*.csv.
Pakai: python mini41_imbalance.py <rep_awal> <rep_akhir> <n_proses> <out.csv> <level,dipisah,koma | asli> <dataset,.. | semua>
"""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini21_final as F
sys.argv = _argv
import exp41_gate_confirm as G, bridge_metrics as BM
from mini12_optuna import union, lab

NAMES = ("LICD", "KDD98", "Homesite", "Census-Income-KDD", "Santander2016")
LEVELS = (0.20, 0.10, 0.05, 0.02)
METRICS = ("wF1", "bacc", "macroF1")
OFF, PREV_SEED, EPS = 1_041_000_000, 20261015, 0.005
U = union(("HGB", "DT", "SVC"))
_D = {}


def prevalence(X, y, pi, seed):
    y = np.asarray(y); cnt = np.bincount(y, minlength=2); mn = int(cnt.argmin())
    imn, imj = np.flatnonzero(y == mn), np.flatnonzero(y != mn); rng = np.random.RandomState(seed)
    if cnt[mn] / len(y) > pi:
        imn = rng.choice(imn, int(round(pi / (1 - pi) * len(imj))), replace=False)
    else:
        imj = rng.choice(imj, int(round((1 - pi) / pi * len(imn))), replace=False)
    keep = np.sort(np.concatenate([imn, imj]))
    return X.iloc[keep].reset_index(drop=True), pd.Series(y[keep], name="target")


def data(ds, lev):
    key = (ds, lev)
    if key not in _D:
        _D.clear(); X, y = F.load(*F.DS[ds])
        if lev != "asli":
            X, y = prevalence(X, y, lev, PREV_SEED + 100 * LEVELS.index(lev) + NAMES.index(ds))
        Xd, Xr, yd, yr = train_test_split(X, y, test_size=G.ref_size(len(X)), stratify=y, random_state=G.LOCKED_REF_SEED)
        _D[key] = (Xd.reset_index(drop=True), yd.reset_index(drop=True), Xr.reset_index(drop=True), np.asarray(yr))
    return _D[key]


def seed_of(ds, lev, rep):
    if lev == "asli":
        return F.OFF + 1000 * list(F.DS).index(ds) + rep  # seed MINI-21/32
    return OFF + 10000 * LEVELS.index(lev) + 1000 * NAMES.index(ds) + rep


def run_job(job):
    try:
        return _trial(job)
    except Exception as e:
        return [dict(dataset=job[0], level=job[1], rep=job[2], metric=m, status="ERROR " + str(e)[:150]) for m in METRICS]


def _trial(job):
    ds, lev, rep = job; seed = seed_of(ds, lev, rep)
    Xd, yd, Xr, yr = data(ds, lev)
    TOT, SS = (G.TOTAL_TRAIN, G.SEARCH_SIZE) if len(Xd) >= G.TOTAL_TRAIN + 50 else (1000, 259)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=TOT, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS)) if np.bincount(ys[i], minlength=2).min() >= 10]
    info = dict(dataset=ds, level=lev, rep=rep, seed=seed, n_bags=len(bags), min_search=int(np.bincount(ys, minlength=2).min()),
                min_pool=int(np.bincount(yp, minlength=2).min()), min_ref=int(np.bincount(yr, minlength=2).min()), ref_n=len(yr),
                prev_ref=float(np.bincount(yr, minlength=2).min() / len(yr)))
    if not bags:
        return [dict(info, metric=m, status="abstain") for m in METRICS]
    cols = list(Xs.columns); P, Pr = {}, {}
    for mdl in ("HGB", "DT", "SVC"):
        fs = [G.fit_proba_m(mdl, {}, Xs.iloc[i].reset_index(drop=True), ys[i], cols) for i in bags]
        P[mdl] = np.column_stack([f(Xp) for f in fs]).mean(1); Pr[mdl] = np.column_stack([f(Xr) for f in fs]).mean(1)
    base, baser = (P["HGB"] >= .5).astype(int), (Pr["HGB"] >= .5).astype(int)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp))
    out = []
    for k, m in enumerate(METRICS):
        pk = lambda idx: F.pick(yp, P, idx, U, base, m)
        cnt = np.zeros(8, int)
        for a, b in folds:
            c = pk(a); cnt += G.counts8(yp[b], base[b], base[b] if c is None else lab(P, c, b))
        cfg = pk(np.arange(len(yp)))
        cpool = G.counts8(yp, base, base if cfg is None else lab(P, cfg))
        lo = float(np.quantile(BM.delta(np.random.default_rng(seed + 61 + k).multinomial(cnt.sum(), cnt / cnt.sum(), size=20000), m), .05))
        cr = G.counts8(yr, baser, baser if cfg is None else lab(Pr, cfg))
        dr = BM.delta(np.random.default_rng(seed + 987163 + k).dirichlet(cr + .5, size=20000), m)
        a2, b2 = np.quantile(dr, [.025, .975]); ref = float(BM.delta(cr, m)); pt = float(BM.delta(cpool, m))
        out.append(dict(info, metric=m, status="ok", cfg=str(cfg), pt_pool=pt, lo_cf_boot=lo, ref=ref,
                        naive=int(pt > 0), bridge=int(lo > EPS), harmful=int(ref <= 0),
                        kelas="membaik" if a2 > 0 else "memburuk" if b2 < 0 else "tidak-pasti"))
    return out


COLS = ["dataset", "level", "rep", "seed", "metric", "status", "n_bags", "min_search", "min_pool", "min_ref", "ref_n", "prev_ref",
        "cfg", "pt_pool", "lo_cf_boot", "ref", "naive", "bridge", "harmful", "kelas"]

if __name__ == "__main__":
    a, b, Pn, outn = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    levs = ["asli"] if sys.argv[5] == "asli" else [float(x) for x in sys.argv[5].split(",")]
    names = list(NAMES) if sys.argv[6] == "semua" else sys.argv[6].split(",")
    assert all(n in NAMES for n in names) and all(l == "asli" or l in LEVELS for l in levs)
    jobs = [(ds, lev, r) for ds in names for lev in levs for r in range(a, b)]; t0 = time.time()
    with Pool(Pn) as pool:
        res = [x for rows in pool.imap(run_job, jobs, chunksize=1) for x in rows]
    p = Path(outn) if "/" in outn else HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, restval=""); w.writeheader(); w.writerows(res)
    print(f"MINI-41 {names} level {levs} rep {a}-{b}: {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

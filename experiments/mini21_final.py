"""MINI-21: validasi akhir pada dataset baru yang belum pernah disentuh (audit out/audit_baru.csv).
Kebijakan sudah lulus di pengembangan. Gate beku: cross-fit 3, bootstrap 20000, terima bila
kuantil-5% delta > .005; baseline M@0.5; kebenaran adalah delta di referensi terkunci.
  union_wF1, union_bacc, union_macroF1 : ruang union MINI-11 (HGB/DT/SVC x threshold).
                                         Pemilihan, gate, dan kebenaran memakai metrik tersebut (MINI-16).
  fsthr_KNN, fsthr_SVC                  : {M, M_mi25, M_mi50} x threshold, wF1 (MINI-19). FS = MI di SEARCH.
Ukuran latih (MINI-17/20): TOTAL 2.430 (630/1.800) bila dev >= 2.480, selain itu 1.000 (259/741).
Seed 1_021_000_000 + 1000*idx + rep, dengan idx urutan di DS. Ada pengaman error per trial.
Pakai: python mini21_final.py <R> <n_proses> <out> <nama_ds,dipisah,koma | semua>"""
import csv, sys, time, warnings, pickle
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import exp41_gate_confirm as G
import bridge_metrics as BM
from mini12_optuna import union, lab
from mini19_fs_lebar import mi_top

OFF = 1_021_000_000
# Nama: (sumber, id, kolom yang dibuang)
DS = {"pc3": ("openml", 1050, ()), "MIC": ("openml", 45648, ()), "Marketing_Campaign": ("openml", 46940, ()),
      "Pumpkin_Seeds": ("openml", 46951, ()), "seismic-bumps": ("openml", 46956, ()), "Rice": ("uci", 545, ()),
      "Census-Income-KDD": ("openml", 42750, ()), "Santander2016": ("openml", 46859, ()), "KDD98": ("openml", 42343, ()),
      "LICD": ("openml", 46792, ()), "Homesite": ("openml", 46857, ("QuoteNumber",))}
TH = [round(t, 2) for t in np.arange(.05, .96, .05)]
_D = {}


def load(src, i, drop):
    if src == "openml":
        X, y = G.load_openml_binary(i)
    else:
        df = pd.read_pickle(Path.home() / "uci_cache" / f"uci_{i}.pkl")
        y0 = df.pop("target").astype(str); cats = sorted(y0.unique()); assert len(cats) == 2
        y = pd.Series(pd.Categorical(y0, categories=cats).codes, name="target"); X = df
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]):
                s = X[c].astype(object).where(X[c].notna(), "__MISSING__").astype(str)
                X[c] = pd.Categorical(s, categories=sorted(s.unique())).codes
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.drop(columns=[c for c in drop if c in X.columns])
    return X.reset_index(drop=True), y.reset_index(drop=True)


def data(name):
    if name not in _D:
        _D.clear(); X, y = load(*DS[name])
        Xd, Xr, yd, yr = train_test_split(X, y, test_size=G.ref_size(len(X)), stratify=y, random_state=G.LOCKED_REF_SEED)
        _D[name] = (Xd.reset_index(drop=True), yd.reset_index(drop=True), Xr.reset_index(drop=True), np.asarray(yr))
    return _D[name]


def pick(y, P, idx, cands, base, m):
    best, g0 = None, 0.0
    for cfg in cands:
        g = float(BM.delta(G.counts8(y[idx], base[idx], lab(P, cfg, idx)), m))
        if g > g0 + 1e-12:
            best, g0 = cfg, g
    return best


def trial(job):
    try:
        return _trial(job)
    except Exception as e:
        return [dict(dataset=job[0], rep=job[1], TOTAL=-1, design="ERROR", metric="", cfg=str(e)[:150], D=0,
                     lo95=np.nan, ref=np.nan, kelas="error")]


def _trial(job):
    ds, rep = job; seed = OFF + 1000 * list(DS).index(ds) + rep
    Xd, yd, Xr, yr = data(ds)
    TOT, SS = (G.TOTAL_TRAIN, G.SEARCH_SIZE) if len(Xd) >= G.TOTAL_TRAIN + 50 else (1000, 259)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=TOT, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS))
            if np.bincount(ys[i], minlength=2).min() >= 10]
    if not bags:
        return [dict(dataset=ds, rep=rep, TOTAL=TOT, design="SKIP", metric="", cfg="", D=0, lo95=np.nan, ref=np.nan, kelas="skip")]
    cols = list(Xs.columns)
    F = {"": cols, "_mi25": mi_top(Xs, ys, .25, seed), "_mi50": mi_top(Xs, ys, .5, seed + 1)}
    P, Pr = {}, {}
    for m, sfxs in (("HGB", ("",)), ("DT", ("",)), ("KNN", tuple(F)), ("SVC", tuple(F))):
        for s in sfxs:
            fs = [G.fit_proba_m(m, {}, Xs.iloc[i].reset_index(drop=True), ys[i], F[s]) for i in bags]
            P[m + s] = np.column_stack([f(Xp) for f in fs]).mean(1)
            Pr[m + s] = np.column_stack([f(Xr) for f in fs]).mean(1)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp))
    out = []

    def row(design, M, cands, metric, k):
        base, baser = (P[M] >= .5).astype(int), (Pr[M] >= .5).astype(int)
        cnt = np.zeros(8, int)
        for a, b in folds:
            cfg = pick(yp, P, a, cands, base, metric)
            cnt += G.counts8(yp[b], base[b], base[b] if cfg is None else lab(P, cfg, b))
        boot = BM.delta(np.random.default_rng(seed + 61 + k).multinomial(len(yp), cnt / len(yp), size=20000), metric)
        cfg = pick(yp, P, np.arange(len(yp)), cands, base, metric)
        cr = G.counts8(yr, baser, baser if cfg is None else lab(Pr, cfg))
        dr = BM.delta(np.random.default_rng(seed + 987163).dirichlet(cr + .5, size=20000), metric)
        lo, hi = np.quantile(dr, [.025, .975])
        out.append(dict(dataset=ds, rep=rep, TOTAL=TOT, design=design, metric=metric, cfg=str(cfg),
                        D=int(cnt[[1, 2, 5, 6]].sum()), lo95=float(np.quantile(boot, .05)), ref=float(BM.delta(cr, metric)),
                        kelas="membaik" if lo > 0 else "memburuk" if hi < 0 else "tidak-pasti"))

    U = union(("HGB", "DT", "SVC"))
    for k, m in enumerate(("wF1", "bacc", "macroF1")):
        row(f"union_{m}", "HGB", U, m, k)
    for k, M in enumerate(("KNN", "SVC")):
        row(f"fsthr_{M}", M, [({M + s: 1}, t) for s in F for t in TH if (s, t) != ("", .5)], "wF1", 3 + k)
    return out


if __name__ == "__main__":
    R, Pn, outn = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    names = list(DS) if sys.argv[4] == "semua" else sys.argv[4].split(",")
    assert all(n in DS for n in names) and R <= 1000
    t0 = time.time(); jobs = [(ds, r) for ds in names for r in range(R)]
    with Pool(Pn) as pool:
        res = [x for rows in pool.imap(trial, jobs, chunksize=2) for x in rows]
    p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print(f"{len(names)} dataset, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

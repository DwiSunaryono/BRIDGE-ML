"""MINI-19 (PENGEMBANGAN): kapan seleksi fitur (FS) berguna? Menguji dua klaim teori:
 (a) FS membantu pada data LEBAR (p >= 100), (b) FS membantu model berbasis jarak (KNN, SVC).
FS = filter mutual information (MI) dihitung HANYA di SEARCH (data latih model; tak menyentuh POOL/ref):
     simpan top-25% atau top-50% fitur (minimal 5), model dilatih ulang (bag sama, 32x).
Model dasar M in {HGB, KNN, SVC}; baseline = M@0.5 (per baris desain).
Baris per trial:
  raw_<M>_mi25 / raw_<M>_mi50 : Δ ref Mfs@0.5 vs M@0.5 (tanpa gate; utk persentase menang/kalah)
  fs_<M>      : gate atas kandidat {M_mi25@.5, M_mi50@.5}  (FS sendirian)
  fsthr_<M>   : gate atas {M, M_mi25, M_mi50} x threshold grid (FS + threshold)
  union (HGB) : ruang MINI-11; union_FS: union + {HGB_mi25/50 x thr} + {HGB_mi25/50 + DT x thr}
Gate beku: cross-fit 3, bootstrap 20000, TERIMA iff kuantil-5% > .005.
Dataset (aturan ditetapkan SEBELUM run): 8 data LEBAR baru (biner, p >= 100, N >= 3600, belum dipakai)
+ 18 dataset DEV MINI-05. Seed 1_019_000_000. Pakai: python mini19_fs_lebar.py <R> <n_proses> [rep_awal] [lebar] [out]"""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import exp41_gate_confirm as G
from mini01_threshold_crossfit import gate
from mini05_ensemble import DS as DS18
from mini12_optuna import truth, union, lab, pick

OFF = 1_019_000_000
WIDE = {"Bioresponse": 4134, "hiva_agnostic": 1039, "sylva_agnostic": 1040, "christine": 41142,
        "philippine": 41145, "nomao": 1486, "SpeedDating": 40536, "KDDCup09_upselling": 1114}
DS = dict(WIDE, **DS18)
TH = [round(t, 2) for t in np.arange(.05, .96, .05)]
_D = {}


def data(did):
    if did not in _D:
        _D.clear(); _D[did] = G.make_dev_and_locked_reference(did)
    return _D[did]


def mi_top(Xs, ys, frac, seed):
    Z = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(Xs)
    mi = mutual_info_classif(Z, ys, random_state=seed % (2**32 - 1))
    k = max(5, int(round(frac * Xs.shape[1])))
    return [Xs.columns[i] for i in np.argsort(-mi, kind="stable")[:k]]


def trial(job):
    ds, did, rep = job; seed = OFF + 1000 * list(DS).index(ds) + rep
    Xd, yd, Xr, yr = data(did)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=G.TOTAL_TRAIN, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=G.SEARCH_SIZE, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS))
            if np.bincount(ys[i], minlength=2).min() >= 10]
    cols = list(Xs.columns)
    F = {"": cols, "_mi25": mi_top(Xs, ys, .25, seed), "_mi50": mi_top(Xs, ys, .5, seed + 1)}
    P, Pr = {}, {}
    for m in ("HGB", "DT", "KNN", "SVC"):
        for sfx, feats in F.items():
            if m == "DT" and sfx:
                continue
            fs = [G.fit_proba_m(m, {}, Xs.iloc[i].reset_index(drop=True), ys[i], feats) for i in bags]
            P[m + sfx] = np.column_stack([f(Xp) for f in fs]).mean(1)
            Pr[m + sfx] = np.column_stack([f(Xr) for f in fs]).mean(1)
    yr_ = np.asarray(yr)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp))
    out = []

    def row(design, M, cands, k):
        base, baser = (P[M] >= .5).astype(int), (Pr[M] >= .5).astype(int)
        cnt = np.zeros(8, int)
        for a, b in folds:
            cfg = pick(yp, P, a, cands, base)
            cnt += G.counts8(yp[b], base[b], base[b] if cfg is None else lab(P, cfg, b))
        l99, l95 = gate(cnt, len(yp), seed + 51 + k)
        cfg = pick(yp, P, np.arange(len(yp)), cands, base)
        d, kls = truth(yr_, baser, baser if cfg is None else lab(Pr, cfg), seed)
        out.append(dict(dataset=ds, lebar=ds in WIDE, p=len(cols), rep=rep, design=design, cfg=str(cfg),
                        D=int(cnt[[1, 2, 5, 6]].sum()), lo95=l95, ref=d, kelas=kls))

    k = 0
    for M in ("HGB", "KNN", "SVC"):
        baser = (Pr[M] >= .5).astype(int)
        for sfx in ("_mi25", "_mi50"):
            d, kls = truth(yr_, baser, (Pr[M + sfx] >= .5).astype(int), seed)
            out.append(dict(dataset=ds, lebar=ds in WIDE, p=len(cols), rep=rep, design=f"raw_{M}{sfx}", cfg="",
                            D=-1, lo95=np.nan, ref=d, kelas=kls))
        row(f"fs_{M}", M, [({M + "_mi25": 1}, .5), ({M + "_mi50": 1}, .5)], k); k += 1
        row(f"fsthr_{M}", M, [({M + s: 1}, t) for s in F for t in TH if (s, t) != ("", .5)], k); k += 1
    U = union(("HGB", "DT", "SVC"))
    UF = U + [({"HGB" + s: 1}, t) for s in ("_mi25", "_mi50") for t in TH] \
        + [({"HGB" + s: w, "DT": 1 - w}, t) for s in ("_mi25", "_mi50") for w in (.75, .5, .25) for t in TH]
    row("union", "HGB", U, k); row("union_FS", "HGB", UF, k + 1)
    return out


if __name__ == "__main__":
    R, Pn = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    r0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # rep awal (perluasan: seed tetap unik, rep < 1000)
    pilih = WIDE if len(sys.argv) > 4 and sys.argv[4] == "lebar" else DS
    outn = sys.argv[5] if len(sys.argv) > 5 else "mini19_fs_lebar_dev.csv"
    assert r0 + R <= 1000
    jobs = [(ds, did, r) for ds, did in pilih.items() for r in range(r0, r0 + R)]
    with Pool(Pn) as pool:
        res = [x for rows in pool.map(trial, jobs, chunksize=2) for x in rows]
    p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print(f"{len(DS)} dataset, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

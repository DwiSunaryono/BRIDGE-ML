"""MINI-32: ablasi faktorial 2x2x2 komponen BRIDGE (PRAREG_MINI32.md): setiap komponen diuji
secara terpisah.
Faktor: (A) sumber bukti: POOL yang sama (konfigurasi akhir dinilai di data yang memilihnya) vs
cross-fit 3 lipat; (B) statistik: delta titik vs batas bawah bootstrap (kuantil 5%, 20.000);
(C) margin: 0 vs 0,005.
8 sel; sel (cross-fit, bootstrap, 0,005) = BRIDGE; (POOL, titik, 0) = naif; (POOL, titik, 0,005) =
VALMARGIN 0,005; (cross-fit, bootstrap, 0) = boot0. Keputusan dan kebenaran dinilai pada konfigurasi
deploy yang sama (dipilih di seluruh POOL).
Data: 11 dataset baru (seed MINI-21/24, R=100) dan populasi Q (seed MINI-25). Kolom lo_cf_boot harus
sama dengan lo95 run asal.
Pakai: python mini32_ablasi.py <nyata|Q> <awal> <akhir> <n_proses> <out>   (nyata: indeks dataset;
Q: indeks tugas)"""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini21_final as F
sys.argv = _argv
import exp41_gate_confirm as G, bridge_metrics as BM
from mini12_optuna import union, lab
MODE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("nyata", "Q") else "nyata"
if MODE == "Q":
    import mini25_sintetis as Q  # mengganti F.OFF, F.DS, F.data dengan populasi Q (seed MINI-25)
U = union(("HGB", "DT", "SVC"))


def run_job(job):
    try:
        return _trial(job)
    except Exception as e:
        return dict(dataset=job[0], rep=job[1], cfg="ERROR " + str(e)[:150])


def _trial(job):
    ds, rep = job; seed = F.OFF + 1000 * list(F.DS).index(ds) + rep
    Xd, yd, Xr, yr = F.data(ds)
    TOT, SS = (G.TOTAL_TRAIN, G.SEARCH_SIZE) if len(Xd) >= G.TOTAL_TRAIN + 50 else (1000, 259)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=TOT, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS)) if np.bincount(ys[i], minlength=2).min() >= 10]
    cols = list(Xs.columns); P, Pr = {}, {}
    for m in ("HGB", "DT", "SVC"):
        fs = [G.fit_proba_m(m, {}, Xs.iloc[i].reset_index(drop=True), ys[i], cols) for i in bags]
        P[m] = np.column_stack([f(Xp) for f in fs]).mean(1); Pr[m] = np.column_stack([f(Xr) for f in fs]).mean(1)
    base, baser = (P["HGB"] >= .5).astype(int), (Pr["HGB"] >= .5).astype(int)
    pk = lambda idx: F.pick(yp, P, idx, U, base, "wF1")
    cnt = np.zeros(8, int)
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp):
        c = pk(a); cnt += G.counts8(yp[b], base[b], base[b] if c is None else lab(P, c, b))
    cfg = pk(np.arange(len(yp)))
    cpool = G.counts8(yp, base, base if cfg is None else lab(P, cfg))  # konfigurasi akhir dinilai di POOL yang sama
    q = lambda c, s: float(np.quantile(BM.delta(np.random.default_rng(s).multinomial(c.sum(), c / c.sum(), size=20000), "wF1"), .05))
    cr = G.counts8(yr, baser, baser if cfg is None else lab(Pr, cfg))
    return dict(dataset=ds, rep=rep, cfg=str(cfg), ref=float(BM.delta(cr, "wF1")),
                pt_pool=float(BM.delta(cpool, "wF1")), lo_pool_boot=q(cpool, seed + 62) if cfg is not None else float("nan"),
                pt_cf=float(BM.delta(cnt, "wF1")), lo_cf_boot=q(cnt, seed + 61))


if __name__ == "__main__":
    a, b, Pn, outn = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    names = list(F.DS)
    jobs = ([(names[i], r) for i in range(a, b) for r in range(100)] if MODE == "nyata" else [(f"T{i:04d}", 0) for i in range(a, b)])
    t0 = time.time()
    with Pool(Pn) as pool:
        res = pool.map(run_job, jobs, chunksize=2)
    keys = list(next(r for r in res if "ref" in r)); p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(res)
    print(f"{MODE} {a}-{b}, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

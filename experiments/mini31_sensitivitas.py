"""MINI-31: sensitivitas parameter BRIDGE dan model dasar XGBoost pada populasi sintetis Q (tugas i.i.d.,
Proposisi 1). PRAREG_MINI31.md.
Per tugas (T0000-T1399, data dan seed identik MINI-25): model dilatih sekali; konfigurasi deploy dipilih di
seluruh POOL (sama untuk semua varian), sehingga kebenaran sama; yang divariasikan hanya bukti gate:
  K lipatan cross-fit in {2, 3, 5}; kuantil in {0,025; 0,05; 0,10}; jumlah bootstrap in {2.000; 20.000} (K=3, q=0,05).
  Margin tetap 0,005. Varian baku (K=3, q=0,05, B=20.000) harus identik dengan kolom lo95 MINI-25.
Mode "xgb": model dasar XGBoost (baseline XGB@0,5; kandidat union XGB/DT/SVC x threshold), gate baku.
Pakai: python mini31_sensitivitas.py <mode: param|xgb> <tugas_awal> <tugas_akhir> <n_proses> <out>"""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import exp41_gate_confirm as G
import bridge_metrics as BM
from mini12_optuna import union, lab
import mini25_sintetis as Q
F = Q.F
MODE = sys.argv[1] if __name__ == "__main__" and len(sys.argv) > 1 else "param"


def run_job(job):
    try:
        return _trial(job)
    except Exception as e:
        return dict(dataset=job[0], cfg="ERROR " + str(e)[:150])


def _trial(job):
    ds, rep = job; seed = F.OFF + 1000 * list(F.DS).index(ds) + rep  # = seed MINI-25
    Xd, yd, Xr, yr = F.data(ds)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=G.TOTAL_TRAIN, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=G.SEARCH_SIZE, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS)) if np.bincount(ys[i], minlength=2).min() >= 10]
    B0 = "XGB" if MODE == "xgb" else "HGB"; names = (B0, "DT", "SVC"); cols = list(Xs.columns); P, Pr = {}, {}
    for m in names:
        fs = [G.fit_proba_m(m, {}, Xs.iloc[i].reset_index(drop=True), ys[i], cols) for i in bags]
        P[m] = np.column_stack([f(Xp) for f in fs]).mean(1); Pr[m] = np.column_stack([f(Xr) for f in fs]).mean(1)
    base, baser = (P[B0] >= .5).astype(int), (Pr[B0] >= .5).astype(int); U = union(names)
    pk = lambda idx: F.pick(yp, P, idx, U, base, "wF1")
    cfg = pk(np.arange(len(yp))); cr = G.counts8(yr, baser, baser if cfg is None else lab(Pr, cfg))
    dr = BM.delta(np.random.default_rng(seed + 987163).dirichlet(cr + .5, size=20000), "wF1"); lo, hi = np.quantile(dr, [.025, .975])
    out = dict(dataset=ds, mode=MODE, cfg=str(cfg), ref=float(BM.delta(cr, "wF1")), kelas="membaik" if lo > 0 else "memburuk" if hi < 0 else "tidak-pasti")
    Ks = (2, 3, 5) if MODE == "param" else (3,)
    for K in Ks:
        cnt = np.zeros(8, int)
        for a, b in StratifiedKFold(K, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp):
            c = pk(a); cnt += G.counts8(yp[b], base[b], base[b] if c is None else lab(P, c, b))
        boot = BM.delta(np.random.default_rng(seed + 61).multinomial(len(yp), cnt / len(yp), size=20000), "wF1")
        for q in ((.025, .05, .10) if MODE == "param" else (.05,)):
            out[f"lo_K{K}_q{q}"] = float(np.quantile(boot, q))
        if K == 3 and MODE == "param":
            out["lo_K3_q0.05_B2000"] = float(np.quantile(boot[:2000], .05))
    return out


if __name__ == "__main__":
    a, b, Pn, outn = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    jobs = [(f"T{i:04d}", 0) for i in range(a, b)]; t0 = time.time()
    with Pool(Pn) as pool:
        res = pool.map(run_job, jobs, chunksize=2)
    keys = list(next(r for r in res if "ref" in r)); p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(res)
    print(f"{MODE} tugas {a}-{b}, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

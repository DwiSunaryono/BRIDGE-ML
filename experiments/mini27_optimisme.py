"""MINI-27 (deskriptif, motivasi Gambar 1): metrik optimism gap dan rank inversion pada data ini.
Seed dan model identik MINI-21 (11 dataset baru, kebijakan union, wF1). Per trial, untuk 173 kandidat union:
  dpool_c = delta wF1 di seluruh POOL (estimasi validasi naif), dref_c = delta wF1 di referensi terkunci (kebenaran).
Disimpan: rho_spearman(dpool, dref) atas kandidat; kandidat terbaik menurut POOL: dpool_best, dref_best,
optimism gap = dpool_best - dref_best; estimasi cross-fit BRIDGE (delta titik dari counts out-of-fold) untuk konfigurasi
yang di-deploy, sehingga bias estimasi naif vs cross-fit bisa dibandingkan pada keputusan yang sama.
Bukan uji hipotesis (tidak ada kriteria lulus); murni deskriptif.
Pakai: python mini27_optimisme.py <R> <n_proses> <out> <nama,dipisah,koma | semua>"""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import exp41_gate_confirm as G
import bridge_metrics as BM
from mini12_optuna import union, lab
import mini21_final as F


def trial(job):
    try:
        return _trial(job)
    except Exception as e:
        return dict(dataset=job[0], rep=job[1], cfg="ERROR " + str(e)[:150])


def _trial(job):
    ds, rep = job; seed = F.OFF + 1000 * list(F.DS).index(ds) + rep  # = seed MINI-21
    Xd, yd, Xr, yr = F.data(ds)
    TOT, SS = (G.TOTAL_TRAIN, G.SEARCH_SIZE) if len(Xd) >= G.TOTAL_TRAIN + 50 else (1000, 259)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=TOT, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS))
            if np.bincount(ys[i], minlength=2).min() >= 10]
    cols = list(Xs.columns); P, Pr = {}, {}
    for m in ("HGB", "DT", "SVC"):
        fs = [G.fit_proba_m(m, {}, Xs.iloc[i].reset_index(drop=True), ys[i], cols) for i in bags]
        P[m] = np.column_stack([f(Xp) for f in fs]).mean(1); Pr[m] = np.column_stack([f(Xr) for f in fs]).mean(1)
    base, baser = (P["HGB"] >= .5).astype(int), (Pr["HGB"] >= .5).astype(int)
    U = union(("HGB", "DT", "SVC"))
    dpool = np.array([float(BM.delta(G.counts8(yp, base, lab(P, c)), "wF1")) for c in U])
    dref = np.array([float(BM.delta(G.counts8(yr, baser, lab(Pr, c)), "wF1")) for c in U])
    rho = float(stats.spearmanr(dpool, dref).statistic)
    b = int(np.argmax(dpool))
    cnt = np.zeros(8, int)
    for a, bb in StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp):
        cfg = F.pick(yp, P, a, U, base, "wF1")
        cnt += G.counts8(yp[bb], base[bb], base[bb] if cfg is None else lab(P, cfg, bb))
    cfg = F.pick(yp, P, np.arange(len(yp)), U, base, "wF1")
    return dict(dataset=ds, rep=rep, TOTAL=TOT, rho=rho, dpool_best=float(dpool[b]), dref_best=float(dref[b]),
                gap=float(dpool[b] - dref[b]), cfg=str(cfg), dcf=float(BM.delta(cnt, "wF1")),
                dref_deploy=0.0 if cfg is None else float(dref[U.index(cfg)]),
                n_pos_pool=int((dpool > 0).sum()), n_pos_ref=int((dref > 0).sum()))


if __name__ == "__main__":
    R, Pn, outn = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    names = list(F.DS) if sys.argv[4] == "semua" else sys.argv[4].split(",")
    jobs = [(ds, r) for ds in names for r in range(R)]; t0 = time.time()
    with Pool(Pn) as pool:
        res = pool.map(trial, jobs, chunksize=2)
    keys = list(next(r for r in res if "rho" in r)); p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(res)
    print(f"{len(names)} dataset, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

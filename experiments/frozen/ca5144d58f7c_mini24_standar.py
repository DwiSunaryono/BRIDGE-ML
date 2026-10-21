"""MINI-24 (perluasan MINI-23, kode trial sama + 2 aturan standar): BRIDGE vs ATURAN TERIMA STANDAR (kebijakan union, wF1) pada 11 dataset BARU MINI-21.
Seed IDENTIK MINI-21 -> konfigurasi yang di-deploy (dipilih di seluruh POOL) & kebenaran referensi sama persis;
yang dibandingkan HANYA aturan terima/tolak. Semua aturan ditetapkan SEBELUM run (PRAREG_MINI23.md):
  naif        : terima bila pemilihan di POOL menemukan konfigurasi dgn wF1 validasi > baseline (praktik umum)
  mcnemar_pool: pilih di POOL, lalu uji McNemar eksak satu sisi (p < .05) di POOL yang SAMA (praktik umum)
  mcnemar_cf  : uji McNemar eksak satu sisi (p < .05) pada counts cross-fit 3 lipat (pemilihan jujur)
  crt         : corrected resampled t-test (Nadeau & Bengio 2003), 10 split 2/3-1/3 POOL; tiap split pilih di
                2/3, ukur Δ wF1 di 1/3; terima bila rata2 > 0 dan p satu sisi < .05
  boot0       : BRIDGE tanpa margin (kuantil-5% bootstrap > 0)            [ablasi]
  bridge      : BRIDGE beku (kuantil-5% > .005)                            [harus identik MINI-21 union_wF1]
  crt_final   : corrected resampled t-test STANDAR: konfigurasi AKHIR (dipilih di seluruh POOL) dievaluasi pada
                10 split 2/3-1/3 yg sama TANPA pilih ulang; terima bila rata2 > 0 & p < .05
  t5x2        : uji 5x2cv paired t (Dietterich 1998) pada konfigurasi AKHIR (Δ wF1 per lipatan); p satu sisi < .05
Pakai: python mini23_pembanding.py <R> <n_proses> <out> <nama,dipisah,koma | semua>"""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import exp41_gate_confirm as G
import bridge_metrics as BM
from mini12_optuna import union, lab
import mini21_final as F
from sklearn.model_selection import train_test_split


def mcn(y, b, c):
    """McNemar eksak satu sisi: kandidat benar & baseline salah (n10) vs sebaliknya (n01)."""
    n10 = int(((c == y) & (b != y)).sum()); n01 = int(((b == y) & (c != y)).sum())
    return 1.0 if n10 + n01 == 0 else float(stats.binomtest(n10, n10 + n01, .5, alternative="greater").pvalue)


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
    U = union(("HGB", "DT", "SVC")); pk = lambda idx: F.pick(yp, P, idx, U, base, "wF1")
    cnt = np.zeros(8, int); cf_c = base.copy()
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp):
        cfg = pk(a); cb = base[b] if cfg is None else lab(P, cfg, b)
        cnt += G.counts8(yp[b], base[b], cb); cf_c[b] = cb
    boot = BM.delta(np.random.default_rng(seed + 61).multinomial(len(yp), cnt / len(yp), size=20000), "wF1")
    lo95 = float(np.quantile(boot, .05))
    cfg = pk(np.arange(len(yp)))
    cpool = base if cfg is None else lab(P, cfg)
    dl = []
    for a, b in StratifiedShuffleSplit(10, test_size=1 / 3, random_state=(seed + 5) % (2**32 - 1)).split(yp, yp):
        c2 = pk(a); dl.append(0.0 if c2 is None else float(BM.delta(G.counts8(yp[b], base[b], lab(P, c2, b)), "wF1")))
    dl = np.asarray(dl); J = len(dl); sd = dl.std(ddof=1)
    dfin = []
    for a, b in StratifiedShuffleSplit(10, test_size=1 / 3, random_state=(seed + 5) % (2**32 - 1)).split(yp, yp):
        dfin.append(float(BM.delta(G.counts8(yp[b], base[b], cpool[b]), "wF1")))
    dfin = np.asarray(dfin); sf = dfin.std(ddof=1)
    p_fin = (float(stats.t.sf(dfin.mean() / np.sqrt((1 / J + .5) * sf ** 2), J - 1)) if sf > 0
             else (0.0 if dfin.mean() > 0 else 1.0))
    p5, s2 = [], []
    for i in range(5):
        dd = []
        for a, b in StratifiedKFold(2, shuffle=True, random_state=(seed + 100 + i) % (2**32 - 1)).split(yp, yp):
            dd.append(float(BM.delta(G.counts8(yp[b], base[b], cpool[b]), "wF1")))
        p5.append(dd[0]); s2.append((dd[0] - np.mean(dd)) ** 2 + (dd[1] - np.mean(dd)) ** 2)
    den = np.sqrt(np.mean(s2))
    p_5x2 = float(stats.t.sf(p5[0] / den, 5)) if den > 0 else (0.0 if p5[0] > 0 else 1.0)
    if sd > 0:
        tt = dl.mean() / np.sqrt((1 / J + (1 / 3) / (2 / 3)) * sd ** 2); p_crt = float(stats.t.sf(tt, J - 1))
    else:
        p_crt = 0.0 if dl.mean() > 0 else 1.0
    cr = G.counts8(yr, baser, baser if cfg is None else lab(Pr, cfg))
    dr = BM.delta(np.random.default_rng(seed + 987163).dirichlet(cr + .5, size=20000), "wF1")
    lo, hi = np.quantile(dr, [.025, .975])
    p_pool, p_cf = mcn(yp, base, cpool), mcn(yp, base, cf_c)
    return dict(dataset=ds, rep=rep, TOTAL=TOT, cfg=str(cfg), lo95=lo95, ref=float(BM.delta(cr, "wF1")),
                kelas="membaik" if lo > 0 else "memburuk" if hi < 0 else "tidak-pasti",
                p_mcn_pool=p_pool, p_mcn_cf=p_cf, crt_mean=float(dl.mean()), p_crt=p_crt,
                naif=int(cfg is not None), mcnemar_pool=int(cfg is not None and p_pool < .05),
                mcnemar_cf=int(cfg is not None and p_cf < .05), crt=int(cfg is not None and dl.mean() > 0 and p_crt < .05),
                boot0=int(cfg is not None and lo95 > 0), bridge=int(lo95 > .005),
                p_crt_final=p_fin, p_5x2=p_5x2, crt_final=int(cfg is not None and dfin.mean() > 0 and p_fin < .05),
                t5x2=int(cfg is not None and p_5x2 < .05))


if __name__ == "__main__":
    R, Pn, outn = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    names = list(F.DS) if sys.argv[4] == "semua" else sys.argv[4].split(",")
    jobs = [(ds, r) for ds in names for r in range(R)]; t0 = time.time()
    with Pool(Pn) as pool:
        res = pool.map(trial, jobs, chunksize=2)
    keys = list(next(r for r in res if "lo95" in r))
    p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(res)
    print(f"{len(names)} dataset, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

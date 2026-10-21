"""MINI-30: KONFIRMASI KESEHATAN dgn METRIK KLINIS (PRAREG_MINI30.md). Gate BRIDGE beku (cross-fit 3, bootstrap 20000,
TERIMA iff kuantil-5% Δ > 0,005), kebijakan union (HGB/DT/SVC x threshold), baseline HGB@0,5.
Metrik (pemilihan, gate, kebenaran memakai metrik yg sama): bacc, nb10, nb20, F2 (bridge_metrics_klinis) + wF1.
Per baris disimpan 8-counts referensi konfigurasi deploy -> Δ sensitivitas & spesifisitas (deskriptif).
12 tugas kesehatan dari 6 sumber (health_data, _b, _c); seed 1_030_000_000 + 1000*idx + rep (BARU).
Pakai: python mini30_klinis.py <R> <n_proses> <out> <nama,dipisah,koma | semua>"""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import exp41_gate_confirm as G
import bridge_metrics_klinis as K
from mini12_optuna import union, lab
import health_data as H, health_data_b as HB, health_data_c as HC

OFF = 1_030_000_000
DS = ["mammography", "actg175"] + list(H.NH_TGT) + list(H.BR_TGT) + ["covid_mx", "wids_icu"]
KLASTER = dict(H.KLASTER, **HB.KLASTER, **HC.KLASTER)
METRIK = ("bacc", "nb10", "nb20", "F2", "wF1")
U = union(("HGB", "DT", "SVC"))
_D = {}


def load(name):
    return HB.load(name) if name == "covid_mx" else HC.load(name) if name == "wids_icu" else H.load(name)


def data(name):
    if name not in _D:
        _D.clear(); X, y = load(name)
        Xd, Xr, yd, yr = train_test_split(X, y, test_size=G.ref_size(len(X)), stratify=y, random_state=G.LOCKED_REF_SEED)
        _D[name] = (Xd.reset_index(drop=True), yd.reset_index(drop=True), Xr.reset_index(drop=True), np.asarray(yr))
    return _D[name]


def pick(y, P, idx, base, m):
    best, g0 = None, 0.0
    for cfg in U:
        g = float(K.delta(G.counts8(y[idx], base[idx], lab(P, cfg, idx)), m))
        if g > g0 + 1e-12:
            best, g0 = cfg, g
    return best, g0


def trial(job):
    try:
        return _trial(job)
    except Exception as e:
        return [dict(dataset=job[0], rep=job[1], metric="", cfg="ERROR " + str(e)[:150], kelas="error")]


def _trial(job):
    ds, rep = job; seed = OFF + 1000 * DS.index(ds) + rep
    Xd, yd, Xr, yr = data(ds)
    TOT, SS = (G.TOTAL_TRAIN, G.SEARCH_SIZE) if len(Xd) >= G.TOTAL_TRAIN + 50 else (1000, 259)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=TOT, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS))
            if np.bincount(ys[i], minlength=2).min() >= 10]
    if not bags:
        return [dict(dataset=ds, rep=rep, metric="", cfg="SKIP", kelas="skip")]
    cols = list(Xs.columns); P, Pr = {}, {}
    for m in ("HGB", "DT", "SVC"):
        fs = [G.fit_proba_m(m, {}, Xs.iloc[i].reset_index(drop=True), ys[i], cols) for i in bags]
        P[m] = np.column_stack([f(Xp) for f in fs]).mean(1); Pr[m] = np.column_stack([f(Xr) for f in fs]).mean(1)
    base, baser = (P["HGB"] >= .5).astype(int), (Pr["HGB"] >= .5).astype(int)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp))
    out = []
    for k, m in enumerate(METRIK):
        cnt = np.zeros(8, int)
        for a, b in folds:
            cfg, _ = pick(yp, P, a, base, m)
            cnt += G.counts8(yp[b], base[b], base[b] if cfg is None else lab(P, cfg, b))
        boot = K.delta(np.random.default_rng(seed + 61 + k).multinomial(len(yp), cnt / len(yp), size=20000), m)
        cfg, gpool = pick(yp, P, np.arange(len(yp)), base, m)
        cr = G.counts8(yr, baser, baser if cfg is None else lab(Pr, cfg))
        dr = K.delta(np.random.default_rng(seed + 987163 + k).dirichlet(cr + .5, size=20000), m)
        lo, hi = np.quantile(dr, [.025, .975])
        out.append(dict(dataset=ds, sumber=KLASTER[ds], rep=rep, TOTAL=TOT, metric=m, cfg=str(cfg),
                        lo95=float(np.quantile(boot, .05)), dpool=gpool, ref=float(K.delta(cr, m)),
                        kelas="membaik" if lo > 0 else "memburuk" if hi < 0 else "tidak-pasti",
                        d_sens=float(K.delta(cr, "sens")), d_spes=float(K.delta(cr, "spes")),
                        counts_ref=" ".join(map(str, cr.tolist()))))
    return out


if __name__ == "__main__":
    R, Pn, outn = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    names = DS if sys.argv[4] == "semua" else sys.argv[4].split(","); assert all(n in DS for n in names)
    jobs = [(ds, r) for ds in names for r in range(R)]; t0 = time.time()
    with Pool(Pn) as pool:
        res = [x for rows in pool.imap(trial, jobs, chunksize=1) for x in rows]
    keys = list(next(r for r in res if "lo95" in r)); p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(res)
    print(f"{len(names)} tugas, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

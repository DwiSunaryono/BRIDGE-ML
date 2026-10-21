"""MINI-36: jaringan saraf (MLP) sebagai model dasar, uji prediksi dari MINI-34 (PRAREG_MINI36.md).
MLP_ES  : MLPClassifier(64,32), early_stopping=True, validation_fraction=0,1, n_iter_no_change=10 (aturan sama dengan HGB), max_iter 300
MLP_noES: sama, early_stopping=False, tol=0, dilatih ~300 epoch tanpa kendali iterasi (padanan XGB 300 ronde)
Kandidat union (MLP, DT, SVC); gate baku. Tugas Q T1500-T2899 (pembagian identik MINI-33/34). Diagnostik skill_pool seperti MINI-34.
Pakai: python mini36_mlp.py <awal> <akhir> <n_proses> <out>
"""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini33_xgb_setara as M33
sys.argv = _argv
import exp41_gate_confirm as G, bridge_metrics as BM, data as Dd
from mini12_optuna import union, lab
from mini34_xgb_faktor import ll
F = M33.F
KOND = {"MLP_ES": dict(early_stopping=True, validation_fraction=.1, n_iter_no_change=10),
        "MLP_noES": dict(early_stopping=False, tol=0.0, n_iter_no_change=300)}


def fit_mlp(Xf, yf, ubah):
    cols = list(Xf.columns); num_c, cat_c = Dd.column_kinds(Xf)
    pre = Dd.make_preprocessor(num_c, cat_c); Z = pre.fit_transform(Xf, yf)
    est = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42, **ubah).fit(Z, yf)
    return (lambda Xn: est.predict_proba(pre.transform(Xn[cols]))[:, 1]), est.n_iter_


def run_job(job):
    try:
        return _trial(job)
    except Exception as e:
        return [dict(dataset=job[0], base="", cfg="ERROR " + str(e)[:150])]


def _trial(job):
    ds, rep = job; seed = F.OFF + 1000 * list(F.DS).index(ds) + rep
    Xd, yd, Xr, yr = F.data(ds)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=G.TOTAL_TRAIN, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=G.SEARCH_SIZE, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp); yr = np.asarray(yr)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS)) if np.bincount(ys[i], minlength=2).min() >= 10]
    cols = list(Xs.columns); P, Pr, EP = {}, {}, {}
    for m in ("DT", "SVC", *KOND):
        if m in KOND:
            r = [fit_mlp(Xs.iloc[i].reset_index(drop=True), ys[i], KOND[m]) for i in bags]; fs = [f for f, _ in r]; EP[m] = float(np.mean([e for _, e in r]))
        else:
            fs = [G.fit_proba_m(m, {}, Xs.iloc[i].reset_index(drop=True), ys[i], cols) for i in bags]
        P[m] = np.column_stack([f(Xp) for f in fs]).mean(1); Pr[m] = np.column_stack([f(Xr) for f in fs]).mean(1)
    ll0 = ll(yp, np.full(len(yp), ys.mean())); out = []
    for B0 in KOND:
        U = union((B0, "DT", "SVC")); base, baser = (P[B0] >= .5).astype(int), (Pr[B0] >= .5).astype(int)
        pk = lambda idx: F.pick(yp, P, idx, U, base, "wF1")
        cnt = np.zeros(8, int)
        for a, b in StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp):
            c = pk(a); cnt += G.counts8(yp[b], base[b], base[b] if c is None else lab(P, c, b))
        lo95 = float(np.quantile(BM.delta(np.random.default_rng(seed + 61).multinomial(len(yp), cnt / len(yp), size=20000), "wF1"), .05))
        cfg = pk(np.arange(len(yp))); cpool = G.counts8(yp, base, base if cfg is None else lab(P, cfg))
        cr = G.counts8(yr, baser, baser if cfg is None else lab(Pr, cfg))
        dr = BM.delta(np.random.default_rng(seed + 987163).dirichlet(cr + .5, size=20000), "wF1"); lo, hi = np.quantile(dr, [.025, .975])
        llp = ll(yp, P[B0])
        out.append(dict(dataset=ds, base=B0, cfg=str(cfg), lo95=lo95, pt_pool=float(BM.delta(cpool, "wF1")), ref=float(BM.delta(cr, "wF1")),
                        kelas="membaik" if lo > 0 else "memburuk" if hi < 0 else "tidak-pasti", ll_pool=llp, ll_nol=ll0, skill_pool=1 - llp / ll0,
                        ekstrem_pool=100 * float(np.mean((P[B0] < .01) | (P[B0] > .99))), ll_ref=ll(yr, Pr[B0]), epoch=EP[B0]))
    return out


if __name__ == "__main__":
    a, b, Pn, outn = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    jobs = [(f"T{i:04d}", 0) for i in range(a, b)]; t0 = time.time()
    with Pool(Pn) as pool:
        res = [x for rows in pool.imap(run_job, jobs, chunksize=1) for x in rows]
    keys = list(next(r for r in res if "ref" in r)); p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(res)
    print(f"tugas {a}-{b}, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

"""MINI-37 (PRAREG_MINI37.md): (a) sensitivitas JUMLAH BAG (16/32/64) & UKURAN SEARCH (360/630/1.000 dari latih 2.430)
pada populasi Q (tugas & seed MINI-25; varian baku 32/630 HARUS identik dgn MINI-25); (b) PRESISI: galat Monte Carlo
kuantil 5% (20 seed bootstrap, B=2.000 vs 20.000) + peluang keputusan berubah, dan galat baku Δ referensi (bootstrap
multinomial 2.000 atas hitungan referensi). Mode nyata: hanya (b) utk konfigurasi baku, 11 dataset baru x ulangan 0-9.
Pakai: python mini37_bag_search_presisi.py <Q|nyata> <awal> <akhir> <n_proses> <out>"""
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
    import mini25_sintetis as Q  # mengganti F.OFF, F.DS, F.data dgn populasi Q (seed MINI-25)
U = union(("HGB", "DT", "SVC"))
VARIAN = [("B32_S630", 32, 630), ("B16_S630", 16, 630), ("B64_S630", 64, 630), ("B32_S360", 32, 360), ("B32_S1000", 32, 1000)]


def run_job(job):
    try:
        return _trial(job)
    except Exception as e:
        return [dict(dataset=job[0], rep=job[1], varian="", cfg="ERROR " + str(e)[:150])]


def q05(c, seed, B):
    return float(np.quantile(BM.delta(np.random.default_rng(seed).multinomial(c.sum(), c / c.sum(), size=B), "wF1"), .05))


def _gate(yp, P, base, seed):
    pk = lambda idx: F.pick(yp, P, idx, U, base, "wF1")
    cnt = np.zeros(8, int)
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp):
        c = pk(a); cnt += G.counts8(yp[b], base[b], base[b] if c is None else lab(P, c, b))
    return cnt, pk(np.arange(len(yp)))


def _trial(job):
    ds, rep = job; seed = F.OFF + 1000 * list(F.DS).index(ds) + rep
    Xd, yd, Xr, yr = F.data(ds)
    TOT, SS0 = (G.TOTAL_TRAIN, G.SEARCH_SIZE) if len(Xd) >= G.TOTAL_TRAIN + 50 else (1000, 259)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=TOT, stratify=yd, random_state=seed)
    varian = VARIAN if MODE == "Q" else VARIAN[:1]
    out, fitted = [], {}
    for nama, nb, ss in varian:
        ss = ss if MODE == "Q" else SS0
        if ss not in fitted:
            Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=ss, stratify=ytr, random_state=seed)
            Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
            rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
            draws = [rng.choice(len(ys), size=sub, replace=False) for _ in range(64 if ss == 630 or MODE != "Q" else 32)]
            nmax = len(draws); cols = list(Xs.columns); fits = {}
            ok = [np.bincount(ys[i], minlength=2).min() >= 10 for i in draws]
            for m in ("HGB", "DT", "SVC"):
                fits[m] = [(G.fit_proba_m(m, {}, Xs.iloc[i].reset_index(drop=True), ys[i], cols) if o else None) for i, o in zip(draws, ok)]
            fitted[ss] = (yp, Xp, fits, ok)
        yp, Xp, fits, ok = fitted[ss]
        P, Pr = {}, {}
        for m in ("HGB", "DT", "SVC"):
            fs = [f for f, o in zip(fits[m][:nb], ok[:nb]) if o]  # == aturan asli: ambil nb draw pertama lalu saring
            P[m] = np.column_stack([f(Xp) for f in fs]).mean(1); Pr[m] = np.column_stack([f(Xr) for f in fs]).mean(1)
        base, baser = (P["HGB"] >= .5).astype(int), (Pr["HGB"] >= .5).astype(int)
        cnt, cfg = _gate(yp, P, base, seed)
        lo95 = q05(cnt, seed + 61, 20000)
        yrr = np.asarray(yr); cr = G.counts8(yrr, baser, baser if cfg is None else lab(Pr, cfg))
        row = dict(dataset=ds, rep=rep, varian=nama, bags=nb, search=ss, cfg=str(cfg), lo95=lo95, ref=float(BM.delta(cr, "wF1")))
        if nama == "B32_S630":  # (b) presisi
            q20 = [q05(cnt, seed + 5000 + s, 20000) for s in range(20)]; q2 = [q05(cnt, seed + 6000 + s, 2000) for s in range(20)]
            acc = lo95 > .005
            se_ref = float(np.std(BM.delta(np.random.default_rng(seed + 7).multinomial(cr.sum(), cr / cr.sum(), size=2000), "wF1")))
            row.update(sd_q_B20000=float(np.std(q20)), sd_q_B2000=float(np.std(q2)),
                       flip_B20000=float(np.mean([(q > .005) != acc for q in q20])), flip_B2000=float(np.mean([(q > .005) != acc for q in q2])),
                       n_ref=int(cr.sum()), se_ref_delta=se_ref)
        out.append(row)
    return out


if __name__ == "__main__":
    a, b, Pn, outn = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    names = list(F.DS)
    jobs = ([(names[i], r) for i in range(a, b) for r in range(10)] if MODE == "nyata" else [(f"T{i:04d}", 0) for i in range(a, b)])
    t0 = time.time()
    with Pool(Pn) as pool:
        res = [x for rows in pool.imap(run_job, jobs, chunksize=1) for x in rows]
    keys = list(next(r for r in res if "ref" in r and "sd_q_B20000" in r)); p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(res)
    print(f"{MODE} {a}-{b}, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

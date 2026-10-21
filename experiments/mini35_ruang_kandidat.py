"""MINI-35: ukuran ruang kandidat x sumber bukti (PRAREG_MINI35.md), menindaklanjuti
ablasi MINI-32 (H1 tidak terbukti: bootstrap pada POOL yang sama dengan margin 0,005
lulus pada K=173). Teori: bias pemilihan (winner's curse) tumbuh dengan ukuran ruang
kandidat K; bukti POOL yang sama tidak mengoreksinya, cross-fit mengoreksinya. Ruang
(anggota HGB/DT/SVC sama, hanya lebih rapat):
  K173  : union MINI-12 (ambang 0,05-0,95 langkah 0,05; bobot 2-model 0,25; simpleks 3-model 0,25 x ambang 0,30-0,60 langkah 0,05)
  Ksedang: ambang langkah 0,02; bobot 2-model langkah 0,1; simpleks 0,1 x ambang 0,30-0,70 langkah 0,02
  Kbesar : ambang langkah 0,01; bobot 2-model langkah 0,05; simpleks 0,05 x ambang 0,30-0,70 langkah 0,01
Per trial dan ruang: cfg (dipilih di seluruh POOL), ref, pt_pool, lo_pool_boot, pt_cf, lo_cf_boot (seed sama MINI-32).
Data dan seed identik MINI-32 (nyata: 11 dataset x 100; Q: T0000-T1399). K173 harus mereplikasi MINI-32 persis.
Pakai: python mini35_ruang_kandidat.py <nyata|Q> <awal> <akhir> <n_proses> <out>"""
import csv, itertools, sys, time, warnings
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


def rapat(st_t, st_w2, st_s, st_t3):
    h, dt, sv = "HGB", "DT", "SVC"; r = lambda a, b, s: [round(x, 4) for x in np.arange(a, b + 1e-9, s)]
    TH, TH3 = r(.05, .95, st_t), r(.30, .70, st_t3); W2 = r(st_w2, 1 - st_w2, st_w2); g = r(0, 1, st_s)
    SX = [w for w in itertools.product(g, repeat=3) if abs(sum(w) - 1) < 1e-9 and w != (1, 0, 0)]
    return ([({h: 1}, t) for t in TH if t != .5] + [({h: w, dt: round(1 - w, 4)}, t) for w in W2 for t in TH]
            + [({h: a, dt: b, sv: c}, t) for a, b, c in SX for t in TH3])


RUANG = {"K173": union(("HGB", "DT", "SVC")), "Ksedang": rapat(.02, .1, .1, .02), "Kbesar": rapat(.01, .05, .05, .01)}


def run_job(job):
    try:
        return _trial(job)
    except Exception as e:
        return [dict(dataset=job[0], rep=job[1], K="", cfg="ERROR " + str(e)[:150])]


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
    q = lambda c, s: float(np.quantile(BM.delta(np.random.default_rng(s).multinomial(c.sum(), c / c.sum(), size=20000), "wF1"), .05))
    out = []
    for K, U in RUANG.items():
        pk = lambda idx: F.pick(yp, P, idx, U, base, "wF1")
        cnt = np.zeros(8, int)
        for a, b in StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp):
            c = pk(a); cnt += G.counts8(yp[b], base[b], base[b] if c is None else lab(P, c, b))
        cfg = pk(np.arange(len(yp))); cpool = G.counts8(yp, base, base if cfg is None else lab(P, cfg))
        cr = G.counts8(yr, baser, baser if cfg is None else lab(Pr, cfg))
        out.append(dict(dataset=ds, rep=rep, K=K, nK=len(U), cfg=str(cfg), ref=float(BM.delta(cr, "wF1")),
                        pt_pool=float(BM.delta(cpool, "wF1")), lo_pool_boot=q(cpool, seed + 62) if cfg is not None else float("nan"),
                        pt_cf=float(BM.delta(cnt, "wF1")), lo_cf_boot=q(cnt, seed + 61)))
    return out


if __name__ == "__main__":
    a, b, Pn, outn = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    names = list(F.DS)
    jobs = ([(names[i], r) for i in range(a, b) for r in range(100)] if MODE == "nyata" else [(f"T{i:04d}", 0) for i in range(a, b)])
    t0 = time.time()
    with Pool(Pn) as pool:
        res = [x for rows in pool.imap(run_job, jobs, chunksize=1) for x in rows]
    keys = list(next(r for r in res if "ref" in r)); p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(res)
    print(f"{MODE} {a}-{b}, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

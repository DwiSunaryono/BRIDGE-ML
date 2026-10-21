"""MINI-29: pembanding ambang tetap pada kenaikan validasi (VALMARGIN_t: terima bila
delta_POOL konfigurasi terpilih > t). Menjawab pertanyaan mengapa tidak memakai ambang
sederhana (PRAREG_MINI29.md).
Skrip hanya menghitung delta_POOL konfigurasi yang dipilih di seluruh POOL; pemilihan identik dengan
run asal. Keputusan BRIDGE dan kebenaran referensi diambil dari CSV run asal (MINI-20, MINI-21,
MINI-25) dan dicek identik lewat kolom cfg.
Mode:
  ukuran : 27 dataset dev, TOTAL 1000/2430/6000, seed MINI-20 (mini17_trainsize, offset 1_020_000_000), wF1
  metrik : 11 dataset baru, seed MINI-21, metrik wF1/bacc/macroF1 (kebijakan union)
  Q      : populasi sintetis MINI-25 (1.400 tugas), wF1
Pakai: python mini29_valmargin.py <mode> <n_proses> <out> [bagian i/n]"""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
MODE = sys.argv[1] if __name__ == "__main__" else "ukuran"
_argv = sys.argv; sys.argv = [sys.argv[0], "1", "1", "x", "1020000000"]
import mini17_trainsize as S17
sys.argv = _argv
import exp41_gate_confirm as G
import bridge_metrics as BM
from mini12_optuna import union, lab
import mini21_final as F
U = union(("HGB", "DT", "SVC"))


def bagged(Xs, ys, seed):
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    fn = {m: [] for m in ("HGB", "DT", "SVC")}
    for _ in range(G.B_BAGS):
        idx = rng.choice(len(ys), size=sub, replace=False)
        if np.bincount(ys[idx], minlength=2).min() < 10:
            continue
        for m in fn:
            fn[m].append(G.fit_proba_m(m, {}, Xs.iloc[idx].reset_index(drop=True), ys[idx], list(Xs.columns)))
    return fn


def pick_gain(yp, P, base, metric):
    best, g0 = None, 0.0
    for cfg in U:
        g = float(BM.delta(G.counts8(yp, base, lab(P, cfg)), metric))
        if g > g0 + 1e-12:
            best, g0 = cfg, g
    return best, g0


def t_ukuran(job):
    ds, did, k, rep = job; TOT = S17.SIZES[k]; SS = int(round(0.259 * TOT))
    seed = S17.OFF + 1000 * list(S17.DS).index(ds) + 100 * k + rep
    Xd, yd, Xr, yr = S17.data(did)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=TOT, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    fn = bagged(Xs, ys, seed)
    if not fn["HGB"]:
        return [dict(dataset=ds, TOTAL=TOT, rep=rep, metric="wF1", cfg="SKIP", dpool=np.nan)]
    P = {m: np.column_stack([f(Xp) for f in fs]).mean(1) for m, fs in fn.items()}
    cfg, g = pick_gain(yp, P, (P["HGB"] >= .5).astype(int), "wF1")
    return [dict(dataset=ds, TOTAL=TOT, rep=rep, metric="wF1", cfg=str(cfg), dpool=g)]


def t_metrik(job):
    ds, rep = job; seed = F.OFF + 1000 * list(F.DS).index(ds) + rep
    Xd, yd, Xr, yr = F.data(ds)
    TOT, SS = (G.TOTAL_TRAIN, G.SEARCH_SIZE) if len(Xd) >= G.TOTAL_TRAIN + 50 else (1000, 259)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=TOT, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    fn = bagged(Xs, ys, seed)
    P = {m: np.column_stack([f(Xp) for f in fs]).mean(1) for m, fs in fn.items()}
    base = (P["HGB"] >= .5).astype(int); out = []
    for m in ("wF1", "bacc", "macroF1"):
        cfg, g = pick_gain(yp, P, base, m)
        out.append(dict(dataset=ds, TOTAL=TOT, rep=rep, metric=m, cfg=str(cfg), dpool=g))
    return out


def t_Q(job):
    return [dict(r, metric="wF1") for r in t_metrik(job) if r["metric"] == "wF1"]


FUN = {"ukuran": t_ukuran, "metrik": t_metrik, "Q": t_Q}


def run_job(job):  # Fungsi tingkat modul agar bisa di-pickle multiprocessing; closure tidak bisa
    try:
        return FUN[MODE](job)
    except Exception as e:
        return [dict(dataset=job[0], rep=job[-1], metric="", cfg="ERROR " + str(e)[:120], dpool=np.nan)]


if __name__ == "__main__":
    Pn, outn = int(sys.argv[2]), sys.argv[3]
    part = (int(sys.argv[4].split("/")[0]), int(sys.argv[4].split("/")[1])) if len(sys.argv) > 4 else (0, 1)
    if MODE == "ukuran":
        jobs = [(ds, did, k, r) for ds, did in S17.DS.items() for k in range(3) for r in range(100)]
        jobs.sort(key=lambda j: -S17.SIZES[j[2]])
    elif MODE == "metrik":
        jobs = [(ds, r) for ds in F.DS for r in range(100)]
    else:  # Mode Q
        import mini25_sintetis as Q
        F.OFF, F.DS, F.data = Q.F.OFF, Q.F.DS, Q.F.data
        jobs = [(f"T{i:04d}", 0) for i in range(1400)]
    jobs = jobs[part[0]::part[1]]; t0 = time.time()
    with Pool(Pn) as pool:
        res = [x for rows in pool.imap(run_job, jobs, chunksize=2) for x in rows]
    p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "TOTAL", "rep", "metric", "cfg", "dpool"], restval=""); w.writeheader(); w.writerows(res)
    print(f"{MODE} bagian {part}, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

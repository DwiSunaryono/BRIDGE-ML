"""MINI-39 (PRAREG_MINI39.md): hasil hyperparameter tuning sebagai UPDATE yang dinilai gate, LINTAS LEARNER.

Pertanyaan: apakah BRIDGE tetap menyaring update hasil tuning Optuna pada empat keluarga learner tabular
(HGB, DT, SVC, KNN), dibanding penerimaan naif?
Per trial dan per learner L (urutan k: HGB=0, DT=1, SVC=2, KNN=3):
  default L  : models.build_estimator(L, {})  (setelan proyek, sama seperti seluruh paper)
  tuned L    : Optuna TPE (seed (seed+k) mod 2^31-1), 25 search iterations, CV internal StratifiedKFold(3, shuffle,
               random_state = seed mod 2^32-1) HANYA di SEARCH, objektif rata-rata weighted-F1 @0,5,
               ruang = config.SEARCH_SPACES[L] via models.suggest_params (cara MINI-12 tune(), digeneralisasi ke L)
  update     : default L -> tuned L (keduanya @0,5), bag identik (32 draw, 80% SEARCH, buang bila kelas < 10)
  naif       : terima bila Δ weighted-F1 (tuned - default) pada SELURUH POOL > 0
  BRIDGE     : kandidat sudah dipilih di SEARCH -> tidak ada pemilihan di POOL -> bukti = counts seluruh POOL
               (persis desain tune_HGB MINI-12); bootstrap multinomial 20.000, terima iff kuantil-5% > 0,005
  kebenaran  : Δ weighted-F1 di referensi terkunci; harmful = Δ <= 0; kelas Dirichlet(+0,5) 95% (spt MINI-12)
Arm R : 11 dataset baru MINI-21 (data & referensi terkunci MINI-21), 100 trial, seed 1_039_000_000 + 1000*idx + rep.
Arm Q : populasi Q MINI-25 (generator beku), tugas BARU T2900-T4299, 1 trial/tugas, seed 1_039_100_000 + i.
Pakai: python mini39_tuning_lintas.py <R|Q> <awal> <akhir> <n_proses> <out.csv>
  R: awal/akhir = rentang rep (0-100) untuk semua 11 dataset; Q: awal/akhir = indeks tugas (2900-4300)."""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np, optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
ARM = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("R", "Q") else "R"
_argv = sys.argv; sys.argv = sys.argv[:1]
if ARM == "Q":
    import mini25_sintetis as Qm  # mengganti F.data dgn generator Q beku (seed tugas 7_000_000+i, split 8_000_000+i)
    F = Qm.F
    F.DS = {f"T{i:04d}": ("sintetis", i, ()) for i in range(4300)}
else:
    import mini21_final as F
sys.argv = _argv
import models
import exp41_gate_confirm as G
from mini01_threshold_crossfit import gate
from mini12_optuna import truth

LEARNERS = ("HGB", "DT", "SVC", "KNN")
OFF_R, OFF_Q = 1_039_000_000, 1_039_100_000
N_ITER, EPS = 25, 0.005


def tune(L, k, Xs, ys, seed):
    cols = list(Xs.columns)
    cv = list(StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(Xs, ys))

    def obj(tr):
        prm = models.suggest_params(tr, L, len(ys) * 2 // 3)
        sc = []
        for a, b in cv:
            f = G.fit_proba_m(L, prm, Xs.iloc[a].reset_index(drop=True), ys[a], cols)
            pr = f(Xs.iloc[b].reset_index(drop=True))
            sc.append(f1_score(ys[b], (pr >= .5).astype(int), average="weighted"))
        return float(np.mean(sc))
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=(seed + k) % (2**31 - 1)))
    st.optimize(obj, n_trials=N_ITER)
    return dict(st.best_params), float(st.best_value)


def seed_of(ds, rep):
    return (OFF_Q + int(ds[1:])) if ARM == "Q" else (OFF_R + 1000 * list(F.DS).index(ds) + rep)


def trial(job):
    ds, rep = job; seed = seed_of(ds, rep)
    try:
        Xd, yd, Xr, yr = F.data(ds)
        TOT, SS = (G.TOTAL_TRAIN, G.SEARCH_SIZE) if len(Xd) >= G.TOTAL_TRAIN + 50 else (1000, 259)
        Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=TOT, stratify=yd, random_state=seed)
        Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
        Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
        yr = np.asarray(yr)
        rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
        bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS))
                if np.bincount(ys[i], minlength=2).min() >= 10]
    except Exception as e:
        return [dict(arm=ARM, dataset=ds, rep=rep, seed=seed, learner=L, error="SETUP " + str(e)[:150]) for L in LEARNERS]
    cols = list(Xs.columns); out = []
    for k, L in enumerate(LEARNERS):
        t0 = time.time()
        row = dict(arm=ARM, dataset=ds, rep=rep, seed=seed, learner=L, TOTAL=TOT, search_n=len(ys), pool_n=len(yp),
                   ref_n=len(yr), n_bags=len(bags))
        try:
            if not bags:
                raise RuntimeError("SKIP no eligible bag")
            best, cvbest = tune(L, k, Xs, ys, seed)
            P, Pr = {}, {}
            for v, prm in (("def", {}), ("tun", best)):
                fs = [G.fit_proba_m(L, prm, Xs.iloc[i].reset_index(drop=True), ys[i], cols) for i in bags]
                P[v] = np.column_stack([f(Xp) for f in fs]).mean(1); Pr[v] = np.column_stack([f(Xr) for f in fs]).mean(1)
            bD, bT = (P["def"] >= .5).astype(int), (P["tun"] >= .5).astype(int)
            bDr, bTr = (Pr["def"] >= .5).astype(int), (Pr["tun"] >= .5).astype(int)
            cnt = G.counts8(yp, bD, bT); Dpool = float(G.delta_from_counts(cnt))
            lo99, lo95 = gate(cnt, len(yp), seed + 21 + k)
            ref, kelas = truth(yr, bDr, bTr, seed + k)
            row.update(best_params=str(best), cv_best=cvbest,
                       wf1_pool_def=float(f1_score(yp, bD, average="weighted")), wf1_pool_tun=float(f1_score(yp, bT, average="weighted")),
                       D_pool=Dpool, naive=int(Dpool > 0), lo95=lo95, bridge=int(lo95 > EPS), ref=ref, kelas=kelas, harmful=int(ref <= 0),
                       wf1_ref_def=float(f1_score(yr, bDr, average="weighted")), wf1_ref_tun=float(f1_score(yr, bTr, average="weighted")),
                       **{f"c{j}": int(cnt[j]) for j in range(8)}, error="")
        except Exception as e:
            row.update(error=str(e)[:200])
        row["secs"] = round(time.time() - t0, 1); out.append(row)
    return out


COLS = ["arm", "dataset", "rep", "seed", "learner", "TOTAL", "search_n", "pool_n", "ref_n", "n_bags", "best_params", "cv_best",
        "wf1_pool_def", "wf1_pool_tun", "D_pool", "naive", "lo95", "bridge", "ref", "kelas", "harmful", "wf1_ref_def",
        "wf1_ref_tun"] + [f"c{j}" for j in range(8)] + ["error", "secs"]

if __name__ == "__main__":
    a, b, Pn, outn = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    jobs = ([(f"T{i:04d}", 0) for i in range(a, b)] if ARM == "Q"
            else [(ds, r) for ds in F.DS for r in range(a, b)])
    t0 = time.time()
    with Pool(Pn) as pool:
        res = [x for rows in pool.imap(trial, jobs, chunksize=1) for x in rows]
    p = Path(outn) if "/" in outn else HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, restval=""); w.writeheader(); w.writerows(res)
    print(f"MINI-39 {ARM} {a}-{b}: {len(jobs)} trial x {len(LEARNERS)} learner, {time.time()-t0:.0f} dtk -> {p}")

"""MINI-40 (PRAREG_MINI40.md): hasil hyperparameter tuning sebagai UPDATE yang dinilai gate, MULTI-KELAS, lintas learner.

Perluasan MINI-39 ke K >= 3 kelas pada 5 dataset multi-kelas MINI-22 (Fashion-MNIST, DryBean, USPS, mnist_784,
Kuzushiji-MNIST; loader dan referensi terkunci MINI-22 apa adanya). Fungsi multi-kelas MINI-18 dipakai apa adanya:
counts K^3 (cells), Δ weighted-F1 multi-kelas (wf1_delta), bootstrap multinomial atas sel tak-nol (boot_delta).
Per trial dan per learner L (urutan k: HGB=0, DT=1, SVC=2, KNN=3):
  default L : models.build_estimator(L, {}); prediksi = argmax probabilitas bag-rata
  tuned L   : Optuna TPE (seed (seed+k) mod 2^31-1), 25 search iterations, CV internal StratifiedKFold(3, shuffle,
              random_state = seed mod 2^32-1) HANYA di SEARCH, objektif rata-rata weighted-F1 multi-kelas dari argmax,
              ruang = config.SEARCH_SPACES[L] via models.suggest_params
  bag       : 32 draw, 80% SEARCH, aturan multi-kelas MINI-18/22 (buang bila < 2 kelas atau kelas yang muncul < 2 contoh)
  naif      : terima bila Δ weighted-F1 (tuned - default) pada SELURUH POOL > 0
  BRIDGE    : kandidat dipilih di SEARCH -> bukti = counts K^3 seluruh POOL; bootstrap 20.000 (seed+71+k);
              terima iff kuantil-5% > 0,005
  kebenaran : Δ weighted-F1 di referensi terkunci; harmful = Δ <= 0; kelas dari Dirichlet(cr) 95% (cara MINI-18)
60 trial per dataset (seperti MINI-22), seed 1_040_000_000 + 1000*idx + rep.
Pakai: python mini40_tuning_multikelas.py <rep_awal> <rep_akhir> <n_proses> <out.csv> <nama,dipisah,koma | semua>"""
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
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini22_mc_final  # noqa: F401  (memasang loader dan daftar dataset MINI-22 ke modul MINI-18)
import mini18_multiclass as M
sys.argv = _argv
import models, data as D
import exp41_gate_confirm as G

LEARNERS = ("HGB", "DT", "SVC", "KNN")
OFF, N_ITER, EPS = 1_040_000_000, 25, 0.005
DS = dict(M.DS)


def fit_proba_p(model, params, Xf, yf, K):
    cols = list(Xf.columns); num_c, cat_c = D.column_kinds(Xf)
    pre = D.make_preprocessor(num_c, cat_c); Z = pre.fit_transform(Xf, yf)
    est = models.build_estimator(model, params, len(yf), final=True); est.fit(Z, yf)
    cl = np.asarray(est.classes_, int)

    def f(Xn):
        out = np.zeros((len(Xn), K)); out[:, cl] = est.predict_proba(pre.transform(Xn[cols])); return out
    return f


def tune(L, k, Xs, ys, seed, K):
    cv = list(StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(Xs, ys))

    def obj(tr):
        prm = models.suggest_params(tr, L, len(ys) * 2 // 3)
        sc = []
        for a, b in cv:
            f = fit_proba_p(L, prm, Xs.iloc[a].reset_index(drop=True), ys[a], K)
            sc.append(f1_score(ys[b], f(Xs.iloc[b].reset_index(drop=True)).argmax(1), average="weighted"))
        return float(np.mean(sc))
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=(seed + k) % (2**31 - 1)))
    st.optimize(obj, n_trials=N_ITER)
    return dict(st.best_params), float(st.best_value)


def trial(job):
    ds, rep = job; seed = OFF + 1000 * list(DS).index(ds) + rep
    try:
        Xd, yd, Xr, yr, K = M.data(DS[ds])
        Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=G.TOTAL_TRAIN, stratify=yd, random_state=seed)
        Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=G.SEARCH_SIZE, stratify=ytr, random_state=seed)
        Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
        rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
        bags = []
        for _ in range(G.B_BAGS):
            idx = rng.choice(len(ys), size=sub, replace=False)
            cc = np.bincount(ys[idx], minlength=K); cc = cc[cc > 0]
            if len(cc) >= 2 and cc.min() >= 2:
                bags.append(idx)
    except Exception as e:
        return [dict(dataset=ds, rep=rep, seed=seed, learner=L, error="SETUP " + str(e)[:150]) for L in LEARNERS]
    out = []
    for k, L in enumerate(LEARNERS):
        t0 = time.time(); row = dict(dataset=ds, rep=rep, seed=seed, learner=L, K=K, search_n=len(ys), pool_n=len(yp),
                                    ref_n=len(yr), n_bags=len(bags))
        try:
            if not bags:
                raise RuntimeError("SKIP no eligible bag")
            best, cvbest = tune(L, k, Xs, ys, seed, K)
            P, Pr = {}, {}
            for v, prm in (("def", {}), ("tun", best)):
                fs = [fit_proba_p(L, prm, Xs.iloc[i].reset_index(drop=True), ys[i], K) for i in bags]
                P[v] = sum(f(Xp) for f in fs) / len(fs); Pr[v] = sum(f(Xr) for f in fs) / len(fs)
            bD, bT, bDr, bTr = P["def"].argmax(1), P["tun"].argmax(1), Pr["def"].argmax(1), Pr["tun"].argmax(1)
            cnt = M.cells(yp, bD, bT, K); Dpool = float(M.wf1_delta(cnt, K))
            lo95 = float(np.quantile(M.boot_delta(cnt, K, np.random.default_rng(seed + 71 + k)), .05))
            cr = M.cells(yr, bDr, bTr, K); ref = float(M.wf1_delta(cr, K))
            lo, hi = np.quantile(M.boot_delta(cr, K, np.random.default_rng(seed + 987163 + k), dirichlet=True), [.025, .975])
            row.update(best_params=str(best), cv_best=cvbest,
                       wf1_pool_def=float(f1_score(yp, bD, average="weighted")), wf1_pool_tun=float(f1_score(yp, bT, average="weighted")),
                       D_pool=Dpool, naive=int(Dpool > 0), lo95=lo95, bridge=int(lo95 > EPS), ref=ref,
                       kelas="membaik" if lo > 0 else "memburuk" if hi < 0 else "tidak-pasti", harmful=int(ref <= 0),
                       wf1_ref_def=float(f1_score(yr, bDr, average="weighted")), wf1_ref_tun=float(f1_score(yr, bTr, average="weighted")),
                       error="")
        except Exception as e:
            row.update(error=str(e)[:200])
        row["secs"] = round(time.time() - t0, 1); out.append(row)
    return out


COLS = ["dataset", "rep", "seed", "learner", "K", "search_n", "pool_n", "ref_n", "n_bags", "best_params", "cv_best", "wf1_pool_def",
        "wf1_pool_tun", "D_pool", "naive", "lo95", "bridge", "ref", "kelas", "harmful", "wf1_ref_def", "wf1_ref_tun", "error", "secs"]

if __name__ == "__main__":
    a, b, Pn, outn = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    names = list(DS) if sys.argv[5] == "semua" else sys.argv[5].split(","); assert all(n in DS for n in names)
    jobs = [(ds, r) for ds in names for r in range(a, b)]; t0 = time.time()
    with Pool(Pn) as pool:
        res = [x for rows in pool.imap(trial, jobs, chunksize=1) for x in rows]
    p = Path(outn) if "/" in outn else HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, restval=""); w.writeheader(); w.writerows(res)
    print(f"MINI-40 {names} rep {a}-{b}: {len(jobs)} trial x {len(LEARNERS)} learner, {time.time()-t0:.0f} dtk -> {p}")

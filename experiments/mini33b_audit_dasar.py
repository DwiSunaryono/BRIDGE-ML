"""MINI-33b: audit model dasar, deskriptif dan tanpa keputusan gate. Membandingkan HGB, XGB bawaan (MINI-31), dan XGB_eq (MINI-33).

Tugas dan pembagian data identik dengan MINI-33 (seed sama), sehingga log-loss HGB/XGB_eq harus sama dengan
logloss_base MINI-33 sebagai cek konsistensi. Per model, dirata-ratakan 32 bag dan dinilai pada data referensi
terkunci: AUC, log-loss, persentase probabilitas ekstrem (<0,01 atau >0,99), wF1 pada ambang 0,5, jumlah ronde
boosting terpakai (rata-rata bag), dan waktu latih per bag (detik, 1 core).
Pakai: python mini33b_audit_dasar.py <awal> <akhir> <n_proses> <out>
"""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini33_xgb_setara as M33
sys.argv = _argv
import exp41_gate_confirm as G, data as Dd, models
from xgboost import XGBClassifier
F = M33.F


def fit_one(m, Xf, yf):
    cols = list(Xf.columns); num_c, cat_c = Dd.column_kinds(Xf)
    pre = Dd.make_preprocessor(num_c, cat_c); Z = pre.fit_transform(Xf, yf); t = time.perf_counter()
    if m == "XGBeq":
        Ztr, Zva, ytr, yva = train_test_split(Z, yf, test_size=.1, stratify=yf, random_state=42)
        est = XGBClassifier(n_estimators=300, learning_rate=.1, grow_policy="lossguide", max_leaves=31, max_depth=0,
                            min_child_weight=5, reg_lambda=0.0, early_stopping_rounds=10, eval_metric="logloss",
                            tree_method="hist", random_state=42, n_jobs=1)
        est.fit(Ztr, ytr, eval_set=[(Zva, yva)], verbose=False); rounds = est.best_iteration + 1
    else:
        est = models.build_estimator(m, {}, len(yf), final=True); est.fit(Z, yf)
        rounds = getattr(est, "n_iter_", None) or est.get_booster().num_boosted_rounds()
    dt = time.perf_counter() - t; pos = list(est.classes_).index(1)
    return (lambda Xn: est.predict_proba(pre.transform(Xn[cols]))[:, pos]), rounds, dt


def run_job(job):
    try:
        return _trial(job)
    except Exception as e:
        return [dict(dataset=job[0], model="ERROR " + str(e)[:150])]


def _trial(job):
    ds, rep = job; seed = F.OFF + 1000 * list(F.DS).index(ds) + rep
    Xd, yd, Xr, yr = F.data(ds)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=G.TOTAL_TRAIN, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=G.SEARCH_SIZE, stratify=ytr, random_state=seed)
    Xs, ys = Xs.reset_index(drop=True), np.asarray(ys); yr = np.asarray(yr)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS)) if np.bincount(ys[i], minlength=2).min() >= 10]
    out = []
    for m in ("HGB", "XGB", "XGBeq"):
        r = [fit_one(m, Xs.iloc[i].reset_index(drop=True), ys[i]) for i in bags]
        p = np.column_stack([f(Xr) for f, _, _ in r]).mean(1); pc = np.clip(p, 1e-6, 1 - 1e-6)
        out.append(dict(dataset=ds, model=m, auc=roc_auc_score(yr, p), logloss=float(-np.mean(yr * np.log(pc) + (1 - yr) * np.log(1 - pc))),
                        extreme_pct=100 * float(np.mean((p < .01) | (p > .99))), wf1=f1_score(yr, (p >= .5).astype(int), average="weighted"),
                        rounds=float(np.mean([k for _, k, _ in r])), fit_s_per_bag=float(np.mean([d for _, _, d in r]))))
    return out


if __name__ == "__main__":
    a, b, Pn, outn = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    jobs = [(f"T{i:04d}", 0) for i in range(a, b)]; t0 = time.time()
    with Pool(Pn) as pool:
        res = [x for rows in pool.imap(run_job, jobs, chunksize=2) for x in rows]
    keys = list(next(r for r in res if "auc" in r)); p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(res)
    print(f"tugas {a}-{b}, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

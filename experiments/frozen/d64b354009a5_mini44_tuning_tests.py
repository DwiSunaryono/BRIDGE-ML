"""MINI-44 (PRAREG_MINI44.md): is the frozen design beaten by tuning? Three pre-registered questions on the same decisions,
paired with the stored runs (same seeds, same outer split, same SEARCH/POOL split, same bags):
  part 1, frozen gate (identity check): the paper's procedure; its 5% lower bound and reference gain must equal the
          stored MINI-24 (A) / MINI-43 (D) values. The same bootstrap draws also give the 2.5% and 10% quantiles, so every
          (q, eps) in Q_GRID x EPS_GRID is decided on the full POOL (sensitivity of q and eps on real data).
  H2,     gate tuned per task: POOL is split into POOL_in (2/3) and CHECK (1/3); the gate (cross-fit, bootstrap) runs on
          POOL_in for every (q, eps); the candidate chosen on POOL_in is scored on CHECK; the (q, eps) with the largest
          CHECK value (gain if accepted, 0 if not) is chosen (ties: default, then the most conservative), and applied on
          the full POOL. Deployed configuration = the frozen gate's (only the acceptance threshold changes).
  H1,     tuned baseline: HGB tuned by Optuna TPE on SEARCH only (25 iterations, 3-fold CV inside SEARCH, weighted-F1,
          the MINI-39 HGB space and budget); the tuned HGB replaces the default HGB everywhere (bags, current model,
          candidates); DT, SVC, candidate space and gate unchanged.
Every finished trial is appended to the output CSV at once; a rerun skips finished trials.
Usage: python mini44_tuning_tests.py <out.csv> <n_proc> [i/n] [--smoke]"""
import csv, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pkg" / "src")); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd, optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
optuna.logging.set_verbosity(optuna.logging.WARNING)
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini21_final as F
import mini43_supporting_D as D43
sys.argv = _argv
from bridge_gate import protocol as P_, core

Q_GRID, EPS_GRID, DEF = (0.025, 0.05, 0.10), (0.0025, 0.005, 0.0075, 0.01), (0.05, 0.005)
N_ITER = 25
A_DS = list(F.DS)
ORDER = [("A", k) for k in A_DS] + [(g, k) for g, k in D43.ORDER]
TRIALS = {"A": 100, "bin": 100, "mc": 60}
COLS = (["setting", "dataset", "rep", "seed", "K", "total"]
        + ["f_lb05", "f_cfg", "f_gain_pool", "f_gain_ref", "f_base_ref", "f_dep_ref", "f_identical"]
        + [f"f_q{q}" for q in Q_GRID]
        + ["t_q", "t_eps", "t_inner_cfg", "t_check_gain", "t_accept"]
        + ["h_params", "h_cv", "h_lb05", "h_cfg", "h_gain_pool", "h_gain_ref", "h_base_ref", "h_dep_ref", "h_accept", "h_naive"]
        + ["status", "seconds"])


def stored():
    a = pd.concat([pd.read_csv(HERE / "out" / f"mini24_m{i}.csv", float_precision="round_trip") for i in (1, 2, 3)])
    d = pd.concat([pd.read_csv(HERE / "out" / f"mini43_m{i}.csv", float_precision="round_trip") for i in (1, 2, 3)])
    s = {(r.dataset, int(r.rep)): (float(r.lo95), float(r.ref)) for r in a.itertuples()}
    s.update({(r.dataset, int(r.rep)): (float(r.lower_bound), float(r.gain_ref)) for r in d.itertuples()})
    return s


_D = {}


def data(kind, name):
    if (kind, name) not in _D:
        _D.clear()
        if kind == "A":
            Xd, yd, Xr, yr = F.data(name); _D[(kind, name)] = (Xd, np.asarray(yd, int), Xr, np.asarray(yr, int), 2)
        else:
            _D[(kind, name)] = D43.data(kind, name)
    return _D[(kind, name)]


def seed_of(kind, name, rep):
    if kind == "A":
        return F.OFF + 1000 * A_DS.index(name) + rep                       # = MINI-21 / MINI-24
    return D43.OFF + 1000 * [k for _, k in D43.ORDER].index(name) + rep    # = MINI-43


def boot(cnt, K, metric, seed):
    """The bootstrap draws of core.lower_bound (same stream), returned in full."""
    c = np.asarray(cnt); rng = np.random.default_rng(seed); n = int(c.sum())
    if K == 2:
        return core.delta(rng.multinomial(n, c / n, size=core.REPLICATES), 2, metric)
    nz = np.flatnonzero(c); out = []
    for s in range(0, core.REPLICATES, 2500):
        m = min(2500, core.REPLICATES - s)
        W = np.zeros((m, K ** 3)); W[:, nz] = rng.multinomial(n, c[nz] / n, size=m); out.append(core.delta(W, K, metric))
    return np.concatenate(out)


def wf1(y, p):
    return float(f1_score(y, p, average="weighted"))


def hgb(params):
    def make(name, n):
        if name == "HGB" and params is not None:
            return HistGradientBoostingClassifier(**params, early_stopping=True, n_iter_no_change=10, random_state=P_.MODEL_SEED)
        return _ORIG(name, n)
    return make


_ORIG = P_._learner


def tune(Xs, ys, K, seed):
    cv = list(StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(Xs, ys))
    S = {"learning_rate": (0.01, 0.3), "max_iter": (50, 300), "max_leaf_nodes": (3, 63), "min_samples_leaf": (1, 50),
         "l2_regularization": (1e-8, 10.0)}

    def obj(tr):
        prm = {"learning_rate": tr.suggest_float("learning_rate", *S["learning_rate"], log=True),
               "max_iter": tr.suggest_int("max_iter", *S["max_iter"]),
               "max_leaf_nodes": tr.suggest_int("max_leaf_nodes", *S["max_leaf_nodes"]),
               "min_samples_leaf": tr.suggest_int("min_samples_leaf", *S["min_samples_leaf"]),
               "l2_regularization": tr.suggest_float("l2_regularization", *S["l2_regularization"], log=True)}
        sc = []
        for a, b in cv:
            Xa, Xb = Xs.iloc[a].reset_index(drop=True), Xs.iloc[b].reset_index(drop=True)
            pre = P_._preprocessor(Xa); est = HistGradientBoostingClassifier(**prm, early_stopping=True, n_iter_no_change=10,
                                                                              random_state=P_.MODEL_SEED)
            est.fit(pre.fit_transform(Xa, ys[a]), ys[a]); pr = est.predict_proba(pre.transform(Xb))
            pred = est.classes_[pr.argmax(1)]
            sc.append(wf1(ys[b], pred))
        return float(np.mean(sc))
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=(seed + 101) % (2**31 - 1)))
    st.optimize(obj, n_trials=N_ITER)
    return dict(st.best_params), float(st.best_value)


def fit_all(Xs, ys, K, seed, params):
    """Bags and learners exactly as protocol.run; params = tuned HGB settings or None (paper default)."""
    P_._learner = hgb(params)
    try:
        rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(P_.BAG_FRACTION * len(ys)); bags = []
        for _ in range(P_.BAGS):
            i = rng.choice(len(ys), size=sub, replace=False); cc = np.bincount(ys[i], minlength=K)
            if K == 2:
                ok = cc.min() >= 10
            else:
                cc = cc[cc > 0]; ok = len(cc) >= 2 and cc.min() >= 2
            if ok:
                bags.append(i)
        return {m: [P_._Fitted(m, Xs.iloc[i].reset_index(drop=True), ys[i], K) for i in bags] for m in ("HGB", "DT", "SVC")}
    finally:
        P_._learner = _ORIG


def avg(fitted, Xe, K):
    if K == 2:
        return {m: np.column_stack([f.proba(Xe) for f in fs]).mean(1) for m, fs in fitted.items()}
    return {m: sum(f.proba(Xe) for f in fs) / len(fs) for m, fs in fitted.items()}


def gate_parts(yp, P, K, seed, metric, idx=None, fold_seed=None, boot_seed=None):
    """Cross-fit counts on POOL (or the POOL rows idx), bootstrap draws, configuration chosen on those rows."""
    idx = np.arange(len(yp)) if idx is None else idx
    base = (P["HGB"] >= .5).astype(int) if K == 2 else P["HGB"].argmax(1)
    C = P_.candidates(K); cnt = np.zeros(K ** 3, int); y = yp[idx]
    Pi = {m: v[idx] for m, v in P.items()}; bi = base[idx]
    for a, b in StratifiedKFold(3, shuffle=True, random_state=(seed if fold_seed is None else fold_seed) % (2**32 - 1)).split(y, y):
        cfg = P_._pick(y, bi, Pi, a, C, K, metric)
        cnt += core.counts(y[b], bi[b], bi[b] if cfg is None else P_.labels(Pi, cfg, K, b), K)
    bs = seed + (P_.BOOT_SEED_OFFSET_BIN + P_.METRICS_BINARY.index(metric) if K == 2 else P_.BOOT_SEED_OFFSET_MC)
    draws = boot(cnt, K, metric, bs if boot_seed is None else boot_seed)
    cfg = P_._pick(y, bi, Pi, np.arange(len(y)), C, K, metric)
    g = 0.0 if cfg is None else float(core.delta(core.counts(y, bi, P_.labels(Pi, cfg, K), K), K, metric))
    return draws, cfg, g, base


def ref_eval(fitted, cfg, Xr, yr, K, metric):
    Pr = avg(fitted, Xr.reset_index(drop=True), K)
    br = (Pr["HGB"] >= .5).astype(int) if K == 2 else Pr["HGB"].argmax(1)
    dep = br if cfg is None else P_.labels(Pr, cfg, K)
    return float(core.delta(core.counts(yr, br, dep, K), K, metric)), wf1(yr, br), wf1(yr, dep)


def run_one(job):
    kind, name, rep, want = job; seed = seed_of(kind, name, rep); t0 = time.time(); metric = "wF1"
    setting = "A" if kind == "A" else ("D" if kind == "bin" else "D-mc")
    try:
        Xd, yd, Xr, yr, K = data(kind, name)
        TOT, SS = (P_.TOTAL_TRAIN, P_.SEARCH_SIZE) if len(yd) >= P_.TOTAL_TRAIN + 50 else (P_.SMALL_TOTAL, P_.SMALL_SEARCH)
        X = Xd.reset_index(drop=True)
        Xtr, _, ytr, _ = train_test_split(X, pd.Series(yd), train_size=TOT, stratify=yd, random_state=seed)
        Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
        Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
        row = dict(setting=setting, dataset=name, rep=rep, seed=seed, K=K, total=TOT)
        # ---- part 1: frozen gate, identity check, q x eps on the full POOL ----
        fz = fit_all(Xs, ys, K, seed, None); P = avg(fz, Xp, K)
        draws, cfg, g, base = gate_parts(yp, P, K, seed, metric)
        qs = {q: float(np.quantile(draws, q)) for q in Q_GRID}
        gr, bref, dref = ref_eval(fz, cfg, Xr, yr, K, metric)
        row.update(f_lb05=repr(qs[.05]), f_cfg=str(cfg), f_gain_pool=repr(g), f_gain_ref=repr(gr), f_base_ref=bref, f_dep_ref=dref,
                   f_identical=int(repr(qs[.05]) == repr(want[0]) and repr(gr) == repr(want[1])),
                   **{f"f_q{q}": repr(v) for q, v in qs.items()})
        # ---- H2: gate tuned per task on POOL_in / CHECK, applied on the full POOL ----
        pin, chk = train_test_split(np.arange(len(yp)), train_size=2 / 3, stratify=yp, random_state=(seed + 13) % (2**32 - 1))
        d_in, cfg_in, _, _ = gate_parts(yp, P, K, seed, metric, idx=np.sort(pin), fold_seed=seed + 17, boot_seed=seed + 23)
        q_in = {q: float(np.quantile(d_in, q)) for q in Q_GRID}
        if cfg_in is None:
            gchk = 0.0
        else:
            ci = np.sort(chk); Pc = {m: v[ci] for m, v in P.items()}
            gchk = float(core.delta(core.counts(yp[ci], base[ci], P_.labels(Pc, cfg_in, K), K), K, metric))
        val = {(q, e): (gchk if (cfg_in is not None and q_in[q] > e) else 0.0) for q in Q_GRID for e in EPS_GRID}
        best = max(val.values()); tied = [k for k, v in val.items() if v == best]
        tq, te = DEF if DEF in tied else sorted(tied, key=lambda k: (-k[1], k[0]))[0]
        row.update(t_q=tq, t_eps=te, t_inner_cfg=str(cfg_in), t_check_gain=repr(gchk), t_accept=int(cfg is not None and qs[tq] > te))
        del fz, P
        # ---- H1: tuned HGB baseline ----
        prm, cvb = tune(Xs, ys, K, seed)
        tz = fit_all(Xs, ys, K, seed, prm); Pt = avg(tz, Xp, K)
        d_t, cfg_t, g_t, _ = gate_parts(yp, Pt, K, seed, metric)
        lb_t = float(np.quantile(d_t, .05)); gr_t, bref_t, dref_t = ref_eval(tz, cfg_t, Xr, yr, K, metric)
        row.update(h_params=str(prm), h_cv=cvb, h_lb05=repr(lb_t), h_cfg=str(cfg_t), h_gain_pool=repr(g_t), h_gain_ref=repr(gr_t),
                   h_base_ref=bref_t, h_dep_ref=dref_t, h_accept=int(cfg_t is not None and lb_t > DEF[1]), h_naive=int(g_t > 0),
                   status="ok", seconds=round(time.time() - t0, 1))
        return row
    except Exception as e:
        return dict(setting=setting, dataset=name, rep=rep, seed=seed, status="ERROR " + str(e)[:150], seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    out, n_proc = Path(sys.argv[1]), int(sys.argv[2]); S = stored()
    jobs = [(kind, name, rep, S[(name, rep)]) for kind, name in ORDER for rep in range(TRIALS[kind])]
    if "--smoke" in sys.argv:
        jobs = [j for j in jobs if (j[1], j[2]) in (("LICD", 0), ("compas", 0), ("DryBean", 0))]
    pos = [a for a in sys.argv[3:] if "/" in a]
    if pos:
        i, n = map(int, pos[0].split("/")); jobs = jobs[i - 1::n]
    done = set()
    if out.exists():
        done = {(r["dataset"], int(r["rep"])) for r in csv.DictReader(open(out)) if r["status"] == "ok"}
    todo = [j for j in jobs if (j[1], j[2]) not in done]
    print(f"{len(jobs)} trials in this part, {len(jobs) - len(todo)} already done", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as f, Pool(n_proc) as pool:
        w = csv.DictWriter(f, fieldnames=COLS, restval="")
        if new:
            w.writeheader()
        for k, r in enumerate(pool.imap_unordered(run_one, todo, chunksize=1)):
            w.writerow(r); f.flush()
            if k % 50 == 0 or r["status"] != "ok":
                print(f"{k + 1}/{len(todo)} {r['dataset']} rep {r['rep']}: {r['status']} {r['seconds']} s", flush=True)
    print("done", flush=True)

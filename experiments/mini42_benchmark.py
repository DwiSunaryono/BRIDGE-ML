"""MINI-42 (PRAREG_MINI42.md, draft): BRIDGE on the OpenML benchmark suites
(CC18, AutoML Benchmark, TabArena), using every classification dataset in its
native form (native classes, native rows) so results can be grouped by dataset
character.

Per dataset (OpenML data_id) and trial rep:
  outer split : locked reference = ref_size(N) rows, capped at REF_CAP (cost rule), seed LOCKED_REF_SEED, stratified
  training    : 2,430 rows (SEARCH 630) when the development part has at least 2,480 rows;
                otherwise 1,000 (SEARCH 259) when it has at least 1,050; otherwise all
                development rows (SEARCH = round(0.259 * rows)); the size tier is recorded
  update (a)  : the paper's candidate space with default learners
                K = 2 uses the MINI-32 trial (union HGB/DT/SVC x threshold, bags with at least 10 per class)
                K >= 3 uses the MINI-22 trial (ensemble weights and class scales, bags with at least 2 per occurring class)
  update (b)  : hyperparameter tuning of learner L as the update (MINI-39 for K = 2, MINI-40 for K >= 3), L in TUNE_LEARNERS
  rules       : naive = full-POOL gain > 0; BRIDGE = 5% bootstrap quantile > 0.005 (cross-fitted counts for (a),
                whole-POOL counts for (b)); truth = gain on the locked reference; harmful = gain <= 0;
                class from the Dirichlet 95% interval
Seeds: 1_042_000_000 + 1000 * (position of the dataset in the frozen list) + rep.
Each finished trial is appended to the output CSV immediately.
Usage: python mini42_benchmark.py <rep_start> <rep_end> <n_proc> <out.csv> <data_ids,comma | list_file> [tune learners, e.g. HGB]"""
import csv, os, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini21_final as F
import mini18_multiclass as M
import mini40_tuning_multikelas as T40
import mini39_tuning_lintas as T39
sys.argv = _argv
import models, exp41_gate_confirm as G, bridge_metrics as BM
import bridge_gate as bridge  # every gate decision below uses this package
from mini12_optuna import union, lab, truth
from mini01_threshold_crossfit import gate

OFF, REF_CAP, EPS, N_ITER = 1_042_000_000, 20_000, 0.005, 25
U = union(("HGB", "DT", "SVC"))
_D = {}
LIST = HERE / "mini42_datasets.csv"   # frozen list: data_id, name, position


def data(did):
    if did not in _D:
        _D.clear(); X, y = M.load(did)
        nref = min(G.ref_size(len(X)), REF_CAP)
        Xd, Xr, yd, yr = train_test_split(X, y, test_size=nref, stratify=y, random_state=G.LOCKED_REF_SEED)
        _D[did] = (Xd.reset_index(drop=True), np.asarray(yd), Xr.reset_index(drop=True), np.asarray(yr), int(y.max()) + 1, len(X))
    return _D[did]


def sizes(ndev):
    if ndev >= G.TOTAL_TRAIN + 50:
        return G.TOTAL_TRAIN, G.SEARCH_SIZE, "full"
    if ndev >= 1050:
        return 1000, 259, "moderate"
    return ndev, int(round(0.259 * ndev)), "small"


def split(did, seed):
    Xd, yd, Xr, yr, K, N = data(did)
    TOT, SS, tier = sizes(len(yd))
    if TOT < len(yd):
        Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=TOT, stratify=yd, random_state=seed)
    else:
        Xtr, ytr = Xd, yd
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
    return (Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp), Xr, yr, K,
            dict(N=N, K=K, ref_n=len(yr), TOTAL=TOT, search_n=SS, pool_n=len(yp), tier=tier,
                 min_search=int(np.bincount(np.asarray(ys), minlength=K).min()), min_pool=int(np.bincount(np.asarray(yp), minlength=K).min())))


def update_a_binary(Xs, ys, Xp, yp, Xr, yr, seed):
    """The MINI-32 trial in real-data mode, unchanged."""
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS)) if np.bincount(ys[i], minlength=2).min() >= 10]
    if not bags:
        return dict(status="abstain", n_bags=0)
    cols = list(Xs.columns); P, Pr = {}, {}
    for m in ("HGB", "DT", "SVC"):
        fs = [G.fit_proba_m(m, {}, Xs.iloc[i].reset_index(drop=True), ys[i], cols) for i in bags]
        P[m] = np.column_stack([f(Xp) for f in fs]).mean(1); Pr[m] = np.column_stack([f(Xr) for f in fs]).mean(1)
    base, baser = (P["HGB"] >= .5).astype(int), (Pr["HGB"] >= .5).astype(int)
    cm = {str(c): c for c in U}
    d = bridge.gate_select(yp, base, {k: lab(P, c) for k, c in cm.items()}, "wF1", fold_seed=seed, boot_seed=seed + 61, K=2)
    cfg = None if d.chosen is None else cm[d.chosen]; lo = d.lower_bound
    cpool = G.counts8(yp, base, base if cfg is None else lab(P, cfg))
    cr = G.counts8(yr, baser, baser if cfg is None else lab(Pr, cfg))
    dr = BM.delta(np.random.default_rng(seed + 987163).dirichlet(cr + .5, size=20000), "wF1"); a2, b2 = np.quantile(dr, [.025, .975])
    return dict(status="ok", n_bags=len(bags), cfg=str(cfg), pt_pool=float(BM.delta(cpool, "wF1")), lo95=lo, ref=float(BM.delta(cr, "wF1")),
                kelas="membaik" if a2 > 0 else "memburuk" if b2 < 0 else "tidak-pasti",
                base_wf1_ref=float(f1_score(yr, baser, average="weighted")))


def update_a_multi(Xs, ys, Xp, yp, Xr, yr, K, seed):
    """The MINI-22 trial (MINI-18 functions), unchanged; adds the naive full-POOL gain."""
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    fn = {m: [] for m in ("HGB", "DT", "SVC")}; nb = 0
    for _ in range(G.B_BAGS):
        idx = rng.choice(len(ys), size=sub, replace=False)
        cc = np.bincount(ys[idx], minlength=K); cc = cc[cc > 0]
        if len(cc) < 2 or cc.min() < 2:
            continue
        nb += 1
        for m in fn:
            fn[m].append(M.fit_proba(m, Xs.iloc[idx].reset_index(drop=True), ys[idx], K))
    if not nb:
        return dict(status="abstain", n_bags=0)
    P = {m: sum(f(Xp) for f in fs) / len(fs) for m, fs in fn.items()}
    Pr = {m: sum(f(Xr) for f in fs) / len(fs) for m, fs in fn.items()}
    base, baser = P["HGB"].argmax(1), Pr["HGB"].argmax(1); cm = {str(c): c for c in M.cands(K)}
    d = bridge.gate_select(yp, base, {k: M.lab(P, c) for k, c in cm.items()}, "wF1", fold_seed=seed, boot_seed=seed + 71, K=K)
    cfg = None if d.chosen is None else cm[d.chosen]; l95 = d.lower_bound; pt = d.gain_pool
    cr = M.cells(yr, baser, baser if cfg is None else M.lab(Pr, cfg), K); d = float(M.wf1_delta(cr, K))
    lo, hi = np.quantile(M.boot_delta(cr, K, np.random.default_rng(seed + 987163), dirichlet=True), [.025, .975])
    return dict(status="ok", n_bags=nb, cfg=str(cfg), pt_pool=pt, lo95=l95, ref=d,
                kelas="membaik" if lo > 0 else "memburuk" if hi < 0 else "tidak-pasti",
                base_wf1_ref=float(f1_score(yr, baser, average="weighted")))


def update_b(L, k, Xs, ys, Xp, yp, Xr, yr, K, seed):
    """Tuning of learner L as the update: MINI-39 trial body for K = 2, MINI-40 for K >= 3."""
    if K == 2:
        rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
        bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS)) if np.bincount(ys[i], minlength=2).min() >= 10]
        if not bags:
            return dict(status="abstain", n_bags=0)
        best, _ = T39.tune(L, k, Xs, ys, seed); cols = list(Xs.columns); P, Pr = {}, {}
        for v, prm in (("def", {}), ("tun", best)):
            fs = [G.fit_proba_m(L, prm, Xs.iloc[i].reset_index(drop=True), ys[i], cols) for i in bags]
            P[v] = np.column_stack([f(Xp) for f in fs]).mean(1); Pr[v] = np.column_stack([f(Xr) for f in fs]).mean(1)
        bD, bT, bDr, bTr = ((P["def"] >= .5).astype(int), (P["tun"] >= .5).astype(int), (Pr["def"] >= .5).astype(int), (Pr["tun"] >= .5).astype(int))
        d = bridge.gate(yp, bD, bT, "wF1", seed=seed + 21 + k, K=2); pt, lo95 = d.gain_pool, d.lower_bound
        ref, kelas = truth(yr, bDr, bTr, seed + k)
        return dict(status="ok", n_bags=len(bags), cfg=str(best), pt_pool=pt, lo95=float(lo95), ref=float(ref), kelas=kelas)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys)); bags = []
    for _ in range(G.B_BAGS):
        idx = rng.choice(len(ys), size=sub, replace=False); cc = np.bincount(ys[idx], minlength=K); cc = cc[cc > 0]
        if len(cc) >= 2 and cc.min() >= 2:
            bags.append(idx)
    if not bags:
        return dict(status="abstain", n_bags=0)
    best, _ = T40.tune(L, k, Xs, ys, seed, K); P, Pr = {}, {}
    for v, prm in (("def", {}), ("tun", best)):
        fs = [T40.fit_proba_p(L, prm, Xs.iloc[i].reset_index(drop=True), ys[i], K) for i in bags]
        P[v] = sum(f(Xp) for f in fs) / len(fs); Pr[v] = sum(f(Xr) for f in fs) / len(fs)
    bD, bT, bDr, bTr = P["def"].argmax(1), P["tun"].argmax(1), Pr["def"].argmax(1), Pr["tun"].argmax(1)
    d = bridge.gate(yp, bD, bT, "wF1", seed=seed + 71 + k, K=K); pt, lo95 = d.gain_pool, d.lower_bound
    cr = M.cells(yr, bDr, bTr, K); lo, hi = np.quantile(M.boot_delta(cr, K, np.random.default_rng(seed + 987163 + k), dirichlet=True), [.025, .975])
    return dict(status="ok", n_bags=len(bags), cfg=str(best), pt_pool=pt, lo95=lo95, ref=float(M.wf1_delta(cr, K)),
                kelas="membaik" if lo > 0 else "memburuk" if hi < 0 else "tidak-pasti")


COLS = ["data_id", "rep", "seed", "update", "status", "N", "K", "ref_n", "TOTAL", "search_n", "pool_n", "tier", "min_search", "min_pool",
        "n_bags", "cfg", "pt_pool", "lo95", "ref", "kelas", "naive", "bridge", "harmful", "base_wf1_ref", "error", "secs"]
TUNE = []


def run_job(job):
    did, pos, rep = job; seed = int(os.environ.get("MINI42_OFF", OFF)) + 1000 * pos + rep; rows = []  # MINI42_OFF: identity tests only
    try:
        Xs, ys, Xp, yp, Xr, yr, K, info = split(did, seed)
    except Exception as e:
        return [dict(data_id=did, rep=rep, seed=seed, update="setup", status="ERROR", error=str(e)[:200])]
    tasks = [("a", None, 0)] + [(f"tune_{L}", L, ("HGB", "DT", "SVC", "KNN").index(L)) for L in TUNE]  # fixed k as in MINI-39/40
    for upd, L, k in tasks:
        t0 = time.time(); row = dict(data_id=did, rep=rep, seed=seed, update=upd, **info)
        try:
            r = (update_a_binary(Xs, ys, Xp, yp, Xr, yr, seed) if (upd == "a" and K == 2) else
                 update_a_multi(Xs, ys, Xp, yp, Xr, yr, K, seed) if upd == "a" else update_b(L, k, Xs, ys, Xp, yp, Xr, yr, K, seed))
            row.update(r)
            if r["status"] == "ok":
                row.update(naive=int(r["pt_pool"] > 0), bridge=int(r["lo95"] > EPS), harmful=int(r["ref"] <= 0))
        except Exception as e:
            row.update(status="ERROR", error=str(e)[:200])
        row["secs"] = round(time.time() - t0, 1); rows.append(row)
    return rows


if __name__ == "__main__":
    a, b, Pn, outn, sel = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
    TUNE[:] = sys.argv[6].split(",") if len(sys.argv) > 6 and sys.argv[6] else []
    lst = pd.read_csv(LIST) if LIST.exists() else None
    if sel.endswith(".csv"):
        dd = pd.read_csv(sel); jobs_ds = [(int(r.data_id), int(r.position)) for r in dd.itertuples()]
    else:
        toks = [x.split(":") for x in sel.split(",")]  # "data_id" or "data_id:position" (explicit position: identity tests, timing)
        cv = lambda s: int(s) if s.isdigit() else s
        pos = dict(zip(lst.data_id, lst.position)) if lst is not None else {}
        jobs_ds = [(cv(x[0]), int(x[1]) if len(x) > 1 else int(pos[cv(x[0])])) for x in toks]
    jobs = [(d, p, r) for d, p in jobs_ds for r in range(a, b)]
    p = Path(outn) if "/" in outn else HERE / "out" / outn; new = not p.exists(); t0 = time.time()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, restval="", extrasaction="ignore")
        if new:
            w.writeheader()
        with Pool(Pn) as pool:
            for rows in pool.imap_unordered(run_job, jobs, chunksize=1):
                w.writerows(rows); f.flush(); os.fsync(f.fileno())
    print(f"MINI-42 {len(jobs_ds)} dataset, rep {a}-{b}: {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

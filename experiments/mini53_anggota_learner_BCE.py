"""MINI-53 (PRAREG_MINI53.md): ensemble members, base learners and the joint seven-family space on (B), (C) and (E),
so that the conditions of use of MINI-45, MINI-46 and MINI-50 (run on (A) and (D)) cover every collection.

Seven arms per trial, all on the same split, bags and seeds as the stored run of the collection:
  DEF    : the paper's procedure (base HGB, members DT and SVC); must reproduce the stored decision exactly;
  LRNB   : members logistic regression and naive Bayes (MINI-45);
  MLPKNN : members MLP (64, 32) with early stopping and KNN (k = 5) (MINI-45);
  XGB0   : base learner XGBoost project default, 300 rounds, no early stopping (MINI-46);
  XGBES  : base learner XGBoost with early stopping (MINI-46);
  MLPES  : base learner MLP (64, 32) with early stopping (MINI-46);
  ALL7   : joint space of the seven member families, 711 binary / 42 + 32K multiclass configurations (MINI-50).
A member or base-learner swap fills the slots of the paper's space, so the candidate structure is unchanged.

Pipelines (each collection keeps its own stored procedure):
  B : MINI-25 primary batch, tasks T0000-T1399, one trial, weighted-F1 (mini24_standar trial code);
  C : MINI-30, 12 clinical tasks x 200 trials, net benefit at p_t = 0.2 (primary metric of (C));
  E : setting (E), 14 OpenML datasets (100 trials binary, 60 multiclass), weighted-F1 (bridge_gate.protocol.run).
Every finished trial is appended to the output CSV at once; a rerun skips finished trials.
Usage: python mini53_anggota_learner_BCE.py <out.csv> <n_proc> <plan.csv> <machine> [--smoke]"""
import csv, os, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pkg" / "src")); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini44_tuning_tests as M44
import mini45_member_families as M45
import mini46_base_learners_real as M46
import mini50_all_families as M50
import mini52_ruang_kandidat_E as M52
import mini25_sintetis as M25            # sets mini21_final.OFF / DS / data to the synthetic population
import mini21_final as F
import mini30_klinis as M30
import exp41_gate_confirm as G
import bridge_metrics as BM
import bridge_metrics_klinis as KM
from mini12_optuna import union, lab, TH, SIMPLEX
sys.argv = _argv
from bridge_gate import core, protocol as P_

ARMS = ("DEF", "LRNB", "MLPKNN", "XGB0", "XGBES", "MLPES", "ALL7")
SLOTS = {"DEF": ("HGB", "DT", "SVC"), "LRNB": ("HGB", "LR", "NB"), "MLPKNN": ("HGB", "MLP", "KNN"),
         "XGB0": ("XGB0", "DT", "SVC"), "XGBES": ("XGBES", "DT", "SVC"), "MLPES": ("MLP", "DT", "SVC")}
LEARNERS = ("HGB", "DT", "SVC", "LR", "NB", "MLP", "KNN", "XGB0", "XGBES")
ALL7 = {"HGB": "HGB", "DT": "DT", "SVC": "SVC", "LR": "LR", "NB": "NB", "MLP": "MLP", "KNN": "KNN", "XGB": "XGBES"}
C_METRIC, C_K = "nb20", M30.METRIK.index("nb20")
COLS = (["setting", "dataset", "source", "rep", "seed", "K", "total"]
        + [f"{a}_{c}" for a in ARMS for c in ("lb05", "cfg", "gain_pool", "gain_ref", "accept", "naive")]
        + ["identical", "status", "seconds"])


def estimator(name):
    """The member and base-learner settings of MINI-45 and MINI-46, unchanged."""
    if name in ("LR", "NB", "MLP", "KNN"):
        return M45.member(name)
    if name == "XGB0":
        return M46.XGB(False)
    if name == "XGBES":
        return M46.XGB(True)
    raise ValueError(name)


# ---------- pipeline G: (B) MINI-25 / mini24_standar and (C) MINI-30 ----------
def fit_g(name, Xb, yb, cols):
    if name in ("HGB", "DT", "SVC"):
        return G.fit_proba_m(name, {}, Xb, yb, cols)
    num_c, cat_c = G.data.column_kinds(Xb[cols]); pre = G.data.make_preprocessor(num_c, cat_c)
    est = estimator(name); est.fit(pre.fit_transform(Xb[cols], yb), yb)
    pos = list(est.classes_).index(1) if 1 in list(est.classes_) else 1
    return lambda Xn: est.predict_proba(pre.transform(Xn[cols]))[:, pos]


def space_g():
    """MINI-50 joint space with the grids of the (B)/(C) pipeline (identical to protocol's)."""
    thr = [({"HGB": 1}, t) for t in TH if t != .5]
    e2 = [({"HGB": w, x: 1 - w}, t) for x in M50.MEMBERS for w in (.75, .5, .25) for t in TH]
    e3 = [({"HGB": a, p: b, q: c}, t) for p, q in M50.PAIRS for a, b, c in SIMPLEX for t in (.3, .35, .4, .45, .5, .55, .6)]
    return thr + e2 + e3


def run_g(kind, name, rep):
    if kind == "B":
        seed = F.OFF + 1000 * list(F.DS).index(name) + rep; Xd, yd, Xr, yr = M25.data(name); metric = "wF1"
        delta = lambda c: BM.delta(c, "wF1"); boot_seed = seed + 61; src = name
    else:
        seed = M30.OFF + 1000 * M30.DS.index(name) + rep; Xd, yd, Xr, yr = M30.data(name); metric = C_METRIC
        delta = lambda c: KM.delta(c, C_METRIC); boot_seed = seed + 61 + C_K; src = M30.KLASTER[name]
    TOT, SS = (G.TOTAL_TRAIN, G.SEARCH_SIZE) if len(Xd) >= G.TOTAL_TRAIN + 50 else (1000, 259)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=TOT, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS))
            if np.bincount(ys[i], minlength=2).min() >= 10]
    if not bags:
        raise ValueError("SKIP: no usable bag")
    cols = list(Xs.columns); P, Pr = {}, {}
    for m in LEARNERS:
        fs = [fit_g(m, Xs.iloc[i].reset_index(drop=True), ys[i], cols) for i in bags]
        P[m] = np.column_stack([f(Xp) for f in fs]).mean(1); Pr[m] = np.column_stack([f(Xr) for f in fs]).mean(1)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp))

    def pick(Pa, idx, cands, base):
        best, g0 = None, 0.0
        for cfg in cands:
            g = float(delta(G.counts8(yp[idx], base[idx], lab(Pa, cfg, idx))))
            if g > g0 + 1e-12:
                best, g0 = cfg, g
        return best

    row = dict(setting=kind, dataset=name, source=src, rep=rep, seed=seed, K=2, total=TOT)
    for arm in ARMS:
        if arm == "ALL7":
            Pa = {k: P[v] for k, v in ALL7.items()}; Pra = {k: Pr[v] for k, v in ALL7.items()}; cands = space_g()
        else:
            h, d, s = SLOTS[arm]; Pa = {"HGB": P[h], "DT": P[d], "SVC": P[s]}; Pra = {"HGB": Pr[h], "DT": Pr[d], "SVC": Pr[s]}
            cands = union(("HGB", "DT", "SVC"))
        base, baser = (Pa["HGB"] >= .5).astype(int), (Pra["HGB"] >= .5).astype(int)
        cnt = np.zeros(8, int)
        for a, b in folds:
            cfg = pick(Pa, a, cands, base)
            cnt += G.counts8(yp[b], base[b], base[b] if cfg is None else lab(Pa, cfg, b))
        bt = delta(np.random.default_rng(boot_seed).multinomial(len(yp), cnt / len(yp), size=20000))
        lb = float(np.quantile(bt, .05)); cfg = pick(Pa, np.arange(len(yp)), cands, base)
        g = 0.0 if cfg is None else float(delta(G.counts8(yp, base, lab(Pa, cfg))))
        gr = float(delta(G.counts8(yr, baser, baser if cfg is None else lab(Pra, cfg))))
        row.update({f"{arm}_lb05": repr(lb), f"{arm}_cfg": str(cfg), f"{arm}_gain_pool": repr(g), f"{arm}_gain_ref": repr(gr),
                    f"{arm}_accept": int(lb > .005), f"{arm}_naive": int(cfg is not None)})
    return row


# ---------- pipeline P: (E), bridge_gate.protocol.run ----------
_E = list(csv.DictReader(open(HERE / "settingE_daftar_beku.csv")))


def setup_e(did, seed):
    """M52.setup generalised to K classes: same encoding, class codes, training draw and SEARCH/POOL split."""
    X, ys = M52.frame(M52.SPLIT / f"{did}_dev.pkl"); Xr, yrs = M52.frame(M52.SPLIT / f"{did}_ref.pkl"); n = len(X)
    both = P_.encode(pd.concat([X, Xr], ignore_index=True)); X, Xr = both.iloc[:n], both.iloc[n:]
    names = sorted(set(ys) | set(yrs)); code = {c: i for i, c in enumerate(names)}; K = max(len(names), 2)
    y, yr = ys.map(code).to_numpy().astype(int), yrs.map(code).to_numpy().astype(int)
    if len(y) >= P_.TOTAL_TRAIN + 50:
        tot, ss = P_.TOTAL_TRAIN, P_.SEARCH_SIZE
    elif len(y) >= P_.SMALL_TOTAL + 50:
        tot, ss = P_.SMALL_TOTAL, P_.SMALL_SEARCH
    else:
        tot, ss = len(y), int(round(P_.SEARCH_SIZE / P_.TOTAL_TRAIN * len(y)))
    X = X.reset_index(drop=True)
    if tot < len(y):
        Xtr, _, ytr, _ = train_test_split(X, pd.Series(y), train_size=tot, stratify=y, random_state=seed)
    else:
        Xtr, ytr = X, pd.Series(y)
    Xs, Xp, ys_, yp = train_test_split(Xtr, ytr, train_size=ss, stratify=ytr, random_state=seed)
    return (Xs.reset_index(drop=True), np.asarray(ys_), Xp.reset_index(drop=True), np.asarray(yp),
            Xr.reset_index(drop=True), yr, K, tot)


_ORIG = P_._learner


def run_e(name, rep):
    pos = next(i for i, r in enumerate(_E) if r["name"] == name); did = int(_E[pos]["data_id"])
    seed = M52.OFF + 1000 * pos + rep; Xs, ys, Xp, yp, Xr, yr, K, TOT = setup_e(did, seed)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(P_.BAG_FRACTION * len(ys)); bags = []
    for _ in range(P_.BAGS):
        i = rng.choice(len(ys), size=sub, replace=False); cc = np.bincount(ys[i], minlength=K)
        if K == 2:
            ok = cc.min() >= 10
        else:
            cc = cc[cc > 0]; ok = len(cc) >= 2 and cc.min() >= 2
        if ok:
            bags.append(i)
    if not bags:
        raise ValueError("SKIP: no usable bag")
    P_._learner = lambda nm, n: estimator(nm) if nm in ("LR", "NB", "MLP", "KNN", "XGB0", "XGBES") else _ORIG(nm, n)
    try:
        fitted = {m: [P_._Fitted(m, Xs.iloc[i].reset_index(drop=True), ys[i], K) for i in bags] for m in LEARNERS}
    finally:
        P_._learner = _ORIG
    Pall = M44.avg(fitted, Xp, K)
    row = dict(setting="E", dataset=name, source=name, rep=rep, seed=seed, K=K, total=TOT)
    orig_cand = P_.candidates
    try:
        for arm in ARMS:
            if arm == "ALL7":
                fz = {k: fitted[v] for k, v in ALL7.items()}; P_.candidates = M50.space
            else:
                h, d, s = SLOTS[arm]; fz = {"HGB": fitted[h], "DT": fitted[d], "SVC": fitted[s]}; P_.candidates = orig_cand
            Pa = {k: Pall[v] for k, v in (ALL7.items() if arm == "ALL7" else zip(("HGB", "DT", "SVC"), SLOTS[arm]))}
            draws, cfg, g, _ = M44.gate_parts(yp, Pa, K, seed, "wF1"); lb = float(np.quantile(draws, .05))
            gr, _, _ = M44.ref_eval(fz, cfg, Xr, yr, K, "wF1")
            row.update({f"{arm}_lb05": repr(lb), f"{arm}_cfg": str(cfg), f"{arm}_gain_pool": repr(g), f"{arm}_gain_ref": repr(gr),
                        f"{arm}_accept": int(cfg is not None and lb > .005), f"{arm}_naive": int(g > 0)})
    finally:
        P_.candidates = orig_cand
    return row


# ---------- stored default decisions (identity check) ----------
def stored():
    rd = lambda p: pd.read_csv(p, float_precision="round_trip")
    s = {}
    b = pd.concat([rd(f) for f in sorted((HERE / "out").glob("mini25_m*.csv"))])
    s.update({("B", r.dataset, int(r.rep)): (float(r.lo95), float(r.ref)) for r in b.itertuples()})
    c = pd.concat([rd(f) for f in sorted((HERE / "out").glob("mini30_m*.csv"))]); c = c[c.metric == C_METRIC]
    s.update({("C", r.dataset, int(r.rep)): (float(r.lo95), float(r.ref)) for r in c.itertuples()})
    e = pd.concat([rd(f) for f in sorted((HERE / "out").glob("settingE_*laptop.csv")) if "luar" not in f.name])
    ids = {int(r["data_id"]): r["name"] for r in _E}
    s.update({("E", ids[int(r.data_id)], int(r.rep)): (float(r.lower_bound), float(r.gain_ref)) for r in e.itertuples()})
    return s


_S = None


def run_one(job):
    global _S
    kind, name, rep = job; t0 = time.time()
    if _S is None:
        _S = stored()
    try:
        row = run_e(name, rep) if kind == "E" else run_g(kind, name, rep)
        want = _S.get((kind, name, rep))
        row["identical"] = int(want is not None and float(row["DEF_lb05"]) == want[0] and float(row["DEF_gain_ref"]) == want[1])
        row.update(status="ok", seconds=round(time.time() - t0, 1)); return row
    except Exception as e:
        st = "SKIP" if str(e).startswith("SKIP") else "ERROR " + str(e)[:150]
        return dict(setting=kind, dataset=name, rep=rep, status=st, seconds=round(time.time() - t0, 1))


def all_jobs():
    jb = [("B", f"T{i:04d}", 0) for i in range(1400)]
    jc = [("C", n, r) for n in M30.DS for r in range(200)]
    je = [("E", r["name"], k) for r in _E for k in range(100 if int(float(r["K"])) == 2 else 60)]
    return jb + jc + je


if __name__ == "__main__":
    out, n_proc, plan, machine = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    jobs = all_jobs()
    if "--smoke" in sys.argv:
        jobs = [("B", "T0000", 0), ("C", "actg175", 0), ("E", "spambase", 0)]
    else:
        pl = pd.read_csv(plan); mine = set(zip(pl[pl.machine == machine].setting, pl[pl.machine == machine].dataset))
        jobs = [j for j in jobs if (j[0], j[1]) in mine]
    done = set()
    if out.exists():
        done = {(r["setting"], r["dataset"], int(r["rep"])) for r in csv.DictReader(open(out)) if r["status"] in ("ok", "SKIP")}
    todo = [j for j in jobs if j not in done]
    print(f"{len(jobs)} trials in this part, {len(jobs) - len(todo)} already done", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as f, Pool(n_proc, maxtasksperchild=20) as pool:
        w = csv.DictWriter(f, fieldnames=COLS, restval="")
        if new:
            w.writeheader()
        for k, r in enumerate(pool.imap_unordered(run_one, todo, chunksize=1)):
            w.writerow(r); f.flush()
            print(f"{k + 1}/{len(todo)} {r['setting']} {r['dataset']} rep {r['rep']}: {r['status']} id={r.get('identical', '')} {r['seconds']} s", flush=True)
    print("done", flush=True)

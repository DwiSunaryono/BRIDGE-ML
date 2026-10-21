"""MINI-48 (PRAREG_MINI48.md): comparator rules, honest holdout and ablation cells as defined for
(A) and (B) in MINI-24, MINI-26 and MINI-32, applied to (C) the 12 clinical tasks (net benefit
p_t = 0.2, the MINI-30 primary metric; same seeds, split, bags and candidates as MINI-30) and to
(D) the 39 supporting datasets (weighted-F1; the package procedure with the MINI-43 seeds).
Every rule judges the same deployed configuration (chosen on the whole POOL), except the holdouts,
which deploy the configuration they select. Identity check: the BRIDGE lower bound and reference
gain equal the stored run.
Rules (one-sided level 0.05; definitions and seed offsets as in MINI-24/26/32):
  naive, mcnemar_pool, mcnemar_cf, crt (corrected t with per-split re-selection), crt_final
  (corrected t, final config), t5x2 (5x2 cv paired t), boot0 (bootstrap bound > 0), bridge
  (> 0.005), ho23 / ho12 (selection + honest holdout); ablation cells: pt_pool (point, same
  POOL) = naive gain, lo_pool_boot (bootstrap bound on the same POOL), pt_cf (point,
  cross-fitted), lo_cf_boot = BRIDGE's bound.
Usage: python mini48_comparators_CD.py <out.csv> <n_proc> [i/n] [--smoke]"""
import csv, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pkg" / "src")); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
_argv = sys.argv; sys.argv = sys.argv[:1]
import exp41_gate_confirm as G
import bridge_metrics_klinis as KM
from mini12_optuna import lab as lab12
import mini30_klinis as M30
import mini44_tuning_tests as M44
import mini43_supporting_D as D43
sys.argv = _argv
from bridge_gate import protocol as P_, core

K_NB20 = M30.METRIK.index("nb20")
ORDER = [("C", n) for n in M30.DS] + [(g, n) for g, n in D43.ORDER]
TRIALS = {"C": 200, "bin": 100, "mc": 60}
RULES = ["naive", "mcnemar_pool", "mcnemar_cf", "crt", "crt_final", "t5x2", "boot0", "bridge", "ho23", "ho12"]
COLS = (["setting", "dataset", "source", "rep", "seed", "K", "identical", "cfg", "lo95", "ref", "gain_pool"] + RULES
        + ["ho23_ref", "ho12_ref", "pt_pool", "lo_pool_boot", "pt_cf", "lo_cf_boot", "status", "seconds"])


def mcn(y, b, c):
    n10 = int(((c == y) & (b != y)).sum()); n01 = int(((b == y) & (c != y)).sum())
    return 1.0 if n10 + n01 == 0 else float(stats.binomtest(n10, n10 + n01, .5, alternative="greater").pvalue)


def stored():
    c = pd.concat([pd.read_csv(HERE / "out" / f"mini30_m{i}.csv", float_precision="round_trip", keep_default_na=False) for i in (1, 2, 3)])
    c = c[c.metric == "nb20"]
    s = {(r.dataset, int(r.rep)): (float(r.lo95), float(r.ref)) for r in c.itertuples()}
    d = pd.concat([pd.read_csv(HERE / "out" / f"mini43_m{i}.csv", float_precision="round_trip") for i in (1, 2, 3)])
    s.update({(r.dataset, int(r.rep)): (float(r.lower_bound), float(r.gain_ref)) for r in d.itertuples()})
    return s


def setup_C(name, seed):
    """MINI-30 pipeline for one trial (net benefit p_t = 0.2)."""
    Xd, yd, Xr, yr = M30.data(name)
    TOT, SS = (G.TOTAL_TRAIN, G.SEARCH_SIZE) if len(Xd) >= G.TOTAL_TRAIN + 50 else (1000, 259)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=TOT, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS)) if np.bincount(ys[i], minlength=2).min() >= 10]
    if not bags:
        raise ValueError("SKIP: no usable bag")
    cols = list(Xs.columns); P, Pr = {}, {}
    for m in ("HGB", "DT", "SVC"):
        fs = [G.fit_proba_m(m, {}, Xs.iloc[i].reset_index(drop=True), ys[i], cols) for i in bags]
        P[m] = np.column_stack([f(Xp) for f in fs]).mean(1); Pr[m] = np.column_stack([f(Xr) for f in fs]).mean(1)
    base, baser = (P["HGB"] >= .5).astype(int), (Pr["HGB"] >= .5).astype(int)
    k = K_NB20
    return dict(yp=yp, yr=np.asarray(yr), base=base, baser=baser, K=2,
                pick=lambda idx: M30.pick(yp, P, idx, base, "nb20")[0],
                lab=lambda cfg, idx=None: (base if idx is None else base[idx]) if cfg is None else (lab12(P, cfg) if idx is None else lab12(P, cfg, idx)),
                labr=lambda cfg: baser if cfg is None else lab12(Pr, cfg),
                cnt=lambda y, b, c: G.counts8(y, b, c), dl=lambda c: KM.delta(c, "nb20"),
                boot=lambda c, s: KM.delta(np.random.default_rng(s).multinomial(int(c.sum()), c / c.sum(), size=20000), "nb20"),
                seed_cf=seed + 61 + k, seed_ref=seed + 987163 + k, ref_dirichlet=True)


def setup_D(kind, name, seed):
    """Package pipeline for one trial (weighted-F1), identical to MINI-43."""
    Xd, yd, Xr, yr, K = M44.data(kind, name)
    TOT, SS = (P_.TOTAL_TRAIN, P_.SEARCH_SIZE) if len(yd) >= P_.TOTAL_TRAIN + 50 else (P_.SMALL_TOTAL, P_.SMALL_SEARCH)
    Xtr, _, ytr, _ = train_test_split(Xd.reset_index(drop=True), pd.Series(yd), train_size=TOT, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    fz = M44.fit_all(Xs, ys, K, seed, None); P = M44.avg(fz, Xp, K); Pr = M44.avg(fz, Xr.reset_index(drop=True), K)
    base = (P["HGB"] >= .5).astype(int) if K == 2 else P["HGB"].argmax(1)
    baser = (Pr["HGB"] >= .5).astype(int) if K == 2 else Pr["HGB"].argmax(1)
    C = P_.candidates(K); metric = "wF1"
    bs = seed + (P_.BOOT_SEED_OFFSET_BIN + P_.METRICS_BINARY.index(metric) if K == 2 else P_.BOOT_SEED_OFFSET_MC)
    return dict(yp=yp, yr=np.asarray(yr, int), base=base, baser=baser, K=K,
                pick=lambda idx: P_._pick(yp, base, P, idx, C, K, metric),
                lab=lambda cfg, idx=None: (base if idx is None else base[idx]) if cfg is None else P_.labels(P, cfg, K, idx),
                labr=lambda cfg: baser if cfg is None else P_.labels(Pr, cfg, K),
                cnt=lambda y, b, c: core.counts(y, b, c, K), dl=lambda c: core.delta(c, K, metric),
                boot=lambda c, s: M44.boot(c, K, metric, s), seed_cf=bs, seed_ref=None, ref_dirichlet=False)


def rules(T, seed):
    yp, base = T["yp"], T["base"]; pick, lab, cnt, dl = T["pick"], T["lab"], T["cnt"], T["dl"]
    c_cf = np.zeros_like(cnt(yp[:2], base[:2], base[:2])); cf_lab = base.copy()
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp):
        cf = pick(a); cb = lab(cf, b); c_cf = c_cf + cnt(yp[b], base[b], cb); cf_lab[b] = cb
    lo95 = float(np.quantile(T["boot"](c_cf, T["seed_cf"]), .05))
    cfg = pick(np.arange(len(yp))); cpool = lab(cfg); c_pool = cnt(yp, base, cpool); gpool = float(dl(c_pool))
    ref = float(dl(cnt(T["yr"], T["baser"], T["labr"](cfg))))
    sss = list(StratifiedShuffleSplit(10, test_size=1 / 3, random_state=(seed + 5) % (2**32 - 1)).split(yp, yp))
    dlist = np.asarray([0.0 if (c2 := pick(a)) is None else float(dl(cnt(yp[b], base[b], lab(c2, b)))) for a, b in sss])
    dfin = np.asarray([float(dl(cnt(yp[b], base[b], cpool[b]))) for a, b in sss]); J = len(dlist)
    sd, sf = dlist.std(ddof=1), dfin.std(ddof=1)
    p_crt = float(stats.t.sf(dlist.mean() / np.sqrt((1 / J + .5) * sd ** 2), J - 1)) if sd > 0 else (0.0 if dlist.mean() > 0 else 1.0)
    p_fin = float(stats.t.sf(dfin.mean() / np.sqrt((1 / J + .5) * sf ** 2), J - 1)) if sf > 0 else (0.0 if dfin.mean() > 0 else 1.0)
    p5, s2 = [], []
    for i in range(5):
        dd = [float(dl(cnt(yp[b], base[b], cpool[b]))) for a, b in StratifiedKFold(2, shuffle=True, random_state=(seed + 100 + i) % (2**32 - 1)).split(yp, yp)]
        p5.append(dd[0]); s2.append((dd[0] - np.mean(dd)) ** 2 + (dd[1] - np.mean(dd)) ** 2)
    den = np.sqrt(np.mean(s2)); p_5x2 = float(stats.t.sf(p5[0] / den, 5)) if den > 0 else (0.0 if p5[0] > 0 else 1.0)
    p_pool, p_cf = mcn(yp, base, cpool), mcn(yp, base, cf_lab)
    has = cfg is not None
    out = dict(cfg=str(cfg), lo95=repr(lo95), ref=repr(ref), gain_pool=repr(gpool),
               naive=int(has), mcnemar_pool=int(has and p_pool < .05), mcnemar_cf=int(has and p_cf < .05),
               crt=int(has and dlist.mean() > 0 and p_crt < .05), crt_final=int(has and dfin.mean() > 0 and p_fin < .05),
               t5x2=int(has and p_5x2 < .05), boot0=int(has and lo95 > 0), bridge=int(lo95 > .005),
               pt_pool=repr(gpool), lo_pool_boot=repr(float(np.quantile(T["boot"](c_pool, seed + 62), .05))) if has else "nan",
               pt_cf=repr(float(dl(c_cf))), lo_cf_boot=repr(lo95))
    for nm, ts, k in (("ho23", 1 / 3, 0), ("ho12", 1 / 2, 1)):
        a, b = next(StratifiedShuffleSplit(1, test_size=ts, random_state=(seed + 7 + k) % (2**32 - 1)).split(yp, yp))
        ch = pick(a)
        if ch is None:
            out.update({nm: 0, nm + "_ref": "0.0"}); continue
        lh = float(np.quantile(T["boot"](cnt(yp[b], base[b], lab(ch, b)), seed + 67 + k), .05))
        out.update({nm: int(lh > .005), nm + "_ref": repr(float(dl(cnt(T["yr"], T["baser"], T["labr"](ch)))))})
    return out


def run_one(job):
    kind, name, rep, want = job; t0 = time.time()
    if kind == "C":
        seed = M30.OFF + 1000 * M30.DS.index(name) + rep; setting, src = "C", M30.KLASTER[name]
    else:
        seed = M44.seed_of(kind, name, rep); setting, src = ("D" if kind == "bin" else "D-mc"), name
    try:
        T = setup_C(name, seed) if kind == "C" else setup_D(kind, name, seed)
        r = rules(T, seed)
        ident = int(want is not None and r["lo95"] == repr(want[0]) and r["ref"] == repr(want[1]))
        return dict(setting=setting, dataset=name, source=src, rep=rep, seed=seed, K=T["K"], identical=ident, **r,
                    status="ok", seconds=round(time.time() - t0, 1))
    except Exception as e:
        st = "SKIP" if str(e).startswith("SKIP") else "ERROR " + str(e)[:150]
        return dict(setting=setting, dataset=name, source=src, rep=rep, seed=seed, status=st, seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    out, n_proc = Path(sys.argv[1]), int(sys.argv[2]); S = stored()
    jobs = [(kind, name, rep, S.get((name, rep))) for kind, name in ORDER for rep in range(TRIALS[kind])]
    if "--smoke" in sys.argv:
        jobs = [j for j in jobs if (j[1], j[2]) in (("mammography", 0), ("brfss_diabetes", 0), ("compas", 0), ("DryBean", 0))
                or (j[0] == "C" and j[2] == 0 and j[1] == M30.DS[3])]
    pos = [a for a in sys.argv[3:] if "/" in a]
    if pos:
        i, n = map(int, pos[0].split("/")); jobs = jobs[i - 1::n]
    done = set()
    if out.exists():
        done = {(r["dataset"], int(r["rep"])) for r in csv.DictReader(open(out)) if r["status"] in ("ok", "SKIP")}
    todo = [j for j in jobs if (j[1], j[2]) not in done]
    print(f"{len(jobs)} trials in this part, {len(jobs) - len(todo)} already done", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as f, Pool(n_proc) as pool:
        w = csv.DictWriter(f, fieldnames=COLS, restval="")
        if new:
            w.writeheader()
        for k, r in enumerate(pool.imap_unordered(run_one, todo, chunksize=1)):
            w.writerow(r); f.flush()
            if k % 50 == 0 or not r["status"].startswith(("ok", "SKIP")):
                print(f"{k + 1}/{len(todo)} {r['dataset']} rep {r['rep']}: {r['status']} {r['seconds']} s", flush=True)
    print("done", flush=True)

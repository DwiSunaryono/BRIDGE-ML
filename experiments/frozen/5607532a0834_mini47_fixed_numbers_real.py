"""MINI-47 (PRAREG_MINI47.md): fixed numbers of the protocol that were never varied on real data, varied one at a time on
the same trials as MINI-24 (A) and MINI-43 (D) (same seeds, outer split and, unless the arm changes it, SEARCH/POOL split):
  DEF    : the paper's protocol (identity check against the stored lower bound and reference gain);
  SEED   : learner random state = the trial seed (instead of the fixed 42) for HGB, DT and SVC;
  FRAC60 : bags of 60% of SEARCH;  FRAC100: no subsampling (one fit on the whole SEARCH, the limit of bags of 100%);
  BAG16  : the first 16 of the paper's bag draws;  BAG64: the paper's 32 draws followed by 32 more from the same sequence
           (the MINI-37 rule: bag n is the n-th draw of one random sequence, with the >= 10 per class filter);
  S360, S1000: SEARCH of 360 or 1,000 of the 2,430 training rows (POOL = the rest).
Gate, candidate space, metric and margin unchanged. Every finished trial is appended to the CSV at once.
Usage: python mini47_fixed_numbers_real.py <out.csv> <n_proc> [i/n] [--smoke]"""
import csv, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pkg" / "src")); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini44_tuning_tests as M44
sys.argv = _argv
from bridge_gate import protocol as P_

ARMS = ("DEF", "SEED", "FRAC60", "FRAC100", "BAG16", "BAG64", "S360", "S1000")
COLS = (["setting", "dataset", "rep", "seed", "K", "total", "def_identical"]
        + [f"{a}_{c}" for a in ARMS for c in ("lb05", "cfg", "gain_pool", "gain_ref", "accept", "naive", "bags")]
        + ["status", "seconds"])


def draws(ys, K, seed, n, frac):
    """The first n bag draws of the paper's random sequence, each with its keep flag (binary: >= 10 per class)."""
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(frac * len(ys)); out = []
    for _ in range(n):
        i = rng.choice(len(ys), size=sub, replace=False); cc = np.bincount(ys[i], minlength=K)
        if K == 2:
            ok = cc.min() >= 10
        else:
            cc = cc[cc > 0]; ok = len(cc) >= 2 and cc.min() >= 2
        out.append((i, ok))
    return out


def fit(Xs, ys, K, idx_list, mseed=None):
    old = P_.MODEL_SEED
    if mseed is not None:
        P_.MODEL_SEED = mseed
    try:
        return {m: [P_._Fitted(m, Xs.iloc[i].reset_index(drop=True), ys[i], K) for i in idx_list] for m in ("HGB", "DT", "SVC")}
    finally:
        P_.MODEL_SEED = old


def split(Xd, yd, seed, SS):
    TOT = P_.TOTAL_TRAIN
    Xtr, _, ytr, _ = train_test_split(Xd.reset_index(drop=True), pd.Series(yd), train_size=TOT, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
    return Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)


def decide(fz, Xp, yp, Xr, yr, K, seed, metric="wF1"):
    P = M44.avg(fz, Xp, K); d, cfg, g, _ = M44.gate_parts(yp, P, K, seed, metric)
    lb = float(np.quantile(d, .05)); gr, _, _ = M44.ref_eval(fz, cfg, Xr, yr, K, metric)
    return dict(lb05=repr(lb), cfg=str(cfg), gain_pool=repr(g), gain_ref=repr(gr), accept=int(cfg is not None and lb > .005), naive=int(g > 0),
                bags=len(fz["HGB"]))


def run_one(job):
    kind, name, rep, want = job; seed = M44.seed_of(kind, name, rep); t0 = time.time()
    setting = "A" if kind == "A" else ("D" if kind == "bin" else "D-mc")
    try:
        Xd, yd, Xr, yr, K = M44.data(kind, name)
        if len(yd) < P_.TOTAL_TRAIN + 50:
            raise ValueError("fewer rows than the 2,430-row protocol")
        row = dict(setting=setting, dataset=name, rep=rep, seed=seed, K=K, total=P_.TOTAL_TRAIN)
        Xs, ys, Xp, yp = split(Xd, yd, seed, P_.SEARCH_SIZE)
        dr = draws(ys, K, seed, 64, P_.BAG_FRACTION)
        keep32 = [i for i, ok in dr[:32] if ok]; keep_extra = [i for i, ok in dr[32:] if ok]; keep16 = [i for i, ok in dr[:16] if ok]
        f32 = fit(Xs, ys, K, keep32)
        r = decide(f32, Xp, yp, Xr, yr, K, seed); row.update({f"DEF_{k}": v for k, v in r.items()})
        row["def_identical"] = int(r["lb05"] == repr(want[0]) and r["gain_ref"] == repr(want[1]))
        n16 = len(keep16); r = decide({m: v[:n16] for m, v in f32.items()}, Xp, yp, Xr, yr, K, seed); row.update({f"BAG16_{k}": v for k, v in r.items()})
        fx = fit(Xs, ys, K, keep_extra); r = decide({m: f32[m] + fx[m] for m in f32}, Xp, yp, Xr, yr, K, seed)
        row.update({f"BAG64_{k}": v for k, v in r.items()}); del f32, fx
        r = decide(fit(Xs, ys, K, keep32, mseed=seed % (2**31 - 1)), Xp, yp, Xr, yr, K, seed); row.update({f"SEED_{k}": v for k, v in r.items()})
        k60 = [i for i, ok in draws(ys, K, seed, 32, 0.6) if ok]
        r = decide(fit(Xs, ys, K, k60), Xp, yp, Xr, yr, K, seed); row.update({f"FRAC60_{k}": v for k, v in r.items()})
        one = fit(Xs, ys, K, [np.arange(len(ys))])   # no subsampling: one fit on the whole SEARCH
        r = decide(one, Xp, yp, Xr, yr, K, seed); r["bags"] = 32; row.update({f"FRAC100_{k}": v for k, v in r.items()})
        for arm, SS in (("S360", 360), ("S1000", 1000)):
            Xs2, ys2, Xp2, yp2 = split(Xd, yd, seed, SS)
            kk = [i for i, ok in draws(ys2, K, seed, 32, P_.BAG_FRACTION) if ok]
            r = decide(fit(Xs2, ys2, K, kk), Xp2, yp2, Xr, yr, K, seed); row.update({f"{arm}_{k}": v for k, v in r.items()})
        row.update(status="ok", seconds=round(time.time() - t0, 1)); return row
    except Exception as e:
        return dict(setting=setting, dataset=name, rep=rep, seed=seed, status="ERROR " + str(e)[:150], seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    out, n_proc = Path(sys.argv[1]), int(sys.argv[2]); S = M44.stored()
    jobs = [(kind, name, rep, S[(name, rep)]) for kind, name in M44.ORDER for rep in range(M44.TRIALS[kind])]
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

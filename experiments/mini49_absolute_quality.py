"""MINI-49 (PRAREG_MINI49.md): absolute model quality behind MINI-46.

For the first 20 trials of every dataset in (A) and (D), the four base-learner arms (HGB = the paper, XGB0, XGBES,
MLPES) are re-run with the MINI-46 seeds and settings. The gate decision must equal the stored one as an identity
check, and the weighted-F1 on the locked reference is recorded for the baseline, the model deployed by the naive rule,
and the model deployed by BRIDGE. This tests whether a learner on which the gate fails (XGBoost without iteration
control) gives a worse model than HGB in the first place.
Usage: python mini49_absolute_quality.py <out.csv> <n_proc> [i/n] [--mc-shards=2,3] [--smoke]
"""
import csv, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pkg" / "src")); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini44_tuning_tests as M44
import mini46_base_learners_real as M46
sys.argv = _argv
from bridge_gate import protocol as P_

ARMS = ("HGB",) + M46.ARMS
REPS = 20
COLS = (["setting", "dataset", "rep", "seed", "K"]
        + [f"{a}_{c}" for a in ARMS for c in ("identical", "accept", "naive", "base_wf1", "naive_wf1", "bridge_wf1")] + ["status", "seconds"])


def stored():
    s = {}
    a = pd.concat([pd.read_csv(HERE / "out" / f"mini24_m{i}.csv", float_precision="round_trip") for i in (1, 2, 3)])
    s.update({("HGB", r.dataset, int(r.rep)): (float(r.lo95), float(r.ref)) for r in a.itertuples()})
    d = pd.concat([pd.read_csv(HERE / "out" / f"mini43_m{i}.csv", float_precision="round_trip") for i in (1, 2, 3)])
    s.update({("HGB", r.dataset, int(r.rep)): (float(r.lower_bound), float(r.gain_ref)) for r in d.itertuples()})
    m = pd.concat([pd.read_csv(HERE / "out" / f"mini46_m{i}.csv", float_precision="round_trip") for i in (1, 2, 3)])
    for arm in M46.ARMS:
        s.update({(arm, r.dataset, int(r.rep)): (float(getattr(r, f"{arm}_lb05")), float(getattr(r, f"{arm}_gain_ref"))) for r in m.itertuples()})
    return s


S = None


def wf1(y, p):
    return float(f1_score(y, p, average="weighted"))


def run_one(job):
    global S
    if S is None:
        S = stored()
    kind, name, rep = job; seed = M44.seed_of(kind, name, rep); t0 = time.time(); metric = "wF1"
    setting = "A" if kind == "A" else ("D" if kind == "bin" else "D-mc")
    try:
        Xd, yd, Xr, yr, K = M44.data(kind, name)
        TOT, SS = (P_.TOTAL_TRAIN, P_.SEARCH_SIZE) if len(yd) >= P_.TOTAL_TRAIN + 50 else (P_.SMALL_TOTAL, P_.SMALL_SEARCH)
        Xtr, _, ytr, _ = train_test_split(Xd.reset_index(drop=True), pd.Series(yd), train_size=TOT, stratify=yd, random_state=seed)
        Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
        Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
        yr = np.asarray(yr, int); row = dict(setting=setting, dataset=name, rep=rep, seed=seed, K=K); others = None
        for arm in ARMS:
            fz = M44.fit_all(Xs, ys, K, seed, None) if arm == "HGB" else M46.fit_arm(Xs, ys, K, seed, arm, others)
            others = {"DT": fz["DT"], "SVC": fz["SVC"]}
            P = M44.avg(fz, Xp, K); d, cfg, g, _ = M44.gate_parts(yp, P, K, seed, metric); lb = float(np.quantile(d, .05))
            Pr = M44.avg(fz, Xr.reset_index(drop=True), K)
            br = (Pr["HGB"] >= .5).astype(int) if K == 2 else Pr["HGB"].argmax(1)
            dep = br if cfg is None else P_.labels(Pr, cfg, K)
            gr = float(M44.core.delta(M44.core.counts(yr, br, dep, K), K, metric)) if hasattr(M44, "core") else None
            want = S.get((arm, name, rep))
            acc, nv = int(cfg is not None and lb > .005), int(g > 0)
            row.update({f"{arm}_identical": int(want is not None and repr(lb) == repr(want[0]) and (gr is None or repr(gr) == repr(want[1]))),
                        f"{arm}_accept": acc, f"{arm}_naive": nv, f"{arm}_base_wf1": wf1(yr, br),
                        f"{arm}_naive_wf1": wf1(yr, dep if nv else br), f"{arm}_bridge_wf1": wf1(yr, dep if acc else br)})
        row.update(status="ok", seconds=round(time.time() - t0, 1)); return row
    except Exception as e:
        return dict(setting=setting, dataset=name, rep=rep, seed=seed, status="ERROR " + str(e)[:150], seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    out, n_proc = Path(sys.argv[1]), int(sys.argv[2])
    jobs = [(kind, name, rep) for kind, name in M44.ORDER for rep in range(REPS)]
    if "--smoke" in sys.argv:
        jobs = [j for j in jobs if (j[1], j[2]) in (("LICD", 0), ("compas", 0))]
    pos = [a for a in sys.argv[3:] if "/" in a]
    if pos:
        i, n = map(int, pos[0].split("/"))
        mcs = [int(x) for x in next((a.split("=")[1] for a in sys.argv if a.startswith("--mc-shards=")), ",".join(map(str, range(1, n + 1)))).split(",")]
        jb, jm = [j for j in jobs if j[0] != "mc"], [j for j in jobs if j[0] == "mc"]
        jobs = jb[i - 1::n] + (jm[mcs.index(i)::len(mcs)] if i in mcs else [])
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

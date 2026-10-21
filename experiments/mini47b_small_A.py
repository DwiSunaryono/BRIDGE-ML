"""MINI-47b (DEVIASI_MINI47.md, PRAREG_MINI47b.md): the MINI-47 arms defined under the small-data protocol, run on the
five datasets of (A) that MINI-47 skipped (fewer than 2,480 rows: 1,000 training rows, SEARCH 259, POOL 741, as in
MINI-24). Arms DEF, SEED, FRAC60, FRAC100, BAG16 and BAG64 follow mini47_fixed_numbers_real.py; S360 and S1000 are not
defined for 1,000 training rows and are left empty. Same seeds as MINI-24; identity check against the stored decision.
Usage: python mini47b_small_A.py <out.csv> <n_proc> [--smoke]"""
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
import mini47_fixed_numbers_real as M47
sys.argv = _argv
from bridge_gate import protocol as P_

SMALL = ("MIC", "Marketing_Campaign", "Pumpkin_Seeds", "pc3", "seismic-bumps")
COLS = M47.COLS


def split_small(Xd, yd, seed):
    Xtr, _, ytr, _ = train_test_split(Xd.reset_index(drop=True), pd.Series(yd), train_size=P_.SMALL_TOTAL, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=P_.SMALL_SEARCH, stratify=ytr, random_state=seed)
    return Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)


def run_one(job):
    kind, name, rep, want = job; seed = M44.seed_of(kind, name, rep); t0 = time.time()
    try:
        Xd, yd, Xr, yr, K = M44.data(kind, name)
        assert len(yd) < P_.TOTAL_TRAIN + 50
        row = dict(setting="A", dataset=name, rep=rep, seed=seed, K=K, total=P_.SMALL_TOTAL)
        Xs, ys, Xp, yp = split_small(Xd, yd, seed)
        dr = M47.draws(ys, K, seed, 64, P_.BAG_FRACTION)
        keep32 = [i for i, ok in dr[:32] if ok]; keep_extra = [i for i, ok in dr[32:] if ok]; keep16 = [i for i, ok in dr[:16] if ok]
        f32 = M47.fit(Xs, ys, K, keep32)
        r = M47.decide(f32, Xp, yp, Xr, yr, K, seed); row.update({f"DEF_{k}": v for k, v in r.items()})
        row["def_identical"] = int(r["lb05"] == repr(want[0]) and r["gain_ref"] == repr(want[1]))
        n16 = len(keep16); r = M47.decide({m: v[:n16] for m, v in f32.items()}, Xp, yp, Xr, yr, K, seed); row.update({f"BAG16_{k}": v for k, v in r.items()})
        fx = M47.fit(Xs, ys, K, keep_extra); r = M47.decide({m: f32[m] + fx[m] for m in f32}, Xp, yp, Xr, yr, K, seed)
        row.update({f"BAG64_{k}": v for k, v in r.items()}); del f32, fx
        r = M47.decide(M47.fit(Xs, ys, K, keep32, mseed=seed % (2**31 - 1)), Xp, yp, Xr, yr, K, seed); row.update({f"SEED_{k}": v for k, v in r.items()})
        k60 = [i for i, ok in M47.draws(ys, K, seed, 32, 0.6) if ok]
        r = M47.decide(M47.fit(Xs, ys, K, k60), Xp, yp, Xr, yr, K, seed); row.update({f"FRAC60_{k}": v for k, v in r.items()})
        one = M47.fit(Xs, ys, K, [np.arange(len(ys))])
        r = M47.decide(one, Xp, yp, Xr, yr, K, seed); r["bags"] = 32; row.update({f"FRAC100_{k}": v for k, v in r.items()})
        row.update(status="ok", seconds=round(time.time() - t0, 1)); return row
    except Exception as e:
        return dict(setting="A", dataset=name, rep=rep, seed=seed, status="ERROR " + str(e)[:150], seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    out, n_proc = Path(sys.argv[1]), int(sys.argv[2]); S = M44.stored()
    jobs = [(kind, name, rep, S[(name, rep)]) for kind, name in M44.ORDER if name in SMALL for rep in range(M44.TRIALS[kind])]
    if "--smoke" in sys.argv:
        jobs = jobs[:2]
    done = set()
    if out.exists():
        done = {(r["dataset"], int(r["rep"])) for r in csv.DictReader(open(out)) if r["status"] == "ok"}
    todo = [j for j in jobs if (j[1], j[2]) not in done]
    print(f"{len(jobs)} trials, {len(jobs) - len(todo)} already done", flush=True)
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

"""Check that bridge_gate.protocol.run, the procedure the bridge-gate program runs by default, gives trial by trial
the same configuration, bound, reference gain, and reference class as the paper's confirmation runs:
  binary     : MINI-21 (union policy; metrics wF1, bacc, macroF1), stored in out/mini21_m*.csv
  multiclass : MINI-22, stored in out/mini22_m*.csv
Data, outer split, and seeds come from the original run scripts; only the decision procedure comes from the package.
Every result line is appended to out/verify_protocol/<kind>[.<i>of<n>].log as soon as it is known, and a rerun skips
trials already in that log. --shard i/n takes every n-th trial starting at the i-th one for splitting over machines.
Usage: python verify_protocol.py <binary|multiclass> <trials per dataset> [n_proc] [datasets,comma | all] [--shard i/n]
"""
import csv, glob, os, sys, time
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pkg" / "src")); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import numpy as np
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini21_final as F
import mini22_mc_final as M22
M = M22.M          # mini18_multiclass with the MINI-22 datasets, seeds, and loader
sys.argv = _argv
from bridge_gate import protocol
CLASS = {"membaik": "better", "memburuk": "worse", "tidak-pasti": "uncertain"}


def stored(kind):
    rows = {}
    for f in [HERE / "out" / f"{'mini21' if kind == 'binary' else 'mini22'}_m{i}.csv" for i in (1, 2, 3)]:   # output files of the run
        for r in csv.DictReader(open(f)):
            if kind == "binary" and r["design"].startswith("union_"):
                rows[(r["dataset"], int(r["rep"]), r["metric"])] = r
            elif kind == "multiclass" and not r["cfg"].startswith("ERROR"):
                rows[(r["dataset"], int(r["rep"]), "wF1")] = r
    return rows


def check(job):
    kind, ds, rep, metrics, want = job
    if kind == "binary":
        seed = F.OFF + 1000 * list(F.DS).index(ds) + rep; Xd, yd, Xr, yr = F.data(ds); K = 2
    else:
        seed = M.OFF + 1000 * list(M.DS).index(ds) + rep; Xd, yd, Xr, yr, K = M.data(M.DS[ds])
    out = []
    for m in metrics:
        r = protocol.run(Xd, np.asarray(yd), seed, metric=m, X_ref=Xr, y_ref=yr, K=K); w = want[m]
        got = dict(cfg=str(r.chosen), lo95=repr(r.lower_bound), ref=repr(r.gain_ref), kelas=r.ref_class)
        exp = dict(cfg=w["cfg"], lo95=repr(float(w["lo95"])), ref=repr(float(w["ref"])), kelas=CLASS[w["kelas"]])
        bad = [k for k in got if got[k] != exp[k]]
        out.append((ds, rep, m, "IDENTICAL" if not bad else "DIFFERENT " + "; ".join(f"{k}: {got[k]} vs {exp[k]}" for k in bad)))
    return out


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--shard")]; shard = [x for x in sys.argv[1:] if x.startswith("--shard")]
    if shard and "=" not in shard[0]:
        k = sys.argv.index(shard[0]); args = [x for x in args if x != sys.argv[k + 1]]; shard = sys.argv[k + 1]
    else:
        shard = shard[0].split("=", 1)[1] if shard else None
    kind, R = args[0], int(args[1]); n_proc = int(args[2]) if len(args) > 2 else 4
    S = stored(kind); every = list(F.DS) if kind == "binary" else list(M.DS)
    names = args[3].split(",") if len(args) > 3 and args[3] != "all" else every
    jobs = []
    for ds in names:
        for rep in range(R):
            want = {m: S[(ds, rep, m)] for m in protocol.METRICS_BINARY if (ds, rep, m) in S} if kind == "binary" else \
                   ({"wF1": S[(ds, rep, "wF1")]} if (ds, rep, "wF1") in S else {})
            if want:
                jobs.append((kind, ds, rep, list(want), want))
    tag = ""
    if shard:
        i, n = map(int, shard.split("/")); jobs = jobs[i - 1::n]; tag = f".{i}of{n}"
    log = HERE / "out" / "verify_protocol" / f"{kind}{tag}.log"; log.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            p = line.split()
            if len(p) >= 5 and p[1] == "rep":
                done.add((p[0], int(p[2])))
    todo = [j for j in jobs if (j[1], j[2]) not in done]
    print(f"{kind}{tag}: {len(jobs)} trials in this part, {len(jobs) - len(todo)} already checked", flush=True)
    t0 = time.time(); n_ok = n = 0
    with Pool(n_proc) as pool, open(log, "a") as f:
        for res in pool.imap_unordered(check, todo):
            for ds, rep, m, v in res:
                n += 1; n_ok += v == "IDENTICAL"; line = f"{ds:20s} rep {rep:3d} {m:8s} {v}"
                print(line, flush=True); f.write(line + "\n"); f.flush()
    lines = [l for l in log.read_text().splitlines() if " rep " in l]
    bad = [l for l in lines if "IDENTICAL" not in l]
    summary = f"{kind}{tag}: {len(lines) - len(bad)} of {len(lines)} decisions identical ({time.time() - t0:.0f} s this run)"
    print(summary); open(log, "a").write("# " + summary + "\n")
    sys.exit(0 if not bad else 1)

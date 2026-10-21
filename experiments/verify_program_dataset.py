"""Check the bridge-gate program in its default mode (the paper's validated procedure) as a user runs it.
The development rows and the locked reference rows of a paper trial are written to files, then
    python -m bridge_gate --dataset dev.<fmt> --reference ref.<fmt> --seed <trial seed> [--metric m] --out res
runs in its own process. The configuration, bound, reference gain and reference class in res.json must equal the
stored trial (binary: MINI-21 union policy; multiclass: MINI-22). fmt = pkl (exact) or csv (as most users have it).
Usage: python verify_program_dataset.py <binary|multiclass> <dataset> <rep> [metric] [pkl|csv]"""
import json, subprocess, sys, tempfile, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_protocol as V
import numpy as np, pandas as pd

kind, ds, rep = sys.argv[1], sys.argv[2], int(sys.argv[3])
metric = sys.argv[4] if len(sys.argv) > 4 else "wF1"; fmt = sys.argv[5] if len(sys.argv) > 5 else "pkl"
if kind == "binary":
    seed = V.F.OFF + 1000 * list(V.F.DS).index(ds) + rep; Xd, yd, Xr, yr = V.F.data(ds)
else:
    seed = V.M.OFF + 1000 * list(V.M.DS).index(ds) + rep; Xd, yd, Xr, yr, K = V.M.data(V.M.DS[ds])
w = V.stored(kind)[(ds, rep, metric)]
with tempfile.TemporaryDirectory() as tmp:
    for name, X, y in (("dev", Xd, yd), ("ref", Xr, yr)):
        df = X.copy(); df["target"] = np.asarray(y)
        (df.to_pickle if fmt == "pkl" else lambda p: df.to_csv(p, index=False))(f"{tmp}/{name}.{fmt}")
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "bridge_gate", "--dataset", f"{tmp}/dev.{fmt}", "--reference", f"{tmp}/ref.{fmt}",
                    "--target", "target", "--seed", str(seed), "--metric", metric, "--no-progress", "--out", f"{tmp}/res"],
                   check=True, stdout=subprocess.DEVNULL)
    r = json.load(open(f"{tmp}/res.json"))
got = dict(cfg=r["chosen_config"], lo95=repr(r["lower_bound"]), ref=repr(r["gain_ref"]), kelas=r["ref_class"])
exp = dict(cfg=w["cfg"], lo95=repr(float(w["lo95"])), ref=repr(float(w["ref"])), kelas=V.CLASS[w["kelas"]])
bad = [k for k in got if got[k] != exp[k]]
print(f"{kind} {ds} rep {rep} {metric} via program ({fmt}, {time.time() - t0:.0f} s): "
      + ("IDENTICAL" if not bad else "DIFFERENT " + "; ".join(f"{k}: {got[k]} vs {exp[k]}" for k in bad)))
sys.exit(0 if not bad else 1)

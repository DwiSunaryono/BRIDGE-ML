"""Verify the installed bridge package against the paper's stored outputs.

Run the MINI-42 trial code with every gate decision made by the installed package on the identity cases below, store
the outputs in out/verify_pkg/, and compare them number by number with the stored outputs of the original runs:
  1. binary, paper candidate space : pc3, MIC, seismic-bumps, reps 0-19, seeds of MINI-21/32  -> out/mini32_nyata_m*.csv
  2. multiclass, paper candidates  : DryBean, reps 0-19, seeds of MINI-22                      -> out/mini22_m*.csv
  3. binary, tuning as the update  : pc3, MIC, reps 0-2, HGB/DT/SVC/KNN, seeds of MINI-39       -> out/mini39_R_*.csv
  4. multiclass, tuning            : DryBean, reps 0-2, HGB/DT/SVC/KNN, seeds of MINI-40        -> out/mini40_m_laptop.csv
The package's own unit tests run as well. Writes out/verify_pkg/REPORT.txt and exits 0 only when every check passes.
Usage (after `pip install ./pkg`): python experiments/verify_bridge_package.py [n_processes]
"""
import glob, os, subprocess, sys, time
from importlib import metadata
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent; O = HERE / "out"; V = O / "verify_pkg"; V.mkdir(parents=True, exist_ok=True)
NP = sys.argv[1] if len(sys.argv) > 1 else "16"
lines, fails = [], []


def say(s):
    print(s, flush=True); lines.append(s)


def check(name, cond, info=""):
    say(("PASS  " if cond else "FAIL  ") + name + (f"   [{info}]" if info else ""))
    if not cond:
        fails.append(name)


def run(tag, off, reps, ids, tune=""):
    out = V / f"{tag}.csv"
    if out.exists():
        out.unlink()
    env = dict(os.environ, MINI42_OFF=str(off), OMP_NUM_THREADS="1")
    cmd = [sys.executable, str(HERE / "mini42_benchmark.py"), str(reps[0]), str(reps[1]), NP, str(out), ids] + ([tune] if tune else [])
    t0 = time.time(); r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode:
        say(r.stderr[-1500:])
    say(f"      ran {tag}: {time.time() - t0:.0f} s")
    return pd.read_csv(out, float_precision="round_trip")


rd = lambda *fs: pd.concat([pd.read_csv(f, float_precision="round_trip") for f in fs], ignore_index=True)
say(f"bridge-gate {metadata.version('bridge-gate')} imported from {__import__('bridge_gate').__file__}")
check("package is the installed copy (not imported from the source folder pkg/src)",
      "pkg/src" not in __import__("bridge_gate").__file__, __import__("bridge_gate").__file__)

# 0. unit tests of the package
r = subprocess.run([sys.executable, str(HERE.parent / "pkg" / "tests" / "test_basic.py")], capture_output=True, text=True)
check("package unit tests", r.returncode == 0, r.stdout.strip().replace("\n", "; "))

# 1. binary, paper candidate space vs MINI-32
a = run("binary_paper_space", 1_021_000_000, (0, 20), "1050:0,45648:1,46956:4")
a["dataset"] = a.data_id.map({1050: "pc3", 45648: "MIC", 46956: "seismic-bumps"})
m = a.merge(rd(*sorted(glob.glob(str(O / "mini32_nyata_m*.csv")))), on=["dataset", "rep"], suffixes=("", "_paper"))
check("1 binary: all 60 trials present", len(m) == 60, f"{len(m)} rows")
check("1 binary: lower bound identical", (m.lo95 == m.lo_cf_boot).all())
check("1 binary: reference gain identical", (m.ref == m.ref_paper).all())
check("1 binary: POOL gain (naive rule) identical", (m.pt_pool == m.pt_pool_paper).all())
check("1 binary: BRIDGE decision identical", (m.bridge == (m.lo_cf_boot > .005).astype(int)).all(), f"{int(m.bridge.sum())} accepted")

# 2. multiclass, paper candidates vs MINI-22
c = run("multiclass_paper_space", 1_022_000_000, (0, 20), "uci602:1")
d = rd(*[O / f"mini22_m{i}.csv" for i in (1, 2, 3)]); d = d[d.dataset == "DryBean"]
m = c.merge(d, on="rep", suffixes=("", "_paper"))
check("2 multiclass: all 20 trials present", len(m) == 20, f"{len(m)} rows")
check("2 multiclass: lower bound identical", (m.lo95 == m.lo95_paper).all())
check("2 multiclass: reference gain identical", (m.ref == m.ref_paper).all())
check("2 multiclass: selected configuration identical", (m.cfg == m.cfg_paper).all())

# 3. binary tuning vs MINI-39
t = run("binary_tuning", 1_039_000_000, (0, 3), "1050:0,45648:1", "HGB,DT,SVC,KNN")
t = t[t["update"].str.startswith("tune")].copy(); t["learner"] = t["update"].str[5:]; t["dataset"] = t.data_id.map({1050: "pc3", 45648: "MIC"})
m = t.merge(rd(*sorted(glob.glob(str(O / "mini39_R_*.csv")))), on=["dataset", "rep", "learner"], suffixes=("", "_paper"))
check("3 binary tuning: all 24 learner-trials present", len(m) == 24, f"{len(m)} rows")
check("3 binary tuning: tuned parameters identical", (m.cfg == m.best_params).all())
check("3 binary tuning: lower bound identical", (m.lo95 == m.lo95_paper).all())
check("3 binary tuning: reference gain identical", (m.ref == m.ref_paper).all())
check("3 binary tuning: POOL gain identical", (m.pt_pool == m.D_pool).all())

# 4. multiclass tuning vs MINI-40
t = run("multiclass_tuning", 1_040_000_000, (0, 3), "uci602:1", "HGB,DT,SVC,KNN")
t = t[t["update"].str.startswith("tune")].copy(); t["learner"] = t["update"].str[5:]
m = t.merge(rd(O / "mini40_m_laptop.csv"), on=["rep", "learner"], suffixes=("", "_paper"))
check("4 multiclass tuning: all 12 learner-trials present", len(m) == 12, f"{len(m)} rows")
check("4 multiclass tuning: tuned parameters identical", (m.cfg == m.best_params).all())
check("4 multiclass tuning: lower bound identical", (m.lo95 == m.lo95_paper).all())
check("4 multiclass tuning: reference gain identical", (m.ref == m.ref_paper).all())

say(f"\nRESULT: {'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' CHECK(S) FAILED: ' + '; '.join(fails)}")
(V / "REPORT.txt").write_text("\n".join(lines) + "\n")
sys.exit(1 if fails else 0)

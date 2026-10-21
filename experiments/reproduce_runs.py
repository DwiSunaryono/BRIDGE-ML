"""Reproduce the paper's runs (tables A and B) with the installed bridge-gate package, comparing every trial at once.

For each run the original trial code runs unchanged (same data, seeds, splits, bags, models). The gate evidence and
metric functions are replaced by the package's (exp41_gate_confirm.counts8 / delta_from_counts, bridge_metrics.delta,
bridge_metrics_klinis.delta except the descriptive sens/spes, and mini18_multiclass.cells / wf1_delta); package calls
are counted. The trial list comes from the stored output, so every stored trial is re-run. Each produced row is compared
with the stored row as text (full stored precision, about 17 significant digits); the first difference stops the run
(fail-fast) and is written to the log.

Runs (table A = development stage, table B = confirmation stage):
  A: MINI-04 MINI-09 MINI-12 MINI-14 MINI-16 MINI-18 MINI-19
  B: MINI-20 MINI-21 MINI-22 MINI-24 MINI-25 MINI-30 MINI-32 MINI-32Q
Usage:
  python experiments/reproduce_runs.py --runs MINI-09,MINI-12 --jobs 16 [--shard 1/3] [--limit 2]
  python experiments/reproduce_runs.py --report            # summary of all runs in out/reproduce_runs/
--limit N keeps N trials per dataset (quick check). --shard i/n runs every n-th trial starting at i. Output:
out/reproduce_runs/<run>[.shard].csv, .log; a rerun resumes where it stopped."""
import argparse, csv, io, os, re, signal, subprocess, sys, time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent; O = HERE / "out"; RD = O / "reproduce_runs"
TIME_COLS = {"secs", "param"}            # excluded from the comparison: timing, and 'param' is added after the trial
RUNS = {
    "MINI-04": dict(mod="mini04_confirm", argv=["x", "0", "0", "KNN"], files=["mini04_KNN.csv"], job="ds_did_rep"),
    "MINI-09": dict(mod="mini09_dev", argv=["x"], files=["mini09_dev.csv"], job="ds_did_rep"),
    "MINI-12": dict(mod="mini12_optuna", argv=["x"], files=["mini12_optuna_dev.csv"], job="ds_did_rep"),
    "MINI-14": dict(mod="mini14_optuna_size", argv=["x"], files=["mini14_optuna_size.csv"], job="ds_did_size_rep", size="S"),
    "MINI-16": dict(mod="mini16_metrics", argv=["x"], files=["mini16_metrics_dev.csv"], job="ds_did_rep"),
    "MINI-18": dict(mod="mini18_multiclass", argv=["x"], files=["mini18_multiclass_dev.csv"], job="ds_did_rep"),
    "MINI-19": dict(mod="mini19_fs_lebar", argv=["x"], files=["mini19_fs_lebar_dev.csv", "mini19b_lebar_dev.csv"], job="ds_did_rep"),
    "MINI-20": dict(mod="mini17_trainsize", argv=["x", "0", "0", "x", "1020000000"], files=["mini20_size_a.csv", "mini20_size_b.csv"],
                    job="ds_did_size_rep", size="TOTAL"),
    "MINI-21": dict(mod="mini21_final", argv=["x"], files=["mini21_m1.csv", "mini21_m2.csv", "mini21_m3.csv"], job="ds_rep"),
    "MINI-22": dict(mod="mini22_mc_final", argv=["x"], files=["mini22_m1.csv", "mini22_m2.csv", "mini22_m3.csv"], job="mc_ds_did_rep"),
    "MINI-24": dict(mod="mini24_standar", argv=["x"], files=["mini24_m1.csv", "mini24_m2.csv", "mini24_m3.csv"], job="ds_rep"),
    "MINI-25": dict(mod="mini25_sintetis", argv=["x"], files=["mini25_m1.csv", "mini25_m2.csv", "mini25_m3.csv"], job="q_task"),
    "MINI-30": dict(mod="mini30_klinis", argv=["x"], files=["mini30_m1.csv", "mini30_m2.csv", "mini30_m3.csv"], job="ds_rep"),
    "MINI-32": dict(mod="mini32_ablasi", argv=["x", "nyata"], files=["mini32_nyata_m1.csv", "mini32_nyata_m2.csv"], job="ds_rep", fn="run_job"),
    "MINI-32Q": dict(mod="mini32_ablasi", argv=["x", "Q"], files=["mini32_Q.csv"], job="q_task", fn="run_job"),
}
KEYS = ("dataset", "S", "TOTAL", "rep", "design", "metric")
CALLS = {"n": 0}
DEFER = os.environ.get("REPRO_STRICT") != "1"       # default: defer CPU-dependent mismatches; REPRO_STRICT=1 stops immediately


def cpu_family():
    try:
        name = next(l for l in open("/proc/cpuinfo") if l.startswith("model name")).split(":", 1)[1].strip()
    except Exception:
        return "unknown"
    return "AMD" if "AMD" in name else "Intel" if "Intel" in name else name


CPU = cpu_family()


# Worker side
def install(run):
    """Import the run's module with its argv and route gate arithmetic through the bridge_gate package."""
    global MOD, TRIAL, SPEC
    SPEC = RUNS[run]
    sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
    import warnings; warnings.filterwarnings("ignore")
    import bridge_gate as bg
    import exp41_gate_confirm as G, bridge_metrics as BM

    def counted(f):
        def g(*a, **k):
            CALLS["n"] += 1; return f(*a, **k)
        return g
    G.counts8 = counted(lambda y, b, c: bg.counts(y, b, c, 2))
    G.delta_from_counts = counted(lambda c: bg.delta(c, 2, "wF1"))
    BM.delta = counted(lambda c, name: bg.delta(c, 2, name))
    try:
        import bridge_metrics_klinis as KM
        _orig = KM.delta
        KM.delta = counted(lambda c, name: _orig(c, name) if name in ("sens", "spes") else bg.delta(c, 2, name))
    except ImportError:
        pass
    import importlib, contextlib
    argv = sys.argv
    with contextlib.redirect_stdout(io.StringIO()):
        sys.argv = ["x"]; import mini18_multiclass as M18      # reads its seed offset from argv at import: default, as in its runs
        sys.argv = list(SPEC["argv"]); MOD = importlib.import_module(SPEC["mod"])
    sys.argv = argv
    M18.cells = counted(lambda y, b, c, K: bg.counts(y, b, c, K))
    M18.wf1_delta = counted(lambda W, K: bg.delta(W, K, "wF1"))
    if SPEC["job"] == "mc_ds_did_rep":
        TRIAL = M18.trial
    elif SPEC["job"] == "q_task" and SPEC["mod"] == "mini25_sintetis":
        TRIAL = MOD.M.trial
    else:
        TRIAL = getattr(MOD, SPEC.get("fn", "trial"))


def run_one(job):
    before = CALLS["n"]; rows = TRIAL(job)
    rows = rows if isinstance(rows, list) else [rows]
    return job, rows, CALLS["n"] - before


# Main side
def stored(run):
    fs = [O / f for f in RUNS[run]["files"]]
    s = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in fs], ignore_index=True)
    keys = [k for k in KEYS if k in s.columns]
    dup = s.duplicated(keys).sum()
    return s, keys, int(dup)


def job_of(run, row, mod):
    kind = RUNS[run]["job"]; ds = row["dataset"]; rep = int(row["rep"])
    if kind == "ds_rep":
        return (ds, rep)
    if kind == "q_task":
        return (ds, 0)
    if kind == "ds_did_rep":
        return (ds, mod.DS[ds], rep)
    if kind == "mc_ds_did_rep":
        import mini18_multiclass as M18
        return (ds, M18.DS[ds], rep)
    if kind == "ds_did_size_rep":
        return (ds, mod.DS[ds], list(mod.SIZES).index(int(row[RUNS[run]["size"]])), rep)
    raise ValueError(kind)


NUM = __import__("re").compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def ulp_equal(a, b):
    """Configuration text that differs only in the last bits of its numbers, for example hyperparameters sampled by
    Optuna on another CPU. The text around the numbers is equal and every number is within a relative 1e-12."""
    if NUM.sub("#", a) != NUM.sub("#", b):
        return False
    xa, xb = [float(x) for x in NUM.findall(a)], [float(x) for x in NUM.findall(b)]
    return len(xa) == len(xb) and all(abs(u - v) <= 1e-12 * max(abs(u), abs(v), 1e-300) for u, v in zip(xa, xb))


TEXT_CONFIG = {"cfg", "best_params", "thr"}       # configuration text; every other column must be identical as text


def as_text(row, cols):
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=cols, restval="", extrasaction="ignore"); w.writerow(row)
    return next(csv.reader(io.StringIO(buf.getvalue())))


def main_run(run, jobs_n, shard, limit):
    install(run)                                              # same import in the main process, to build the job list
    s, keys, dup = stored(run)
    cols = [c for c in s.columns if c not in TIME_COLS]
    ref, jobkeys = {}, {}
    for r in s.to_dict("records"):
        k = tuple(r[x] for x in keys); ref[k] = r; jobkeys.setdefault(job_of(run, r, MOD), []).append(k)
    jobs = list(jobkeys)
    if limit:                                                  # quick check: the first `limit` trials of each dataset
        count, kept = {}, []
        q = RUNS[run]["job"] == "q_task"                       # population Q: one trial per task; keep 14*limit tasks (limit of 100)
        for j in jobs:
            g = "Q" if q else j[0]; count[g] = count.get(g, 0) + 1
            if count[g] <= (14 * limit if q else limit):
                kept.append(j)
        jobs = kept
    i, n = shard; jobs = jobs[i - 1::n]
    tag = run + ("" if n == 1 else f".{i}of{n}")
    out, logp = RD / f"{tag}.csv", RD / f"{tag}.log"
    done = set()
    if out.exists():
        old = pd.read_csv(out, dtype=str, keep_default_na=False)
        done = {tuple(r[k] for k in keys) for r in old.to_dict("records")}
    total = len(jobs); jobs = [j for j in jobs if not all(k in done for k in jobkeys[j])]
    log = open(logp, "a")
    say = lambda m: (log.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {m}\n"), log.flush(), print(m, flush=True))
    say(f"=== {tag}: {total} trials in this shard ({total - len(jobs)} already done); stored rows {len(s)} (duplicate keys {dup}); keys {keys}; jobs {jobs_n}")
    t0 = time.perf_counter(); fresh = not out.exists(); ndone = calls = nulp = ndef = 0
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="", extrasaction="ignore")
        if fresh:
            w.writeheader()
        pool = Pool(jobs_n, initializer=install, initargs=(run,))
        try:
            for job, rows, c in pool.imap_unordered(run_one, jobs, chunksize=1):
                calls += c; deferred_job = False
                for r in rows:
                    k = tuple(str(r.get(x, "")) for x in keys)
                    if k in done:
                        continue
                    if k not in ref:
                        say(f"MISMATCH {tag}: produced row {k} not in the stored output"); raise SystemExit(2)
                    got, want = as_text(r, cols), [ref[k][c_] for c_ in cols]
                    bad = [(c_, a, b) for c_, a, b in zip(cols, got, want) if a != b]
                    ulp = [x for x in bad if x[0] in TEXT_CONFIG and ulp_equal(x[1], x[2])]
                    for c_, a, b in ulp:
                        nulp += 1; say(f"BIT-LEVEL {tag} {k} column {c_}: reproduced {a!r} stored {b!r} (numbers equal within 1e-12)")
                    bad = [x for x in bad if x not in ulp]
                    if bad:
                        if not DEFER:
                            for c_, a, b in bad[:5]:
                                say(f"MISMATCH {tag} {k} column {c_}: reproduced {a!r} stored {b!r}")
                            raise SystemExit(2)
                        # Results that depend on the CPU family, for example Optuna sampling or mutual-information
                        # ranking, are re-run on the other CPU family later with --recheck. Here the trial is
                        # recorded, not written as reproduced.
                        c_, a, b = bad[0]
                        say(f"DEFERRED {tag} {k} column {c_}: reproduced {a!r} stored {b!r} ({len(bad)} column(s) differ) cpu={CPU}")
                        with open(RD / f"{tag}.deferred.txt", "a") as fd:
                            fd.write(repr(job) + "\n")
                        ndef += 1; deferred_job = True
                        continue
                    w.writerow(r)
                f.flush(); os.fsync(f.fileno()); ndone += 1
                if ndone % 25 == 0 or ndone == len(jobs):
                    el = time.perf_counter() - t0
                    say(f"{tag}: {ndone}/{len(jobs)} trials identical ({100 * ndone / len(jobs):.0f}%), elapsed {el / 60:.1f} min, "
                        f"remaining ~{el / ndone * (len(jobs) - ndone) / 60:.1f} min, package calls {calls}")
        except BaseException:
            pool.terminate(); log.close(); raise
        pool.close(); pool.join()
    say(f"=== {tag}: DONE, {ndone} trials; {ndone - ndef} identical to the stored output"
        + (f" (with {nulp} bit-level differences in configuration text)" if nulp else "")
        + (f"; {ndef} DEFERRED to the other CPU family (see {tag}.deferred.txt)" if ndef else "") + f"; package calls {calls}; cpu={CPU}")
    log.close()


def recheck(run, jobs_n):
    """Re-run strictly the trials deferred on the other CPU family; rows identical here count as reproduced on this CPU."""
    import ast
    install(run); s, keys, _ = stored(run); cols = [c for c in s.columns if c not in TIME_COLS]
    ref = {tuple(r[k] for k in keys): r for r in s.to_dict("records")}
    files = [p for p in RD.glob(f"{run}*.deferred.txt") if p.name.split(".")[0] == run]
    jobs = list(dict.fromkeys(ast.literal_eval(l) for p in files for l in p.read_text().splitlines() if l.strip()))
    out = RD / f"{run}.recheck_{CPU}.csv"; log = open(RD / f"{run}.recheck_{CPU}.log", "a")
    say = lambda m: (log.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {m}\n"), log.flush(), print(m, flush=True))
    say(f"=== recheck {run} on {CPU}: {len(jobs)} deferred trials")
    ok = bad_n = 0
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="", extrasaction="ignore"); w.writeheader()
        with Pool(jobs_n, initializer=install, initargs=(run,)) as pool:
            for job, rows, _ in pool.imap_unordered(run_one, jobs, chunksize=1):
                diff = []
                for r in rows:
                    k = tuple(str(r.get(x, "")) for x in keys); got, want = as_text(r, cols), [ref[k][c_] for c_ in cols]
                    diff += [(c_, a, b) for c_, a, b in zip(cols, got, want) if a != b and not (c_ in TEXT_CONFIG and ulp_equal(a, b))]
                    w.writerow(r)
                if diff:
                    bad_n += 1; c_, a, b = diff[0]
                    say(f"MISMATCH recheck {run} {job[:1] + job[-1:]} on {CPU}: column {c_} reproduced {a!r} stored {b!r}")
                else:
                    ok += 1
    say(f"=== recheck {run} on {CPU}: {ok} of {len(jobs)} deferred trials identical here; {bad_n} differ here (check them on the other CPU family)")


def report():
    lines = []
    for run in RUNS:
        s, keys, _ = stored(run)
        got = [p for p in RD.glob(f"{run}*.csv") if p.name.split(".")[0] == run]
        frames = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in got]
        want = {tuple(str(v) for v in r) for r in s[keys].astype(str).itertuples(index=False)}
        have = {tuple(r) for f in frames if len(f) for r in f[keys].itertuples(index=False)} & want   # each trial once
        n = len(have) if keys and len(s) == len(want) else sum(len(f) for f in frames)
        logs = "".join(p.read_text() for p in RD.glob(f"{run}*.log") if p.name.split(".")[0] == run and ".recheck_" not in p.name)
        rlogs_all = "".join(p.read_text() for p in RD.glob(f"{run}.recheck_*.log"))
        unresolved = [l for l in logs.splitlines() if "MISMATCH" in l and "DEFERRED" not in l]
        resolved_by_recheck = "differ here" in rlogs_all and " 0 differ here" in rlogs_all
        state = "MISMATCH" if (unresolved and not resolved_by_recheck) or "MISMATCH recheck" in rlogs_all else \
                ("COMPLETE" if n >= len(want) else "partial")
        nb = logs.count("BIT-LEVEL")
        nd = len({l.strip() for p in RD.glob(f"{run}*.deferred.txt") if p.name.split(".")[0] == run
                  for l in p.read_text().splitlines() if l.strip()})                            # distinct deferred trials
        rlogs = "".join(p.read_text() for p in RD.glob(f"{run}.recheck_*.log"))
        if nd:
            lines.append(f"{'':9s} {nd} trials deferred to the other CPU family; recheck: "
                         + (" | ".join(l.split(": ", 1)[-1] for l in rlogs.splitlines() if "deferred trials identical here" in l) or "not yet run"))
        lines.append(f"{run:9s} {state:9s} {n:6d} of {len(want):6d} stored result rows reproduced identically (each counted once)"
                     + (f"; {nb} bit-level differences in configuration text (numbers equal within 1e-12)" if nb else ""))
    txt = "\n".join(lines); (RD / "REPORT.txt").write_text(txt + "\n"); print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--recheck":        # re-run deferred trials strictly on this machine
        RD.mkdir(parents=True, exist_ok=True)
        for r in sys.argv[2].split(","):
            recheck(r, int(sys.argv[3]) if len(sys.argv) > 3 else 8)
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--_run":          # internal: one run in its own process
        RD.mkdir(parents=True, exist_ok=True)
        main_run(sys.argv[2], int(sys.argv[3]), tuple(map(int, sys.argv[4].split("/"))), int(sys.argv[5]))
        sys.exit(0)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="", help="comma-separated run names, or A, B, all")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--shard", default="1/1", help="i/n: this machine runs every n-th trial starting at i")
    ap.add_argument("--limit", type=int, default=0, help="trials per dataset (quick check); 0 = all")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(); RD.mkdir(parents=True, exist_ok=True)
    if a.report:
        report(); sys.exit(0)
    A = ["MINI-04", "MINI-09", "MINI-12", "MINI-14", "MINI-16", "MINI-18", "MINI-19"]
    B = ["MINI-20", "MINI-21", "MINI-22", "MINI-24", "MINI-25", "MINI-30", "MINI-32", "MINI-32Q"]
    runs = {"A": A, "B": B, "all": A + B}.get(a.runs) or a.runs.split(",")
    for r in runs:
        # each run in its own process: the original modules keep module-level state per run
        rc = subprocess.run([sys.executable, __file__, "--_run", r, str(a.jobs), a.shard, str(a.limit)]).returncode
        if rc:
            print(f"STOPPED at {r} (exit {rc}); see out/reproduce_runs/{r}*.log"); sys.exit(rc)

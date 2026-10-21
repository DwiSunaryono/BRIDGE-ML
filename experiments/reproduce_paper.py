"""Reproduce the paper's main results table (Table 8, tab:summary) with the installed bridge-gate package.

Each part re-runs the original trials (same data, seeds, splits, bags and base models as the original runs), but every
gate decision is made by the installed package. Two checks:
  1. per trial: lower bound, reference gain, POOL gain and selected configuration identical to the stored outputs;
  2. per table row: decisions, naive accepted (% harmful), BRIDGE accepted (% harmful) and BRIDGE's upper bound,
     recomputed from the reproduced trials, equal to the numbers in paper_assets/tab0_summary.tex.
Parts:
  real       11 new datasets x 100 trials (original: MINI-32, real-data mode; stored out/mini32_nyata_m*.csv)
  synthetic  population Q, 1,400 tasks (original: MINI-32, Q mode; stored out/mini32_Q.csv)
  clinical   12 clinical tasks x 200 trials, net benefit p_t = 0.2 (original: MINI-30; stored out/mini30_m*.csv)
Every finished trial is appended to out/reproduce/<part>.csv at once, so a stopped run resumes where it stopped.
Progress is written to out/reproduce/<part>.log every 25 trials (follow with: tail -f out/reproduce/<part>.log).
The parts can run on different machines; copy the <part>.csv files into one out/reproduce/ and run --part compare.
Usage (after `pip install bridge_gate-*.whl`):
  python experiments/reproduce_paper.py --part all --jobs 16
  python experiments/reproduce_paper.py --part all --jobs 16 --reps 5       # quick partial check (5 trials per dataset, 70 tasks of Q)
Report: out/reproduce/REPORT.txt (PASS/FAIL per check). Exit code 0 only if every check passes.
"""
import argparse, csv, os, re, subprocess, sys, time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; O = HERE / "out"; R = O / "reproduce"
PAPER_TABLE = HERE.parent / "paper_assets" / "tab0_summary.tex"
ROWNAME = {"real": "11 new datasets", "synthetic": "Synthetic population", "clinical": "Clinical, net benefit"}
COLS = ["part", "dataset", "rep", "seed", "cfg", "pt_pool", "lo95", "ref", "error"]
VIA_PROGRAM = os.environ.get("BRIDGE_VIA_PROGRAM") == "1"     # set by --via-program; read by the worker processes
SUFFIX = "_program" if VIA_PROGRAM else ""


# Worker: one part, run in its own process
def worker_setup(part):
    global A, C, bridge, lab, U, G, BMx
    sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
    import warnings; warnings.filterwarnings("ignore")
    import bridge_gate as bridge_mod; bridge = bridge_mod
    argv = sys.argv; sys.argv = ["x", "Q" if part == "synthetic" else "nyata"]
    if part == "clinical":
        import mini30_klinis as C_mod; C = C_mod; A = None
        from mini12_optuna import lab as lab_f, union
        U = union(("HGB", "DT", "SVC"))
    else:
        import mini32_ablasi as A_mod; A = A_mod; C = None
        from mini12_optuna import lab as lab_f
        U = A.U
    sys.argv = argv
    import exp41_gate_confirm as G_mod; G = G_mod; lab = lab_f


def jobs_of(part, reps):
    if part == "clinical":
        return [(ds, r) for ds in C.DS for r in range(reps)]
    if part == "synthetic":
        return [(f"T{i:04d}", 0) for i in range(min(1400, reps))]   # reps = number of tasks here
    return [(ds, r) for ds in A.F.DS for r in range(reps)]


def trial(job):
    ds, rep = job
    try:
        if C is not None:
            M, data, seed, metric, boot_off = C, C.data, C.OFF + 1000 * C.DS.index(ds) + rep, "nb20", 61 + 2  # nb20 is metric k=2 in MINI-30
        else:
            M, data, seed, metric, boot_off = A.F, A.F.data, A.F.OFF + 1000 * list(A.F.DS).index(ds) + rep, "wF1", 61
        Xd, yd, Xr, yr = data(ds)[:4]
        TOT, SS = (G.TOTAL_TRAIN, G.SEARCH_SIZE) if len(Xd) >= G.TOTAL_TRAIN + 50 else (1000, 259)
        from sklearn.model_selection import train_test_split
        Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=TOT, stratify=yd, random_state=seed)
        Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
        Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
        rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
        bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS)) if np.bincount(ys[i], minlength=2).min() >= 10]
        if not bags:
            return dict(part="", dataset=ds, rep=rep, seed=seed, error="SKIP no eligible bag")
        cols = list(Xs.columns); P, Pr = {}, {}
        for m in ("HGB", "DT", "SVC"):
            fs = [G.fit_proba_m(m, {}, Xs.iloc[i].reset_index(drop=True), ys[i], cols) for i in bags]
            P[m] = np.column_stack([f(Xp) for f in fs]).mean(1); Pr[m] = np.column_stack([f(Xr) for f in fs]).mean(1)
        base, baser = (P["HGB"] >= .5).astype(int), (Pr["HGB"] >= .5).astype(int)
        cm = {str(c): c for c in U}
        if VIA_PROGRAM:          # the decision is made by the bridge-gate program, called as a separate process
            import tempfile, json as _json
            with tempfile.TemporaryDirectory() as tmp:
                f = Path(tmp) / "pool.csv"
                pd.DataFrame(dict(y=yp, current=base, **{k: lab(P, c) for k, c in cm.items()})).to_csv(f, index=False)
                cmd = [sys.executable, "-m", "bridge_gate", "--predictions", str(f), "--metric", metric, "--fold-seed", str(seed),
                       "--boot-seed", str(seed + boot_off), "--out", str(Path(tmp) / "result"), "--no-progress"]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode:
                    raise RuntimeError("bridge-gate program failed: " + r.stderr[-300:])
                res = _json.loads((Path(tmp) / "result.json").read_text())
            chosen, gain, lo = res["chosen"], res["gain_pool_chosen"], res["lower_bound"]
        else:                    # the decision is made by the bridge_gate library
            d = bridge.gate_select(yp, base, {k: lab(P, c) for k, c in cm.items()}, metric, fold_seed=seed, boot_seed=seed + boot_off, K=2)
            chosen, gain, lo = d.chosen, d.gain_pool, d.lower_bound
        cfg = None if chosen is None else cm[chosen]
        cr = bridge.counts(yr, baser, baser if cfg is None else lab(Pr, cfg), 2)
        return dict(dataset=ds, rep=rep, seed=seed, cfg=str(cfg), pt_pool=gain, lo95=lo,
                    ref=float(bridge.delta(cr, 2, metric)), error="")
    except Exception as e:
        return dict(dataset=ds, rep=rep, error=str(e)[:200])


def run_part(part, jobs_n, reps):
    worker_setup(part)
    out = R / f"{part}{SUFFIX}.csv"; done = set()
    if out.exists():
        old = pd.read_csv(out, keep_default_na=False)
        done = {(str(a), int(b)) for a, b in zip(old.dataset, old.rep)}
    todo = [j for j in jobs_of(part, reps) if (str(j[0]), int(j[1])) not in done]
    print(f"[{part}] {len(done)} trials already done, {len(todo)} to run", flush=True)
    t0 = time.perf_counter(); tty = sys.stderr.isatty()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, restval="", extrasaction="ignore")
        if not done:
            w.writeheader()
        with Pool(jobs_n, initializer=worker_setup, initargs=(part,)) as pool:
            for i, row in enumerate(pool.imap_unordered(trial, todo, chunksize=1), 1):
                row["part"] = part; w.writerow(row); f.flush(); os.fsync(f.fileno())
                if i % 25 == 0 or i == len(todo):
                    el = time.perf_counter() - t0
                    print(f"[{part}] {i}/{len(todo)} trials ({100 * i / len(todo):.0f}%), elapsed {el / 60:.1f} min, "
                          f"remaining ~{el / i * (len(todo) - i) / 60:.1f} min, {datetime.now():%H:%M:%S}", flush=True)
                if tty:
                    el = time.perf_counter() - t0; rem = el / i * (len(todo) - i)
                    sys.stderr.write(f"\r[{part}] {'#' * int(24 * i / len(todo)):24s} {i}/{len(todo)} "
                                     f"{100 * i / len(todo):3.0f}%  elapsed {el / 60:5.1f} min  remaining ~{rem / 60:5.1f} min ")
                    sys.stderr.flush()
    if tty:
        sys.stderr.write("\n")
    print(f"[{part}] finished in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


# Comparison (main process)
def paper_row(part):
    """Return the numbers printed in the paper's Table 8 row for this part."""
    txt = PAPER_TABLE.read_text()
    line = next(l for l in txt.splitlines() if l.strip().startswith(ROWNAME[part]))
    c = [x.strip() for x in line.split("&")]
    num = lambda s: float(s.replace(",", "").replace("{,}", ""))
    m1, m2 = re.match(r"([\d,]+) \(([\d.]+)\)", c[2]), re.match(r"([\d,]+) \(([\d.]+)\)", c[3])
    return dict(decisions=int(num(c[1])), naive_acc=int(num(m1.group(1))), naive_pct=float(m1.group(2)),
                bridge_acc=int(num(m2.group(1))), bridge_pct=float(m2.group(2)), bound=float(re.match(r"([\d.]+)", c[5]).group(1)))


def stored(part):
    rd = lambda fs: pd.concat([pd.read_csv(O / f, keep_default_na=False, float_precision="round_trip") for f in fs], ignore_index=True)
    if part == "real":
        s = rd(["mini32_nyata_m1.csv", "mini32_nyata_m2.csv"]); s["lo95"] = s.lo_cf_boot
    elif part == "synthetic":
        s = rd(["mini32_Q.csv"]); s = s[~s.cfg.astype(str).str.startswith("ERROR")]; s["lo95"] = s.lo_cf_boot
    else:
        s = rd(["mini30_m1.csv", "mini30_m2.csv", "mini30_m3.csv"]); s = s[(s.metric == "nb20") & ~s.kelas.isin(["error", "skip"])]
        s["pt_pool"] = s.dpool
    for c_ in ("lo95", "ref", "pt_pool"):
        s[c_] = pd.to_numeric(s[c_], errors="coerce")
    return s


def compare(part, lines, fails):
    sys.path.insert(0, str(HERE))
    from verify_exp41 import kl_ratio_up as up
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):     # the analysis module prints its own report when imported
        from analisis_mini34 import prop1
    r = pd.read_csv(R / f"{part}{SUFFIX}.csv", keep_default_na=False, float_precision="round_trip")
    for c_ in ("lo95", "ref", "pt_pool"):
        r[c_] = pd.to_numeric(r[c_], errors="coerce")
    ok_rows = r[r.error.astype(str) == ""]; s = stored(part)
    m = ok_rows.merge(s, on=["dataset", "rep"], suffixes=("", "_paper"))
    norm = lambda x: x.astype(str).str.replace(r"np\.float64\(([^)]*)\)", r"\1", regex=True).replace({"nan": "None", "": "None"})

    def check(name, cond, info=""):
        lines.append(("PASS  " if cond else "FAIL  ") + name + (f"   [{info}]" if info else ""))
        if not cond:
            fails.append(name)
    check(f"{part}: trials without error", len(ok_rows) == len(r), f"{len(r) - len(ok_rows)} with error")
    check(f"{part}: every reproduced trial found in the stored outputs", len(m) == len(ok_rows), f"{len(m)} of {len(ok_rows)}")
    check(f"{part}: lower bound identical", bool((m.lo95 == m.lo95_paper).all()))
    check(f"{part}: reference gain identical", bool((m.ref == m.ref_paper).all()))
    check(f"{part}: POOL gain identical", bool((m.pt_pool == m.pt_pool_paper).all()))
    check(f"{part}: selected configuration identical", bool((norm(m.cfg) == norm(m.cfg_paper)).all()))
    full = len(ok_rows) == len(s)
    if not full:
        lines.append(f"      {part}: {len(ok_rows)} of {len(s)} trials reproduced; the table-row check needs the full run (--reps default)")
        return
    NT = len(ok_rows); nv = ok_rows[ok_rows.pt_pool > 0]; br = ok_rows[ok_rows.lo95 > .005]
    k = int((br.ref <= 0).sum()); n = len(br)
    bound = 100 * (prop1(k, n) if part == "synthetic" else up(k, n, NT, 1))
    got = dict(decisions=NT, naive_acc=len(nv), naive_pct=round(100 * (nv.ref <= 0).mean(), 1),
               bridge_acc=n, bridge_pct=round(100 * k / max(n, 1), 1), bound=round(bound, 2))
    want = paper_row(part)
    for key in want:
        check(f"{part}: Table 8 '{key}' = paper", abs(got[key] - want[key]) < 1e-9, f"reproduced {got[key]}, paper {want[key]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="all", choices=["all", "real", "synthetic", "clinical", "compare"])
    ap.add_argument("--jobs", type=int, default=8, help="parallel processes (at most the number of physical cores)")
    ap.add_argument("--reps", type=int, default=None, help="trials per dataset/task (default: as in the paper, 100 real, 200 clinical)")
    ap.add_argument("--fresh", action="store_true", help="discard earlier reproduced trials and start again")
    ap.add_argument("--via-program", action="store_true", help="every decision is made by calling the bridge-gate program "
                    "(python -m bridge_gate --predictions ...) as a separate process, instead of the library")
    a = ap.parse_args()
    if a.via_program and os.environ.get("BRIDGE_VIA_PROGRAM") != "1":
        os.environ["BRIDGE_VIA_PROGRAM"] = "1"              # re-run with the flag in the environment of every process
        sys.exit(subprocess.run([sys.executable, __file__] + sys.argv[1:]).returncode)
    R.mkdir(parents=True, exist_ok=True)
    parts = ["real", "synthetic", "clinical"] if a.part in ("all", "compare") else [a.part]
    t_start = datetime.now()
    if a.part != "compare":
        for p in parts:
            if a.fresh and (R / f"{p}{SUFFIX}.csv").exists():
                (R / f"{p}{SUFFIX}.csv").unlink()
            reps = (a.reps or 200) if p == "clinical" else (a.reps or 100) if p == "real" else (1400 if a.reps is None else min(1400, 14 * a.reps))
            # each part in its own process: the original modules set module-level state per mode
            with open(R / f"{p}{SUFFIX}.log", "a") as log:
                log.write(f"=== {datetime.now():%Y-%m-%d %H:%M:%S} start part {p}, jobs {a.jobs}, reps {reps}\n"); log.flush()
                subprocess.run([sys.executable, __file__, "--_worker", p, str(a.jobs), str(reps)], check=True, stdout=log, stderr=log)
                log.write(f"=== {datetime.now():%Y-%m-%d %H:%M:%S} end part {p}\n")
            print(f"part {p}: done (log: {R / (p + SUFFIX + '.log')})")
    import bridge_gate
    lines = [f"bridge-gate {bridge_gate.__version__} from {bridge_gate.__file__}; decisions made by the "
             + ("bridge-gate PROGRAM (python -m bridge_gate --predictions), one process per trial" if VIA_PROGRAM else "library"),
             f"start {t_start:%Y-%m-%d %H:%M:%S}, end {datetime.now():%Y-%m-%d %H:%M:%S}"]; fails = []
    for p in parts:
        if (R / f"{p}{SUFFIX}.csv").exists():
            compare(p, lines, fails)
        else:
            lines.append(f"      {p}: not run")
    lines.append(f"\nRESULT: {'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' CHECK(S) FAILED: ' + '; '.join(fails)}")
    (R / f"REPORT{SUFFIX}.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_worker":
        run_part(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        main()

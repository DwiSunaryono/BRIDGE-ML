"""MINI-43 (PRAREG_MINI43.md): the frozen BRIDGE procedure on the supporting group (D) of the paper: the 34 confirmation
datasets (binary, 100 trials each) and the 5 new multiclass datasets (60 trials each), with fresh seeds, through the
bridge_gate package (protocol.run = the paper's procedure, identical to the MINI-21/MINI-22 trial code).
Per decision it records what Table 1 and Figure 1 need: naive and BRIDGE acceptance, the validation (POOL) gain of the
selected configuration, its reference gain and harmful status.
Seed of trial rep of dataset at position p (binary list, then multiclass list): 1_044_000_000 + 1000 * p + rep.
Every finished trial is appended to the output CSV at once; a rerun skips finished trials.
Usage: python mini43_supporting_D.py <out.csv> <n_proc> [i/n]"""
import csv, os, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pkg" / "src")); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
_argv = sys.argv; sys.argv = sys.argv[:1]
import exp41_gate_confirm as G
import mini22_mc_final as M22
sys.argv = _argv
from bridge_gate import protocol

OFF = 1_044_000_000
BIN = {}
for f in ("exp40_manifest/datasets.csv", "exp41_manifest/datasets.csv"):
    for r in pd.read_csv(HERE / f).itertuples():
        BIN.setdefault(r.dataset, int(r.data_id))
MC = dict(M22.M.DS)                      # Fashion-MNIST, DryBean, USPS, mnist_784, Kuzushiji-MNIST
ORDER = [("bin", k) for k in BIN] + [("mc", k) for k in MC]
TRIALS = {"bin": 100, "mc": 60}
COLS = ["group", "dataset", "rep", "seed", "K", "total", "accept", "naive_accept", "chosen", "lower_bound", "gain_pool",
        "gain_ref", "ref_class", "status", "seconds"]
_D = {}


def data(kind, name):
    key = (kind, name)
    if key not in _D:
        _D.clear()
        if kind == "bin":
            Xd, yd, Xr, yr = G.make_dev_and_locked_reference(BIN[name]); K = 2
        else:
            Xd, yd, Xr, yr, K = M22.M.data(MC[name])
        _D[key] = (Xd, np.asarray(yd, int), Xr, np.asarray(yr, int), K)
    return _D[key]


def run_one(job):
    kind, name, pos, rep = job; seed = OFF + 1000 * pos + rep; t0 = time.time()
    try:
        Xd, yd, Xr, yr, K = data(kind, name)
        r = protocol.run(Xd, yd, seed, metric="wF1", X_ref=Xr, y_ref=yr, K=K)
        return dict(group="D" if kind == "bin" else "D-mc", dataset=name, rep=rep, seed=seed, K=K, total=r.total,
                    accept=int(r.accept), naive_accept=int(r.naive_accept), chosen=str(r.chosen), lower_bound=repr(r.lower_bound),
                    gain_pool=repr(r.gain_pool), gain_ref=repr(r.gain_ref), ref_class=r.ref_class, status="ok",
                    seconds=round(time.time() - t0, 1))
    except Exception as e:
        return dict(group=kind, dataset=name, rep=rep, seed=seed, status="ERROR " + str(e)[:150], seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    out, n_proc = Path(sys.argv[1]), int(sys.argv[2])
    jobs = [(kind, name, pos, rep) for pos, (kind, name) in enumerate(ORDER) for rep in range(TRIALS[kind])]
    if len(sys.argv) > 3:
        i, n = map(int, sys.argv[3].split("/")); jobs = jobs[i - 1::n]
    done = set()
    if out.exists():
        done = {(r["dataset"], int(r["rep"])) for r in csv.DictReader(open(out)) if r["status"] == "ok"}
    todo = [j for j in jobs if (j[1], j[3]) not in done]
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

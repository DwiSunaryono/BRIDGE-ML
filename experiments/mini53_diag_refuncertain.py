"""MINI-53 post hoc diagnostic (not pre-registered): for every harmful acceptance of the LRNB and MLPKNN arms on (E),
recompute the deployed configuration's reference counts and the 95% Bayesian-bootstrap interval of the reference gain,
with the same function and seed offset as bridge_gate.protocol.run (ref_class: better / worse / uncertain).
The trial is re-run with mini53_anggota_learner_BCE.run_e, whose decisions reproduce the stored rows.
Usage: python mini53_diag_refuncertain.py <n_proc>"""
import sys, glob
from multiprocessing import Pool
import numpy as np, pandas as pd
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini53_anggota_learner_BCE as M
import mini44_tuning_tests as M44
from bridge_gate import core, protocol as P_
sys.argv = _argv

captured = {}
_orig_ref = M44.ref_eval


def ref_eval_capture(fitted, cfg, Xr, yr, K, metric):
    Pr = M44.avg(fitted, Xr.reset_index(drop=True), K)
    br = (Pr["HGB"] >= .5).astype(int) if K == 2 else Pr["HGB"].argmax(1)
    dep = br if cfg is None else P_.labels(Pr, cfg, K)
    captured.setdefault("cr", []).append(core.counts(yr, br, dep, K))
    return _orig_ref(fitted, cfg, Xr, yr, K, metric)


def one(job):
    name, rep, arm = job
    M44.ref_eval = ref_eval_capture; captured.clear()
    row = M.run_e(name, rep)
    k = M.ARMS.index(arm); cr = captured["cr"][k]; K = int(row["K"])
    lo, hi = P_._dirichlet_interval(cr, K, "wF1", int(row["seed"]) + P_.REF_SEED_OFFSET)
    return dict(dataset=name, rep=rep, arm=arm, gain_ref=float(row[f"{arm}_gain_ref"]), ref_lo=lo, ref_hi=hi,
                ref_class="better" if lo > 0 else "worse" if hi < 0 else "uncertain", ref_rows=int(np.asarray(cr).sum()))


if __name__ == "__main__":
    fs = [f for f in sorted(glob.glob("out/mini53_*.csv")) if "analysis" not in f]
    d = pd.concat([pd.read_csv(f, float_precision="round_trip") for f in fs]); d = d[d.status == "ok"].drop_duplicates(["setting", "dataset", "rep"])
    e = d[d.setting == "E"]
    jobs = [(r.dataset, int(r.rep), a) for a in ("LRNB", "MLPKNN") for r in e.itertuples()
            if getattr(r, f"{a}_accept") == 1 and getattr(r, f"{a}_gain_ref") <= 0]
    with Pool(int(sys.argv[1])) as p:
        out = pd.DataFrame(p.map(one, jobs, chunksize=1))
    out.to_csv("out/mini53_diag_refuncertain.csv", index=False); print(out.to_string(index=False))

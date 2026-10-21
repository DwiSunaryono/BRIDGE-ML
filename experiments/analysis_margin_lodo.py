"""Post hoc reanalysis of stored outputs (no new runs): a fixed validation margin chosen without seeing the evaluated dataset.

The leave-one-setting-out analysis (analysis_traceable_controls.py) holds out one MINI-29 setting, but the same dataset
appears in several settings (three training sizes, three metrics). Here the held-out unit is the dataset itself:
for each real dataset d, the margin t is chosen from the pre-registered grid 0.005..0.050 on all seven settings with
every row of d removed, by the same rule (maximize accepted updates subject to all seven settings passing the
criterion); t is then applied to the rows of d in every setting. The synthetic tasks of Q are independent, so they are
held out in 10 blocks of 140 tasks. The decisions of all held-out units are pooled per setting and judged by the
pre-registered criterion, beside BRIDGE on the same decisions. The false-improvement rates of the margin and of
BRIDGE are compared with the paired cluster bootstrap (held-out units as clusters; B = 10,000, seed 20261021).
The interval and p-value are conditional on the chosen margins: the margin of one held-out unit is chosen on data that
include the other held-out units, and the bootstrap does not repeat the margin choice. The paper therefore reports this
comparison as descriptive, without a significance claim.
Usage: python analysis_margin_lodo.py -> out/analysis_margin_lodo.csv"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
with contextlib.redirect_stdout(io.StringIO()):
    import analisis_mini29 as A
import analysis_mini44 as A44
A44.SEED, A44.B = 20261021, 10_000

S = {"dev 1,000 rows wF1": (A.U[A.U.TOTAL == 1000], False), "dev 2,430 rows wF1": (A.U[A.U.TOTAL == 2430], False),
     "dev 6,000 rows wF1": (A.U[A.U.TOTAL == 6000], False), "new wF1": (A.Mt[A.Mt.metric == "wF1"], False),
     "new balanced accuracy": (A.Mt[A.Mt.metric == "bacc"], False), "new macro-F1": (A.Mt[A.Mt.metric == "macroF1"], False),
     "Q wF1": (A.Q, True)}
S = {k: (s.reset_index(drop=True), p1) for k, (s, p1) in S.items()}


def unit(k, s):
    """Held-out unit: the dataset name on real data, a block of 140 tasks on Q."""
    if k.startswith("Q"):
        idx = s.dataset.astype(str).str[1:].astype(int)
        return ("Qblock" + (idx // 140).astype(str)).to_numpy()
    return s.dataset.astype(str).to_numpy()


U = {k: unit(k, s) for k, (s, _) in S.items()}
units = sorted(set(np.concatenate(list(U.values()))))


def choose(held):
    best, by = None, -1
    for t in A.GRID:
        res = []
        for k, (s, p1) in S.items():
            keep = U[k] != held
            if keep.sum() == 0:
                continue
            ss = s[keep]
            res.append(A.stats(ss, (ss.cfg != "None") & (ss.dpool > t), "dataset", p1))
        if all(r["lulus"] for r in res) and sum(r["terima"] for r in res) > by:
            best, by = t, sum(r["terima"] for r in res)
    return best


def main():
    chosen = {u: choose(u) for u in units}
    rows = []
    for k, (s, p1) in S.items():
        t_row = np.array([chosen[u] if chosen[u] is not None else np.inf for u in U[k]])
        acc = (s.cfg != "None").to_numpy() & (s.dpool.to_numpy() > t_row)
        m = A.stats(s, acc, "dataset", p1); b = A.stats(s, s.lo95 > .005, "dataset", p1)
        ts = sorted({chosen[u] for u in set(U[k])}, key=lambda v: (v is None, v))
        ref = s.ref.to_numpy(); br = (s.lo95 > .005).to_numpy(); x = s.assign(dataset=U[k])
        fir = lambda acc: (lambda ix: 100 * (ref[ix][acc[ix]] <= 0).mean() if acc[ix].any() else np.nan)
        _, lo, hi, tb = A44.boot(x, lambda ix: fir(acc)(ix) - fir(br)(ix))
        p_diff = min(1.0, max(2 * min((tb <= 0).mean(), (tb >= 0).mean()), 1 / len(tb)))
        gen_m, gen_b = 100 * (acc & (ref > .005)).mean(), 100 * (br & (ref > .005)).mean()
        rows.append(dict(setting=k, margin_range=f"{min(ts)}-{max(ts)}", margin_above_eps_per100=round(gen_m, 1),
                         bridge_above_eps_per100=round(gen_b, 1), false_pct_margin_minus_bridge_lo=round(lo, 2),
                         false_pct_margin_minus_bridge_hi=round(hi, 2), p_false_diff=p_diff, margin_accepted=m["terima"], margin_false=m["salah"],
                         margin_false_pct=round(100 * m["salah"] / max(m["terima"], 1), 2), margin_bound=m["batas"],
                         margin_LOCO=m["LOCO"], margin_yield=m["yield_"], margin_pass=m["lulus"],
                         bridge_accepted=b["terima"], bridge_false=b["salah"], bridge_false_pct=round(100 * b["salah"] / max(b["terima"], 1), 2),
                         bridge_bound=b["batas"], bridge_LOCO=b["LOCO"], bridge_yield=b["yield_"], bridge_pass=b["lulus"]))
    d = pd.DataFrame(rows); d.to_csv(HERE / "out" / "analysis_margin_lodo.csv", index=False)
    pd.set_option("display.width", 250); print(d.to_string(index=False))
    print("margins chosen:", pd.Series(chosen).value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()

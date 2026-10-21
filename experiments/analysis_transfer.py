"""Post hoc reanalysis of stored per-trial outputs (no new runs): does the cross-fit evidence carry over to the configuration
that is finally deployed?

The gate accepts when the lower 5% bootstrap quantile of the cross-fitted gain (lo) exceeds the margin. The deployed
configuration is then re-selected on all of POOL, and its gain on the locked reference (ref) is measured. Among the
updates that BRIDGE accepts, per collection:
  honored  : % of accepted updates whose reference gain is at least the lower bound (ref >= lo);
  shortfall: mean of max(lo - ref, 0) over accepted updates (how far the final configuration falls short of the bound);
  excess   : median of ref - lo over accepted updates (positive = the bound was conservative for the final configuration).
95% cluster-bootstrap percentile intervals over datasets (A, D, E) or tasks (B, C); B = 10,000, seed 20261021.
Inputs: out/mini24_m*.csv (A), out/mini25_m*.csv (B), out/mini48_m*.csv (C, D), out/settingE_*laptop.csv (E).
Usage: python analysis_transfer.py -> out/analysis_transfer.csv"""
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_mini44 as A44

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
A44.SEED, A44.B = 20261021, 10_000


def rd(pattern):
    return pd.concat([pd.read_csv(f, float_precision="round_trip") for f in sorted(OUT.glob(pattern))]).reset_index(drop=True)


def collections():
    a = rd("mini24_m*.csv"); b = rd("mini25_m*.csv"); b["dataset"] = np.arange(len(b)).astype(str)
    m = rd("mini48_m*.csv"); m["S"] = m.setting.replace({"D-mc": "D"})
    e = pd.concat([pd.read_csv(OUT / f) for f in ("settingE_laptop.csv", "settingE_part2_laptop.csv")]).drop_duplicates(["data_id", "rep"])
    e = e[e.status == "ok"]
    e = pd.DataFrame(dict(dataset=e.data_id.astype(str), lo95=e.lower_bound, ref=e.gain_ref, bridge=e.accept))
    out = {"(A)": a, "(B)": b, "(C)": m[m.S == "C"], "(D)": m[m.S == "D"], "(E)": e}
    return {k: v[v.bridge == 1][["dataset", "lo95", "ref"]].reset_index(drop=True) for k, v in out.items()}


def main():
    rows = []
    for S, x in collections().items():
        lo, ref = x.lo95.to_numpy(), x.ref.to_numpy()
        row = dict(collection=S, accepted=len(x))
        for q, f in (("honored", lambda s: 100 * (ref[s] >= lo[s]).mean()),
                     ("shortfall", lambda s: np.maximum(lo[s] - ref[s], 0).mean()),
                     ("excess", lambda s: np.median(ref[s] - lo[s]))):
            est, l, h, _ = A44.boot(x, f); row |= {q: est, f"{q}_lo": l, f"{q}_hi": h}
        row |= dict(mean_lo=lo.mean(), mean_ref=ref.mean())
        rows.append(row)
    d = pd.DataFrame(rows); d.to_csv(OUT / "analysis_transfer.csv", index=False)
    pd.set_option("display.width", 220); print(d.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""Locked analysis of MINI-53 (PRAREG_MINI53.md), in the format of analysis_arms.py (MINI-45/46).

Per arm and collection: naive accepted / harmful %, the BRIDGE criterion, BRIDGE minus naive harmful proportion with a
95% cluster-bootstrap interval, TOST (+-2 pp) against DEF with Holm over all arm x collection comparisons, and yield and
net-gain ratios to DEF. Clusters: task (B), source (C), dataset (E). Bootstrap B = 10,000, seed 20261026.
Criterion: (C), (E) as in analysis_arms (KL-ratio bound, LOCO, LOCO_0 < 10%, yield >= 5, >= 3 clusters with yield >= 5);
(B) Proposition 1 bound < 10% and yield >= 5 (every task is independent, so LOCO is not defined).
Only trials whose DEF arm reproduces the stored decision (identical = 1) enter the analysis.
Usage: python analisis_mini53.py -> out/analysis_mini53.csv"""
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_mini44 as A44
from summarize_paper import prop1

HERE = Path(__file__).resolve().parent
ARMS = ("DEF", "LRNB", "MLPKNN", "XGB0", "XGBES", "MLPES", "ALL7")
EQ = 2.0


def load():
    fs = sorted((HERE / "out").glob("mini53_*.csv"))
    d = pd.concat([pd.read_csv(f, float_precision="round_trip") for f in fs if "MANIFEST" not in f.name and "analysis" not in f.name])
    print("rows in all files", len(d), d.status.str[:5].value_counts().to_dict())
    d = d[d.status == "ok"].drop_duplicates(["setting", "dataset", "rep"])   # one ok row per trial (helpers may repeat a trial)
    print("unique ok trials", len(d), "non-identical", int((d.identical != 1).sum()))
    return d[d.identical == 1].reset_index(drop=True)


def main():
    A44.SEED = 20261026
    d = load(); rows = []
    for S, x in d.groupby("setting"):
        x = x.reset_index(drop=True)
        u = (x.source if S == "C" else x.dataset).to_numpy()
        x = x.assign(dataset=u)                     # bootstrap clusters (A44.boot resamples by the 'dataset' column)
        da, dr = (x.DEF_accept == 1).to_numpy(), x.DEF_gain_ref.to_numpy()
        for arm in ARMS:
            acc, nv, ref = (x[f"{arm}_accept"] == 1).to_numpy(), (x[f"{arm}_naive"] == 1).to_numpy(), x[f"{arm}_gain_ref"].to_numpy()
            wrong = ref <= 0
            if S == "B":
                n, k = int(acc.sum()), int((acc & wrong).sum()); y = 100 * n / len(x); ub = 100 * prop1(k, n)
                c = dict(accepted=n, harmful=k, harm_pct=100 * k / max(n, 1), kl=ub, loco=np.nan, r0=np.nan, yield_=y,
                         ds_yield5=np.nan, meets=bool(ub < 10 and y >= 5))
            else:
                c = A44.crit(acc, wrong, u)
            e, lo, hi, _ = A44.boot(x, lambda s: A44.hp(acc, wrong)(s) - A44.hp(nv, wrong)(s))
            r = dict(setting=S, arm=arm, decisions=len(x), naive_acc=int(nv.sum()), naive_harm=int((nv & wrong).sum()),
                     naive_harm_pct=100 * (nv & wrong).sum() / max(nv.sum(), 1), **{f"bridge_{k}": v for k, v in c.items()},
                     bridge_minus_naive_pp=e, bmn_lo=lo, bmn_hi=hi, yield_ratio=acc.sum() / max(da.sum(), 1),
                     netgain_ratio=ref[acc].sum() / dr[da].sum())
            if arm != "DEF":
                t, tlo, thi, tb = A44.boot(x, lambda s: A44.hp(acc, wrong)(s) - A44.hp(da, dr <= 0)(s))
                r |= dict(vs_default_pp=t, vd_lo=tlo, vd_hi=thi, p_tost=max((tb <= -EQ).mean(), (tb >= EQ).mean()))
            rows.append(r)
    out = pd.DataFrame(rows); v = out.p_tost.notna(); out.loc[v, "p_tost_holm"] = A44.holm(out.loc[v, "p_tost"])
    out["equivalent"] = out.p_tost_holm < .05
    out.to_csv(HERE / "out" / "analysis_mini53.csv", index=False)
    pd.set_option("display.width", 260)
    print(out[["setting", "arm", "decisions", "naive_harm_pct", "bridge_accepted", "bridge_harmful", "bridge_harm_pct", "bridge_kl",
               "bridge_loco", "bridge_r0", "bridge_yield_", "bridge_ds_yield5", "bridge_meets", "bridge_minus_naive_pp", "bmn_lo",
               "bmn_hi", "vs_default_pp", "p_tost_holm", "equivalent", "netgain_ratio"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()

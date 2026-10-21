"""Locked analysis shared by MINI-45, MINI-46 and MINI-47 (see their PRAREG). Every arm is compared with the paired
default arm, the stored MINI-24 (A) and MINI-43 (D) decisions on the same trials, with settings (A) and (D) analysed
separately and clusters defined by dataset. Per arm and setting the table reports naive accepted/harmful, the BRIDGE
criterion (KL-ratio bound, LOCO, LOCO_0, yield, datasets with yield >= 5), BRIDGE minus naive harmful proportion with
a 95% cluster-bootstrap interval, TOST (+-2 pp) of BRIDGE's harmful proportion against the default arm with Holm over
all arm x setting comparisons, and yield and net-gain ratios to the default.
Usage: python analysis_arms.py <45|46|47>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_mini44 as A44

HERE = Path(__file__).resolve().parent
RUNS = {"45": ("mini45", ("LR-NB", "MLP-KNN"), 20261019), "46": ("mini46", ("XGB0", "XGBES", "MLPES"), 20261020),
        "47": ("mini47", ("SEED", "FRAC60", "FRAC100", "BAG16", "BAG64", "S360", "S1000"), 20261021)}
EQ, B = 2.0, 10_000


def default_arm():
    rd = lambda f: pd.read_csv(HERE / "out" / f, float_precision="round_trip")
    a = pd.concat([rd(f"mini24_m{i}.csv") for i in (1, 2, 3)])
    a = pd.DataFrame(dict(dataset=a.dataset, rep=a.rep, d_naive=(a.cfg.notna() & (a.cfg != "None")).astype(int), d_accept=(a.lo95 > .005).astype(int), d_ref=a.ref))
    d = pd.concat([rd(f"mini43_m{i}.csv") for i in (1, 2, 3)])
    d = pd.DataFrame(dict(dataset=d.dataset, rep=d.rep, d_naive=d.naive_accept, d_accept=d.accept, d_ref=d.gain_ref))
    return pd.concat([a, d])


def main(run):
    name, arms, seed = RUNS[run]; A44.SEED = seed
    m = pd.concat([pd.read_csv(HERE / "out" / f"{name}_m{i}.csv", float_precision="round_trip") for i in (1, 2, 3)])
    m = m[m.status == "ok"].merge(default_arm(), on=["dataset", "rep"], how="left", validate="1:1")
    assert m.d_ref.notna().all(); m["S"] = m.setting.replace({"D-mc": "D"}); rows = []
    for S, x in m.groupby("S"):
        x = x.reset_index(drop=True); u = x.dataset.to_numpy()
        da, dr = (x.d_accept == 1).to_numpy(), x.d_ref.to_numpy()
        for arm in ("default",) + arms:
            if arm == "default":
                acc, nv, ref = da, (x.d_naive == 1).to_numpy(), dr
            else:
                acc, nv, ref = (x[f"{arm}_accept"] == 1).to_numpy(), (x[f"{arm}_naive"] == 1).to_numpy(), x[f"{arm}_gain_ref"].to_numpy()
            c = A44.crit(acc, ref <= 0, u)
            e, lo, hi, _ = A44.boot(x, lambda s: A44.hp(acc, ref <= 0)(s) - A44.hp(nv, ref <= 0)(s))
            r = dict(setting=S, arm=arm, decisions=len(x), naive_acc=int(nv.sum()), naive_harm=int((nv & (ref <= 0)).sum()),
                     naive_harm_pct=100 * (nv & (ref <= 0)).sum() / max(nv.sum(), 1), **{f"bridge_{k}": v for k, v in c.items()},
                     bridge_minus_naive_pp=e, bmn_lo=lo, bmn_hi=hi, yield_ratio=acc.sum() / da.sum(), netgain_ratio=ref[acc].sum() / dr[da].sum())
            if arm != "default":
                t, tlo, thi, tb = A44.boot(x, lambda s: A44.hp(acc, ref <= 0)(s) - A44.hp(da, dr <= 0)(s))
                r |= dict(vs_default_pp=t, vd_lo=tlo, vd_hi=thi, p_tost=max((tb <= -EQ).mean(), (tb >= EQ).mean()))
            rows.append(r)
    d = pd.DataFrame(rows); v = d.p_tost.notna(); d.loc[v, "p_tost_holm"] = A44.holm(d.loc[v, "p_tost"]); d["equivalent"] = d.p_tost_holm < .05
    d.to_csv(HERE / "out" / f"analysis_{name}.csv", index=False)
    pd.set_option("display.width", 260)
    print(d[["setting", "arm", "naive_acc", "naive_harm_pct", "bridge_accepted", "bridge_harmful", "bridge_harm_pct", "bridge_kl", "bridge_loco",
             "bridge_r0", "bridge_yield_", "bridge_ds_yield5", "bridge_meets", "bridge_minus_naive_pp", "bmn_lo", "bmn_hi", "vs_default_pp", "vd_lo",
             "vd_hi", "p_tost_holm", "equivalent", "yield_ratio", "netgain_ratio"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])

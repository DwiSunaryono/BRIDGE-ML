"""Locked combined analysis for MINI-47 and MINI-47b.

MINI-47 contains six large Phase-A datasets and all 39 setting-D datasets.
MINI-47b supplies the five small Phase-A datasets for DEF, SEED, FRAC60,
FRAC100, BAG16 and BAG64. S360 and S1000 remain defined on six Phase-A
datasets only. See PRAREG_MINI47.md and DEVIASI_MINI47.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_mini44 as A44
import analysis_arms as AA

HERE = Path(__file__).resolve().parent
ARMS = ("SEED", "FRAC60", "FRAC100", "BAG16", "BAG64", "S360", "S1000")
EQ = 2.0
A44.SEED = 20261021


def load():
    main = pd.concat([
        pd.read_csv(HERE / "out" / f"mini47_m{i}.csv", float_precision="round_trip")
        for i in (1, 2, 3)
    ], ignore_index=True)
    assert len(main) == 4800
    bad = main.status != "ok"
    assert bad.sum() == 500
    assert set(main.loc[bad, "dataset"]) == {
        "MIC", "Marketing_Campaign", "Pumpkin_Seeds", "pc3", "seismic-bumps"
    }
    main = main.loc[~bad].copy()
    extra = pd.read_csv(HERE / "out" / "mini47b.csv", float_precision="round_trip")
    assert len(extra) == 500 and (extra.status == "ok").all()
    d = pd.concat([main, extra], ignore_index=True, sort=False)
    assert not d.duplicated(["dataset", "rep"]).any()
    assert len(d) == 4800
    default = AA.default_arm()
    d = d.merge(default, on=["dataset", "rep"], how="left", validate="1:1")
    assert d.d_ref.notna().all()
    d["S"] = d.setting.replace({"D-mc": "D"})
    return d


def evaluate(x, arm):
    if arm == "default":
        y = x.copy()
        acc = (y.d_accept == 1).to_numpy()
        nv = (y.d_naive == 1).to_numpy()
        ref = y.d_ref.to_numpy()
    else:
        col = f"{arm}_accept"
        y = x.loc[x[col].notna()].copy()
        acc = (y[col] == 1).to_numpy()
        nv = (y[f"{arm}_naive"] == 1).to_numpy()
        ref = y[f"{arm}_gain_ref"].to_numpy()
    return y.reset_index(drop=True), acc, nv, ref


def main():
    d = load()
    rows = []
    for setting, group in d.groupby("S"):
        group = group.reset_index(drop=True)
        yd, ad, _, rd = evaluate(group, "default")
        for arm in ("default",) + ARMS:
            y, acc, nv, ref = evaluate(group, arm)
            units = y.dataset.to_numpy()
            c = A44.crit(acc, ref <= 0, units)
            e, lo, hi, _ = A44.boot(
                y, lambda s: A44.hp(acc, ref <= 0)(s) - A44.hp(nv, ref <= 0)(s)
            )
            row = dict(
                setting=setting, arm=arm, decisions=len(y), datasets=y.dataset.nunique(),
                naive_acc=int(nv.sum()), naive_harm=int((nv & (ref <= 0)).sum()),
                naive_harm_pct=100 * (nv & (ref <= 0)).sum() / max(int(nv.sum()), 1),
                **{f"bridge_{k}": v for k, v in c.items()},
                bridge_minus_naive_pp=e, bmn_lo=lo, bmn_hi=hi,
            )
            if arm == "default":
                row |= dict(yield_ratio=1.0, netgain_ratio=1.0)
            else:
                key = yd[["dataset", "rep"]].apply(tuple, axis=1)
                pos = {k: i for i, k in enumerate(key)}
                take = np.array([pos[k] for k in y[["dataset", "rep"]].apply(tuple, axis=1)])
                acc0, ref0 = ad[take], rd[take]
                t, tlo, thi, tb = A44.boot(
                    y, lambda s: A44.hp(acc, ref <= 0)(s) - A44.hp(acc0, ref0 <= 0)(s)
                )
                row |= dict(
                    vs_default_pp=t, vd_lo=tlo, vd_hi=thi,
                    p_tost=max((tb <= -EQ).mean(), (tb >= EQ).mean()),
                    yield_ratio=acc.sum() / max(acc0.sum(), 1),
                    netgain_ratio=ref[acc].sum() / ref0[acc0].sum(),
                )
            rows.append(row)
    out = pd.DataFrame(rows)
    mask = out.p_tost.notna()
    out.loc[mask, "p_tost_holm"] = A44.holm(out.loc[mask, "p_tost"])
    out["equivalent"] = out.p_tost_holm < .05
    out.to_csv(HERE / "out" / "analysis_mini47_combined.csv", index=False)
    pd.set_option("display.width", 280)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()


"""Locked analysis for MINI-50 (PRAREG_MINI50.md): acceptance, harm and
default-arm comparisons per setting, with bootstrap intervals and Holm correction.
Writes out/analysis_mini50.csv and out/analysis_mini50_families.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_mini44 as A44
import analysis_arms as AA

HERE = Path(__file__).resolve().parent
A44.SEED = 20261020
EQ = 2.0


def load():
    m = pd.read_csv(HERE / "out" / "mini50_merged.csv", float_precision="round_trip")
    assert len(m) == 4800 and (m.status == "ok").all()
    assert not m.duplicated(["dataset", "rep"]).any()
    assert m.dataset.nunique() == 50
    m["S"] = m.setting.replace({"D-mc": "D"})
    d = AA.default_arm()
    m = m.merge(d, on=["dataset", "rep"], validate="1:1", how="left")
    assert m.d_ref.notna().all()
    return m


def main():
    m = load()
    rows = []
    families = []
    for setting, x in m.groupby("S"):
        x = x.reset_index(drop=True)
        unit = x.dataset.to_numpy()
        ref = x.gain_ref.to_numpy()
        acc = (x.accept == 1).to_numpy()
        nv = (x.naive == 1).to_numpy()
        dref = x.d_ref.to_numpy()
        dacc = (x.d_accept == 1).to_numpy()
        dnv = (x.d_naive == 1).to_numpy()
        c = A44.crit(acc, ref <= 0, unit)
        cd = A44.crit(dacc, dref <= 0, unit)
        bn, bnlo, bnhi, _ = A44.boot(
            x, lambda s: A44.hp(acc, ref <= 0)(s) - A44.hp(nv, ref <= 0)(s)
        )
        vd, vdlo, vdhi, vdb = A44.boot(
            x, lambda s: A44.hp(acc, ref <= 0)(s) - A44.hp(dacc, dref <= 0)(s)
        )
        nn, nnlo, nnhi, _ = A44.boot(
            x, lambda s: A44.hp(nv, ref <= 0)(s) - A44.hp(dnv, dref <= 0)(s)
        )
        rows.append(dict(
            setting=setting, arm="all_families", decisions=len(x), datasets=x.dataset.nunique(),
            naive_acc=int(nv.sum()), naive_harm=int((nv & (ref <= 0)).sum()),
            naive_harm_pct=100 * (nv & (ref <= 0)).sum() / max(nv.sum(), 1),
            **{f"bridge_{k}": v for k, v in c.items()},
            bridge_minus_naive_pp=bn, bmn_lo=bnlo, bmn_hi=bnhi,
            default_bridge_accepted=cd["accepted"], default_bridge_harmful=cd["harmful"],
            default_bridge_harm_pct=cd["harm_pct"],
            vs_default_pp=vd, vd_lo=vdlo, vd_hi=vdhi,
            p_tost=max((vdb <= -EQ).mean(), (vdb >= EQ).mean()),
            yield_ratio=acc.sum() / dacc.sum(),
            netgain_ratio=ref[acc].sum() / dref[dacc].sum(),
            naive_vs_default_pp=nn, nvd_lo=nnlo, nvd_hi=nnhi,
            naive_default_harm_pct=100 * (dnv & (dref <= 0)).sum() / dnv.sum(),
        ))
        z = x.assign(harmful=x.gain_ref <= 0)
        for fam, g in z.groupby("family", dropna=False):
            families.append(dict(
                setting=setting, family=fam, selected=len(g),
                naive_accepted=int(g.naive.sum()),
                bridge_accepted=int(g.accept.sum()),
                bridge_harmful=int(((g.accept == 1) & g.harmful).sum()),
                mean_gain_ref=float(g.gain_ref.mean()),
            ))
    out = pd.DataFrame(rows)
    out["p_tost_holm"] = A44.holm(out.p_tost)
    out["equivalent"] = out.p_tost_holm < .05
    fam = pd.DataFrame(families).sort_values(["setting", "selected"], ascending=[True, False])
    out.to_csv(HERE / "out" / "analysis_mini50.csv", index=False)
    fam.to_csv(HERE / "out" / "analysis_mini50_families.csv", index=False)
    pd.set_option("display.width", 280)
    print(out.round(3).to_string(index=False))
    print("\nFAMILIES")
    print(fam.to_string(index=False))


if __name__ == "__main__":
    main()


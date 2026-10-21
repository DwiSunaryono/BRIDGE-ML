"""Locked analyses for Setting E and exploratory Setting E-luar.

Reads stored result parts from out/ and writes one summary CSV per setting.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

import analysis_mini44 as A44

HERE = Path(__file__).resolve().parent
A44.SEED = 20261023


def read_parts(prefix):
    files = sorted((HERE / "out").glob(f"{prefix}*laptop.csv"))
    assert files
    d = pd.concat([pd.read_csv(f, float_precision="round_trip") for f in files], ignore_index=True)
    assert (d.status == "ok").all()
    assert not d.duplicated(["data_id", "rep"]).any()
    return d


def metadata(name):
    x = pd.read_csv(HERE / name)
    x["data_id"] = x.data_id.astype(int)
    x["K"] = x.K.astype(int)
    return x[["data_id", "name", "K"]]


def basic_counts(x):
    acc = x.accept.astype(bool).to_numpy()
    nv = x.naive_accept.astype(bool).to_numpy()
    harm = (x.gain_ref <= 0).to_numpy()
    assert np.all(~acc | nv)
    n_naive, h_naive = int(nv.sum()), int((nv & harm).sum())
    n_bridge, h_bridge = int(acc.sum()), int((acc & harm).sum())
    expected = n_bridge * h_naive / n_naive if n_naive else np.nan
    p = hypergeom.cdf(h_bridge, n_naive, h_naive, n_bridge) if n_naive and n_bridge else np.nan
    return acc, nv, harm, dict(
        decisions=len(x), datasets=x.dataset.nunique(),
        naive_accepted=n_naive, naive_harmful=h_naive,
        naive_harm_pct=100 * h_naive / max(n_naive, 1),
        bridge_accepted=n_bridge, bridge_harmful=h_bridge,
        bridge_harm_pct=100 * h_bridge / max(n_bridge, 1),
        random_expected_harmful=expected, random_p=p,
        gain_retained_pct=100 * x.loc[acc, "gain_ref"].sum() / x.loc[nv, "gain_ref"].sum(),
    )


def analyze_group(x, label, criterion=True):
    x = x.reset_index(drop=True)
    acc, nv, harm, row = basic_counts(x)
    if criterion and x.dataset.nunique() >= 3:
        c = A44.crit(acc, harm, x.dataset.to_numpy())
        row |= {f"bridge_{k}": v for k, v in c.items()}
        e, lo, hi, _ = A44.boot(
            x, lambda s: A44.hp(acc, harm)(s) - A44.hp(nv, harm)(s)
        )
        gain = x.gain_ref.to_numpy()
        _, grlo, grhi, _ = A44.boot(
            x, lambda s: 100 * gain[s][acc[s]].sum() / gain[s][nv[s]].sum()
        )
        row |= dict(bridge_minus_naive_pp=e, bmn_lo=lo, bmn_hi=hi,
                    gain_retained_lo=grlo, gain_retained_hi=grhi)
    else:
        row |= dict(
            bridge_kl=np.nan, bridge_loco=np.nan, bridge_r0=np.nan,
            bridge_yield_=100 * acc.mean(), bridge_ds_yield5=np.nan,
            bridge_meets=np.nan, bridge_minus_naive_pp=np.nan,
            bmn_lo=np.nan, bmn_hi=np.nan,
            gain_retained_lo=np.nan, gain_retained_hi=np.nan,
        )
    row["group"] = label
    return row


def trial_zero(d, label):
    z = d[d.rep == 0].copy()
    acc, nv, harm, row = basic_counts(z)
    row |= dict(group=label, bridge_yield_=100 * acc.mean())
    return row


def main():
    e = read_parts("settingE_").merge(metadata("settingE_daftar_beku.csv"), on=["data_id", "K"], validate="many_to_one")
    assert len(e) == 1280 and e.data_id.nunique() == 14
    e["dataset"] = e.data_id.astype(str) + ":" + e.name
    erows = [analyze_group(e, "E_all")]
    erows.append(analyze_group(e[e.K == 2], "E_binary"))
    erows.append(analyze_group(e[e.K > 2], "E_multiclass"))
    eout = pd.DataFrame(erows)
    et0 = pd.DataFrame([trial_zero(e, "E_trial0")])
    eout.to_csv(HERE / "out" / "analysis_settingE.csv", index=False)
    et0.to_csv(HERE / "out" / "analysis_settingE_trial0.csv", index=False)

    q = read_parts("settingEluar_").merge(metadata("settingEluar_daftar_beku.csv"), on=["data_id", "K"], validate="many_to_one")
    assert len(q) == 660 and q.data_id.nunique() == 11
    q["dataset"] = q.data_id.astype(str) + ":" + q.name
    qrows = [analyze_group(q, "E_luar_all")]
    for k, g in q.groupby("K"):
        qrows.append(analyze_group(g, f"E_luar_K{k}", criterion=g.data_id.nunique() >= 3))
    qout = pd.DataFrame(qrows)
    qt0 = pd.DataFrame([trial_zero(q, "E_luar_trial0")])
    qout.to_csv(HERE / "out" / "analysis_settingEluar.csv", index=False)
    qt0.to_csv(HERE / "out" / "analysis_settingEluar_trial0.csv", index=False)

    pd.set_option("display.width", 280)
    print("SETTING E")
    print(eout.round(4).to_string(index=False))
    print("\nSETTING E TRIAL 0")
    print(et0.round(4).to_string(index=False))
    print("\nSETTING E-LUAR")
    print(qout.round(4).to_string(index=False))
    print("\nSETTING E-LUAR TRIAL 0")
    print(qt0.round(4).to_string(index=False))


if __name__ == "__main__":
    main()


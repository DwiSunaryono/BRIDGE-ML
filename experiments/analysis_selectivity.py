"""Post hoc exploratory check of whether BRIDGE's low harmful proportion follows from accepting fewer updates.

Reanalysis of stored per-trial outputs only, with no new runs. In every setting the updates BRIDGE accepts are a
subset of those the naive rule accepts with the same deployed configuration, so two quantities are exact.
(1) Random-selection control: draw as many updates at random from the naive rule's accepted updates and count the
    harmful ones. Reported globally and, on real data, within each dataset (clinical: each task) using BRIDGE's
    own count and an exact hypergeometric convolution. Tasks of Q are i.i.d., so the global version applies there.
(2) Difference in harmful proportion (BRIDGE minus naive, percentage points) with a cluster-bootstrap interval over
    datasets (real), tasks (Q), or clinical tasks, plus the coarser clinical source-level interval. The gain-loss
    trade-off reports the share of the naive rule's net gain that BRIDGE keeps, the share of harmful updates and of
    loss (summed negative reference gain) that BRIDGE removes, and mean gain per accepted update. All 95% percentile
    intervals come from a cluster bootstrap over datasets (real), sources (clinical), or tasks (Q) with B = 10,000.
Usage: python analysis_selectivity.py -> prints and writes out/analysis_selectivity.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

import summarize_paper as SP

HERE = Path(__file__).resolve().parent
B, SEED = 10_000, 20261014


def random_control(m, nv, br, harm, unit=None):
    """Return the expected harmful count and P(X <= observed) when BRIDGE's count is drawn at random from the naive rule's accepted updates."""
    groups = [(None, np.ones(len(m), bool))] if unit is None else [(u, (m[unit] == u).to_numpy()) for u in m[unit].unique()]
    dist, exp = np.array([1.0]), 0.0
    for _, g in groups:
        N, K, n = int((nv & g).sum()), int((nv & harm & g).sum()), int((br & g).sum())
        if n == 0:
            continue
        dist = np.convolve(dist, hypergeom.pmf(np.arange(n + 1), N, K, n)); exp += n * K / N
    k = int((br & harm).sum())
    return exp, float(dist[:k + 1].sum())


def tradeoff(ref, nv, br):
    np.seterr(invalid="ignore", divide="ignore")
    gn, gb = ref[nv].sum(), ref[br].sum()
    ln, lb = -np.minimum(ref[nv], 0).sum(), -np.minimum(ref[br], 0).sum()
    hn, hb = (ref[nv] <= 0).sum(), (ref[br] <= 0).sum()
    return dict(gain_kept=100 * gb / gn, harmful_removed=100 * (hn - hb) / hn, loss_removed=100 * (ln - lb) / ln if ln > 0 else np.nan,
                gain_per_acc_bridge=gb / max(br.sum(), 1), gain_per_acc_naive=gn / max(nv.sum(), 1))


def diff_ci(m, nv, br, unit):
    """Return BRIDGE minus naive harmful proportion in percentage points with a cluster-bootstrap 95% percentile interval."""
    rng = np.random.default_rng(SEED); harm = (m.ref <= 0).to_numpy()
    idx = {u: np.flatnonzero((m[unit] == u).to_numpy()) for u in m[unit].unique()}; keys = list(idx)
    f = lambda s: 100 * (harm[s][br[s]].mean() - harm[s][nv[s]].mean())
    d = []
    for _ in range(B):
        s = np.concatenate([idx[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        if br[s].any() and nv[s].any():
            d.append(f(s))
    allr = np.arange(len(m))
    return dict(diff_pp=f(allr), diff_lo=np.percentile(d, 2.5), diff_hi=np.percentile(d, 97.5))


def cluster_ci(m, nv, br, unit):
    rng = np.random.default_rng(SEED); ref = m.ref.to_numpy()
    idx = {u: np.flatnonzero((m[unit] == u).to_numpy()) for u in m[unit].unique()}; keys = list(idx)
    out = []
    for _ in range(B):
        s = np.concatenate([idx[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        out.append(tradeoff(ref[s], nv[s], br[s]))
    d = pd.DataFrame(out)
    return {f"{c}_lo": np.nanpercentile(d[c], 2.5) for c in d} | {f"{c}_hi": np.nanpercentile(d[c], 97.5) for c in d}


def main():
    m24 = SP.rd("mini24_m1.csv", "mini24_m2.csv", "mini24_m3.csv")
    q25 = SP.rd("mini25_m1.csv", "mini25_m2.csv", "mini25_m3.csv"); q25 = q25.assign(task=np.arange(len(q25)))
    m30 = SP.rd("mini30_m1.csv", "mini30_m2.csv", "mini30_m3.csv"); m30 = m30[~m30.kelas.isin(["error", "skip"])]
    m30 = m30[m30.metric == "nb20"].reset_index(drop=True)
    m43 = pd.concat([pd.read_csv(HERE / "out" / f"mini43_m{i}.csv") for i in (1, 2, 3)])  # (D): MINI-43, 39 datasets
    m43 = m43.assign(cfg=m43.chosen.where(m43.naive_accept == 1, "None"), lo95=m43.lower_bound, ref=m43.gain_ref).reset_index(drop=True)
    rows = []
    for lbl, m, unit, within in (("11 new datasets", m24, "dataset", "dataset"), ("Synthetic population Q", q25, "task", None),
                                 ("Clinical, net benefit", m30, "sumber", "dataset"), ("39 supporting datasets", m43, "dataset", "dataset")):
        nv, br, harm = (m.cfg != "None").to_numpy(), (m.lo95 > .005).to_numpy(), (m.ref <= 0).to_numpy()
        assert not (br & ~nv).any()  # accepted updates of BRIDGE are a subset of the naive rule's
        eg, pg = random_control(m, nv, br, harm)
        ew, pw = random_control(m, nv, br, harm, within) if within else (np.nan, np.nan)
        r = dict(setting=lbl, decisions=len(m), naive_accepted=int(nv.sum()), naive_harmful=int((nv & harm).sum()),
                 bridge_accepted=int(br.sum()), bridge_harmful=int((br & harm).sum()),
                 random_expected_harmful=eg, random_p=pg, random_within_unit_expected=ew, random_within_unit_p=pw)
        r |= tradeoff(m.ref.to_numpy(), nv, br) | cluster_ci(m, nv, br, unit) | diff_ci(m, nv, br, within or unit)
        if unit != (within or unit):  # clinical: also report the coarser source-level interval with 6 clusters
            r |= {f"{k}_by_{unit}": v for k, v in diff_ci(m, nv, br, unit).items()}
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "out" / "analysis_selectivity.csv", index=False)
    pd.set_option("display.width", 250)
    print(d.round(4).T.to_string())


if __name__ == "__main__":
    main()

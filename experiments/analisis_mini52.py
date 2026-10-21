"""Locked MINI-52 analysis (PRAREG_MINI52.md): candidate-space size and the eight ablation cells on (E), binary."""
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_mini44 as A44

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
B, SEED = 10_000, 20261024
SPACES = ("K173", "Ksedang", "Kbesar")
RULES = {
    "naive": ("pt_pool", 0.0),
    "same_pool_boot_005": ("lo_pool_boot", 0.005),
    "BRIDGE": ("lo_cf_boot", 0.005),
}
ABLATION = {
    "Naive": ("pt_pool", 0.0),
    "Fixed margin": ("pt_pool", 0.005),
    "Same-POOL bootstrap": ("lo_pool_boot", 0.0),
    "Same-POOL cell": ("lo_pool_boot", 0.005),
    "Cross-fit, point": ("pt_cf", 0.0),
    "Cross-fit, point, margin": ("pt_cf", 0.005),
    "BRIDGE without margin": ("lo_cf_boot", 0.0),
    "BRIDGE": ("lo_cf_boot", 0.005),
}


def read_all():
    paths = sorted(OUT.glob("mini52_m*.csv"))
    d = pd.concat([pd.read_csv(p, float_precision="round_trip") for p in paths], ignore_index=True)
    assert len(d) == 3_300, len(d)
    assert (d.status == "ok").all()
    assert not d.duplicated(["data_id", "rep", "space"]).any()
    assert d[d.space == "K173"].identical.eq(1).all()
    d["dataset"] = d.data_id.astype(str) + ":" + d.dataset
    return d


def hp(acc, wrong, idx):
    a = acc[idx]
    return 100 * wrong[idx][a].mean() if a.any() else np.nan


def cluster_boot(x, fn):
    rng = np.random.default_rng(SEED)
    units = x.dataset.unique()
    groups = [np.flatnonzero((x.dataset == u).to_numpy()) for u in units]
    values = []
    for _ in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        value = fn(idx)
        if np.isfinite(value):
            values.append(value)
    return np.percentile(values, [2.5, 97.5]) if values else (np.nan, np.nan)


def criterion(x, acc, wrong):
    unit = x.dataset.astype(str).to_numpy()
    result = A44.crit(acc, wrong, unit)
    per = pd.Series(unit).value_counts()
    result["ds_yield5"] = int(sum(100 * (acc & (unit == u)).sum() / n >= 5 for u, n in per.items()))
    result["meets"] = bool(result["meets"] and result["ds_yield5"] >= 3)
    return result


def cell(x, column, threshold):
    has = x.cfg.notna().to_numpy() & (x.cfg.astype(str) != "None").to_numpy()
    score = x[column].to_numpy(float)
    return has, score, has & (score > threshold)


def main():
    d = read_all()
    rows, contrasts, ablation, cache = [], [], [], {}
    for space in SPACES:
        x = d[d.space == space].sort_values(["dataset", "rep"]).reset_index(drop=True)
        wrong = x.ref.to_numpy() <= 0
        k_bridge = int(cell(x, "lo_cf_boot", .005)[2].sum())
        for rule, (column, threshold) in RULES.items():
            has, score, acc = cell(x, column, threshold)
            c = criterion(x, acc, wrong)
            lo, hi = cluster_boot(x, lambda idx, a=acc, w=wrong: hp(a, w, idx))
            eligible = np.flatnonzero(has & np.isfinite(score))
            top = eligible[np.argsort(score[eligible])[::-1]][:k_bridge]
            matched = 100 * wrong[top].mean() if len(top) else np.nan
            rows.append(dict(setting="E", space=space, nK=int(x.nK.iloc[0]), rule=rule, decisions=len(x),
                             harm_ci_lo=lo, harm_ci_hi=hi, harm_at_bridge_yield=matched,
                             total_gain=float(x.ref.to_numpy()[acc].sum()), **c))
            cache[(space, rule)] = (x, acc, wrong)
        if space == "K173":
            for name, (column, threshold) in ABLATION.items():
                _, _, acc = cell(x, column, threshold)
                c = criterion(x, acc, wrong)
                ablation.append(dict(setting="E", cell=name, total_gain=float(x.ref.to_numpy()[acc].sum()), **c))
    for rule in RULES:
        xb, ab, wb = cache[("K173", rule)]
        for space in ("Ksedang", "Kbesar"):
            x, acc, wrong = cache[(space, rule)]
            assert (x[["dataset", "rep"]].to_numpy() == xb[["dataset", "rep"]].to_numpy()).all()
            lo, hi = cluster_boot(x, lambda idx, a=acc, w=wrong, aa=ab, ww=wb: hp(a, w, idx) - hp(aa, ww, idx))
            est = hp(acc, wrong, np.arange(len(x))) - hp(ab, wb, np.arange(len(x)))
            contrasts.append(dict(setting="E", rule=rule, space=space, minus_K173_pp=est, lo=lo, hi=hi))
    result, contrast, abl = pd.DataFrame(rows), pd.DataFrame(contrasts), pd.DataFrame(ablation)
    result.to_csv(OUT / "analysis_mini52.csv", index=False)
    contrast.to_csv(OUT / "analysis_mini52_contrasts.csv", index=False)
    abl.to_csv(OUT / "analysis_mini52_ablation.csv", index=False)
    pd.set_option("display.width", 280)
    print(result.round(3).to_string(index=False))
    print("\nCONTRASTS")
    print(contrast.round(3).to_string(index=False))
    print("\nABLATION (K173)")
    print(abl.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Locked MINI-51 analysis (PRAREG_MINI51.md)."""
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


def read_all():
    paths = sorted(OUT.glob("mini51_m*.csv"))
    if not paths:
        raise RuntimeError("no MINI-51 shards")
    d = pd.concat([pd.read_csv(p, float_precision="round_trip") for p in paths], ignore_index=True)
    assert len(d) == 17_400, len(d)
    assert (d.status == "ok").all()
    assert not d.duplicated(["setting", "dataset", "rep", "space"]).any()
    assert d[d.space == "K173"].identical.eq(1).all()
    assert d[["setting", "dataset", "rep"]].drop_duplicates().shape[0] == 5_800
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
    return np.percentile(values, [2.5, 97.5])


def criterion(setting, x, acc, wrong):
    unit = (x.source if setting == "C" else x.dataset).astype(str).to_numpy()
    result = A44.crit(acc, wrong, unit)
    task = x.dataset.astype(str).to_numpy()
    per = pd.Series(task).value_counts()
    result["ds_yield5"] = int(sum(100 * (acc & (task == u)).sum() / n >= 5 for u, n in per.items()))
    if setting == "C":
        result["r0"] = np.nan
        result["meets"] = bool(result["kl"] < 10 and result["loco"] < 10 and result["yield_"] >= 5
                               and result["ds_yield5"] >= 3)
    else:
        result["meets"] = bool(result["meets"] and result["ds_yield5"] >= 3)
    return result


def main():
    d = read_all()
    rows, contrasts = [], []
    for setting, whole in d.groupby("setting"):
        base_cache = {}
        for space in SPACES:
            x = whole[whole.space == space].sort_values(["dataset", "rep"]).reset_index(drop=True)
            has = x.cfg.notna().to_numpy() & (x.cfg.astype(str) != "None").to_numpy()
            wrong = x.ref.to_numpy() <= 0
            bridge = has & (x.lo_cf_boot.to_numpy() > .005)
            k_bridge = int(bridge.sum())
            for rule, (column, threshold) in RULES.items():
                score = x[column].to_numpy(float)
                acc = has & (score > threshold)
                c = criterion(setting, x, acc, wrong)
                lo, hi = cluster_boot(x, lambda idx, a=acc, w=wrong: hp(a, w, idx))
                eligible = np.flatnonzero(has & np.isfinite(score))
                order = eligible[np.argsort(score[eligible])[::-1]]
                top = order[:k_bridge]
                matched = 100 * wrong[top].mean() if len(top) else np.nan
                row = dict(setting=setting, space=space, nK=int(x.nK.iloc[0]), rule=rule,
                           decisions=len(x), harm_ci_lo=lo, harm_ci_hi=hi,
                           harm_at_bridge_yield=matched, total_gain=float(x.ref.to_numpy()[acc].sum()), **c)
                rows.append(row)
                base_cache[(space, rule)] = (x, acc, wrong)
        for rule in RULES:
            xb, ab, wb = base_cache[("K173", rule)]
            for space in ("Ksedang", "Kbesar"):
                x, acc, wrong = base_cache[(space, rule)]
                assert (x[["dataset", "rep"]].to_numpy() == xb[["dataset", "rep"]].to_numpy()).all()
                lo, hi = cluster_boot(x, lambda idx, a=acc, w=wrong, aa=ab, ww=wb:
                                      hp(a, w, idx) - hp(aa, ww, idx))
                est = hp(acc, wrong, np.arange(len(x))) - hp(ab, wb, np.arange(len(x)))
                contrasts.append(dict(setting=setting, rule=rule, space=space,
                                      minus_K173_pp=est, lo=lo, hi=hi))
    result, contrast = pd.DataFrame(rows), pd.DataFrame(contrasts)
    result.to_csv(OUT / "analysis_mini51.csv", index=False)
    contrast.to_csv(OUT / "analysis_mini51_contrasts.csv", index=False)
    pd.set_option("display.width", 280)
    print(result.round(3).to_string(index=False))
    print("\nCONTRASTS")
    print(contrast.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

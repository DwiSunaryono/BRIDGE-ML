"""Locked descriptive analysis for MINI-49 (PRAREG_MINI49.md). Tables are written to out/analysis_mini49_*.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_mini44 as A44

HERE = Path(__file__).resolve().parent
ARMS = ("HGB", "XGB0", "XGBES", "MLPES")
A44.SEED = 20261023


def load():
    d = pd.concat([
        pd.read_csv(HERE / "out" / f"mini49_m{i}.csv", float_precision="round_trip")
        for i in (1, 2, 3)
    ], ignore_index=True)
    assert len(d) == 1000
    assert (d.status == "ok").all()
    assert not d.duplicated(["dataset", "rep"]).any()
    assert d.dataset.nunique() == 50
    for arm in ARMS:
        assert (d[f"{arm}_identical"] == 1).all()
    d["S"] = d.setting.replace({"D-mc": "D"})
    return d


def boot_diff(m, values):
    return A44.boot(m, lambda s: 100 * np.mean(values[s]))[:3]


def main():
    d = load()
    overall = []
    per_dataset = []
    for setting, x in d.groupby("S"):
        x = x.reset_index(drop=True)
        hgb = x.HGB_bridge_wf1.to_numpy()
        for arm in ARMS:
            base = x[f"{arm}_base_wf1"].to_numpy()
            naive = x[f"{arm}_naive_wf1"].to_numpy()
            bridge = x[f"{arm}_bridge_wf1"].to_numpy()
            diff = bridge - hgb
            if arm == "HGB":
                est = lo = hi = 0.0
            else:
                est, lo, hi = boot_diff(x, diff)
            means = x.assign(
                arm_value=bridge, hgb_value=hgb, diff=diff
            ).groupby("dataset", as_index=False).agg(
                arm_mean=("arm_value", "mean"),
                hgb_mean=("hgb_value", "mean"),
                diff_mean=("diff", "mean"),
            )
            overall.append(dict(
                setting=setting, learner=arm, decisions=len(x),
                datasets=x.dataset.nunique(),
                baseline_mean=base.mean(), naive_final_mean=naive.mean(),
                bridge_final_mean=bridge.mean(),
                bridge_minus_hgb_pts=est, ci_lo=lo, ci_hi=hi,
                datasets_better_than_hgb=int((means.diff_mean > 0).sum()),
                datasets_equal_hgb=int(np.isclose(means.diff_mean, 0, atol=1e-12).sum()),
                datasets_worse_than_hgb=int((means.diff_mean < 0).sum()),
            ))
            for r in means.itertuples(index=False):
                per_dataset.append(dict(
                    setting=setting, dataset=r.dataset, learner=arm,
                    bridge_mean=r.arm_mean, hgb_mean=r.hgb_mean,
                    diff_from_hgb=r.diff_mean,
                ))
    per = pd.DataFrame(per_dataset)
    wide = per.pivot_table(index=["setting", "dataset"], columns="learner", values="bridge_mean").reset_index()
    vals = wide[list(ARMS)].to_numpy()
    mx = vals.max(axis=1)
    wide["best_value"] = mx
    wide["best_learner"] = [
        "+".join(a for a, v in zip(ARMS, row) if np.isclose(v, best, atol=1e-12))
        for row, best in zip(vals, mx)
    ]
    counts = (
        wide.assign(best_piece=wide.best_learner.str.split("+"))
        .explode("best_piece")
        .groupby(["setting", "best_piece"]).size()
        .rename("datasets_best").reset_index()
    )
    pd.DataFrame(overall).to_csv(HERE / "out" / "analysis_mini49_overall.csv", index=False)
    per.to_csv(HERE / "out" / "analysis_mini49_per_dataset_long.csv", index=False)
    wide.to_csv(HERE / "out" / "analysis_mini49_per_dataset.csv", index=False)
    counts.to_csv(HERE / "out" / "analysis_mini49_best_counts.csv", index=False)
    pd.set_option("display.width", 240)
    print("OVERALL")
    print(pd.DataFrame(overall).round(4).to_string(index=False))
    print("\nBEST LEARNER COUNTS")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()


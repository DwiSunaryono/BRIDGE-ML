"""Analysis of MINI-54 (PRAREG_MINI54.md): class-balanced metrics on the eight multiclass datasets of (D) and (E).

For each decision metric (weighted F1, balanced accuracy, macro-F1), the naive rule and BRIDGE are scored with the pass
criterion of the paper (analysis_mini44.crit: KL-ratio bound, leave-one-source-out bound, LOCO_0, yield, and at least
three datasets with a yield of 5), with datasets as sources. A false improvement is an accepted update whose reference
gain under the same metric is at most zero. Intervals: cluster bootstrap over datasets, B = 10,000, seed 20261028.
Inputs: out/mini54_*.csv. Usage: python analysis_mini54.py -> out/analysis_mini54.csv, out/analysis_mini54_datasets.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_mini44 as A44

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
A44.SEED, A44.B = 20261028, 10_000
METRICS = ("wF1", "bacc", "macroF1")


def load():
    d = pd.concat([pd.read_csv(f, float_precision="round_trip") for f in sorted(OUT.glob("mini54_*.csv"))
                   if not f.name.startswith("mini54_an")], ignore_index=True)
    d = d[d.status == "ok"].drop_duplicates(["collection", "dataset", "rep"]).reset_index(drop=True)
    d["dataset"] = d.collection + ":" + d.dataset.astype(str)
    return d


def main():
    d = load()
    assert len(d) == 480 and d.identical.eq(1).all(), "every trial must reproduce the stored weighted-F1 decision"
    rows, per = [], []
    for m in METRICS:
        wrong = (d[f"ref_{m}"] <= 0).to_numpy()
        for rule, col in (("naive", f"naive_{m}"), ("bridge", f"accept_{m}")):
            acc = d[col].eq(1).to_numpy()
            c = A44.crit(acc, wrong, d.dataset)
            est, lo, hi, _ = A44.boot(d, A44.hp(acc, wrong))
            rows.append(dict(metric=m, rule=rule, decisions=len(d), **c, fir_lo=lo, fir_hi=hi))
            for ds, g in d.groupby("dataset"):
                a = g[col].eq(1); w = g[f"ref_{m}"] <= 0
                per.append(dict(metric=m, rule=rule, dataset=ds, K=int(g.K.iloc[0]), accepted=int(a.sum()), harmful=int((a & w).sum())))
    r = pd.DataFrame(rows); r.to_csv(OUT / "analysis_mini54.csv", index=False)
    pd.DataFrame(per).to_csv(OUT / "analysis_mini54_datasets.csv", index=False)
    pd.set_option("display.width", 200)
    print(r[["metric", "rule", "accepted", "harmful", "harm_pct", "fir_lo", "fir_hi", "kl", "loco", "r0", "yield_", "ds_yield5", "meets"]].round(2).to_string(index=False))
    print(f"\nmean seconds per trial: {d.seconds.mean():.1f} (max {d.seconds.max():.1f}); identical weighted-F1 decisions: {int(d.identical.sum())}/{len(d)}")


if __name__ == "__main__":
    main()

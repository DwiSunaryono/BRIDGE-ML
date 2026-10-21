"""Locked analysis of MINI-48 (PRAREG_MINI48.md): comparator rules, honest holdouts and the eight ablation cells on
(C) the 12 clinical tasks (net benefit p_t = 0.2; LOCO by source, clusters = tasks) and (D) the 39 supporting datasets
(weighted-F1; LOCO and clusters = datasets). Per rule: accepted, harmful, % harmful, KL-ratio bound (b = 1), LOCO, LOCO_0,
yield, datasets with yield >= 5, criterion; BRIDGE minus each comparator with a 95% cluster-bootstrap interval
(B = 10,000, seed 20261022) and Holm over the comparators of a setting.
Usage: python analysis_mini48.py"""
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_mini44 as A44


def crit_s(S, acc, wrong, unit, task):
    """Pre-registered criterion (Table criterion): on the clinical runs LOCO is by source, the power check LOCO_0 does not
    apply, and the yield condition counts tasks."""
    c = A44.crit(acc, wrong, unit)
    if S == "C":
        tk = pd.Series(task); per = tk.value_counts(); acc = np.asarray(acc, bool)
        c["ds_yield5"] = int(sum(100 * (acc & (tk.to_numpy() == u)).sum() / n >= 5 for u, n in per.items()))
        c["meets"] = bool(c["kl"] < 10 and c["loco"] < 10 and c["yield_"] >= 5 and c["ds_yield5"] >= 3)
    return c

HERE = Path(__file__).resolve().parent
A44.SEED = 20261022
RULES = ["naive", "mcnemar_pool", "mcnemar_cf", "t5x2", "crt_final", "crt", "boot0", "ho23", "ho12", "bridge"]
CELLS = {  # (evidence, statistic, margin) -> column and threshold
    ("same POOL", "point", 0): ("pt_pool", 0.0), ("same POOL", "point", 1): ("pt_pool", .005),
    ("same POOL", "bootstrap", 0): ("lo_pool_boot", 0.0), ("same POOL", "bootstrap", 1): ("lo_pool_boot", .005),
    ("cross-fit", "point", 0): ("pt_cf", 0.0), ("cross-fit", "point", 1): ("pt_cf", .005),
    ("cross-fit", "bootstrap", 0): ("lo_cf_boot", 0.0), ("cross-fit", "bootstrap", 1): ("lo_cf_boot", .005)}


def main():
    m = pd.concat([pd.read_csv(HERE / "out" / f"mini48_m{i}.csv", float_precision="round_trip") for i in (1, 2, 3)])
    assert (m.status == "ok").all(); print("identity check:", int(m.identical.sum()), "of", len(m))
    m["S"] = m.setting.replace({"D-mc": "D"}); rows, cells = [], []
    for S, x in m.groupby("S"):
        x = x.reset_index(drop=True); unit = (x.source if S == "C" else x.dataset).astype(str).to_numpy()
        ref = x.ref.to_numpy(); acc_b = (x.bridge == 1).to_numpy()
        def rule(r):
            if r in ("ho23", "ho12"):
                return (x[r] == 1).to_numpy(), x[f"{r}_ref"].to_numpy()
            return (x[r] == 1).to_numpy(), ref
        res = {}
        for r in RULES:
            acc, rr = rule(r); c = crit_s(S, acc, rr <= 0, unit, x.dataset.to_numpy())
            res[r] = dict(setting=S, rule=r, decisions=len(x), **c, total_gain=float(np.nansum(rr[acc])))
            if r != "bridge":
                e, lo, hi, tb = A44.boot(x, lambda s, acc=acc, rr=rr: A44.hp(acc_b, ref <= 0)(s) - A44.hp(acc, rr <= 0)(s))
                p = 2 * min((tb >= 0).mean(), (tb <= 0).mean())
                res[r] |= dict(bridge_minus_pp=e, lo=lo, hi=hi, p=min(1.0, max(p, 1 / len(tb))))
        d = pd.DataFrame(res.values()); v = d.p.notna(); d.loc[v, "p_holm"] = A44.holm(d.loc[v, "p"]); rows.append(d)
        for (ev, st, mg), (col, thr) in CELLS.items():
            acc = (x[col].to_numpy() > thr) & x.cfg.notna().to_numpy() & (x.cfg.astype(str) != "None").to_numpy()
            c = crit_s(S, acc, ref <= 0, unit, x.dataset.to_numpy())
            cells.append(dict(setting=S, evidence=ev, statistic=st, margin=mg, **c))
    d = pd.concat(rows); a = pd.DataFrame(cells)
    d.to_csv(HERE / "out" / "analysis_mini48_rules.csv", index=False); a.to_csv(HERE / "out" / "analysis_mini48_ablation.csv", index=False)
    pd.set_option("display.width", 250)
    print(d.round(3).to_string(index=False)); print(); print(a.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Margin sensitivity of the frozen gate, recomputed from stored per-trial outputs (no new runs).

The deployed configuration does not depend on the margin, and every trial stores the 5% bootstrap
lower bound (lo95), so the gate decision at any margin eps is lo95 > eps. For each setting and eps
in {0.0025, 0.005, 0.0075, 0.01} the script reports accepted, harmful, the risk bound, LOCO (real
data) and whether the pre-registered criterion of that setting is met.
Real non-clinical settings: KL-ratio bound, LOCO, R0 power check < 10%, yield >= 5/100, >= 3
datasets with yield >= 5/100.
Clinical (primary metric): KL-ratio bound and LOCO-source < 10%, yield >= 5/100, >= 3 tasks with
yield >= 5/100.
Population Q: Proposition 1 bound < 10% and yield >= 5/100.
Usage: python analysis_margin_sweep.py -> prints and writes out/analysis_margin_sweep.csv
"""
from pathlib import Path

import pandas as pd

import summarize_paper as SP
from verify_exp41 import kl_ratio_up as up

HERE = Path(__file__).resolve().parent
EPS = (0.0025, 0.005, 0.0075, 0.01)


def real(m, eps, unit="dataset"):
    acc = m.lo95 > eps; wrong = m.ref <= 0; NT = len(m); per = m.groupby(unit).size()
    n, k = int(acc.sum()), int((acc & wrong).sum())
    loco = SP.loco(acc, wrong, m[unit], per, NT)
    r0 = max(up(0, int((acc & (m[unit] != x)).sum()), NT - nx, 1) if (acc & (m[unit] != x)).sum() else 1.0
             for x, nx in per.items())
    nds = sum(100 * (acc & (m.dataset == x)).sum() / (m.dataset == x).sum() >= 5 for x in m.dataset.unique())
    b = up(k, n, NT, 1)
    ok = b < .1 and loco < .1 and 100 * n / NT >= 5 and nds >= 3 and (r0 < .1 if unit == "dataset" else True)
    return n, k, 100 * b, 100 * loco, 100 * n / NT, bool(ok), 100 * r0, int(nds)


def main():
    S = []
    m21 = SP.rd("mini21_m1.csv", "mini21_m2.csv", "mini21_m3.csv")
    for des, lbl in (("union_wF1", "New datasets, weighted F1"), ("union_bacc", "New datasets, balanced accuracy"),
                     ("union_macroF1", "New datasets, macro-F1"), ("fsthr_KNN", "New datasets, feature selection (KNN)"),
                     ("fsthr_SVC", "New datasets, feature selection (SVC)")):
        S.append((lbl, m21[m21.design == des], "real"))
    m20 = SP.rd("mini20_size_a.csv", "mini20_size_b.csv"); m20 = m20[m20.kelas != "skip"]
    for T, s in m20.groupby("TOTAL"):
        S.append((f"{26 if T == 1000 else 27} datasets, {T:,} training rows", s, "real"))
    S.append(("5 new multiclass datasets", SP.rd("mini22_m1.csv", "mini22_m2.csv", "mini22_m3.csv"), "real"))
    m30 = SP.rd("mini30_m1.csv", "mini30_m2.csv", "mini30_m3.csv"); m30 = m30[~m30.kelas.isin(["error", "skip"])]
    S.append(("Clinical, net benefit p_t=0.2", m30[m30.metric == "nb20"], "clinical"))
    q = SP.rd("mini25_m1.csv", "mini25_m2.csv", "mini25_m3.csv")
    assert ((q.lo95 > .005).astype(int) == q.bridge).all()  # stored decisions equal lo95 > 0.005
    S.append(("Synthetic population $\\mathcal{Q}$", q, "Q"))
    rows = []
    for lbl, m, kind in S:
        for e in EPS:
            if kind == "Q":
                a = m[m.lo95 > e]; n, k = len(a), int((a.ref <= 0).sum()); b = 100 * SP.prop1(k, n)
                rows.append(dict(setting=lbl, eps=e, trials=len(m), accepted=n, harmful=k, bound=b, loco=float("nan"),
                                 yield_=100 * n / len(m), r0_power=float("nan"), units_with_yield_ge5=float("nan"),
                                 meets=bool(b < 10 and 100 * n / len(m) >= 5)))
            else:
                n, k, b, lo, y, ok, r0, nds = real(m, e, "sumber" if kind == "clinical" else "dataset")
                rows.append(dict(setting=lbl, eps=e, trials=len(m), accepted=n, harmful=k, bound=b, loco=lo, yield_=y, r0_power=r0,
                                 units_with_yield_ge5=nds, meets=ok))
    r = pd.DataFrame(rows).rename(columns={"yield_": "yield"})
    r.to_csv(HERE / "out" / "analysis_margin_sweep.csv", index=False)
    pd.set_option("display.width", 200)
    print(r.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

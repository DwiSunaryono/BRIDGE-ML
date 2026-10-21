"""Paired comparison of every tested margin against the default eps = 0.005 (post hoc reanalysis of
stored outputs, no new runs; ruler section 29.2). Specification fixed and committed before the first
computation:
- settings: (A) 11 new datasets (MINI-24), (B) synthetic population Q (MINI-25), (C) clinical tasks,
  net benefit (MINI-30), (D) 39 supporting datasets (MINI-43); the decision at margin eps is
  lo95 > eps on the same trials;
- variants: eps in {0.0025, 0.0075, 0.01} against 0.005 (3 x 4 = 12 comparisons);
- primary outcome: harmful proportion among accepted updates (percentage points, variant minus
  default);
- interval: cluster bootstrap over datasets (A, D), clinical tasks (C) or tasks (Q), B = 10,000,
  seed 20261017;
- equivalence: TOST with equivalence margin +-2 percentage points, bootstrap p = max(P*(d <= -2),
  P*(d >= 2)); Holm correction over the 12 TOST p values; equivalent when the Holm-adjusted
  p < 0.05;
- secondary (descriptive, no test): accepted per 100 decisions and net reference gain kept,
  variant relative to default.
Usage: python analysis_margin_paired.py -> prints and writes out/analysis_margin_paired.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import summarize_paper as SP

HERE = Path(__file__).resolve().parent
B, SEED, EQ, DEF = 10_000, 20261017, 2.0, 0.005
VAR = (0.0025, 0.0075, 0.01)


def load():
    m24 = SP.rd("mini24_m1.csv", "mini24_m2.csv", "mini24_m3.csv")
    q25 = SP.rd("mini25_m1.csv", "mini25_m2.csv", "mini25_m3.csv"); q25 = q25.assign(task=np.arange(len(q25)))
    m30 = SP.rd("mini30_m1.csv", "mini30_m2.csv", "mini30_m3.csv"); m30 = m30[~m30.kelas.isin(["error", "skip"])]
    m30 = m30[m30.metric == "nb20"].reset_index(drop=True)
    m43 = pd.concat([pd.read_csv(HERE / "out" / f"mini43_m{i}.csv") for i in (1, 2, 3)]).reset_index(drop=True)
    m43 = m43.rename(columns={"lower_bound": "lo95", "gain_ref": "ref"})
    return (("(A) 11 new datasets", m24, "dataset"), ("(B) Synthetic tasks", q25, "task"),
            ("(C) Clinical tasks", m30, "dataset"), ("(D) 39 supporting datasets", m43, "dataset"))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(run, 1.0)
    return adj


def main():
    rng = np.random.default_rng(SEED); rows = []
    for lbl, m, unit in load():
        lo, ref = m.lo95.to_numpy(), m.ref.to_numpy(); harm = ref <= 0
        idx = [np.flatnonzero((m[unit] == u).to_numpy()) for u in m[unit].unique()]
        draws = [np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]) for _ in range(B)]
        hp = lambda s, e: 100 * harm[s][lo[s] > e].mean() if (lo[s] > e).any() else np.nan
        for e in VAR:
            d = np.array([hp(s, e) - hp(s, DEF) for s in draws]); d = d[~np.isnan(d)]
            allr = np.arange(len(m)); a0, a1 = lo > DEF, lo > e
            rows.append(dict(setting=lbl, eps=e, decisions=len(m), acc_default=int(a0.sum()), harm_default=int((a0 & harm).sum()),
                             acc_variant=int(a1.sum()), harm_variant=int((a1 & harm).sum()),
                             diff_pp=hp(allr, e) - hp(allr, DEF), ci95_lo=np.percentile(d, 2.5), ci95_hi=np.percentile(d, 97.5),
                             ci90_lo=np.percentile(d, 5), ci90_hi=np.percentile(d, 95),
                             p_tost=max((d <= -EQ).mean(), (d >= EQ).mean()),
                             yield_ratio=a1.sum() / a0.sum(), gain_ratio=ref[a1].sum() / ref[a0].sum()))
    d = pd.DataFrame(rows); d["p_tost_holm"] = holm(d.p_tost); d["equivalent"] = d.p_tost_holm < .05
    d.to_csv(HERE / "out" / "analysis_margin_paired.csv", index=False)
    pd.set_option("display.width", 250); print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

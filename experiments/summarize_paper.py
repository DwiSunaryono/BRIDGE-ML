"""Regenerate every results table of the paper from the CSVs in out/, without model training.

    python summarize_paper.py

Prints Table 1 (acceptance rules on identical decisions, 11 new real datasets), yield-matched
comparison, dataset-level bootstrap, Table 2 (i.i.d. synthetic task population with Proposition 1
bound and conservative bound), Table 3 (BRIDGE vs honest holdout, if MINI-26 outputs exist), and
Table 4 (generalisation claims).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_exp41 import kl_ratio_up  # noqa: E402 (KL-ratio bound used by the experiments)

OUT = HERE / "out"
RULES = ("naif", "mcnemar_pool", "mcnemar_cf", "t5x2", "crt_final", "crt", "boot0", "bridge")
NAMES = {"naif": "naive (validation improves)", "mcnemar_pool": "McNemar on same POOL", "mcnemar_cf": "McNemar on cross-fit",
         "t5x2": "5x2cv paired t", "crt_final": "corrected resampled t", "crt": "corrected t + per-split re-selection",
         "boot0": "BRIDGE without margin", "bridge": "BRIDGE", "ho23": "selection + honest holdout (2/3-1/3)",
         "ho12": "selection + honest holdout (1/2-1/2)"}


def rd(*files):
    return pd.concat([pd.read_csv(OUT / f, keep_default_na=False, float_precision="round_trip",
                                  na_values={c: ["nan", "NaN"] for c in ("lo95", "ref", "ho23_lo95", "ho12_lo95")})
                      for f in files], ignore_index=True)


def kl(p, q):
    q = min(max(q, 1e-15), 1 - 1e-15)
    return (p * np.log(p / q) if p > 0 else 0.0) + ((1 - p) * np.log((1 - p) / (1 - q)) if p < 1 else 0.0)


def prop1(S, M, delta=.05):
    """Proposition 1: S | M ~ Binomial(M, r); upper Chernoff-KL bound on r at level 1 - delta."""
    if M == 0:
        return 1.0
    ph = S / M
    f = lambda u: M * kl(ph, u) - np.log(1 / delta)
    return 1.0 if f(1 - 1e-12) <= 0 else brentq(f, ph if ph >= 1 else ph + 1e-15, 1 - 1e-12)


def loco(acc, wrong, ds, n_trials_per_ds, NT):
    vals = []
    for x, nx in n_trials_per_ds.items():
        keep = ds != x
        vals.append(kl_ratio_up(int((acc & wrong & keep).sum()), int((acc & keep).sum()), NT - nx, 1) if (acc & keep).sum() else 1.0)
    return max(vals)


def rule_table(m, rules, ref_col=lambda r: "ref"):
    NT = len(m); per = m.groupby("dataset").size()
    rows = []
    for r in rules:
        acc = m[r] == 1; rc = ref_col(r); wrong = m[rc] <= 0
        n, k = int(acc.sum()), int((acc & wrong).sum())
        rows.append(dict(rule=NAMES.get(r, r), accepted=n, harmful=k, pct=round(100 * k / max(n, 1), 2),
                         KL_bound=round(100 * kl_ratio_up(k, n, NT, 1), 2),
                         LOCO=round(100 * loco(acc, wrong, m.dataset, per, NT), 2),
                         yield_per_100=round(100 * n / NT, 2), total_gain=round(m.loc[acc, rc].sum(), 2),
                         total_loss=round(-m.loc[acc & (m[rc] < 0), rc].sum(), 3)))  # Positive: summed metric loss of accepted updates
    return pd.DataFrame(rows)


def yield_matched(m, Ns):
    stat = {"mcnemar_pool": -m.p_mcn_pool, "t5x2": -m.p_5x2, "crt_final": -m.p_crt_final, "crt": -m.p_crt, "bridge": m.lo95}
    ok = m.cfg != "None"; rows = []
    for N in Ns:
        for r, s in stat.items():
            a = m.loc[s[ok].sort_values(ascending=False).index[:N]]
            rows.append(dict(N=N, rule=NAMES[r], harmful=int((a.ref <= 0).sum()), total_gain=round(a.ref.sum(), 2)))
    return pd.DataFrame(rows).pivot(index="rule", columns="N")


def holdout_matched(m, Ns):
    rows = []
    for N in Ns:
        for r, st, rc, c in (("bridge", "lo95", "ref", "cfg"), ("ho23", "ho23_lo95", "ho23_ref", "ho23_cfg"),
                             ("ho12", "ho12_lo95", "ho12_ref", "ho12_cfg")):
            ok = m[c] != "None"; a = m.loc[m.loc[ok, st].dropna().sort_values(ascending=False).index[:N]]
            rows.append(dict(N=N, rule=NAMES[r], harmful=int((a[rc] <= 0).sum()), total_gain=round(a[rc].sum(), 2)))
    return pd.DataFrame(rows).pivot(index="rule", columns="N")


def main():
    pd.set_option("display.width", 200)
    real = rd("mini24_m1.csv", "mini24_m2.csv", "mini24_m3.csv")
    print("== TABLE 1: acceptance rules on identical decisions, 11 new real datasets (MINI-23/24) ==")
    print(rule_table(real, RULES).to_string(index=False))
    print("\n== Yield-matched (descriptive): top-N accepts by each rule's own statistic, real data ==")
    print(yield_matched(real, (150, 284, 400)).to_string())
    print("\n== Dataset-level bootstrap (11 datasets resampled 10,000x): 95% upper bound of risk among accepts ==")
    rng = np.random.default_rng(0)
    for r in ("naif", "t5x2", "crt_final", "crt", "boot0", "bridge"):
        g = real.groupby("dataset").apply(lambda s: pd.Series({"n": (s[r] == 1).sum(), "k": ((s[r] == 1) & (s.ref <= 0)).sum()}))
        idx = rng.integers(0, len(g), size=(10000, len(g))); n = g.n.values[idx].sum(1); k = g.k.values[idx].sum(1)
        print(f"  {NAMES[r]:38s} point {g.k.sum() / g.n.sum():.3f}  upper95 {np.nanquantile(np.where(n > 0, k / np.maximum(n, 1), np.nan), .95):.3f}")

    syn = rd("mini25_m1.csv", "mini25_m2.csv", "mini25_m3.csv")
    print("\n== TABLE 2: i.i.d. synthetic task population Q (MINI-25; 1 trial per task) ==")
    rows = []
    for r in RULES:
        a = syn[syn[r] == 1]; M = len(a); S = int((a.ref <= 0).sum()); Sc = int((a.kelas != "membaik").sum())
        rows.append(dict(rule=NAMES[r], accepted=M, harmful=S, prop1_bound=round(100 * prop1(S, M), 2),
                         conservative_harmful=Sc, prop1_conservative=round(100 * prop1(Sc, M), 2),
                         yield_per_100=round(100 * M / len(syn), 2), total_gain=round(a.ref.sum(), 2)))
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n== Yield-matched, population Q ==")
    print(yield_matched(syn, (300, 475, 600)).to_string())

    f26 = sorted(OUT.glob("mini26_nyata_m*.csv"))
    if len(f26) == 3:
        h = rd(*[p.name for p in f26])
        print("\n== TABLE 3a: BRIDGE vs selection + honest holdout, same data budget, 11 new real datasets (MINI-26) ==")
        print(rule_table(h, ("bridge", "ho23", "ho12"), lambda r: "ref" if r == "bridge" else r + "_ref").to_string(index=False))
        print("   yield-matched vs honest holdout (real data):")
        print(holdout_matched(h, (100, 150, 200)).to_string())
    f26q = sorted(OUT.glob("mini26_q_m*.csv"))
    if len(f26q) == 3:
        q = rd(*[p.name for p in f26q]); rows = []
        for r in ("bridge", "ho23", "ho12"):
            rc = "ref" if r == "bridge" else r + "_ref"; kc = "kelas" if r == "bridge" else r + "_kelas"
            a = q[q[r] == 1]; M = len(a); S = int((a[rc] <= 0).sum()); Sc = int((a[kc] != "membaik").sum())
            rows.append(dict(rule=NAMES[r], accepted=M, harmful=S, prop1_bound=round(100 * prop1(S, M), 2),
                             prop1_conservative=round(100 * prop1(Sc, M), 2), yield_per_100=round(100 * M / len(q), 2),
                             total_gain=round(a[rc].sum(), 2)))
        print("\n== TABLE 3b: same comparison on population Q (MINI-26) ==")
        print(pd.DataFrame(rows).to_string(index=False))
        print("   yield-matched vs honest holdout (population Q):")
        print(holdout_matched(q, (200, 261, 339)).to_string())

    print("\n== TABLE 4: generalisation claims ==")
    m21 = rd("mini21_m1.csv", "mini21_m2.csv", "mini21_m3.csv")
    for des in ("union_wF1", "union_bacc", "union_macroF1", "fsthr_KNN", "fsthr_SVC"):
        s = m21[m21.design == des].copy(); s["acc"] = (s.lo95 > .005).astype(int)
        print(rule_table(s, ("acc",)).assign(rule=f"MINI-21 {des}").to_string(index=False, header=des == "union_wF1"))
    m20 = rd("mini20_size_a.csv", "mini20_size_b.csv"); m20 = m20[m20.kelas != "skip"]
    for T, s in m20.groupby("TOTAL"):
        s = s.copy(); s["acc"] = (s.lo95 > .005).astype(int)
        print(rule_table(s, ("acc",)).assign(rule=f"MINI-20 train={T}").to_string(index=False, header=False))
    m22 = rd("mini22_m1.csv", "mini22_m2.csv", "mini22_m3.csv"); m22["acc"] = (m22.lo95 > .005).astype(int)
    print(rule_table(m22, ("acc",)).assign(rule="MINI-22 multiclass").to_string(index=False, header=False))
    m11 = rd("mini11_confirm.csv"); m11 = m11[m11.design == "union"].copy(); m11["acc"] = (m11.lo95 > .005).astype(int)
    print(rule_table(m11, ("acc",)).assign(rule="MINI-11 union (34 ds)").to_string(index=False, header=False))


if __name__ == "__main__":
    main()

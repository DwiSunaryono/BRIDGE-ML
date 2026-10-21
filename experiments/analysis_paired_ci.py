"""Paired cluster-bootstrap tests behind the manuscript's pairwise risk claims, run on stored outputs.

(1) Each rule at its own operating point vs BRIDGE: % harmful(BRIDGE) minus % harmful(comparator), on 11 new real
    datasets (MINI-24, cluster = dataset) and population Q (MINI-25, cluster = task). B = 10,000; percentile 95% CI,
    Bonferroni-widened CI, and Holm-adjusted bootstrap p over the 6-comparator family.
(2) Ablation-superior cell (same-POOL bootstrap lower bound + margin, MINI-32) vs BRIDGE, at each rule's own operating
    point and at matched yield (top-k by each rule's own statistic, k = BRIDGE's accept count in the resample).
(3) BRIDGE vs the best fixed validation margin (accept when the point POOL gain exceeds t, t = 0.025 and 0.02) on the
    training-size grid (MINI-20/29; 1,000/2,430/6,000 rows), the 11 new datasets (weighted F1) and Q: difference in
    % harmful and in yield (accepted per 100 trials), cluster = dataset (task on Q).
Usage: python analysis_paired_ci.py writes out/paired_ci.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from summarize_paper import rd
B, SEED = 10000, 20261012
RULES = [("naif", "naive"), ("mcnemar_pool", "McNemar on same POOL"), ("t5x2", "5x2cv paired t"), ("crt_final", "corrected resampled t"),
         ("crt", "corrected t + per-split reselection"), ("boot0", "BRIDGE without margin")]


def harm(m, mask):
    a = m[mask]; return 100 * (a.ref <= 0).mean() if len(a) else np.nan


def boot_p(x):
    x = x[~np.isnan(x)]; return min(1.0, 2 * min(np.mean(x >= 0), np.mean(x <= 0)))


def holm(p):
    order = np.argsort(p); out = np.empty(len(p)); run = 0.0
    for rank, k in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[k])); out[k] = run
    return out


rows = []
for data, m in (("real", rd("mini24_m1.csv", "mini24_m2.csv", "mini24_m3.csv")), ("Q", rd("mini25_m1.csv", "mini25_m2.csv", "mini25_m3.csv"))):
    for c in [r for r, _ in RULES] + ["bridge"]: m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0).astype(int)
    m = m.reset_index(drop=True); groups = [g.index.values for _, g in m.groupby("dataset")]; G = len(groups)
    rng = np.random.default_rng(SEED); bt = np.empty((B, len(RULES)))
    for b in range(B):
        mm = m.loc[np.concatenate([groups[i] for i in rng.integers(0, G, G)])]
        bt[b] = [harm(mm, mm.bridge == 1) - harm(mm, mm[r] == 1) for r, _ in RULES]
    est = [harm(m, m.bridge == 1) - harm(m, m[r] == 1) for r, _ in RULES]; p = np.array([boot_p(bt[:, k]) for k in range(len(RULES))]); ph = holm(p)
    for k, (r, lab) in enumerate(RULES):
        lo, hi = np.nanpercentile(bt[:, k], [2.5, 97.5]); a = 2.5 / len(RULES); lob, hib = np.nanpercentile(bt[:, k], [a, 100 - a])
        rows.append(dict(analysis="operating point", data=data, clusters=G, comparator=lab, diff_pp=est[k], ci95_lo=lo, ci95_hi=hi,
                         ci_bonf_lo=lob, ci_bonf_hi=hib, p_boot=p[k], p_holm=ph[k]))

# (2) Ablation-superior cell vs BRIDGE (MINI-32 stored outputs)
for data, m in (("real", pd.concat([rd("mini32_nyata_m1.csv"), rd("mini32_nyata_m2.csv")], ignore_index=True)), ("Q", rd("mini32_Q.csv"))):
    m = m.reset_index(drop=True); m["lo_pool_boot"] = pd.to_numeric(m.lo_pool_boot, errors="coerce"); ch = m.cfg != "None"
    m["acc_br"] = (ch & (m.lo_cf_boot > .005)).astype(int); m["acc_pb"] = (ch & (m.lo_pool_boot > .005)).astype(int)
    groups = [g.index.values for _, g in m.groupby("dataset")]; G = len(groups); rng = np.random.default_rng(SEED)

    def matched(mm):
        k = int(mm.acc_br.sum()); c = mm[mm.cfg != "None"]
        top = lambda col: c.sort_values(col, ascending=False).head(k)
        return 100 * (top("lo_cf_boot").ref <= 0).mean() - 100 * (top("lo_pool_boot").ref <= 0).mean()
    op = lambda mm: harm(mm, mm.acc_br == 1) - harm(mm, mm.acc_pb == 1)
    for lab, f in (("operating point", op), ("matched yield", matched)):
        bt = np.empty(B)
        for b in range(B):
            bt[b] = f(m.loc[np.concatenate([groups[i] for i in rng.integers(0, G, G)])])
        lo, hi = np.nanpercentile(bt, [2.5, 97.5])
        rows.append(dict(analysis=f"ablation cell, {lab}", data=data, clusters=G, comparator="same-POOL bootstrap + margin", diff_pp=f(m),
                         ci95_lo=lo, ci95_hi=hi, p_boot=boot_p(bt)))
# (3) BRIDGE vs best fixed validation margin. MINI-20/21/25 decisions are merged with the MINI-29 point POOL gains.
rdn = lambda *f: pd.concat([pd.read_csv(HERE / "out" / x, keep_default_na=False, float_precision="round_trip",
                                        na_values={c: ["nan", "NaN"] for c in ("lo95", "ref", "dpool")}) for x in f], ignore_index=True)
u = rdn("mini29_ukuran_0.csv", "mini29_ukuran_1.csv", "mini29_ukuran_2.csv"); u20 = rdn("mini20_size_a.csv", "mini20_size_b.csv")
U = u20.merge(u[["dataset", "TOTAL", "rep", "cfg", "dpool"]], on=["dataset", "TOTAL", "rep"], suffixes=("", "_29")); U = U[U.kelas != "skip"]
mt = rdn("mini29_metrik.csv"); m21 = rdn("mini21_m1.csv", "mini21_m2.csv", "mini21_m3.csv"); m21 = m21[m21.design == "union_wF1"]
NEW = m21.merge(mt[mt.metric == "wF1"][["dataset", "rep", "cfg", "dpool"]], on=["dataset", "rep"], suffixes=("", "_29"))
q29 = rdn("mini29_Q.csv"); QQ = rdn("mini25_m1.csv", "mini25_m2.csv", "mini25_m3.csv").merge(q29[["dataset", "cfg", "dpool"]], on="dataset", suffixes=("", "_29"))
settings = [(f"{n} training rows", U[U.TOTAL == n]) for n in (1000, 2430, 6000)] + [("11 new datasets", NEW), ("Q", QQ)]
for lab, m in settings:
    m = m.reset_index(drop=True); ch = m.cfg != "None"; groups = [g.index.values for _, g in m.groupby("dataset")]; G = len(groups)
    br = (ch & (m.lo95.astype(float) > .005)).values
    for t in (.025, .02):
        fm = (ch & (m.dpool.astype(float) > t)).values; rng = np.random.default_rng(SEED)
        def f(idx):
            ref = m.ref.values[idx]; a, b = br[idx], fm[idx]
            h = lambda acc: 100 * (ref[acc] <= 0).mean() if acc.any() else np.nan
            return h(a) - h(b), 100 * (a.sum() - b.sum()) / len(idx)
        bt = np.array([f(np.concatenate([groups[i] for i in rng.integers(0, G, G)])) for _ in range(B)]); est = f(np.arange(len(m)))
        for j, what in enumerate(("harmful pp", "yield per 100")):
            lo, hi = np.nanpercentile(bt[:, j], [2.5, 97.5])
            rows.append(dict(analysis=f"fixed margin t={t}, {what}", data=lab, clusters=G, comparator=f"fixed margin {t}", diff_pp=est[j],
                             ci95_lo=lo, ci95_hi=hi, p_boot=boot_p(bt[:, j])))
r = pd.DataFrame(rows); r.to_csv(HERE / "out" / "paired_ci.csv", index=False)
pd.set_option("display.width", 250); print(r.round(4).to_string(index=False))

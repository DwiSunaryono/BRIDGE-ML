"""Supplementary analyses from stored outputs, with no new runs.
1. Clinical consequences per 1,000 sample observations of changes accepted by BRIDGE
   (net benefit p_t = 0.2, MINI-30).
2. Paired differences in percent harmful between rules at equal yield, with cluster
   bootstrap over datasets or tasks.
3. Relation between cross-fitted gate evidence and the deployed configuration
   (MINI-27 + MINI-24).
    python analysis_review4.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summarize_paper import rd

# 1. Clinical consequences
d = rd("mini30_m1.csv", "mini30_m2.csv", "mini30_m3.csv")
s = d[(d.metric == "nb20") & (d.lo95 > .005)].copy()
c = np.array([list(map(int, x.split())) for x in s.counts_ref]).reshape(-1, 2, 2, 2)  # [row, y, baseline, candidate]
base, cand, n, w = c.sum(axis=3), c.sum(axis=2), c.sum(axis=(1, 2, 3)), .2 / .8
s["extra_TP_per_1000"] = 1000 * (cand[:, 1, 1] - base[:, 1, 1]) / n
s["extra_FP_per_1000"] = 1000 * (cand[:, 0, 1] - base[:, 0, 1]) / n
s["flagged_before"], s["flagged_after"] = 1000 * base[:, :, 1].sum(1) / n, 1000 * cand[:, :, 1].sum(1) / n
s["NB_before"] = base[:, 1, 1] / n - base[:, 0, 1] / n * w; s["NB_after"] = cand[:, 1, 1] / n - cand[:, 0, 1] / n * w
pi = cand[:, 1, :].sum(1) / n; s["NB_treat_all"] = pi - (1 - pi) * w
cols = ["extra_TP_per_1000", "extra_FP_per_1000", "flagged_before", "flagged_after", "NB_before", "NB_after", "NB_treat_all"]
print("1. Consequences per 1,000 sample observations (accepted changes, net benefit 0.2)")
print(s.groupby(["sumber", "dataset"])[cols].mean().round(3).to_string()); print("all:", s[cols].mean().round(3).to_dict())

# 2. Paired differences at equal yield
real = rd("mini24_m1.csv", "mini24_m2.csv", "mini24_m3.csv"); Q = rd("mini25_m1.csv", "mini25_m2.csv", "mini25_m3.csv")
def harm_at(m, stat, frac):
    ok = m.cfg != "None"; idx = stat[ok].sort_values(ascending=False).index[:int(round(frac * len(m)))]
    return 100 * (m.loc[idx].ref <= 0).mean()
rng = np.random.default_rng(7)
for nm, m in (("11 new real datasets (cluster = dataset)", real), ("population Q (cluster = task)", Q)):
    groups = {k: g.index.values for k, g in m.groupby("dataset")}; keys = list(groups)
    print(f"\n2. {nm}: % harmful, BRIDGE minus comparator, equal yield, 95% CI (400 bootstrap)")
    for comp, col in (("corrected resampled t", "p_crt_final"), ("5x2cv", "p_5x2")):
        for frac in (.1, .2, .3, .4, .5):
            diffs = []
            for _ in range(400):
                mm = pd.concat([m.loc[groups[k]] for k in rng.choice(keys, size=len(keys), replace=True)], ignore_index=True)
                diffs.append(harm_at(mm, mm.lo95, frac) - harm_at(mm, -mm[col], frac))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            print(f"   vs {comp:22s} yield {int(100*frac):2d}%: {harm_at(m, m.lo95, frac) - harm_at(m, -m[col], frac):+5.2f} pp [{lo:+5.2f}, {hi:+5.2f}]")

# 3. Gate evidence vs deployed configuration
m27 = rd("mini27_m1.csv", "mini27_m2.csv", "mini27_m3.csv")
x = m27.merge(real[["dataset", "rep", "bridge"]], on=["dataset", "rep"]); x = x[x.cfg != "None"]
print("\n3. Gate vs deployed configuration (n =", len(x), ")")
print("   Spearman cross-fit estimate vs deployed truth:", round(stats.spearmanr(x.dcf, x.dref_deploy).statistic, 3),
      "| naive full-POOL estimate:", round(stats.spearmanr(x.dpool_best, x.dref_deploy).statistic, 3))
print("   mean bias cross-fit:", round((x.dcf - x.dref_deploy).mean(), 4), "| naive:", round((x.dpool_best - x.dref_deploy).mean(), 4))
a, r = x[x.bridge == 1], x[x.bridge == 0]
print(f"   accepted: {len(a)}, truly improving {100*(a.dref_deploy>0).mean():.1f}%; rejected: {len(r)}, truly improving {100*(r.dref_deploy>0).mean():.1f}% (mean gain {r.dref_deploy.mean():.4f})")

"""Confirmatory and sensitivity runs cited in the text (MINI-28, MINI-31, MINI-34, MINI-37, MINI-38) whose numbers appear in no
other table. Every number is recomputed here from the stored per-trial outputs; no new runs.
Gate rule: accept iff lo95 > 0.005; harmful = reference gain <= 0; bound = KL-ratio bound (b = 1); LOCO by dataset
(clinical: by source). Naive (clinical): a configuration was selected (cfg != None).
Usage: python analysis_cited_runs.py -> prints and writes out/analysis_cited_runs.csv"""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); O = HERE / "out"
from verify_exp41 import kl_ratio_up as up

rd = lambda *fs: pd.concat([pd.read_csv(O / f, keep_default_na=False, na_values={c: ["nan", "NaN"] for c in ("lo95", "ref")})
                            for f in fs], ignore_index=True)


def counts(s, acc, grp):
    NT = len(s); a = s[acc]; n, k = len(a), int((a.ref <= 0).sum()); per = s.groupby(grp).size()
    lo = max(up(int((a[a[grp] != q].ref <= 0).sum()), int((a[grp] != q).sum()), NT - per[q], 1)
             if (a[grp] != q).sum() else 1.0 for q in per.index)
    return dict(trials=NT, accepted=n, harmful=k, harmful_pct=100 * k / max(n, 1), bound_pct=100 * up(k, n, NT, 1), loco_pct=100 * lo,
                yield_per100=100 * n / NT)


rows = []
# MINI-28/28b/28c: clinical, weighted-F1 stage (union_wF1), LOCO by source
c = rd("mini28_m1.csv", "mini28_m2.csv", "mini28_m3.csv", "mini28b_covid.csv", "mini28c_wids.csv")
c = c[(c.design == "union_wF1") & ~c.kelas.isin(["error", "skip"])].copy()
c["source"] = c.dataset.str.split("_").str[0].replace({"covid": "covid_mx", "wids": "wids_icu"})
rows.append(dict(run="MINI-28", item="clinical, weighted-F1 stage: BRIDGE", sources=c.source.nunique(), tasks=c.dataset.nunique(),
                 **counts(c, c.lo95 > .005, "source")))
rows.append(dict(run="MINI-28", item="clinical, weighted-F1 stage: naive", sources=c.source.nunique(), tasks=c.dataset.nunique(),
                 **counts(c, c.cfg != "None", "source")))
# MINI-31: project-default XGBoost as the base learner on Q (Proposition 1 bound, as in analisis_mini34.prop1)
from analisis_mini34 import prop1
x = rd("mini31_xgb_m1.csv", "mini31_xgb_m2.csv", "mini31_xgb_m3.csv"); x = x[~x.cfg.astype(str).str.startswith("ERROR")].copy()
x["lo"] = x["lo_K3_q0.05"].astype(float); x["ref"] = x.ref.astype(float); a = x[x.lo > .005]; k = int((a.ref <= 0).sum())
rows.append(dict(run="MINI-31", item="project-default XGBoost as the base learner (Q)", trials=len(x), accepted=len(a), harmful=k,
                 harmful_pct=100 * k / max(len(a), 1), bound_pct=100 * prop1(k, len(a)), yield_per100=100 * len(a) / len(x)))
# MINI-34: skill-positive vs skill-negative base learner on POOL, accepted updates pooled over the nine configurations
s = rd("mini34_m1.csv", "mini34_m2.csv", "mini34_m3.csv"); s = s[~s.cfg.astype(str).str.startswith("ERROR")].copy()
s["skill_pool"] = s.skill_pool.astype(float); a = s[s.lo95 > .005]
for lbl, m in (("skill-positive", a.skill_pool > 0), ("skill-negative", a.skill_pool <= 0)):
    x = a[m]; rows.append(dict(run="MINI-34", item=f"accepted updates, base learner {lbl} on POOL (nine configurations)",
                               configurations=s.base.nunique(), accepted=len(x), harmful=int((x.ref <= 0).sum()),
                               harmful_pct=100 * (x.ref <= 0).mean()))
# MINI-34/36: Spearman correlation between the median log-loss skill on POOL and the gate bound across configurations
from scipy.stats import spearmanr
b34 = pd.read_csv(O / "analisis_mini34.csv"); b36 = pd.read_csv(O / "analisis_mini36.csv")
for lbl, z in (("nine HGB and XGBoost configurations (MINI-34)", b34),
               ("eleven configurations incl. two MLPs (MINI-34, MINI-36)", pd.concat([b34, b36], ignore_index=True))):
    rr = spearmanr(z.med_skill_pool, z.batas)
    rows.append(dict(run="MINI-34", item=f"Spearman rho, median skill vs bound, {lbl}", configurations=len(z), rho=rr.statistic, p=rr.pvalue))
# MINI-33b: base-learner audit (descriptive), regularized XGBoost vs HGB weighted-F1 on the reference
au = rd("mini33b_m1.csv", "mini33b_m2.csv"); w = au.pivot_table(index="dataset", columns="model", values="wf1")
rows.append(dict(run="MINI-33b", item="regularized XGBoost: lower base weighted-F1 than HGB", tasks=len(w), share_pct=100 * (w.XGBeq < w.HGB).mean()))
# MINI-17 / MINI-20: skipped trials (fewer than 10 examples of a class per bag) in the 1,000-row training-size study
for run, fs in (("MINI-20", ("mini20_size_a.csv", "mini20_size_b.csv")), ("MINI-17", ("mini17_trainsize_dev.csv",))):
    z = rd(*fs); sk = z[z.kelas == "skip"].groupby("dataset").size()
    rows.append(dict(run=run, item="skipped trials at 1,000 training rows", trials=len(z),
                     skipped=", ".join(f"{k} {v}" for k, v in sk.items())))
# MINI-37: bag count and SEARCH size on Q
v = pd.read_csv(O / "analisis_mini37a.csv")
for r in v.itertuples():
    rows.append(dict(run="MINI-37", item=f"{r.varian.replace('_S', ' bags, SEARCH ').replace('B', '')} rows (Q)", trials=r.tugas, accepted=r.terima,
                     harmful=r.salah, harmful_pct=r.pct, bound_pct=r.batas, yield_per100=r.yield_))
# MINI-37: Monte Carlo error of the decision statistic and reference precision
n = rd("mini37_nyata.csv"); q = rd("mini37_Q_m1.csv", "mini37_Q_m2.csv", "mini37_Q_m3.csv")
q = q[(q.varian == "B32_S630") & ~q.cfg.astype(str).str.startswith("ERROR")] if "varian" in q else q
for nm, d in (("Q", q), ("real data (11 datasets x 10)", n)):
    if "sd_q_B20000" not in d:
        continue
    for col in ("sd_q_B20000", "flip_B20000", "flip_B2000", "se_ref_delta", "n_ref"):
        d[col] = d[col].astype(float)
    rows.append(dict(run="MINI-37", item=f"Monte Carlo error, {nm}", trials=len(d), median_sd=d.sd_q_B20000.median(),
                     flip_per_rerun_pct=100 * d.flip_B20000.mean(), trials_any_flip=int((d.flip_B20000 > 0).sum()),
                     trials_any_flip_pct=100 * (d.flip_B20000 > 0).mean()))
ds = n.assign(se=n.se_ref_delta.astype(float), nr=n.n_ref.astype(float)).groupby("dataset").agg(se=("se", "median"), n_ref=("nr", "first"))
big, small = ds[ds.n_ref >= 40000], ds[ds.n_ref < 40000]
rows.append(dict(run="MINI-37", item="reference SE of the gain, datasets with reference >= 40,000 rows", datasets=len(big),
                 se_min=big.se.min(), se_max=big.se.max(), n_ref_min=big.n_ref.min(), n_ref_max=big.n_ref.max()))
rows.append(dict(run="MINI-37", item="reference SE of the gain, datasets with reference < 40,000 rows", datasets=len(small),
                 se_min=small.se.min(), se_max=small.se.max(), n_ref_min=small.n_ref.min(), n_ref_max=small.n_ref.max()))
# MINI-38: outer-split (reference) seed
m = pd.read_csv(O / "analisis_mini38.csv")
for r in m.itertuples():
    rows.append(dict(run="MINI-38", item=f"reference seed {r.seed_ref}: {r.aturan}", trials=r.trial, accepted=r.terima, harmful=r.salah,
                     harmful_pct=r.pct_salah, bound_pct=r.batas, loco_pct=r.LOCO, yield_per100=r.yield_))
res = pd.DataFrame(rows); pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
print(res.round(5).to_string(index=False)); print(); print(ds.round(5).to_string())
res.to_csv(O / "analysis_cited_runs.csv", index=False)

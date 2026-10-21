"""Development runs cited in the text (MINI-02, MINI-04, MINI-09, MINI-12, MINI-14, MINI-16, MINI-19). Every number
quoted in the paper is recomputed here from the stored per-trial outputs, with no new runs. Gate rule: accept iff
lo95 > 0.005; harmful = reference gain <= 0; bound = KL-ratio bound (b = 1); LOCO by dataset, using the same
functions as analisis_dev.py.
Usage: python analysis_development_runs.py -> prints and writes out/analysis_development_runs.csv"""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); O = HERE / "out"
from verify_exp41 import kl_ratio_up as up


def cell(s, acc=None):
    s = s[s.kelas != "skip"]; NT = len(s); dss = list(s.dataset.unique())
    a = s[s.lo95 > .005] if acc is None else s[acc]; n, k = len(a), int((a.ref <= 0).sum())
    per = {x: (s.dataset == x).sum() for x in dss}
    lo = max(up(int((a[a.dataset != x].ref <= 0).sum()), int((a.dataset != x).sum()), NT - per[x], 1)
             if (a.dataset != x).sum() else 1.0 for x in dss)
    r0 = max(up(0, int((a.dataset != x).sum()), NT - per[x], 1) if (a.dataset != x).sum() else 1.0 for x in dss)  # power check R0
    return dict(trials=NT, datasets=len(dss), accepted=n, harmful=k, bound_pct=100 * up(k, n, NT, 1),
                loco_pct=100 * lo, yield_per100=100 * n / NT, r0_pct=100 * r0)


rd = lambda f: pd.read_csv(O / f, keep_default_na=False, na_values={c: ["nan", "NaN"] for c in ("lo95", "ref")})
rows = []
d = rd("mini02_threshold_crossfit.csv"); rows.append(dict(run="MINI-02", item="frozen gate, threshold-only HGB", **cell(d[d.design == "xfit3"])))
d = rd("mini04_KNN.csv"); rows.append(dict(run="MINI-04", item="threshold-only KNN (cross-fit gate)", **cell(d[d.design == "xfit3"])))
d = rd("mini09_dev.csv")
for des, lbl in (("ens3_KNN", "third ensemble member KNN"), ("ens3_SVC", "third ensemble member SVC")):
    rows.append(dict(run="MINI-09", item=lbl, **cell(d[d.design == des])))
d = rd("mini12_optuna_dev.csv")
for des, lbl in (("tune_HGB", "Optuna-tuned HGB as the update"), ("union_T", "tuned HGB as baseline and member"),
                 ("union_D", "default HGB as baseline and member")):
    rows.append(dict(run="MINI-12", item=lbl, **cell(d[d.design == des])))
t = d[d.design == "tune_HGB"]
rows.append(dict(run="MINI-12", item="mean reference weighted-F1, default vs tuned",
                 mean_default=t.wf1_ref_default.astype(float).mean(), mean_tuned=t.wf1_ref_tuned.astype(float).mean()))
d = rd("mini14_optuna_size.csv")
for S, s in d.groupby("S"):
    r = cell(s); r.update(mean_gain=s.ref.astype(float).mean(), clearly_worse_pct=100 * (s.kelas == "memburuk").mean())
    rows.append(dict(run="MINI-14", item=f"tuning data {S} rows", **r))
r = cell(d); r.update(mean_gain=d.ref.astype(float).mean(), clearly_worse_pct=100 * (d.kelas == "memburuk").mean())
rows.append(dict(run="MINI-14", item="all three tuning sizes", **r))
d = rd("mini16_metrics_dev.csv")
for des, lbl in (("union_acc", "accuracy"), ("union_bacc", "balanced accuracy")):
    rows.append(dict(run="MINI-16", item=lbl, **cell(d[d.design == des])))
w = rd("mini19_fs_lebar_dev.csv"); w = w[w.lebar.astype(str) == "True"]
rows.append(dict(run="MINI-19", item="union policy, 8 wide datasets, first 50 trials", **cell(w[w.design == "union"])))
wb = pd.concat([w[w.design == "union"], rd("mini19b_lebar_dev.csv").query("design == 'union'")])
rows.append(dict(run="MINI-19", item="union policy, 8 wide datasets, extended to 150 trials (MINI-19b)", **cell(wb)))
res = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(res.round(4).to_string(index=False)); res.to_csv(O / "analysis_development_runs.csv", index=False)

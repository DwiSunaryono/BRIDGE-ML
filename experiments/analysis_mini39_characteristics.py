"""MINI-39, post hoc exploratory: task and dataset characteristics against the outcome of tuning updates.

(1) Population Q (T2900-T4299): the generator parameters of each task are known (mechanism, signal s, minority m,
    features p). Per characteristic level: number of tasks; share with a clear winner among the tuned learners (best
    minus second >= 0.005, the paper's margin); share of clear wins per learner; and, pooled over the four learners,
    the naive rule's and BRIDGE's accepted/harmful counts.
(2) The 11 new real datasets: rows, features, minority share, default-HGB reference weighted-F1 (difficulty proxy),
    tuned winner (or tie), and per-learner BRIDGE accepted/harmful counts.
Groupings were chosen after the results were seen; the analysis is descriptive only.
Usage: python analysis_mini39_characteristics.py -> out/mini39_char_Q.csv, out/mini39_char_R.csv
"""
import sys, io, glob, contextlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); O = HERE / "out"
L = ["HGB", "DT", "SVC", "KNN"]; EPS = 0.005
rd = lambda pat: pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""]) for f in sorted(glob.glob(str(O / pat)))], ignore_index=True)


def winners(w):
    srt = np.sort(w[L].values, axis=1)
    return w[L].idxmax(axis=1), (srt[:, -1] - srt[:, -2]) >= EPS


def counts(s, col):
    a = s[s[col] == 1]
    return len(a), int((a.ref <= 0).sum())


# (1) population Q
_argv = sys.argv; sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import mini25_sintetis as Qm
sys.argv = _argv
d = rd("mini39_Q_*.csv")
P = pd.DataFrame([dict(dataset=f"T{i:04d}", mechanism={"pohon": "tree", "linier": "linear"}[Qm.param(i)["jenis"]],
                       s=Qm.param(i)["s"], p=Qm.param(i)["p"], minor=Qm.param(i)["minor"]) for i in range(2900, 4300)])
P["signal"] = pd.qcut(P.s, 3, labels=["weak", "medium", "strong"])
P["features"] = pd.cut(P.p, [4, 15, 27, 40], labels=["5-15", "16-27", "28-40"])
P["minority"] = pd.cut(P.minor, [0, .2, .35, .5], labels=["<0.20", "0.20-0.35", ">0.35"])
w = d.pivot_table(index="dataset", columns="learner", values="wf1_ref_tun").join(P.set_index("dataset"))
w["winner"], w["clear"] = winners(w)
d = d.merge(P, on="dataset")
rows = []
for g in ("mechanism", "signal", "minority", "features"):
    for lev, ww in w.groupby(g, observed=True):
        dd = d[d[g] == lev]; cw = ww[ww.clear]
        nn, nk = counts(dd, "naive"); bn, bk = counts(dd, "bridge")
        rows.append(dict(characteristic=g, level=str(lev), tasks=len(ww), clear_pct=100 * ww.clear.mean(),
                         **{f"win_{l}_pct": 100 * (cw.winner == l).mean() for l in L},
                         naive_acc=nn, naive_harm=nk, bridge_acc=bn, bridge_harm=bk))
pd.DataFrame(rows).to_csv(O / "mini39_char_Q.csv", index=False)

# (2) real datasets
r = rd("mini39_R_*.csv")
import importlib
with contextlib.redirect_stdout(io.StringIO()):
    import mini21_final as F
    F = importlib.reload(F)  # mini25_sintetis replaced F.DS and F.data with the synthetic population
info = []
for ds in F.DS:
    X, y = F.load(*F.DS[ds])
    info.append(dict(dataset=ds, rows=len(X), features=X.shape[1], minority_pct=100 * min(np.bincount(np.asarray(y))) / len(y)))
I = pd.DataFrame(info).set_index("dataset")
wr = r.pivot_table(index="dataset", columns="learner", values="wf1_ref_tun").reindex(I.index)
wr["winner"], wr["clear"] = winners(wr)
I["default_HGB_wF1"] = r[r.learner == "HGB"].groupby("dataset").wf1_ref_def.mean()
I["tuned_winner"] = np.where(wr.clear, wr.winner, "tie")
for l in L:
    I[f"BRIDGE_{l}"] = [("%d/%d" % counts(r[(r.dataset == ds) & (r.learner == l)], "bridge")) for ds in I.index]
I["naive_all"] = [("%d/%d" % counts(r[r.dataset == ds], "naive")) for ds in I.index]
I.sort_values("default_HGB_wF1").to_csv(O / "mini39_char_R.csv")
print(pd.read_csv(O / "mini39_char_Q.csv").round(1).to_string()); print(I.sort_values("default_HGB_wF1").round(3).to_string())

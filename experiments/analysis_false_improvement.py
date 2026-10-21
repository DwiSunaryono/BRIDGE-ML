"""Post hoc reanalysis of stored per-trial outputs (no new runs): does each accepted update improve by at least the margin?

For every acceptance rule and collection, among the updates that the rule accepts:
  false-improvement rate: % whose reference gain is at most eps = 0.005, the minimum improvement fixed in
                          development; these updates looked better on validation but improved the model by less than eps;
  realized gain per update: mean reference gain of the accepted updates (each rule is judged on its own deployed
                          configuration; each holdout deploys its own configuration);
  genuine improvements per 100 decisions: accepted updates whose reference gain exceeds eps, per 100 decisions.
Each quantity receives a 95% cluster-bootstrap percentile interval. BRIDGE is compared with each rule by the paired
cluster bootstrap of test T1 (two-sided p from the resampled differences, floor 1/B), with Holm over the rules of a
collection, separately for the two quantities. Clusters: datasets (A, D, E) and tasks (B, C). B = 10,000, seed 20261021.
Inputs: out/mini26_nyata_m*.csv (A), out/mini26_q_m*.csv (B), out/mini48_m*.csv (C, D), out/settingE_*laptop.csv (E).
Usage: python analysis_false_improvement.py -> out/analysis_false_improvement.csv
       python analysis_false_improvement.py --table -> ../paper_assets/tabS23_false_improvement.tex"""
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_mini44 as A44

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
EPS = 0.005
A44.SEED, A44.B = 20261021, 10_000
RULES = {"naive": "Naive", "mcnemar_pool": "McNemar, same POOL", "mcnemar_cf": "McNemar, cross-fit", "t5x2": "5x2 cv paired t",
         "crt_final": "Corrected resampled t", "crt": "Corrected t, re-selection", "boot0": "BRIDGE without margin",
         "ho23": "Holdout 2/3 + 1/3", "ho12": "Holdout 1/2 + 1/2"}


def rd(pattern):
    return pd.concat([pd.read_csv(f, float_precision="round_trip") for f in sorted(OUT.glob(pattern))]).reset_index(drop=True)


def collections():
    a = rd("mini26_nyata_m*.csv").rename(columns={"naif": "naive"})
    b = rd("mini26_q_m*.csv").rename(columns={"naif": "naive"}); b["dataset"] = np.arange(len(b)).astype(str)
    m = rd("mini48_m*.csv"); m["S"] = m.setting.replace({"D-mc": "D"})
    c, d = m[m.S == "C"].reset_index(drop=True), m[m.S == "D"].reset_index(drop=True)
    e = pd.concat([pd.read_csv(OUT / f) for f in ("settingE_laptop.csv", "settingE_part2_laptop.csv")]).drop_duplicates(["data_id", "rep"])
    e = e[e.status == "ok"].reset_index(drop=True)
    e = pd.DataFrame(dict(dataset=e.data_id.astype(str), ref=e.gain_ref, naive=e.naive_accept, bridge=e.accept))
    return {"(A)": a, "(B)": b, "(C)": c, "(D)": d, "(E)": e}


def main():
    rows = []
    for S, x in collections().items():
        ref = x.ref.to_numpy(); allr = np.arange(len(x))

        def rule(r):
            acc = (x[r] == 1).to_numpy()
            return acc, (x[f"{r}_ref"].to_numpy() if r in ("ho23", "ho12") else ref)

        def fir(acc, rr):
            return lambda s: 100 * (rr[s][acc[s]] <= EPS).mean() if acc[s].any() else np.nan

        def gpu(acc, rr):
            return lambda s: rr[s][acc[s]].mean() if acc[s].any() else np.nan

        def gen(acc, rr):
            return lambda s: 100 * (acc[s] & (rr[s] > EPS)).mean()

        ab, rb = rule("bridge")
        res = []
        for r in ["bridge"] + [r for r in RULES if r in x.columns]:
            acc, rr = rule(r); row = dict(collection=S, rule=RULES.get(r, "BRIDGE"), accepted=int(acc.sum()))
            for q, f in (("fir", fir), ("gpu", gpu), ("gen", gen)):
                est, lo, hi, _ = A44.boot(x, f(acc, rr)); row |= {q: est, f"{q}_lo": lo, f"{q}_hi": hi}
                if r != "bridge":
                    _, _, _, tb = A44.boot(x, lambda s, f=f, acc=acc, rr=rr: f(ab, rb)(s) - f(acc, rr)(s))
                    row[f"{q}_p"] = min(1.0, max(2 * min((tb <= 0).mean(), (tb >= 0).mean()), 1 / len(tb)))
            res.append(row)
        d = pd.DataFrame(res)
        for q in ("fir", "gpu", "gen"):
            v = d[f"{q}_p"].notna(); d.loc[v, f"{q}_p_holm"] = A44.holm(d.loc[v, f"{q}_p"])
        rows.append(d); print(S, "done")
    out = pd.concat(rows); out.to_csv(OUT / "analysis_false_improvement.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))




def table():
    """Appendix table: false-improvement rate and genuine improvements per 100 decisions, rule by collection."""
    d = pd.read_csv(OUT / "analysis_false_improvement.csv"); cols = ["(A)", "(B)", "(C)", "(D)", "(E)"]
    order = ["Naive", "McNemar, same POOL", "McNemar, cross-fit", "5x2 cv paired t", "Corrected resampled t", "Corrected t, re-selection",
             "BRIDGE without margin", "Holdout 2/3 + 1/3", "Holdout 1/2 + 1/2", "BRIDGE"]
    names = {"5x2 cv paired t": r"$5\times2$~cv paired $t$", "Corrected t, re-selection": r"Corrected $t$, re-selection", "Corrected resampled t": r"Corrected resampled $t$", "BRIDGE": r"\textbf{BRIDGE}"}

    def cell(r, q, better_low):
        if r is None:
            return "--"
        v = str(Decimal(str(r[q])).quantize(Decimal("0.1"), ROUND_HALF_UP))
        if r.rule == "BRIDGE":
            return v
        p, bv = r[f"{q}_p_holm"], d[(d.collection == r.collection) & (d.rule == "BRIDGE")][q].iloc[0]
        bridge_better = (bv < r[q]) if better_low else (bv > r[q])
        return v + (r"$^{\circ}$" if p >= .05 else (r"$^{*}$" if bridge_better else r"$^{\dagger}$"))

    lines = []
    for title, q, low in (("Accepted updates gaining at most $\\varepsilon$ (\\%) $\\downarrow$", "fir", True), ("Accepted updates gaining more than $\\varepsilon$, per $100$ decisions $\\uparrow$", "gen", False)):
        lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for rule in order:
            rs = [d[(d.collection == c) & (d.rule == rule)] for c in cols]
            lines.append(f"{names.get(rule, rule)} & " + " & ".join(cell(x.iloc[0] if len(x) else None, q, low) for x in rs) + r" \\")
    tex = (r"""% generated by experiments/analysis_false_improvement.py (post hoc reanalysis of stored runs)
\begin{table}[!t]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\caption{Acceptance rules on (A)--(E): accepted updates below and above the margin on the reference (post hoc).}
\label{tab:falseimp}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Rule & (A) & (B) & (C) & (D) & (E) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}

\vspace{2pt}
\parbox{\linewidth}{\justifying\scriptsize Margin $\varepsilon=0.005$: the minimum improvement fixed in development (Table~\ref{tab:hyper}); a gain at most $\varepsilon$ includes every false improvement (gain $\le 0$). $^{*}$: BRIDGE significantly better than the rule; $^{\circ}$: no significant difference; $^{\dagger}$: the rule significantly better (paired cluster bootstrap of T1, $10{,}000$ resamples, Holm over the rules of a collection, $\alpha=0.05$). --: not run, because the program used for (E) records the decisions of the naive rule and BRIDGE.}
\end{table}
""")
    (HERE.parent / "paper_assets" / "tabS23_false_improvement.tex").write_text(tex)


if __name__ == "__main__":
    import sys
    table() if "--table" in sys.argv else main()

"""Clinical results by source and leave-one-source-group-out sensitivity (confirmatory run MINI-30).

Analysis of existing per-trial outputs only; no new runs. For every clinical metric it reports
(1) accepted / harmful / clearly harmful / reference-uncertain counts per source for BRIDGE and the naive rule,
(2) the KL-ratio bound, the conservative bound (every accepted update that is not clearly improving counted
    as harmful) and the leave-one-source-out (LOCO) bound,
(3) the same summaries after excluding each source in turn (mammography has no patient identifier).
Outcome classes: improving / reference-uncertain / clearly harmful (Dirichlet bootstrap interval of the reference gain).
Usage: python analysis_clinical_by_source.py  -> prints tables and writes out/analysis_clinical_by_source.csv
"""
from pathlib import Path

import pandas as pd

import summarize_paper as SP
from verify_exp41 import kl_ratio_up as up

HERE = Path(__file__).resolve().parent
METRICS = [("nb20", "Net benefit p_t=0.2"), ("bacc", "Balanced accuracy"), ("wF1", "Weighted F1"),
           ("nb10", "Net benefit p_t=0.1"), ("F2", "F2")]
RULES = (("BRIDGE", lambda x: x.lo95 > .005), ("Naive", lambda x: x.cfg != "None"))


def summary(x, acc):
    """Counts and bounds for one rule on trials x, where acc is the boolean acceptance mask."""
    NT = len(x); a = x[acc]; n = len(a); k = int((a.ref <= 0).sum()); kc = int((a.kelas != "membaik").sum())
    per = x.groupby("sumber").size()
    loco = max((up(int((a[a.sumber != q].ref <= 0).sum()), int((a.sumber != q).sum()), NT - per[q], 1)
                if (a.sumber != q).sum() else 1.0) for q in per.index) if len(per) > 1 else float("nan")
    return dict(trials=NT, accepted=n, harmful=k, clearly_harmful=int((a.kelas == "memburuk").sum()),
                reference_uncertain=int((a.kelas == "tidak-pasti").sum()), not_clearly_improving=kc, bound=100 * up(k, n, NT, 1), conservative_bound=100 * up(kc, n, NT, 1),
                loco=100 * loco)


def main():
    m = SP.rd("mini30_m1.csv", "mini30_m2.csv", "mini30_m3.csv"); m = m[~m.kelas.isin(["error", "skip"])]
    rows = []
    for k, lbl in METRICS:
        x = m[m.metric == k]
        for nm, f in RULES:
            for src, xs in x.groupby("sumber"):
                s = summary(xs, f(xs)); rows.append(dict(metric=lbl, rule=nm, subset=f"source={src}", **s))
            subsets = [("all sources", x.sumber.notna())] + [(f"without {s}", x.sumber != s) for s in sorted(x.sumber.unique())]
            for subset, keep in subsets:
                xs = x[keep]; rows.append(dict(metric=lbl, rule=nm, subset=subset, **summary(xs, f(xs))))
    r = pd.DataFrame(rows)
    r.to_csv(HERE / "out" / "analysis_clinical_by_source.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
    print(r.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

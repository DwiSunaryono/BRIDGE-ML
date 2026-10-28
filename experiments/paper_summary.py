"""Print the main result of the paper as one readable table, from CSVs in out/ (no model training).

    python paper_summary.py

Reads the CSVs that reproduce_paper.sh has just rewritten and prints, for each of the five
collections, how many updates the naive rule and BRIDGE accepted, how many of those were false
improvements, and the 95% upper bound on the BRIDGE rate. The detailed tables of every step are
in out/reproduce_log/.
"""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
COLLECTIONS = [  # label in the paper, row of analysis_selectivity.csv (counts)
    ("(A) 11 held-out datasets", "11 new datasets"),
    ("(B) synthetic population", "Synthetic population Q"),
    ("(C) 12 clinical tasks", "Clinical, net benefit"),
    ("(D) 39 extended datasets", "39 supporting datasets"),
]


def rows():
    s = pd.read_csv(OUT / "analysis_selectivity.csv").set_index("setting")
    lodo = pd.read_csv(OUT / "analysis_margin_lodo.csv").set_index("setting")
    rules = pd.read_csv(OUT / "analysis_mini48_rules.csv")
    kl = rules[rules.rule == "bridge"].set_index("setting")["kl"]
    bound = {"11 new datasets": lodo.loc["new wF1", "bridge_bound"], "Synthetic population Q": lodo.loc["Q wF1", "bridge_bound"],
             "Clinical, net benefit": kl["C"], "39 supporting datasets": kl["D"]}
    # the interval of the reduction comes from the analysis that also gives its Holm-adjusted p value:
    # analysis_paired_ci.py for (A) and (B), analysis_mini48.py for (C) and (D)
    pc = pd.read_csv(OUT / "paired_ci.csv")
    pc = pc[(pc.analysis == "operating point") & (pc.comparator == "naive")].set_index("data")
    nv = rules[rules.rule == "naive"].set_index("setting")
    ci = {"11 new datasets": tuple(pc.loc["real", ["diff_pp", "ci95_lo", "ci95_hi"]]),
          "Synthetic population Q": tuple(pc.loc["Q", ["diff_pp", "ci95_lo", "ci95_hi"]]),
          "Clinical, net benefit": tuple(nv.loc["C", ["bridge_minus_pp", "lo", "hi"]]),
          "39 supporting datasets": tuple(nv.loc["D", ["bridge_minus_pp", "lo", "hi"]])}
    for label, key in COLLECTIONS:
        r = s.loc[key]; d, lo, hi = ci[key]
        yield (label, r.decisions, r.naive_accepted, r.naive_harmful, r.bridge_accepted, r.bridge_harmful,
               -d, -hi, -lo, float(bound[key]))
    e = pd.read_csv(OUT / "analysis_settingE.csv").set_index("group").loc["E_all"]
    yield ("(E) 14 OpenML datasets", e.decisions, e.naive_accepted, e.naive_harmful, e.bridge_accepted, e.bridge_harmful,
           -e.bridge_minus_naive_pp, -e.bmn_hi, -e.bmn_lo, e.bridge_kl)


def main():
    head = (f"{'Collection':26s}{'decisions':>10s} | {'naive rule':^22s} | {'BRIDGE':^22s} | {'lower by':^20s} | {'bound':>6s}")
    sub = (f"{'':26s}{'':>10s} | {'accepted':>9s} {'false':>12s} | {'accepted':>9s} {'false':>12s} | "
           f"{'points [95% CI]':^20s} | {'':>6s}")
    print("MAIN RESULT: false improvements among accepted updates (paper: main results table and Figure 4)\n")
    print(head); print(sub); print("-" * len(head))
    for lab, n, na, nf, ba, bf, d, lo, hi, b in rows():
        print(f"{lab:26s}{int(n):>10,d} | {int(na):>9,d} {f'{int(nf)} ({100 * nf / na:.1f}%)':>12s} | "
              f"{int(ba):>9,d} {f'{int(bf)} ({100 * bf / ba:.1f}%)':>12s} | {f'{d:.1f} [{lo:.1f}, {hi:.1f}]':^20s} | {b:>5.2f}%")
    print("-" * len(head))
    print("accepted: updates the rule let replace the model in use. false: accepted updates that were not better than the")
    print("model in use on the locked reference data (false improvements; ties count as not better), with their share.")
    print("lower by: how far BRIDGE lowers that share, in percentage points, with its 95% interval over datasets or tasks.")
    print("bound: 95% upper bound on the BRIDGE share.")
    print(f"\nDetailed tables of every step: {OUT.name}/reproduce_log/ (VERBOSE=1 bash reproduce_paper.sh prints them here).")


if __name__ == "__main__":
    main()

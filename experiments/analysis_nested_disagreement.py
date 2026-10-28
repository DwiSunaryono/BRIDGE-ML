"""Decisions on which BRIDGE and the nested-resampling rule disagree, on collections (A) to (D), from stored outputs.

For each collection, the updates accepted by one rule and rejected by the other are counted, together with the false
improvements among them (reference gain at most zero). The nested-resampling rule is the corrected resampled t-test with
re-selection in every outer split (column `crt`). Inputs: out/mini26_nyata_m*.csv (A), out/mini26_q_m*.csv (B),
out/mini48_m*.csv (C, D). Usage: python analysis_nested_disagreement.py -> out/analysis_nested_disagreement.csv
"""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "out"


def read(pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(OUT.glob(pattern))], ignore_index=True)


def main():
    m = read("mini48_m*.csv")
    m["setting"] = m.setting.replace({"D-mc": "D"})   # the multiclass datasets of (D) are stored as D-mc, as in analysis_mini48.py
    sets = {"A": read("mini26_nyata_m*.csv"), "B": read("mini26_q_m*.csv"), "C": m[m.setting == "C"], "D": m[m.setting == "D"]}
    rows = []
    for k, d in sets.items():
        if "status" in d:
            d = d[d.status == "ok"]
        wrong = d.ref <= 0
        nested_only = (d.crt == 1) & (d.bridge == 0)
        bridge_only = (d.bridge == 1) & (d.crt == 0)
        rows.append(dict(collection=k, decisions=len(d),
                         nested_accepted=int(d.crt.sum()), nested_false=int(((d.crt == 1) & wrong).sum()),
                         bridge_accepted=int(d.bridge.sum()), bridge_false=int(((d.bridge == 1) & wrong).sum()),
                         nested_only=int(nested_only.sum()), nested_only_false=int((nested_only & wrong).sum()),
                         bridge_only=int(bridge_only.sum()), bridge_only_false=int((bridge_only & wrong).sum())))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "analysis_nested_disagreement.csv", index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()

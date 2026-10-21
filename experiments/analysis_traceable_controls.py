"""Traceable recomputation of two exploratory numbers quoted in Section IV-C from stored outputs only, with no new runs.

1. No-information control: random acceptance at BRIDGE's yield over all decisions (exact expected values), giving the
   harmful proportion among its accepted updates and the expected total gain relative to BRIDGE's
   (real data MINI-24, population Q MINI-25, clinical net benefit MINI-30).
2. Leave-one-setting-out (LOSO) for the fixed validation margin over the seven MINI-29 settings: for each held-out
   setting, choose t in the pre-registered grid 0.005..0.050 that maximizes total accepted updates over the other six
   subject to all six passing, then evaluate it on the held-out setting. BRIDGE needs no such choice.
3. Yield of the post hoc fixed margins 0.020-0.035 in each setting against BRIDGE's yield.
Usage: python analysis_traceable_controls.py -> prints and writes out/analysis_traceable_controls.csv
"""
import sys, io, contextlib
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); O = HERE / "out"
from summarize_paper import rd
with contextlib.redirect_stdout(io.StringIO()):
    import analisis_mini29 as A

rows = []
# 1. random acceptance at BRIDGE's yield
for nm, d, acc in (("real (11 new datasets)", rd("mini24_m1.csv", "mini24_m2.csv", "mini24_m3.csv"), "bridge"),
                   ("Q", rd("mini25_m1.csv", "mini25_m2.csv", "mini25_m3.csv"), "bridge")):
    d = d[~d.cfg.astype(str).str.startswith("ERROR")]
    br = d[d[acc] == 1]; cand = d[d.cfg != "None"]; y = len(br) / len(d)
    rows.append(dict(analysis="random at BRIDGE yield", setting=nm, decisions=len(d), yield_pct=100 * y,
                     random_harmful_pct=100 * (cand.ref <= 0).mean(), random_expected_gain=y * cand.ref.sum(),
                     bridge_gain=br.ref.sum(), gain_ratio=y * cand.ref.sum() / br.ref.sum()))
c = rd("mini30_m1.csv", "mini30_m2.csv", "mini30_m3.csv"); c = c[~c.kelas.isin(["error", "skip"])]; c = c[c.metric == "nb20"]
br = c[c.lo95 > .005]; cand = c[c.cfg != "None"]; y = len(br) / len(c)
rows.append(dict(analysis="random at BRIDGE yield", setting="clinical (net benefit)", decisions=len(c), yield_pct=100 * y,
                 random_harmful_pct=100 * (cand.ref <= 0).mean(), random_expected_gain=y * cand.ref.sum(),
                 bridge_gain=br.ref.sum(), gain_ratio=y * cand.ref.sum() / br.ref.sum()))

# 2. LOSO for the fixed margin over the MINI-29 settings
S = {"dev 1,000 rows wF1": (A.U[A.U.TOTAL == 1000], False), "dev 2,430 rows wF1": (A.U[A.U.TOTAL == 2430], False),
     "dev 6,000 rows wF1": (A.U[A.U.TOTAL == 6000], False), "new wF1": (A.Mt[A.Mt.metric == "wF1"], False),
     "new balanced accuracy": (A.Mt[A.Mt.metric == "bacc"], False), "new macro-F1": (A.Mt[A.Mt.metric == "macroF1"], False),
     "Q wF1": (A.Q, True)}
st = lambda s, acc, p1: A.stats(s, acc, "dataset", p1)
vm = {(k, t): st(s, (s.cfg != "None") & (s.dpool > t), p1) for k, (s, p1) in S.items() for t in A.GRID}
for h, (sh, p1h) in S.items():
    best, by = None, -1
    for t in A.GRID:
        others = [vm[(k, t)] for k in S if k != h]
        if all(o["lulus"] for o in others) and sum(o["terima"] for o in others) > by:
            best, by = t, sum(o["terima"] for o in others)
    ev = vm[(h, best)] if best is not None else None
    bb = st(sh, sh.lo95 > .005, p1h)
    rows.append(dict(analysis="LOSO fixed margin", setting=h, chosen_t=best,
                     margin_bound=ev["batas"] if ev else None, margin_LOCO=ev["LOCO"] if ev else None,
                     margin_pass=ev["lulus"] if ev else False, bridge_bound=bb["batas"], bridge_pass=bb["lulus"]))
# 3. yield of the post hoc fixed margins 0.020-0.035 per setting against BRIDGE's yield
for k, (s, p1) in S.items():
    ys = [vm[(k, t)]["yield_"] for t in (0.02, 0.025, 0.03, 0.035)]
    rows.append(dict(analysis="post hoc margin 0.020-0.035 yield", setting=k, margin_yield_min=min(ys), margin_yield_max=max(ys),
                     margin_pass=all(vm[(k, t)]["lulus"] for t in (0.02, 0.025, 0.03, 0.035)),
                     bridge_yield=st(s, s.lo95 > .005, p1)["yield_"]))
r = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(r.round(3).to_string(index=False)); r.to_csv(O / "analysis_traceable_controls.csv", index=False)

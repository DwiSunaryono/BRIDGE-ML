"""Analisis MINI-37 (PRAREG_MINI37.md): (a) sensitivitas bag dan SEARCH pada Q; (b) presisi B dan
referensi; identitas MINI-25."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; O = HERE / "out"
sys.path.insert(0, str(HERE))
from analisis_mini34 import prop1
rd = lambda f: pd.read_csv(O / f, keep_default_na=False)
Q = pd.concat([rd(f"mini37_Q_m{i}.csv") for i in (1, 2, 3)], ignore_index=True)
err = Q.cfg.str.startswith("ERROR").sum(); Q = Q[~Q.cfg.str.startswith("ERROR")].copy()
for c in ("lo95", "ref"): Q[c] = Q[c].astype(float)
print(f"\n=== MINI-37 === baris {len(Q)}, ERROR {err}, tugas {Q.dataset.nunique()}, varian {Q.varian.nunique()}")
m25 = pd.concat([rd(f"mini25_m{i}.csv") for i in (1, 2, 3)]); j = Q[Q.varian == "B32_S630"].merge(m25, on="dataset", suffixes=("", "_25"))
print("identitas B32_S630 == MINI-25:", len(j), bool((j.lo95 == j.lo95_25.astype(float)).all()), bool((j.cfg == j.cfg_25).all()), bool((j.ref == j.ref_25.astype(float)).all()))
rows = []
for v in ["B16_S630", "B32_S630", "B64_S630", "B32_S360", "B32_S1000"]:
    s = Q[Q.varian == v]; T = len(s); a = s[(s.cfg != "None") & (s.lo95 > .005)]; M, S = len(a), int((a.ref <= 0).sum())
    rows.append(dict(varian=v, tugas=T, terima=M, salah=S, pct=100 * S / max(M, 1), batas=100 * prop1(S, M), yield_=100 * M / T, total_delta=a.ref.sum(),
                     lulus="YA" if prop1(S, M) < .1 and 100 * M / T >= 5 else "TIDAK"))
r = pd.DataFrame(rows); pd.set_option("display.width", 220); print(r.round(2).to_string(index=False)); r.to_csv(O / "analisis_mini37a.csv", index=False)
b = Q[Q.varian == "B32_S630"].copy(); N = rd("mini37_nyata.csv"); N = N[~N.cfg.str.startswith("ERROR")].copy()
out = []
for nama, d in (("Q (1.400 tugas)", b), ("nyata (11 ds x 10)", N)):
    for c in ("sd_q_B20000", "sd_q_B2000", "flip_B20000", "flip_B2000", "se_ref_delta", "n_ref", "lo95"): d[c] = d[c].astype(float)
    acc = d[(d.cfg != "None")]
    out.append(dict(data=nama, trial=len(d), med_sd_q_B20000=d.sd_q_B20000.median(), p95_sd_q_B20000=d.sd_q_B20000.quantile(.95),
                    med_sd_q_B2000=d.sd_q_B2000.median(), flip_B20000_pct=100 * d.flip_B20000.mean(), flip_B2000_pct=100 * d.flip_B2000.mean(),
                    med_se_ref=acc.se_ref_delta.median(), p95_se_ref=acc.se_ref_delta.quantile(.95), max_se_ref=acc.se_ref_delta.max(),
                    med_n_ref=d.n_ref.median(), min_n_ref=d.n_ref.min()))
p = pd.DataFrame(out); print("\n(b) presisi:"); print(p.T.to_string()); p.to_csv(O / "analisis_mini37b.csv", index=False)
print("\nper dataset nyata (median SE ref, n_ref):"); print(N.assign(se=N.se_ref_delta.astype(float)).groupby("dataset").agg(se=("se", "median"), n_ref=("n_ref", "first")).round(5).to_string())

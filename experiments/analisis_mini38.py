"""Analisis MINI-38 (PRAREG_MINI38.md): BRIDGE dan naif per seed referensi (asli 20261125 vs R1, R2) pada 11 dataset baru."""
import sys, io, contextlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); O = HERE / "out"
with contextlib.redirect_stdout(io.StringIO()):
    from analisis_mini32 import up, rd
sets = {"asli (20261125)": pd.concat([rd("mini32_nyata_m1.csv"), rd("mini32_nyata_m2.csv")], ignore_index=True),
        "R1 (1038000001)": pd.concat([rd("mini38_m1R1.csv"), rd("mini38_m2.csv")], ignore_index=True),
        "R2 (1038000002)": pd.concat([rd("mini38_m1R2.csv"), rd("mini38_m3.csv")], ignore_index=True)}
rows = []
for nama, s in sets.items():
    err = int(s.cfg.astype(str).str.startswith("ERROR").sum()); s = s[~s.cfg.astype(str).str.startswith("ERROR")]
    NT = len(s); ubah = s.cfg != "None"; per = s.groupby("dataset").size()
    for aturan, acc in (("naif", ubah & (s.pt_pool > 0)), ("BRIDGE", ubah & (s.lo_cf_boot > .005))):
        a = s[acc]; n, k = len(a), int((a.ref <= 0).sum()); b = up(k, n, NT, 1)
        lo = max(up(int((a[a.dataset != q].ref <= 0).sum()), int((a.dataset != q).sum()), NT - per[q], 1) if (a.dataset != q).sum() else 1.0 for q in per.index)
        rows.append(dict(seed_ref=nama, trial=NT, error=err, aturan=aturan, terima=n, salah=k, pct_salah=100 * k / max(n, 1), batas=100 * b, LOCO=100 * lo,
                         yield_=100 * n / NT, total_delta=a.ref.sum(), lulus=bool(b < .1 and lo < .1 and 100 * n / NT >= 5)))
r = pd.DataFrame(rows); pd.set_option("display.width", 220); print(r.round(2).to_string(index=False)); r.to_csv(O / "analisis_mini38.csv", index=False)

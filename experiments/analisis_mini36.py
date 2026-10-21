"""Analisis MINI-36 (PRAREG_MINI36.md): MLP_ES vs MLP_noES; prediksi P-a/P-b dikunci."""
from pathlib import Path
import numpy as np, pandas as pd
from analisis_mini33 import prop1  # noqa (juga mencetak ringkasan MINI-33)
O = Path(__file__).resolve().parent / "out"
d = pd.concat([pd.read_csv(O / f"mini36_m{i}.csv", keep_default_na=False) for i in (1, 2, 3)], ignore_index=True)
err = d.cfg.str.startswith("ERROR").sum(); d = d[~d.cfg.str.startswith("ERROR")].copy()
for c in ("lo95", "pt_pool", "ref", "skill_pool", "ekstrem_pool", "epoch"): d[c] = d[c].astype(float)
print(f"\n=== MINI-36 === baris {len(d)}, ERROR {err}, tugas {d.dataset.nunique()}")
rows = []
for b in ("MLP_ES", "MLP_noES"):
    s = d[d.base == b]; T = len(s); a = s[s.lo95 > .005]; M, S = len(a), int((a.ref <= 0).sum()); Sc = int((a.kelas != "membaik").sum())
    nv = s[(s.cfg != "None") & (s.pt_pool > 0)]; nS = int((nv.ref <= 0).sum())
    g1, g0 = a[a.skill_pool > 0], a[a.skill_pool <= 0]
    rows.append(dict(kondisi=b, terima=M, salah=S, batas=100 * prop1(S, M), konservatif=100 * prop1(Sc, M), yield_=100 * M / T, total_delta=a.ref.sum(),
                     naif_pct_salah=100 * nS / max(len(nv), 1), med_skill_pool=s.skill_pool.median(), pct_tugas_skill_le0=100 * (s.skill_pool <= 0).mean(),
                     ekstrem_pool=s.ekstrem_pool.mean(), epoch=s.epoch.median(),
                     skillpos=f"{len(g1)}/{int((g1.ref<=0).sum())}", skillneg=f"{len(g0)}/{int((g0.ref<=0).sum())}",
                     lulus="YA" if prop1(S, M) < .1 and 100 * M / T >= 5 else "TIDAK"))
r = pd.DataFrame(rows); pd.set_option("display.width", 260); print(r.round(3).to_string(index=False)); r.to_csv(O / "analisis_mini36.csv", index=False)
pa = r.iloc[0].lulus == "YA" and r.iloc[0].med_skill_pool > 0; pb = r.iloc[1].lulus == "TIDAK" and r.iloc[1].med_skill_pool <= 0
print(f"P-a (MLP_ES lulus & skill>0): {pa} | P-b (MLP_noES gagal & skill<=0): {pb}")

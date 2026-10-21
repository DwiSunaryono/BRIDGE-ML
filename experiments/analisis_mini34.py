"""Analisis MINI-34 (PRAREG_MINI34.md): XGB satu-faktor dan diagnostik skill_pool (ambang 0, tidak disetel)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent; O = HERE / "out"
def kl(p, q):
    q = min(max(q, 1e-15), 1 - 1e-15)
    return (p * np.log(p / q) if p > 0 else 0.0) + ((1 - p) * np.log((1 - p) / (1 - q)) if p < 1 else 0.0)
def prop1(S, M, d=.05):
    if M == 0:
        return 1.0
    ph = S / M; f = lambda u: M * kl(ph, u) - np.log(1 / d)
    return 1.0 if f(1 - 1e-12) <= 0 else brentq(f, ph if ph >= 1 else ph + 1e-15, 1 - 1e-12)
d = pd.concat([pd.read_csv(O / f"mini34_m{i}.csv", keep_default_na=False) for i in (1, 2, 3)], ignore_index=True)
err = d.cfg.str.startswith("ERROR").sum(); d = d[~d.cfg.str.startswith("ERROR")].copy()
for c in ("lo95", "pt_pool", "ref", "skill_pool", "ekstrem_pool", "ll_pool", "ll_ref"): d[c] = d[c].astype(float)
print(f"baris {len(d)}, ERROR {err}, tugas {d.dataset.nunique()}, kondisi {d.base.nunique()}")
m33 = pd.concat([pd.read_csv(O / f"mini33_m{i}.csv", keep_default_na=False) for i in (1, 2, 3)]); j = d.merge(m33, on=["dataset", "base"], suffixes=("", "_33"))
print("replikasi MINI-33 (HGB, XGBeq):", len(j), bool((j.lo95 == j.lo95_33.astype(float)).all()), bool((j.cfg == j.cfg_33).all()))
URUT = ["HGB", "XGBeq", "X0", "X_ES", "X_MCW5", "X_LEAF31", "X_L2_0", "X_N30", "X_N100"]
rows = []
for b in URUT:
    s = d[d.base == b]; T = len(s); a = s[s.lo95 > .005]; M, S = len(a), int((a.ref <= 0).sum()); Sc = int((a.kelas != "membaik").sum())
    nv = s[(s.cfg != "None") & (s.pt_pool > 0)]; nS = int((nv.ref <= 0).sum())
    rows.append(dict(kondisi=b, terima=M, salah=S, batas=100 * prop1(S, M), konservatif=100 * prop1(Sc, M), yield_=100 * M / T, total_delta=a.ref.sum(),
                     naif_terima=len(nv), naif_pct_salah=100 * nS / max(len(nv), 1), naif_batas=100 * prop1(nS, len(nv)),
                     med_skill_pool=s.skill_pool.median(), pct_tugas_skill_le0=100 * (s.skill_pool <= 0).mean(), ekstrem_pool=s.ekstrem_pool.mean(),
                     lulus="YA" if prop1(S, M) < .1 and 100 * M / T >= 5 else "TIDAK"))
r = pd.DataFrame(rows); pd.set_option("display.width", 260); print(r.round(2).to_string(index=False)); r.to_csv(O / "analisis_mini34.csv", index=False)
rho, p = spearmanr(r.med_skill_pool, r.batas); print(f"\n(d-i) Spearman median skill_pool vs batas, 9 kondisi: rho={rho:.3f} p={p:.4f}")
acc = d[d.lo95 > .005]
print("(d-ii) aturan 'pakai gate hanya bila skill_pool > 0' (gabungan 9 kondisi):")
for nm, g in (("skill > 0", acc[acc.skill_pool > 0]), ("skill <= 0", acc[acc.skill_pool <= 0])):
    M, S = len(g), int((g.ref <= 0).sum()); print(f"   {nm}: terima {M}, salah {S} ({100*S/max(M,1):.2f}%), batas Prop.1 {100*prop1(S,M):.2f}%")
print("   per kondisi (skill>0 | skill<=0): terima/salah/batas")
for b in URUT:
    g = acc[acc.base == b]; x = [(len(h), int((h.ref <= 0).sum())) for h in (g[g.skill_pool > 0], g[g.skill_pool <= 0])]
    print(f"   {b:9s} " + " | ".join(f"{M}/{S}/{100*prop1(S,M):.1f}%" for M, S in x))

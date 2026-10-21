"""Analisis MINI-33 (PRAREG_MINI33.md): HGB vs XGB-reg sebagai model dasar, tugas Q baru T1500-T2899. Kriteria terkunci:
batas Prop.1 (delta 0,05) < 10% dan yield >= 5/100. Juga: naif (delta titik POOL > 0 tidak tersedia di sini, jadi tidak
dilaporkan), batas konservatif (tidak-membaik dihitung merugikan), total delta, log-loss model dasar, pembanding MINI-31
(XGB bawaan)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from scipy.optimize import brentq
def kl(p, q):  # salinan analisis_mini29.kl/prop1 (impor langsung menjalankan analisis MINI-29)
    q = min(max(q, 1e-15), 1 - 1e-15)
    return (p * np.log(p / q) if p > 0 else 0.0) + ((1 - p) * np.log((1 - p) / (1 - q)) if p < 1 else 0.0)
def prop1(S, M, d=.05):
    if M == 0:
        return 1.0
    ph = S / M; f = lambda u: M * kl(ph, u) - np.log(1 / d)
    return 1.0 if f(1 - 1e-12) <= 0 else brentq(f, ph if ph >= 1 else ph + 1e-15, 1 - 1e-12)
d = pd.concat([pd.read_csv(HERE / "out" / f"mini33_m{i}.csv", keep_default_na=False) for i in (1, 2, 3)], ignore_index=True)
err = d[d.cfg.str.startswith("ERROR")]; d = d[~d.cfg.str.startswith("ERROR")].copy()
print(f"baris {len(d)}, ERROR {len(err)}, tugas unik {d.dataset.nunique()} (harus 1400), duplikat {d.duplicated(['dataset','base']).sum()}")
for c in ("lo95", "ref", "logloss_base"): d[c] = d[c].astype(float)
rows = []  # merugikan = delta nyata (referensi) <= 0 (definisi MINI-25/31); konservatif = kelas != membaik
def ringkas(nama, s, acc):
    a = s[acc]; M = len(a); S = int((a.ref <= 0).sum()); Sc = int((a.kelas != "membaik").sum()); T = s.dataset.nunique()
    rows.append(dict(kebijakan=nama, tugas=T, diterima=M, yield_per100=100 * M / T, merugikan=S, pct_merugikan=100 * S / max(M, 1),
                     batas_prop1=100 * prop1(S, M), batas_konservatif=100 * prop1(Sc, M), total_delta=a.ref.sum(),
                     median_logloss_dasar=s.logloss_base.median() if "logloss_base" in s else np.nan,
                     LULUS="YA" if M and prop1(S, M) < .10 and 100 * M / T >= 5 else "TIDAK"))
for b in ("HGB", "XGBeq"):
    s = d[d.base == b]; ringkas(f"BRIDGE, dasar {b}", s, s.lo95 > .005)
m31 = pd.concat([pd.read_csv(HERE / "out" / f"mini31_xgb_m{i}.csv", keep_default_na=False) for i in (1, 2, 3)], ignore_index=True)
m31 = m31[~m31.cfg.str.startswith("ERROR")].copy(); m31["lo"] = m31["lo_K3_q0.05"].astype(float); m31["ref"] = m31.ref.astype(float)
ringkas("MINI-31: BRIDGE, XGB bawaan (T0000-T1399)", m31, m31.lo > .005)
r = pd.DataFrame(rows); pd.set_option("display.width", 250); print(r.round(3).to_string(index=False))
r.to_csv(HERE / "out" / "analisis_mini33.csv", index=False)
p = d.pivot(index="dataset", columns="base", values="lo95"); print("\nkesepakatan keputusan HGB vs XGB-reg (tugas sama):")
print(pd.crosstab(p.HGB > .005, p.XGBeq > .005, rownames=["HGB terima"], colnames=["XGBreg terima"]))

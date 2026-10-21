"""Analisis MINI-32 (PRAREG_MINI32.md): ablasi faktorial 2x2x2.

Sel adalah (sumber bukti, statistik, margin); terima bila statistik > margin dan ada konfigurasi yang berubah.
Merugikan berarti delta nyata (referensi) <= 0. Batas: nyata memakai KL rasio (b=1) dan LOCO per dataset;
Q memakai Proposisi 1 (delta 0,05). Cek identitas: lo_cf_boot == lo95 run asal (MINI-21 union_wF1 / MINI-25).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import brentq
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from verify_exp41 import kl_ratio_up as up
O = HERE / "out"
rd = lambda f: pd.read_csv(O / f, keep_default_na=False, float_precision="round_trip", na_values={c: ["nan", "NaN"] for c in ("lo95", "ref", "lo_pool_boot")})
def kl(p, q):
    q = min(max(q, 1e-15), 1 - 1e-15)
    return (p * np.log(p / q) if p > 0 else 0.0) + ((1 - p) * np.log((1 - p) / (1 - q)) if p < 1 else 0.0)
def prop1(S, M, d=.05):
    if M == 0:
        return 1.0
    ph = S / M; f = lambda u: M * kl(ph, u) - np.log(1 / d)
    return 1.0 if f(1 - 1e-12) <= 0 else brentq(f, ph if ph >= 1 else ph + 1e-15, 1 - 1e-12)

N = pd.concat([rd("mini32_nyata_m1.csv"), rd("mini32_nyata_m2.csv")], ignore_index=True); Q = rd("mini32_Q.csv")
for nama, s in (("nyata", N), ("Q", Q)):
    print(nama, "baris", len(s), "ERROR", int(s.cfg.astype(str).str.startswith("ERROR").sum()))
m21 = pd.concat([rd(f"mini21_m{i}.csv") for i in (1, 2, 3)], ignore_index=True); m21 = m21[m21.design == "union_wF1"]
j = N.merge(m21[["dataset", "rep", "lo95", "cfg"]], on=["dataset", "rep"], suffixes=("", "_asal"))
print("identitas nyata:", len(j), bool((j.lo_cf_boot == j.lo95).all()), bool((j.cfg == j.cfg_asal).all()))
m25 = pd.concat([rd(f"mini25_m{i}.csv") for i in (1, 2, 3)], ignore_index=True)
j = Q.merge(m25[["dataset", "lo95", "cfg"]], on="dataset", suffixes=("", "_asal"))
print("identitas Q:", len(j), bool((j.lo_cf_boot == j.lo95).all()), bool((j.cfg == j.cfg_asal).all()))

rows = []
for nama, s in (("nyata (11 dataset x 100)", N), ("Q (1.400 tugas)", Q)):
    NT = len(s); ubah = s.cfg != "None"
    for src in ("POOL-sama", "cross-fit"):
        for st in ("titik", "bootstrap"):
            col = {("POOL-sama", "titik"): "pt_pool", ("POOL-sama", "bootstrap"): "lo_pool_boot", ("cross-fit", "titik"): "pt_cf", ("cross-fit", "bootstrap"): "lo_cf_boot"}[src, st]
            for mg in (0.0, 0.005):
                acc = ubah & (s[col].astype(float) > mg); a = s[acc]; n, k = len(a), int((a.ref <= 0).sum())
                if nama.startswith("Q"):
                    b, lo = prop1(k, n), np.nan
                else:
                    b = up(k, n, NT, 1); per = s.groupby("dataset").size()
                    lo = max(up(int((a[a.dataset != q].ref <= 0).sum()), int((a.dataset != q).sum()), NT - per[q], 1) if (a.dataset != q).sum() else 1.0 for q in per.index)
                label = {("POOL-sama", "titik", 0.0): "naif", ("POOL-sama", "titik", 0.005): "VALMARGIN 0,005",
                         ("cross-fit", "bootstrap", 0.0): "boot0", ("cross-fit", "bootstrap", 0.005): "BRIDGE"}.get((src, st, mg), "")
                rows.append(dict(data=nama, sumber=src, statistik=st, margin=mg, nama=label, terima=n, salah=k, pct_salah=100 * k / max(n, 1),
                                 batas=100 * b, LOCO=100 * lo if lo == lo else np.nan, yield_=100 * n / NT, total_delta=a.ref.sum(),
                                 lulus=bool(b < .1 and (lo != lo or lo < .1) and 100 * n / NT >= 5)))
r = pd.DataFrame(rows); pd.set_option("display.width", 250); print(r.round(2).to_string(index=False)); r.to_csv(O / "analisis_mini32.csv", index=False)

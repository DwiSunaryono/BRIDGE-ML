"""Analisis MINI-35 (PRAREG_MINI35.md): ukuran ruang kandidat x sumber bukti. Merugikan berarti delta nyata <= 0."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); O = HERE / "out"
from analisis_mini32 import prop1, up, rd  # noqa: impor ini juga menjalankan analisis MINI-32 untuk konteks
N = pd.concat([rd("mini35_m1_nyata.csv"), rd("mini35_m2.csv")], ignore_index=True); Q = pd.concat([rd("mini35_m3.csv"), rd("mini35_m1_Q.csv")], ignore_index=True)
print("\n=== MINI-35 ===")
for nama, s in (("nyata", N), ("Q", Q)):
    print(nama, "baris", len(s), "ERROR", int(s.cfg.astype(str).str.startswith("ERROR").sum()), "trial unik", s[["dataset", "rep"]].drop_duplicates().shape[0])
m32 = {"nyata": pd.concat([rd("mini32_nyata_m1.csv"), rd("mini32_nyata_m2.csv")]), "Q": rd("mini32_Q.csv")}
for nama, s in (("nyata", N), ("Q", Q)):
    j = s[s.K == "K173"].merge(m32[nama], on=["dataset", "rep"], suffixes=("", "_32"))
    print("replikasi K173 == MINI-32", nama, len(j), all(((j[c] == j[c + "_32"]) | (j[c].isna() & j[c + "_32"].isna())).all() for c in ("cfg", "ref", "pt_pool", "lo_pool_boot", "pt_cf", "lo_cf_boot")))  # perbandingan NaN == NaN
rows = []
for nama, S in (("nyata", N), ("Q", Q)):
    for K in ("K173", "Ksedang", "Kbesar"):
        s = S[S.K == K]; NT = len(s); ubah = s.cfg != "None"
        sel = {"naif": ubah & (s.pt_pool > 0), "POOL-boot 0,005": ubah & (s.lo_pool_boot.astype(float) > .005), "BRIDGE": ubah & (s.lo_cf_boot > .005)}
        kb = int(sel["BRIDGE"].sum())
        for nm, acc in sel.items():
            a = s[acc]; n, k = len(a), int((a.ref <= 0).sum())
            if nama == "Q":
                b, lo = prop1(k, n), np.nan
            else:
                b = up(k, n, NT, 1); per = s.groupby("dataset").size()
                lo = max(up(int((a[a.dataset != q].ref <= 0).sum()), int((a.dataset != q).sum()), NT - per[q], 1) if (a.dataset != q).sum() else 1.0 for q in per.index)
            col = {"naif": "pt_pool", "POOL-boot 0,005": "lo_pool_boot", "BRIDGE": "lo_cf_boot"}[nm]
            top = s[ubah].assign(v=s[col].astype(float)).sort_values("v", ascending=False).head(kb)
            rows.append(dict(data=nama, K=K, nK=int(s.nK.iloc[0]), aturan=nm, terima=n, salah=k, pct_salah=100 * k / max(n, 1), batas=100 * b,
                             LOCO=100 * lo if lo == lo else np.nan, yield_=100 * n / NT, total_delta=a.ref.sum(),
                             pct_salah_yield_BRIDGE=100 * (top.ref <= 0).mean(), lulus=bool(b < .1 and (lo != lo or lo < .1) and 100 * n / NT >= 5)))
r = pd.DataFrame(rows); pd.set_option("display.width", 260); print(r.round(2).to_string(index=False)); r.to_csv(O / "analisis_mini35.csv", index=False)

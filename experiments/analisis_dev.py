"""Ringkasan standar output pengembangan mini-XX per sel kolom pengelompok.
Per sel: terima (lo95 > .005), salah (ref <= 0), batas KL rasio (b=1: <=1 terima/trial/sel),
LOCO maks, power R0 (k=0 setelah LOCO), yield/100, jumlah dataset dengan yield >= 5/100.
Pakai: python analisis_dev.py <csv> <kolom_sel>"""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_exp41 import kl_ratio_up

d = pd.read_csv(HERE / "out" / sys.argv[1], keep_default_na=False, na_values={c: ["nan", "NaN"] for c in ("lo95", "ref")})
cell = sys.argv[2]
print(f"{'sel':<14}{'trial':>6}{'ds':>4}{'%membaik':>9}{'TERIMA':>7}{'salah':>6}{'KL%':>7}{'LOCO%':>7}{'R0%':>6}{'yield':>6}{'ds>=5':>6}")
for c, s in d.groupby(cell):
    s = s[s.kelas != "skip"]; dss = list(s.dataset.unique()); NT = len(s)
    a = s[s.lo95 > .005]; n, k = len(a), int((a.ref <= 0).sum())
    per = {x: (s.dataset == x).sum() for x in dss}
    kl = kl_ratio_up(k, n, NT, 1) if n else float("nan")
    loco = max(kl_ratio_up(int((a[a.dataset != x].ref <= 0).sum()), int((a.dataset != x).sum()), NT - per[x], 1)
               if (a.dataset != x).sum() else 1.0 for x in dss)
    r0 = max(kl_ratio_up(0, int((a.dataset != x).sum()), NT - per[x], 1) if (a.dataset != x).sum() else 1.0 for x in dss)
    yl = 100 * n / NT; dy = sum(100 * (a.dataset == x).sum() / per[x] >= 5 for x in dss)
    mb = 100 * (s.kelas == "membaik").mean()
    print(f"{str(c):<14}{NT:>6}{len(dss):>4}{mb:>9.1f}{n:>7}{k:>6}{100*kl:>7.2f}{100*loco:>7.2f}{100*r0:>6.2f}{yl:>6.1f}{dy:>6}")

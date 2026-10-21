"""Analisis praregistrasi MINI-28 / 28+28b / 28+28b+28c: kebijakan x (terima/salah, batas KL b=1, LOCO per sumber,
yield, tugas dengan yield >= 5, total delta) dan aturan naif.
Pakai: python analisis_kesehatan.py <file1.csv> [file2.csv ...]"""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_exp41 import kl_ratio_up as up
import health_data as H
KL = dict(H.KLASTER, covid_mx="covid_mx", wids_icu="wids")
d = pd.concat([pd.read_csv(HERE / "out" / f, keep_default_na=False, float_precision="round_trip",
                           na_values={c: ["nan", "NaN"] for c in ("lo95", "ref")}) for f in sys.argv[1:]], ignore_index=True)
print("baris", len(d), "tugas", d.dataset.nunique(), "error", int((d.kelas == "error").sum()), "skip", int((d.kelas == "skip").sum()))
d = d[~d.kelas.isin(["error", "skip"])]; d["sumber"] = d.dataset.map(KL)
for des, s in d.groupby("design"):
    NT = len(s); per = s.groupby("sumber").size()
    for nm, acc in (("BRIDGE", s.lo95 > .005), ("naif", s.cfg != "None")):
        a = s[acc]; n, k = len(a), int((a.ref <= 0).sum())
        loco = max(up(int((a[a.sumber != q].ref <= 0).sum()), int((a.sumber != q).sum()), NT - per[q], 1)
                   if (a.sumber != q).sum() else 1.0 for q in per.index)
        tug = sum(100 * (a.dataset == t).sum() / (s.dataset == t).sum() >= 5 for t in s.dataset.unique())
        lulus = up(k, n, NT, 1) < .1 and loco < .1 and 100 * n / NT >= 5 and tug >= 3
        print(f"{des:14} {nm:6} terima {n:5} salah {k:4} ({100*k/max(n,1):5.1f}%) jelas-buruk {int((a.kelas=='memburuk').sum()):3} "
              f"batas {100*up(k,n,NT,1):6.2f}% LOCO-sumber {100*loco:6.2f}% yield {100*n/NT:5.1f} tugas>=5 {tug:2} "
              f"totalΔ {a.ref.sum():7.2f} {'LULUS' if lulus and nm=='BRIDGE' else ''}")

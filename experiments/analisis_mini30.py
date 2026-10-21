"""Analisis praregistrasi MINI-30 (PRAREG_MINI30.md)."""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_exp41 import kl_ratio_up as up
d = pd.concat([pd.read_csv(HERE / "out" / f"mini30_m{i}.csv", keep_default_na=False, float_precision="round_trip",
                           na_values={c: ["nan", "NaN"] for c in ("lo95", "ref", "dpool", "d_sens", "d_spes")}) for i in (1, 2, 3)],
              ignore_index=True)
print("baris", len(d), "tugas", d.dataset.nunique(), "sumber", d.sumber.nunique(), "error", int((d.kelas == "error").sum()),
      "skip", int((d.kelas == "skip").sum()), "dup", int(d.duplicated(["dataset", "rep", "metric"]).sum()))
d = d[~d.kelas.isin(["error", "skip"])]
for m in ("bacc", "nb20", "nb10", "F2", "wF1"):
    s = d[d.metric == m]; NT = len(s); per = s.groupby("sumber").size()
    for nm, acc in (("BRIDGE", s.lo95 > .005), ("naif", s.cfg != "None")):
        a = s[acc]; n, k = len(a), int((a.ref <= 0).sum())
        loco = max(up(int((a[a.sumber != q].ref <= 0).sum()), int((a.sumber != q).sum()), NT - per[q], 1)
                   if (a.sumber != q).sum() else 1.0 for q in per.index)
        tug = sum(100 * (a.dataset == t).sum() / (s.dataset == t).sum() >= 5 for t in s.dataset.unique())
        lulus = up(k, n, NT, 1) < .1 and loco < .1 and 100 * n / NT >= 5 and tug >= 3
        tag = ("UTAMA " if m in ("bacc", "nb20") else "sekunder") if nm == "BRIDGE" else ""
        print(f"{m:5} {tag:8} {nm:6} terima {n:5} salah {k:4} ({100*k/max(n,1):5.1f}%) jelas-buruk {int((a.kelas=='memburuk').sum()):3} "
              f"batas {100*up(k,n,NT,1):6.2f}% LOCO {100*loco:6.2f}% yield {100*n/NT:5.1f} tugas>=5 {tug:2} "
              f"Δsens {a.d_sens.mean():+.3f} Δspes {a.d_spes.mean():+.3f} tandai-semua {int((a.d_spes < -.5).sum()):4} "
              f"{'LULUS' if lulus and nm == 'BRIDGE' else ('gagal' if nm == 'BRIDGE' else '')}")

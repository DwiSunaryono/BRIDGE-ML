"""Analisis MINI-40 (PRAREG_MINI40.md), ditulis sebelum run, salinan analisis MINI-39 dengan satu arm
multi-kelas "M". Per learner (HGB, DT, SVC, KNN), untuk aturan naif dan BRIDGE:
  keputusan, diterima, harmful, % harmful, batas KL rasio (b=1, alpha .025+.025),
  LOCO-dataset, R0, yield/100, jumlah dataset dengan yield >= 5/100, total Δ referensi.
Deskriptif per learner: rata2/median Δ referensi semua trial, % tuned lebih baik (Δ > 0), % jelas lebih buruk
(kelas 'memburuk'), jumlah error/skip.
Kriteria lulus BRIDGE per learner (kriteria MINI-22, dibekukan di pra-registrasi):
  batas < 10%, LOCO < 10%, R0 < 10%, yield >= 5/100, >= 3 dataset dengan yield >= 5/100
Status tambahan: 'aman, yield rendah' bila semua syarat risiko lolos tetapi syarat yield gagal.
Pakai: python analisis_mini40.py  (membaca out/mini40_m_*.csv)"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); O = HERE / "out"
from verify_exp41 import kl_ratio_up as up

rows, desk = [], []
for arm in ("M",):
    fs = sorted(glob.glob(str(O / "mini40_m_*.csv")))
    if not fs:
        continue
    d = pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""]) for f in fs], ignore_index=True)
    for L in ("HGB", "DT", "SVC", "KNN"):
        s = d[d.learner == L]; ok = s.error.fillna("").astype(str) == ""; err = s[~ok]; s = s[ok].copy()  # 2026-10-15: kolom error kosong terbaca NaN
        NT = len(s); per = s.groupby("dataset").size()
        desk.append(dict(arm=arm, learner=L, trial=NT, error_skip=len(err), mean_ref=s.ref.mean(), median_ref=s.ref.median(),
                         pct_tuned_better=100 * (s.ref > 0).mean(), pct_clearly_worse=100 * (s.kelas == "memburuk").mean()))
        for aturan, col in (("naif", "naive"), ("BRIDGE", "bridge")):
            a = s[s[col] == 1]; n, k = len(a), int((a.ref <= 0).sum()); b = up(k, n, NT, 1) if NT else 1.0
            r = dict(arm=arm, learner=L, aturan=aturan, trial=NT, terima=n, harmful=k, pct_harmful=100 * k / max(n, 1),
                     batas=100 * b, yield_=100 * n / max(NT, 1), total_delta=a.ref.sum())
            ok_y = r["yield_"] >= 5
            if True:
                dss = list(per.index)
                lo = max(up(int((a[a.dataset != q].ref <= 0).sum()), int((a.dataset != q).sum()), NT - per[q], 1)
                         if (a.dataset != q).sum() else 1.0 for q in dss)
                r0 = max(up(0, int((a.dataset != q).sum()), NT - per[q], 1) if (a.dataset != q).sum() else 1.0 for q in dss)
                nds = int(sum(100 * (a.dataset == q).sum() / per[q] >= 5 for q in dss))
                r.update(LOCO=100 * lo, R0=100 * r0, ds_yield5=nds)
                risk_ok = b < .1 and lo < .1 and r0 < .1; ok_y = ok_y and nds >= 3
            else:
                risk_ok = b < .1
            r["lulus"] = bool(risk_ok and ok_y)
            r["status"] = "LULUS" if r["lulus"] else ("aman, yield rendah" if risk_ok else "gagal (risiko)")
            rows.append(r)
res, dsk = pd.DataFrame(rows), pd.DataFrame(desk)
pd.set_option("display.width", 250)
print(res.round(2).to_string(index=False)); print(); print(dsk.round(4).to_string(index=False))
res.to_csv(O / "analisis_mini40.csv", index=False); dsk.to_csv(O / "analisis_mini40_deskriptif.csv", index=False)

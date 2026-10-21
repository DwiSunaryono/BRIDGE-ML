"""Analisis MINI-41 (PRAREG_MINI41.md), ditulis SEBELUM run.
Per sel (level prevalensi x metrik) dan aturan (naif, BRIDGE):
  keputusan (trial 'ok' + 'abstain'; abstain = tidak ada update, tidak diterima), diterima, harmful, % harmful,
  batas KL rasio (b=1), LOCO-dataset, R0, yield/100, jumlah dataset dgn yield >= 5/100, total Δ referensi,
  % jelas lebih buruk di antara yang diterima; error dilaporkan dan dikeluarkan.
Kriteria LULUS BRIDGE per sel (kriteria MINI-22): batas < 10%, LOCO < 10%, R0 < 10%, yield >= 5/100, >= 3 dataset dgn
yield >= 5/100. 'aman, yield rendah' bila syarat risiko lolos tetapi syarat yield gagal. Sel utama: metrik wF1.
Tabel kedua: per sel x dataset (diterima/harmful naif dan BRIDGE), untuk panduan jumlah label minoritas di POOL.
Pakai: python analisis_mini41.py  (membaca out/mini41_m_*.csv)"""
import sys, glob
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); O = HERE / "out"
from verify_exp41 import kl_ratio_up as up

d = pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""]) for f in sorted(glob.glob(str(O / "mini41_m_*.csv")))],
              ignore_index=True)
d["status"] = d.status.astype(str)
rows, per_ds = [], []
for (lev, m), s0 in d.groupby(["level", "metric"], sort=False):
    err = s0[~s0.status.isin(["ok", "abstain"])]; s = s0[s0.status.isin(["ok", "abstain"])].copy()
    for c in ("naive", "bridge", "harmful"):
        s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0).astype(int)
    s["ref"] = pd.to_numeric(s.ref, errors="coerce")
    NT = len(s); per = s.groupby("dataset").size(); dss = list(per.index)
    for aturan, col in (("naif", "naive"), ("BRIDGE", "bridge")):
        a = s[s[col] == 1]; n, k = len(a), int((a.ref <= 0).sum()); b = up(k, n, NT, 1) if NT else 1.0
        lo = max(up(int((a[a.dataset != q].ref <= 0).sum()), int((a.dataset != q).sum()), NT - per[q], 1)
                 if (a.dataset != q).sum() else 1.0 for q in dss)
        r0 = max(up(0, int((a.dataset != q).sum()), NT - per[q], 1) if (a.dataset != q).sum() else 1.0 for q in dss)
        nds = int(sum(100 * (a.dataset == q).sum() / per[q] >= 5 for q in dss)); yl = 100 * n / max(NT, 1)
        risk_ok = b < .1 and lo < .1 and r0 < .1; ok_y = yl >= 5 and nds >= 3
        rows.append(dict(level=lev, metric=m, aturan=aturan, keputusan=NT, abstain=int((s.status == "abstain").sum()),
                         error=len(err), terima=n, harmful=k, pct_harmful=100 * k / max(n, 1), batas=100 * b, LOCO=100 * lo,
                         R0=100 * r0, yield_=yl, ds_yield5=nds, total_delta=a.ref.sum(),
                         pct_jelas_buruk=100 * (a.kelas == "memburuk").mean() if n else 0.0,
                         status="LULUS" if risk_ok and ok_y else ("aman, yield rendah" if risk_ok else "gagal (risiko)")))
        for q in dss:
            aq = a[a.dataset == q]
            per_ds.append(dict(level=lev, metric=m, aturan=aturan, dataset=q, keputusan=int(per[q]), terima=len(aq),
                               harmful=int((aq.ref <= 0).sum()), min_pool_median=float(s[s.dataset == q].min_pool.median())))
res, pds = pd.DataFrame(rows), pd.DataFrame(per_ds)
pd.set_option("display.width", 250)
print(res.round(2).to_string(index=False))
res.to_csv(O / "analisis_mini41.csv", index=False); pds.to_csv(O / "analisis_mini41_per_dataset.csv", index=False)

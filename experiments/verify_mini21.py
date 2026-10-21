"""Verifikasi penuh MINI-21 (validasi akhir, dataset baru) sesuai PRAREG_MINI21.md.
1. Kelengkapan: 11 ds x 100 rep x 5 desain, tanpa duplikat, tanpa error.
2. Statistik dua implementasi: KL rasio/LOCO/R0 per desain dengan brentq (verify_exp41) dan bisection
   (verify_exp41_independent), harus sama (toleransi 1e-9).
3. Replay lintas mesin: 20 trial acak (seed 4242) dijalankan ulang di laptop; semua kolom harus identik
   dengan CSV (sebagian dibuat di komputer lain).
4. Cek independen skor: pada trial replay, delta metrik referensi dihitung ulang dengan sklearn (f1_score
   weighted/macro, balanced_accuracy_score) dari label mentah, harus sama (1e-12).
Keluar non-nol bila ada yang gagal. Pakai: python verify_mini21.py <n_proses>"""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import verify_exp41 as V1
import verify_exp41_independent as V2
import mini21_final as M
import exp41_gate_confirm as G

FAILS = []
DESIGNS = ("union_wF1", "union_bacc", "union_macroF1", "fsthr_KNN", "fsthr_SVC")
SK = {"wF1": lambda y, p: f1_score(y, p, average="weighted"), "macroF1": lambda y, p: f1_score(y, p, average="macro"),
      "bacc": balanced_accuracy_score}


def ok(name, cond, info=""):
    print(("OK  " if cond else "GAGAL ") + name + (f"  [{info}]" if info else ""), flush=True)
    if not cond:
        FAILS.append(name)


def read():
    fs = ["mini21_m1.csv", "mini21_m2.csv", "mini21_m3.csv"]
    return pd.concat([pd.read_csv(HERE / "out" / f, keep_default_na=False, float_precision="round_trip",
                                  na_values={c: ["nan", "NaN"] for c in ("lo95", "ref")}) for f in fs], ignore_index=True)


def stats(s, up):
    dss = list(s.dataset.unique()); NT = len(s); per = s.groupby("dataset").size()
    a = s[s.lo95 > .005]; n, k = len(a), int((a.ref <= 0).sum())
    kl = up(k, n, NT, 1)
    loco = max(up(int((a[a.dataset != x].ref <= 0).sum()), int((a.dataset != x).sum()), NT - per[x], 1) for x in dss)
    r0 = max(up(0, int((a.dataset != x).sum()), NT - per[x], 1) for x in dss)
    return n, k, kl, loco, r0


def replay(job):
    cap = []; orig = G.counts8

    def spy(y, b, c):
        if y is M.data(job[0])[3]:  # identitas objek: hanya panggilan referensi (ukuran bisa sama dengan lipatan)
            cap.append((np.asarray(y).copy(), np.asarray(b).copy(), np.asarray(c).copy()))
        return orig(y, b, c)
    G.counts8 = spy
    try:
        rows = M.trial(job)
    finally:
        G.counts8 = orig
    return rows, cap


def main():
    d = read()
    ok("tanpa error/skip", not d.kelas.isin(["error", "skip"]).any(), str(d.kelas.value_counts().to_dict()))
    ok("kelengkapan 11 x 100 x 5, tanpa duplikat",
       len(d) == 5500 and d.dataset.nunique() == 11 and not d.duplicated(["dataset", "rep", "design"]).any()
       and set(d.design) == set(DESIGNS))
    for des in DESIGNS:
        s1, s2 = stats(d[d.design == des], V1.kl_ratio_up), stats(d[d.design == des], V2.ratio_up)
        ok(f"{des}: brentq == bisection", s1[:2] == s2[:2] and all(abs(x - y) < 1e-9 for x, y in zip(s1[2:], s2[2:])),
           f"{s1[0]}/{s1[1]} KL {100*s1[2]:.3f}% vs {100*s2[2]:.3f}%")
    jobs = [tuple(x) for x in d[["dataset", "rep"]].drop_duplicates().sample(20, random_state=4242).values]
    jobs = [(ds, int(r)) for ds, r in jobs]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 10) as p:
        res = p.map(replay, jobs, chunksize=1)
    mism = skbad = 0
    for (ds, rep), (rows, cap) in zip(jobs, res):
        for i, r in enumerate(rows):
            c = d[(d.dataset == ds) & (d.rep == rep) & (d.design == r["design"])].iloc[0]
            for col in ("TOTAL", "metric", "cfg", "D", "lo95", "ref", "kelas"):
                a, b = c[col], r[col]
                if isinstance(b, float):
                    if not (a == b or (np.isnan(a) and np.isnan(b))):
                        mism += 1
                elif str(a) != str(b):
                    mism += 1
            y, b0, c0 = cap[i]
            sk = SK[r["metric"]](y, c0) - SK[r["metric"]](y, b0)
            if abs(sk - r["ref"]) > 1e-12:
                skbad += 1
    ok("replay 20 trial lintas mesin identik", mism == 0, f"{mism} beda")
    ok("Δ referensi == sklearn (100 baris)", skbad == 0, f"{skbad} beda")
    print("SEMUA LULUS" if not FAILS else f"GAGAL: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

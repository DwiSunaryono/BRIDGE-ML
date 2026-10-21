"""Verifikasi MINI-20 (konfirmasi ukuran latih) sesuai PRAREG_MINI20.md.
1. Kelengkapan: 27 ds x 100 rep x {1000, 2430, 6000}, tanpa duplikat; skip dilaporkan.
2. Statistik dua implementasi (brentq vs bisection) per ukuran.
3. Replay 20 trial acak (seed 4242) di mesin berbeda dari pembuat CSV, harus identik.
Pakai: python verify_mini20.py <n_proses>"""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
sys.argv = sys.argv[:2] + ["x", "out", "1020000000"]  # OFF dibaca mini17 saat impor dari argv[4]
import verify_exp41 as V1
import verify_exp41_independent as V2
import mini17_trainsize as M
assert M.OFF == 1_020_000_000, M.OFF  # Seed dev sempat terpakai dan membuat replay gagal palsu

FAILS = []


def ok(name, cond, info=""):
    print(("OK  " if cond else "GAGAL ") + name + (f"  [{info}]" if info else ""), flush=True)
    if not cond:
        FAILS.append(name)


def stats(s, up):
    s = s[s.kelas != "skip"]; NT = len(s); per = s.groupby("dataset").size()
    a = s[s.lo95 > .005]; n, k = len(a), int((a.ref <= 0).sum())
    loco = max(up(int((a[a.dataset != x].ref <= 0).sum()), int((a.dataset != x).sum()), NT - per[x], 1) for x in per.index)
    return n, k, up(k, n, NT, 1), loco


def main():
    rd = lambda f: pd.read_csv(HERE / "out" / f, keep_default_na=False, float_precision="round_trip",
                               na_values={c: ["nan", "NaN"] for c in ("lo95", "ref")})
    d = pd.concat([rd("mini20_size_a.csv"), rd("mini20_size_b.csv")], ignore_index=True)
    ok("kelengkapan 27 x 100 x 3 tanpa duplikat", len(d) == 8100 and d.dataset.nunique() == 27
       and not d.duplicated(["dataset", "TOTAL", "rep"]).any(), str(d.kelas.value_counts().to_dict()))
    for T in (1000, 2430, 6000):
        s1, s2 = stats(d[d.TOTAL == T], V1.kl_ratio_up), stats(d[d.TOTAL == T], V2.ratio_up)
        ok(f"TOTAL {T}: brentq == bisection, KL & LOCO < 10%",
           s1[:2] == s2[:2] and abs(s1[2] - s2[2]) < 1e-9 and abs(s1[3] - s2[3]) < 1e-9 and s1[2] < .1 and s1[3] < .1,
           f"{s1[0]}/{s1[1]} KL {100*s1[2]:.3f}% LOCO {100*s1[3]:.3f}%")
    smp = d.sample(20, random_state=4242)
    jobs = [(r.dataset, M.DS[r.dataset], M.SIZES.index(int(r.TOTAL)), int(r.rep)) for r in smp.itertuples()]
    with Pool(int(sys.argv[1])) as p:
        res = p.map(M.trial, jobs, chunksize=1)
    mism = 0
    for r, row in zip(smp.itertuples(), res):
        for col in ("cfg", "D", "lo95", "ref", "kelas"):
            a, b = getattr(r, col), row[col]
            if isinstance(b, float):
                mism += not (a == b or (np.isnan(a) and np.isnan(b)))
            else:
                mism += str(a) != str(b)
    ok("replay 20 trial lintas mesin identik", mism == 0, f"{mism} beda")
    print("SEMUA LULUS" if not FAILS else f"GAGAL: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

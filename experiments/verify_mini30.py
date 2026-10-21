"""Verifikasi MINI-30: (1) kelengkapan; (2) batas KL brentq == bisection untuk 5 metrik; (3) replay 12 trial acak
(seed 4242), sebagian berasal dari komputer lain, semua kolom harus identik."""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import verify_exp41 as V1, verify_exp41_independent as V2
import mini30_klinis as M
FAILS = []
def ok(n, c, i=""):
    print(("OK  " if c else "GAGAL ") + n + (f"  [{i}]" if i else ""), flush=True); (c or FAILS.append(n))
d = pd.concat([pd.read_csv(HERE / "out" / f"mini30_m{i}.csv", keep_default_na=False, float_precision="round_trip",
                           na_values={c: ["nan", "NaN"] for c in ("lo95", "ref", "dpool", "d_sens", "d_spes")}) for i in (1, 2, 3)], ignore_index=True)
ok("kelengkapan 12 tugas x 200 x 5 metrik", len(d) == 12000 and not d.duplicated(["dataset", "rep", "metric"]).any() and (d.kelas != "error").all())
for m in M.METRIK:
    s = d[d.metric == m]; a = s[s.lo95 > .005]; n, k = len(a), int((a.ref <= 0).sum())
    b1, b2 = V1.kl_ratio_up(k, n, len(s), 1), V2.ratio_up(k, n, len(s), 1)
    ok(f"{m}: brentq == bisection", abs(b1 - b2) < 1e-9, f"{n}/{k} {100*b1:.3f}% vs {100*b2:.3f}%")
jobs = [(r.dataset, int(r.rep)) for r in d[["dataset", "rep"]].drop_duplicates().sample(12, random_state=4242).itertuples()]
with Pool(12) as p:
    res = p.map(M.trial, jobs, chunksize=1)
mism = 0
for (ds, rep), rows in zip(jobs, res):
    for r in rows:
        c = d[(d.dataset == ds) & (d.rep == rep) & (d.metric == r["metric"])].iloc[0]
        for col in ("cfg", "lo95", "dpool", "ref", "kelas", "d_sens", "d_spes", "counts_ref"):
            a, b = c[col], r[col]
            mism += (not (a == b or (isinstance(b, float) and np.isnan(a) and np.isnan(b)))) if isinstance(b, float) else (str(a) != str(b))
ok("replay 12 trial (60 baris) lintas mesin identik", mism == 0, f"{mism} beda; tugas: {sorted(set(j[0] for j in jobs))}")
print("SEMUA LULUS" if not FAILS else f"GAGAL: {FAILS}")

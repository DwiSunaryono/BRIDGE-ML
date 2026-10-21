"""MINI-28: VALIDASI KESEHATAN BARU (praregistrasi PRAREG_MINI28.md). Kode trial = mini21_final apa adanya
(union wF1/bacc/macroF1, FS+threshold KNN/SVC; gate beku); hanya sumber data (health_data.py) & seed berbeda.
10 tugas dari 4 sumber independen: mammography, actg175, nhanes_* (4), brfss_* (4). Seed 1_028_000_000.
Pakai: python mini28_kesehatan.py <R> <n_proses> <out> <nama,dipisah,koma | semua>"""
import csv, sys, time
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mini21_final as F
import health_data as H

F.OFF = 1_028_000_000
F.DS = {k: ("kesehatan", k, ()) for k in ["mammography", "actg175"] + list(H.NH_TGT) + list(H.BR_TGT)}
F.load = lambda src, name, drop: H.load(name)

if __name__ == "__main__":
    R, Pn, outn = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    names = list(F.DS) if sys.argv[4] == "semua" else sys.argv[4].split(","); assert all(n in F.DS for n in names)
    jobs = [(ds, r) for ds in names for r in range(R)]; t0 = time.time()
    with Pool(Pn) as pool:
        res = [x for rows in pool.imap(F.trial, jobs, chunksize=2) for x in rows]
    p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print(f"{len(names)} tugas, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

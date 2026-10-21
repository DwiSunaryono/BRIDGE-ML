"""MINI-28b: validasi kesehatan tambahan untuk COVID-19 Meksiko (PRAREG_MINI28b.md). Kode trial = mini21_final apa adanya;
sumber data health_data_b.py; seed 1_028_500_000. Pakai: python mini28b_covid.py <R> <n_proses> <out>
"""
import csv, sys, time
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mini21_final as F
import health_data_b as HB

F.OFF = 1_028_500_000
F.DS = {"covid_mx": ("kesehatan", "covid_mx", ())}
F.load = lambda src, name, drop: HB.load(name)

if __name__ == "__main__":
    R, Pn, outn = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    jobs = [("covid_mx", r) for r in range(R)]; t0 = time.time()
    with Pool(Pn) as pool:
        res = [x for rows in pool.imap(F.trial, jobs, chunksize=1) for x in rows]
    p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print(f"covid_mx, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

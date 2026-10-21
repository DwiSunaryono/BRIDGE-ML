"""MINI-38 (PRAREG_MINI38.md): sensitivitas pembagian luar (seed referensi terkunci) pada 11 dataset baru (klaim utama C1).
Pembagian luar D dipecah menjadi (dev | referensi) dan diulang dengan seed referensi baru; per trial SEARCH/POOL/bag diambil ulang
dengan seed trial baru. Kode trial = mini32_ablasi (mode nyata) apa adanya: menghasilkan pt_pool (naif) dan
lo_cf_boot (BRIDGE) pada konfigurasi deploy yang sama. Uji identitas: seed referensi 20261125 + OFF 1_021_000_000
sama dengan mini32 nyata.
Pakai: python mini38_ref_seed.py <ref_seed> <off> <awal_idx> <akhir_idx> <n_proses> <out>"""
import csv, sys, time
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini32_ablasi as A  # mode nyata (bawaan)
sys.argv = _argv
REF_SEED, OFF = int(sys.argv[1]), int(sys.argv[2])
A.G.LOCKED_REF_SEED = REF_SEED; A.F.OFF = OFF; A.F._D.clear()


def run_job(job):
    r = A.run_job(job); r["ref_seed"] = REF_SEED; return r


if __name__ == "__main__":
    a, b, Pn, outn = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    names = list(A.F.DS); jobs = [(names[i], r) for i in range(a, b) for r in range(100)]; t0 = time.time()
    with Pool(Pn) as pool:
        res = pool.map(run_job, jobs, chunksize=2)
    keys = list(next(r for r in res if "ref" in r)); p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(res)
    print(f"ref_seed {REF_SEED} idx {a}-{b}, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

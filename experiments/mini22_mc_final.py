"""MINI-22: validasi akhir multi-kelas pada dataset baru, dijalankan hanya bila MINI-18 lulus.
Kode trial identik mini18_multiclass.py (diimpor apa adanya); hanya daftar dataset, loader UCI, dan seed berbeda.
Dataset (audit 2026-10-10): Fashion-MNIST (40996), DryBean (UCI 602), USPS (41082), mnist_784 (554), Kuzushiji-MNIST (41982).
Dikeluarkan: eye_movements (baris berkelompok per subjek/teks), okcupid-stem (T2: gabungan kategori pekerjaan),
jungle_chess (posisi dihasilkan mesin/solver, sintetis).
Seed 1_022_000_000 + 1000*idx + rep. Pakai: python mini22_mc_final.py <R> <n_proses> <out> <nama,dipisah,koma | semua>
"""
import csv, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini18_multiclass as M
sys.argv = _argv
import exp41_gate_confirm as G

M.OFF = 1_022_000_000
M.DS = {"Fashion-MNIST": 40996, "DryBean": "uci602", "USPS": 41082, "mnist_784": 554, "Kuzushiji-MNIST": 41982}
_load_openml = M.load


def load(did):
    if isinstance(did, str) and did.startswith("uci"):
        df = pd.read_pickle(Path.home() / "uci_cache" / f"uci_{did[3:]}.pkl")
        y0 = df.pop("target").astype(str)
        y = pd.Series(pd.Categorical(y0, categories=sorted(y0.unique())).codes, name="target")
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.reset_index(drop=True), y.reset_index(drop=True)
    return _load_openml(did)


M.load = load

if __name__ == "__main__":
    R, Pn, outn = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    names = list(M.DS) if sys.argv[4] == "semua" else sys.argv[4].split(","); assert all(n in M.DS for n in names)
    t0 = time.time(); jobs = [(ds, M.DS[ds], r) for ds in names for r in range(R)]
    with Pool(Pn) as pool:
        res = pool.map(M.trial, jobs, chunksize=2)
    p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print(f"{len(names)} dataset, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

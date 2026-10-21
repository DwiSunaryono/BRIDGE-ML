"""MINI-25: populasi tugas sintetis dengan kebenaran (hampir) diketahui, menjawab review: unit independen dan jaminan
populasi tugas. Tiap tugas T_i ~ Q diambil i.i.d. (parameter acak dari seed tugas), satu trial per tugas, sehingga
indikator (terima, salah) i.i.d. antar tugas dan batas KL rasio (b=1) adalah batas atas valid untuk
r_Q = E[A*1(Δ_T<=0)] / E[A] pada distribusi Q.
Q = campuran 3 jenis tugas (dipilih seragam per tugas; label stokastik untuk 'pohon' dan 'linier'):
   gauss : make_classification (n_features 5-40, informative 30-100%, class_sep 0.3-1.5, minoritas 0.05-0.5,
           flip_y 0-0.10, cluster/kelas 1-3)
   pohon : x ~ N(0,1)^p, logit = sum_j w_j 1[x_a > t1][x_b > t2] (4-12 interaksi) + bias -> y ~ Bernoulli(sigmoid(s*logit))
   linier: x ~ N(0,1)^p, logit = beta.x (beta ~ N(0,1), sebagian nol) * s + bias -> y ~ Bernoulli
   (s = kekuatan sinyal 0.05-1; gauss: class_sep 0.05-0.6, flip_y 0.1-0.4; bias diatur agar minoritas ~ 0.05-0.5). Desain Q
   ditetapkan dengan target laju salah aturan naif realistis (>= 15%), bukan berdasarkan hasil BRIDGE.
   Dev 3.000 baris (latih 2.430 = 630/1.800), referensi 100.000 baris dari distribusi sama (sampel i.i.d.).
Kode trial = mini24_standar (union, wF1, semua aturan pembanding) apa adanya; hanya sumber data dan seed berbeda.
Pakai: python mini25_sintetis.py <tugas_awal> <tugas_akhir> <n_proses> <out>"""
import csv, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mini21_final as F
import mini24_standar as M

F.OFF = 1_025_000_000
NT = 1500
F.DS = {f"T{i:04d}": ("sintetis", i, ()) for i in range(NT)}
_cache = {}


def param(i):
    r = np.random.default_rng(7_000_000 + i)
    jenis = ("pohon", "linier")[int(r.integers(0, 2))]  # gauss dikeluarkan setelah pilot (naif hampir tak pernah salah)
    p = int(r.integers(5, 41)); minor = float(r.uniform(.05, .5))
    if jenis == "gauss":
        inf = max(2, int(round(r.uniform(.3, 1.0) * p)))
        return dict(jenis=jenis, n_features=p, n_informative=inf, n_redundant=0,
                    n_clusters_per_class=int(r.integers(1, 4)), class_sep=float(r.uniform(.05, .6)),
                    weights=[1 - minor], flip_y=float(r.uniform(.1, .4)), random_state=int(7_500_000 + i))
    return dict(jenis=jenis, p=p, minor=minor, s=float(r.uniform(.05, 1.0)), m=int(r.integers(4, 13)),
                nol=float(r.uniform(0, .7)), seed=int(7_500_000 + i))


def gen(kw, n):
    if kw["jenis"] == "gauss":
        k = {a: b for a, b in kw.items() if a != "jenis"}
        while 2 * k["n_clusters_per_class"] > 2 ** k["n_informative"]:
            k["n_clusters_per_class"] -= 1
        return make_classification(n_samples=n, **k)
    r = np.random.default_rng(kw["seed"]); p = kw["p"]; X = r.normal(size=(n, p))
    if kw["jenis"] == "pohon":
        lg = np.zeros(n)
        for _ in range(kw["m"]):
            a, b = r.integers(0, p, 2); t1, t2 = r.normal(size=2) * .7; w = r.normal() * 2
            lg += w * ((X[:, a] > t1) & (X[:, b] > t2))
    else:
        beta = r.normal(size=p) * (r.uniform(size=p) > kw["nol"]); lg = X @ beta / np.sqrt(max(1, (beta != 0).sum()))
    lg = kw["s"] * (lg - lg.mean()) / (lg.std() + 1e-12)
    b = np.quantile(lg, 1 - kw["minor"])  # bias: kira-kira proporsi minoritas
    pr = 1 / (1 + np.exp(-(lg - b))); y = (r.uniform(size=n) < pr).astype(int)
    return X, y


def data(name):
    if name not in _cache:
        _cache.clear(); i = F.DS[name][1]
        X, y = gen(param(i), 103_000)
        X = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])]); y = pd.Series(y, name="target")
        Xd, Xr, yd, yr = train_test_split(X, y, test_size=100_000, stratify=y, random_state=int(8_000_000 + i))
        _cache[name] = (Xd.reset_index(drop=True), yd.reset_index(drop=True), Xr.reset_index(drop=True), np.asarray(yr))
    return _cache[name]


F.data = data

if __name__ == "__main__":
    a, b, Pn, outn = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    jobs = [(f"T{i:04d}", 0) for i in range(a, b)]; t0 = time.time()
    with Pool(Pn) as pool:
        res = pool.map(M.trial, jobs, chunksize=2)
    keys = list(next(r for r in res if "lo95" in r))
    for r in res:
        r["param"] = str(param(int(r["dataset"][1:])))
    p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys + ["param"], restval=""); w.writeheader(); w.writerows(res)
    print(f"tugas {a}-{b}, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

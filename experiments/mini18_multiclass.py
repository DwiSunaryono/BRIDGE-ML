"""MINI-18 (pengembangan): gate BRIDGE untuk multi-kelas (K>=3). Gate beku diperluas apa adanya:
counts (y, baseline, kandidat) berdimensi K^3 (biner = kasus K=2, 8 counts); wF1 multi-kelas
(rata-rata F1 per kelas berbobot support) dari matriks konfusi; cross-fit 3; bootstrap multinomial
20000 atas sel tak-nol; terima bila kuantil 5% delta wF1 > 0.005.
Baseline: HGB bagged (32x, 80% SEARCH) argmax. Kandidat (analog threshold/ensemble biner):
  E  : bobot simpleks langkah .25 atas proba HGB/DT/SVC, argmax (14, tanpa HGB murni)
  S  : HGB dengan skala kelas p_k * s, s in {0.5, 0.7, 1.4, 2}, argmax (4K)
  ES : (HGB .5 + DT .5) dengan skala kelas yang sama (4K)
Kebenaran referensi: delta wF1 pada referensi terkunci; CI = bootstrap Bayesian Dirichlet(cr) atas
sel tak-nol (prior +.5 di K^3 sel akan mendominasi untuk K besar, jadi tidak dipakai).
Dataset (aturan ditetapkan sebelum run): OpenML multi-kelas N>=3600, kelas terkecil >=30, K<=26.
Seed 1_016_000_000. Rev 11:20: aturan bag (>= 2 per kelas yang muncul) dan pengaman error per trial.
Pakai: python mini18_multiclass.py <R> <n_proses> [out] [offset]
"""
import csv, sys, time, warnings, itertools
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import exp41_gate_confirm as G
import data as D, models

OFF = int(sys.argv[4]) if len(sys.argv) > 4 else 1_016_000_000
DS = {"letter": 6, "pendigits": 32, "optdigits": 28, "satimage": 182, "first-order-theorem": 1475,
      "gas-drift": 1476, "har": 1478, "connect-4": 40668, "covertype": 1596, "JapaneseVowels": 375,
      "texture": 40499, "volkert": 41166, "Diabetes130US": 4541, "isolet": 300, "wall-robot-nav": 1497,
      "artificial-characters": 1459, "GesturePhase": 4538, "walking-activity": 1509}
SIMPLEX = [w for w in itertools.product((0, .25, .5, .75, 1), repeat=3) if abs(sum(w) - 1) < 1e-9 and w != (1, 0, 0)]
SCALES = (.5, .7, 1.4, 2)
_D = {}


def load(did):
    d = fetch_openml(data_id=did, as_frame=True, parser="auto")
    X, y = d.data.copy(), d.target.astype(str)
    y = pd.Series(pd.Categorical(y, categories=sorted(y.unique())).codes, name="target")
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            s = X[col].astype(object).where(X[col].notna(), "__MISSING__").astype(str)
            X[col] = pd.Categorical(s, categories=sorted(s.unique())).codes
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.reset_index(drop=True), y.reset_index(drop=True)


def data(did):
    if did not in _D:
        _D.clear(); X, y = load(did)
        Xd, Xr, yd, yr = train_test_split(X, y, test_size=G.ref_size(len(X)), stratify=y, random_state=G.LOCKED_REF_SEED)
        _D[did] = (Xd.reset_index(drop=True), yd.reset_index(drop=True), Xr.reset_index(drop=True), np.asarray(yr), int(y.max()) + 1)
    return _D[did]


def fit_proba(model, Xf, yf, K):
    cols = list(Xf.columns); num_c, cat_c = D.column_kinds(Xf)
    pre = D.make_preprocessor(num_c, cat_c); Z = pre.fit_transform(Xf, yf)
    est = models.build_estimator(model, {}, len(yf), final=True); est.fit(Z, yf)
    cl = np.asarray(est.classes_, int)

    def f(Xn):
        out = np.zeros((len(Xn), K)); out[:, cl] = est.predict_proba(pre.transform(Xn[cols])); return out
    return f


def cands(K):
    e = [({"HGB": a, "DT": b, "SVC": c}, None) for a, b, c in SIMPLEX]
    s = [({"HGB": 1}, (k, v)) for k in range(K) for v in SCALES]
    es = [({"HGB": .5, "DT": .5}, (k, v)) for k in range(K) for v in SCALES]
    return e + s + es


def lab(P, cfg, idx=None):
    w, sc = cfg
    p = sum(wi * P[m] for m, wi in w.items() if wi)
    if sc is not None:
        p = p.copy(); p[:, sc[0]] *= sc[1]
    p = p if idx is None else p[idx]
    return p.argmax(1)


def cells(y, b, c, K):
    y, b, c = (np.asarray(v, np.int64) for v in (y, b, c))
    return np.bincount((y * K + b) * K + c, minlength=K ** 3)


def wf1_delta(W, K):
    """W: (..., K^3) counts/bobot; hasil delta wF1 (kandidat - baseline), tervektorisasi."""
    T = np.asarray(W, float).reshape(np.shape(W)[:-1] + (K, K, K))

    def wf(M):
        tp = np.diagonal(M, axis1=-2, axis2=-1); row = M.sum(-1); col = M.sum(-2)
        f1 = np.divide(2 * tp, row + col, out=np.zeros_like(tp), where=(row + col) > 0)
        return (row * f1).sum(-1) / np.maximum(row.sum(-1), 1e-300)
    return wf(T.sum(-2)) - wf(T.sum(-1))  # [y,c] kandidat - [y,b] baseline


def boot_delta(cnt, K, rng, B=20000, dirichlet=False):
    nz = np.flatnonzero(cnt); out = []
    for s in range(0, B, 2500):
        n = min(2500, B - s)
        sub = rng.dirichlet(cnt[nz], size=n) if dirichlet else rng.multinomial(cnt.sum(), cnt[nz] / cnt.sum(), size=n)
        W = np.zeros((n, K ** 3)); W[:, nz] = sub; out.append(wf1_delta(W, K))
    return np.concatenate(out)


def pick(y, P, idx, C, base, K):
    best, g0 = None, 0.0
    for cfg in C:
        g = float(wf1_delta(cells(y[idx], base[idx], lab(P, cfg, idx), K), K))
        if g > g0 + 1e-12:
            best, g0 = cfg, g
    return best


def trial(job):
    try:
        return _trial(job)
    except Exception as e:  # catat, jangan buang seluruh run
        return dict(dataset=job[0], K=-1, rep=job[2], cfg="ERROR " + str(e)[:150], D=0, lo95=np.nan, ref=np.nan, kelas="error")


def _trial(job):
    ds, did, rep = job; seed = OFF + 1000 * list(DS).index(ds) + rep
    Xd, yd, Xr, yr, K = data(did)
    Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=G.TOTAL_TRAIN, stratify=yd, random_state=seed)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=G.SEARCH_SIZE, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
    fn = {m: [] for m in ("HGB", "DT", "SVC")}
    for _ in range(G.B_BAGS):
        idx = rng.choice(len(ys), size=sub, replace=False)
        cc = np.bincount(ys[idx], minlength=K); cc = cc[cc > 0]
        if len(cc) < 2 or cc.min() < 2:  # HGB early-stopping (split bertingkat) butuh >= 2 per kelas
            continue
        for m in fn:
            fn[m].append(fit_proba(m, Xs.iloc[idx].reset_index(drop=True), ys[idx], K))
    P = {m: sum(f(Xp) for f in fs) / len(fs) for m, fs in fn.items()}
    Pr = {m: sum(f(Xr) for f in fs) / len(fs) for m, fs in fn.items()}
    base, baser = P["HGB"].argmax(1), Pr["HGB"].argmax(1); C = cands(K)
    cnt = np.zeros(K ** 3, int)
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp):
        cfg = pick(yp, P, a, C, base, K)
        cnt += cells(yp[b], base[b], base[b] if cfg is None else lab(P, cfg, b), K)
    l95 = float(np.quantile(boot_delta(cnt, K, np.random.default_rng(seed + 71)), .05))
    cfg = pick(yp, P, np.arange(len(yp)), C, base, K)
    cr = cells(yr, baser, baser if cfg is None else lab(Pr, cfg), K); d = float(wf1_delta(cr, K))
    lo, hi = np.quantile(boot_delta(cr, K, np.random.default_rng(seed + 987163), dirichlet=True), [.025, .975])
    T = cnt.reshape(K, K, K)
    return dict(dataset=ds, K=K, rep=rep, cfg=str(cfg), D=int(cnt.sum() - np.einsum("ybb->", T)), lo95=l95, ref=d,
                kelas="membaik" if lo > 0 else "memburuk" if hi < 0 else "tidak-pasti")


if __name__ == "__main__":
    R, Pn = int(sys.argv[1]), int(sys.argv[2]); outn = sys.argv[3] if len(sys.argv) > 3 else "mini18_multiclass_dev.csv"
    t0 = time.time(); jobs = [(ds, did, r) for ds, did in DS.items() for r in range(R)]
    with Pool(Pn) as pool:
        res = pool.map(trial, jobs, chunksize=4)
    p = HERE / "out" / outn
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print(f"{len(DS)} dataset, {len(jobs)} trial, {time.time()-t0:.0f} dtk -> {p}")

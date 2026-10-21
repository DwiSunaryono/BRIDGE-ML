"""Audit isi dataset kandidat validasi baru (aturan exp40 §2): asal target (deskripsi), duplikat, kolom ID,
AUC fitur tunggal maks (indikasi bocor), dan kolom waktu. Output: out/audit_baru.csv + deskripsi ringkas."""
import sys, warnings, re
warnings.filterwarnings("ignore")
from multiprocessing import Pool
import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
sys.path.insert(0, ".")
C = {46877: "PostPartum_Depression", 46927: "Fitness_Club", 1050: "pc3", 45648: "MIC", 46938: "good-customer",
     1067: "kc1", 46940: "Marketing_Campaign", 46864: "Skin_Cancer_PAD", 46951: "Pumpkin_Seeds", 46956: "seismic-bumps",
     40978: "Internet-Ads", 42750: "Census-Income-KDD", 46859: "Santander2016", 42343: "KDD98", 46792: "LICD",
     46857: "Homesite", 46858: "IEEE-CIS"}


def one(item):
    did, nm = item
    try:
        d = fetch_openml(data_id=did, as_frame=True, parser="auto")
        X, y = d.data, d.target.astype(str)
        cats = sorted(y.unique()); yb = (y == cats[1]).astype(int).values
        n = len(X); dup = X.duplicated().mean() * 100
        Xy = X.assign(_y=yb); conf = (Xy.duplicated(subset=list(X.columns), keep=False) & ~Xy.duplicated(keep=False)).mean() * 100
        idc = [c for c in X.columns if X[c].nunique() >= 0.98 * n]
        timec = [c for c in X.columns if re.search(r"date|time|year|month|day|dt$", str(c), re.I)]
        best = (0.5, None)
        rng = np.random.default_rng(0); sub = rng.choice(n, size=min(n, 50000), replace=False)
        for c in X.columns:
            v = X[c].iloc[sub]
            v = pd.Series(pd.Categorical(v.astype(str)).codes) if not pd.api.types.is_numeric_dtype(v) else v.astype(float)
            v = v.fillna(v.min() - 1 if v.notna().any() else 0)
            try:
                a = roc_auc_score(yb[sub], v.values); a = max(a, 1 - a)
                if a > best[0]: best = (a, c)
            except Exception:
                pass
        desc = re.sub(r"\s+", " ", (d.DESCR or ""))[:700]
        return dict(did=did, nama=nm, N=n, p=X.shape[1], minor=round(100 * min(yb.mean(), 1 - yb.mean()), 2),
                    kelas=cats, dup_pct=round(dup, 2), konflik_pct=round(conf, 2), kolom_id=idc[:5], kolom_waktu=timec[:6],
                    auc_maks=round(best[0], 4), fitur_auc_maks=best[1], na_pct=round(X.isna().mean().mean() * 100, 1), desc=desc)
    except Exception as e:
        return dict(did=did, nama=nm, error=str(e)[:200])


if __name__ == "__main__":
    with Pool(4) as p:
        res = p.map(one, list(C.items()), chunksize=1)
    pd.DataFrame(res).to_csv("out/audit_baru.csv", index=False)
    for r in res:
        print({k: v for k, v in r.items() if k != "desc"}); print("   DESC:", r.get("desc", "")[:450]); print()

"""Screen OpenML datasets whose target exists but is not marked as default.

Target and drop choices are fixed here before screening. Drops remove identifiers
and variables that directly encode the realised outcome. Raw rows come from
OpenML published Parquet files; no sampling is performed.
"""
import csv
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SPECS = {
    1128: dict(target="Tissue", drop=["ID_REF"]),
    40926: dict(target="class", drop=[]),
    42140: dict(target="class", drop=[]),
    43539: dict(target="Default", drop=[
        "Selected", "LoanNr_ChkDgt", "Name", "ChgOffDate", "MIS_Status",
        "ChgOffPrinGr", "BalanceGross", "xx",
    ]),
    43729: dict(target="not.fully.paid", drop=[]),
    43743: dict(target="V32", drop=["V1"]),
    43826: dict(target="Personal_Loan", drop=[]),
}


def ref_size(n):
    return min(n - 20000, 100000) if n >= 40000 else min(8000, int(0.3 * n))


def load(data_id):
    cache = Path.home() / "tabelC_cache" / f"custom_{data_id}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    url = f"https://data.openml.org/datasets/{data_id // 10000:04d}/{data_id}/dataset_{data_id}.pq"
    raw = Path("/tmp") / f"dataset_{data_id}.pq"
    if not raw.exists():
        urllib.request.urlretrieve(url, raw)
    frame = pd.read_parquet(raw)
    spec = SPECS[data_id]
    y = frame.pop(spec["target"])
    frame = frame.drop(columns=spec["drop"], errors="raise")
    keep = y.notna().to_numpy()
    X, y = frame.loc[keep].reset_index(drop=True), y.loc[keep].astype(str).reset_index(drop=True)
    y = pd.Series(pd.Categorical(y, categories=sorted(y.unique())).codes, name="target")
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            s = X[c].astype(object).where(X[c].notna(), "__MISSING__").astype(str)
            X[c] = pd.Categorical(s, categories=sorted(s.unique())).codes
        X[c] = pd.to_numeric(X[c], errors="coerce")
    result = X, y, int((~keep).sum())
    cache.parent.mkdir(exist_ok=True)
    pd.to_pickle(result, cache)
    return result


def screen(data_id):
    X, y, missing_target = load(data_id)
    n, K = len(y), int(y.max()) + 1
    counts = np.bincount(y, minlength=K)
    duplicate = float(pd.concat([X, y], axis=1).duplicated().mean())
    Z = X.fillna(X.median(numeric_only=True)).fillna(0)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(n, 50000), replace=False)
    target = y.to_numpy()[idx]
    best_auc, best_col = 0.0, ""
    for c in Z.columns:
        values = Z[c].to_numpy()[idx]
        if np.unique(values).size < 2:
            continue
        if K == 2:
            auc = roc_auc_score(target, values)
            auc = max(auc, 1 - auc)
        else:
            auc = float(np.mean([
                max(a, 1 - a)
                for a in (roc_auc_score(target == k, values) for k in range(K))
            ]))
        if auc > best_auc:
            best_auc, best_col = auc, str(c)
    reasons = []
    if duplicate > 0.25:
        reasons.append(f"duplicates {duplicate:.1%} > 25%")
    if K == 2 and counts[0] == counts[1]:
        reasons.append("class ratio exactly 50%")
    if best_auc >= 0.99:
        reasons.append(f"one feature predicts target (AUC {best_auc:.3f}, {best_col})")
    left = n - ref_size(n)
    if K == 2 and left < 1050:
        reasons.append(f"outside coverage: {left} rows after reference (< 1,050)")
    if K > 2 and left < 2480:
        reasons.append(f"outside coverage: {left} rows after reference (< 2,480)")
    if K > 26:
        reasons.append(f"outside coverage: {K} classes (> 26)")
    return dict(data_id=data_id, status="KEEP" if not reasons else "OUT",
                reasons="; ".join(reasons), N=n, K=K, F=X.shape[1],
                minority=float(counts.min() / n), duplicates=round(duplicate, 4),
                max_single_auc=round(best_auc, 4), max_single_feature=best_col,
                missing_target=missing_target)


if __name__ == "__main__":
    rows = [screen(i) for i in SPECS]
    with open(sys.argv[1], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(pd.DataFrame(rows).to_string(index=False))

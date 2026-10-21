"""Screen external UCI candidates with the locked Table-C exclusion rules.

Usage: python screen_e2_external.py OUT.csv
Covers UCI 563 (Iranian Churn) and UCI 179 (SECOM). Raw data are downloaded by ucimlrepo/UCI and are not committed to the manuscript repository.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from ucimlrepo import fetch_ucirepo


def ref_size(n):
    return min(n - 20000, 100000) if n >= 40000 else min(8000, int(0.3 * n))


def coded(X, y):
    keep = y.notna().to_numpy()
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].astype(str).reset_index(drop=True)
    y = pd.Series(pd.Categorical(y, categories=sorted(y.unique())).codes, name="target")
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            s = X[c].astype(object).where(X[c].notna(), "__MISSING__").astype(str)
            X[c] = pd.Categorical(s, categories=sorted(s.unique())).codes
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, y, int((~keep).sum())


TARGETS = {
    198: "Pastry",  # one of the original fault indicators; other outcomes are dropped
    563: "Churn",
    579: "FIBR_PREDS",  # atrial fibrillation complication (original binary outcome)
    967: "label",
}


def load_uci(uci_id):
    d = fetch_ucirepo(id=uci_id)
    if uci_id == 179:  # UCI API puts the separate label/timestamp fields in original.
        X = d.data.original.copy()
        y = X.pop("class")
        X = X.drop(columns=["timestamp"])
    else:
        target = TARGETS[uci_id]
        original = d.data.original.copy()
        if target not in original:
            raise ValueError(f"target {target!r} is absent")
        y = original[target]
        # Use only fields declared as features. This also drops all alternative
        # outcome columns in multi-output UCI datasets.
        X = d.data.features.copy()
    return coded(X, y)


def screen(uci_id):
    X, y, missing_target = load_uci(uci_id)
    n, K = len(y), int(y.max()) + 1
    counts = np.bincount(y, minlength=K)
    duplicate = float(pd.concat([X, y], axis=1).duplicated().mean())
    Z = X.fillna(X.median(numeric_only=True)).fillna(0)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(n, 50000), replace=False)
    best_auc, best_col = 0.0, ""
    for c in Z.columns:
        values, target = Z[c].to_numpy()[idx], y.to_numpy()[idx]
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
    if K != 2:
        reasons.append(f"expected binary target, found K={K}")
    left = n - ref_size(n)
    if left < 1050:
        reasons.append(f"outside coverage: {left} rows after reference (< 1,050)")
    return dict(source_id=f"UCI-{uci_id}", status="KEEP" if not reasons else "OUT",
                reasons="; ".join(reasons), N=n, K=K, F=X.shape[1],
                minority=float(counts.min() / n), duplicates=round(duplicate, 4),
                max_single_auc=round(best_auc, 4), missing_target=missing_target)


if __name__ == "__main__":
    rows = [screen(i) for i in (563, 179, 198, 579, 967)]
    out = Path(sys.argv[1])
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(pd.DataFrame(rows).to_string(index=False))

"""Mini-FDE data.py — SATU-SATUNYA modul yang boleh menyentuh TEST.

Aturan blind (D):
- Split outer stratified 80:20 seed-42. Split stabilitas 0/77 HANYA untuk
  pelaporan OOF-TRAIN, tidak membuka akses TEST tambahan.
- Preprocessing (imputer + scaler + OrdinalEncoder) di-fit ulang PER
  training-fold. Encoder-at-load DILARANG untuk klaim.
- TEST dibaca TEPAT SEKALI pasca-kunci (ditegakkan via counter + guard).
- gate.py DILARANG mengimpor modul ini.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import config as C

_TEST_READ_COUNT = 0  # TEST boleh dibaca tepat sekali per proses run


def test_read_count() -> int:
    return _TEST_READ_COUNT


def reset_test_counter():
    """Reset counter TEST untuk replay (bukan run asli). Hanya untuk replay.py."""
    global _TEST_READ_COUNT
    _TEST_READ_COUNT = 0


def load_dataset(name: str):
    """Muat (X: DataFrame, y: Series). Dev: iris/wine. Sisanya via OpenML data_id."""
    lname = name.lower()
    if lname == "iris":
        from sklearn.datasets import load_iris
        d = load_iris(as_frame=True)
        return d.data, pd.Series(d.target, name="target")
    if lname == "wine":
        from sklearn.datasets import load_wine
        d = load_wine(as_frame=True)
        return d.data, pd.Series(d.target, name="target")
    if lname in C.OPENML_IDS:
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=C.OPENML_IDS[lname], as_frame=True, parser="auto")
        X = d.data if isinstance(d.data, pd.DataFrame) else pd.DataFrame(d.data)
        y = pd.Series(d.target, name="target")
        # Target OpenML sering string/object -> faktorkan deterministik (sorted).
        if not pd.api.types.is_numeric_dtype(y) or y.dtype == object:
            cats = sorted(y.astype(str).unique().tolist())
            y = pd.Series(pd.Categorical(y.astype(str), categories=cats).codes,
                          name="target")
        # Paksa semua kolom numerik; kolom non-numerik di-encode ordinal di tingkat
        # frame agar matriks downstream murni numerik (syarat jalan F).
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                cats = sorted(X[col].astype(str).unique().tolist())
                X[col] = pd.Categorical(X[col].astype(str), categories=cats).codes
            X[col] = pd.to_numeric(X[col], errors="coerce")
        return X, y
    raise ValueError(f"Dataset tak dikenal: {name!r}. Dev: {C.DEV_DATASETS}, "
                     f"klaim: {sorted(C.OPENML_IDS)}")


def column_kinds(X: pd.DataFrame):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def make_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    """Pipeline preprocessing BARU (belum di-fit). Wajib fit per training-fold."""
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), list(num_cols)))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1)),
        ]), list(cat_cols)))
    if not transformers:  # tak ada kolom? (defensif, syarat numerik F menolak ini)
        raise ValueError("Tidak ada kolom fitur.")
    return ColumnTransformer(transformers, remainder="drop")


def outer_split(X, y, seed: int = C.SEED, test_size: float = C.OUTER_TEST_SIZE):
    """Bagi outer stratified. TIDAK menyentuh counter TEST (counter hanya naik
    saat TEST dikunci & dibaca pasca-kunci via lock_test)."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(sss.split(X, y))
    Xtr = X.iloc[tr] if isinstance(X, pd.DataFrame) else np.asarray(X)[tr]
    Xte = X.iloc[te] if isinstance(X, pd.DataFrame) else np.asarray(X)[te]
    ytr = y.iloc[tr] if isinstance(y, pd.Series) else np.asarray(y)[tr]
    yte = y.iloc[te] if isinstance(y, pd.Series) else np.asarray(y)[te]
    return Xtr, Xte, ytr, yte


def lock_test(Xte, yte):
    """Baca TEST TEPAT SEKALI pasca-kunci. Panggilan kedua -> RuntimeError."""
    global _TEST_READ_COUNT
    if _TEST_READ_COUNT >= 1:
        raise RuntimeError("Pelanggaran blind: TEST sudah dibaca 1x; baca ulang DILARANG.")
    _TEST_READ_COUNT += 1
    return Xte, yte


def minority_info(ytr) -> tuple[int, float]:
    """(minority_count, minority_ratio) di outer-TRAIN. Syarat jalan F: count>=10."""
    vals, counts = np.unique(np.asarray(ytr), return_counts=True)
    mc = int(counts.min())
    ratio = float(mc / counts.sum())
    return mc, ratio

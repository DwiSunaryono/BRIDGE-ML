"""Estimator dan ruang pencarian Optuna untuk pipeline Mini-FDE (paket C, beku).

Pencarian selalu memakai probability=False; Platt hanya untuk model final setelah kunci.
Preprocessing di-fit ulang pada setiap training-fold melalui data.make_preprocessor.
TPE berjalan pada ruang kontinu halus (param), bukan pada mask fitur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import config as C
from data import make_preprocessor

ESTIMATORS = ("SVC", "DT", "KNN", "HGB", "XGB")


def build_estimator(name: str, params: dict, n_samples: int, final: bool = False):
    """Membangun estimator untuk nama, params, dan ukuran sampel latih.

    final=True hanya mengaktifkan probability=True pada SVC untuk Platt.
    2026-10-05: class_weight='balanced' pada SVC/DT/HGB dicoba untuk
    ketimpangan kelas pc4 (88%/12%) dan memperburuk hasil. Bobot balanced
    mengoptimalkan akurasi per kelas, sedangkan pipeline memakai weighted-F1
    yang condong ke kelas mayoritas. P1 pc4 turun dari wF1=0.90 ke 0.086
    (model menebak minoritas 282/292 baris), lalu perubahan dibatalkan.
    """
    p = dict(params)
    if name == "SVC":
        kw = dict(C=float(p.get("C", 1.0)), kernel=p.get("kernel", "rbf"),
                  gamma=p.get("gamma", "scale"),
                  max_iter=C.svc_max_iter(n_samples),
                  cache_size=C.SVC_CACHE_SIZE,
                  probability=bool(final), random_state=C.SEED)
        if kw["kernel"] == "poly":
            kw["degree"] = int(p.get("degree", 3))
            kw["coef0"] = float(p.get("coef0", 0.0))
        return SVC(**kw)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=int(p.get("n_neighbors", 5)),
                                    weights=p.get("weights", "uniform"),
                                    p=int(p.get("p", 2)))
    if name == "HGB":
        return HistGradientBoostingClassifier(
            learning_rate=float(p.get("learning_rate", 0.1)),
            max_iter=int(p.get("max_iter", 300)),
            max_leaf_nodes=int(p.get("max_leaf_nodes", 31)),
            min_samples_leaf=int(p.get("min_samples_leaf", 20)),
            l2_regularization=float(p.get("l2_regularization", 0.0)),
            early_stopping=True,
            n_iter_no_change=10,
            random_state=C.SEED)
    if name == "XGB":
        # Nama param lama "max_iter" tidak valid untuk XGBClassifier; xgboost
        # mengabaikannya dan selalu memakai n_estimators=100. early_stopping_rounds
        # tanpa eval_set di .fit() juga gagal ("Must have at least 1 validation
        # dataset"). n_estimators kini dikontrol Optuna dan early stopping dimatikan,
        # sama seperti HGB yang menuning max_iter.
        return XGBClassifier(
            learning_rate=float(p.get("learning_rate", 0.1)),
            n_estimators=int(p.get("max_iter", 300)),
            max_depth=int(p.get("max_depth", 6)),
            min_child_weight=float(p.get("min_child_weight", 1.0)),
            subsample=float(p.get("subsample", 1.0)),
            colsample_bytree=float(p.get("colsample_bytree", 1.0)),
            reg_lambda=float(p.get("reg_lambda", 1.0)),
            random_state=C.SEED,
            eval_metric="logloss",
            n_jobs=1)
    if name == "DT":
        md = p.get("max_depth", None)
        return DecisionTreeClassifier(
            max_depth=None if md is None else int(md),
            min_samples_split=int(p.get("min_samples_split", 2)),
            min_samples_leaf=int(p.get("min_samples_leaf", 1)),
            criterion=p.get("criterion", "gini"), random_state=C.SEED)
    raise ValueError(f"Estimator tak dikenal: {name}")


def suggest_params(trial, name: str, n_train_min: int, prefix: str = "") -> dict:
    """Mengambil ruang search beku dari config.SEARCH_SPACES.

    prefix dipakai pada joint-search multi-anggota dalam satu study (P3 robust)
    agar nama param tidak bertabrakan; "" memberi perilaku lama. Prefix dikupas
    saat estimator dibangun.
    """
    S = C.SEARCH_SPACES[name]
    P = prefix  # awalan nama param untuk joint-search, "" = perilaku lama
    if name == "SVC":
        lo, hi = S["C"][1], S["C"][2]
        out = {"C": trial.suggest_float(P + "C", lo, hi, log=True),
               "kernel": trial.suggest_categorical(P + "kernel", S["kernel"][1]),
               "gamma": trial.suggest_categorical(P + "gamma", S["gamma"][1])}
        # degree/coef0 hanya dipakai bila kernel poly, tetapi tetap di-suggest agar ruang tetap.
        out["degree"] = trial.suggest_int(P + "degree", S["degree"][1], S["degree"][2])
        out["coef0"] = trial.suggest_float(P + "coef0", S["coef0"][1], S["coef0"][2])
        return out
    if name == "KNN":
        hi = max(1, min(S["n_neighbors"][2], n_train_min - 1))
        return {"n_neighbors": trial.suggest_int(P + "n_neighbors", 1, hi),
                "weights": trial.suggest_categorical(P + "weights", S["weights"][1]),
                "p": trial.suggest_categorical(P + "p", S["p"][1])}
    if name == "HGB":
        return {
            "learning_rate": trial.suggest_float(P + "learning_rate", S["learning_rate"][1],
                                                 S["learning_rate"][2], log=True),
            "max_iter": trial.suggest_int(P + "max_iter", S["max_iter"][1],
                                           S["max_iter"][2]),
            "max_leaf_nodes": trial.suggest_int(P + "max_leaf_nodes", S["max_leaf_nodes"][1],
                                                 S["max_leaf_nodes"][2]),
            "min_samples_leaf": trial.suggest_int(P + "min_samples_leaf",
                                                  S["min_samples_leaf"][1],
                                                  S["min_samples_leaf"][2]),
            "l2_regularization": trial.suggest_float(P + "l2_regularization",
                                                     S["l2_regularization"][1],
                                                     S["l2_regularization"][2], log=True)}
    if name == "XGB":
        return {
            "learning_rate": trial.suggest_float(P + "learning_rate", S["learning_rate"][1],
                                                 S["learning_rate"][2], log=True),
            "max_iter": trial.suggest_int(P + "max_iter", S["max_iter"][1],
                                          S["max_iter"][2]),
            "max_depth": trial.suggest_int(P + "max_depth", S["max_depth"][1],
                                           S["max_depth"][2]),
            "min_child_weight": trial.suggest_float(P + "min_child_weight",
                                                    S["min_child_weight"][1],
                                                    S["min_child_weight"][2]),
            "subsample": trial.suggest_float(P + "subsample", S["subsample"][1],
                                             S["subsample"][2]),
            "colsample_bytree": trial.suggest_float(P + "colsample_bytree",
                                                    S["colsample_bytree"][1],
                                                    S["colsample_bytree"][2]),
            "reg_lambda": trial.suggest_float(P + "reg_lambda", S["reg_lambda"][1],
                                              S["reg_lambda"][2], log=True)}
    if name == "DT":
        use_none = trial.suggest_float(P + "dt_no_depth", 0.0, 1.0) < 0.25
        md = None if use_none else trial.suggest_int(P + "max_depth", S["max_depth"][1],
                                                     S["max_depth"][2])
        return {"max_depth": md,
                "min_samples_split": trial.suggest_int(P + "min_samples_split",
                                                       S["min_samples_split"][1],
                                                       S["min_samples_split"][2]),
                "min_samples_leaf": trial.suggest_int(P + "min_samples_leaf",
                                                      S["min_samples_leaf"][1],
                                                      S["min_samples_leaf"][2]),
                "criterion": trial.suggest_categorical(P + "criterion", S["criterion"][1])}
    raise ValueError(name)


def _fold_frames(X, y, cols, tr_idx, va_idx):
    if isinstance(X, pd.DataFrame):
        Xtr, Xva = X.iloc[tr_idx][cols], X.iloc[va_idx][cols]
    else:
        # 2026-10-05: versi lama `X[np.ix_(tr_idx, cols)] if ... else
        # (X[tr_idx], X[va_idx])` salah karena cabang np.ix_ menghasilkan satu
        # array, bukan tuple Xtr, Xva, sehingga unpacking gagal. Jalur ini tidak
        # dipakai pipeline aktual (selalu DataFrame dari data.load_dataset),
        # tetapi diperbaiki agar benar secara umum.
        X = np.asarray(X)
        if isinstance(cols[0], int):
            Xtr, Xva = X[np.ix_(tr_idx, cols)], X[np.ix_(va_idx, cols)]
        else:
            Xtr, Xva = X[tr_idx], X[va_idx]
    yv = np.asarray(y)
    return Xtr, Xva, yv[tr_idx], yv[va_idx]


def oof_predict(build_fn, X, y, cols, cv) -> tuple[np.ndarray, float, list[float]]:
    """Menghitung prediksi out-of-fold di dalam TRAIN dengan preprocessor per fold.

    build_fn(n_fold_train_size) mengembalikan estimator yang belum di-fit.
    Hasil: (oof_preds, weighted-F1 OOF, fold_scores).
    """
    y = np.asarray(y)
    oof = np.empty(len(y), dtype=y.dtype if y.dtype != object else int)
    fold_scores = []
    for tr_idx, va_idx in cv.split(X, y):
        Xtr, Xva, ytr, yva = _fold_frames(X, y, cols, tr_idx, va_idx)
        if isinstance(Xtr, pd.DataFrame):
            num_c = [c for c in cols if pd.api.types.is_numeric_dtype(Xtr[c])]
            cat_c = [c for c in cols if c not in num_c]
            pre = make_preprocessor(num_c, cat_c)
            Ztr, Zva = pre.fit_transform(Xtr, ytr), pre.transform(Xva)
        else:
            pre = make_preprocessor(list(range(Xtr.shape[1])), [])
            Ztr, Zva = pre.fit_transform(Xtr, ytr), pre.transform(Xva)
        est = build_fn(len(ytr))
        est.fit(Ztr, ytr)
        pred = est.predict(Zva)
        oof[va_idx] = pred
        fold_scores.append(C.score_metric(yva, pred))
    oof_f1 = C.score_metric(y, oof)
    return oof, oof_f1, fold_scores


def cv_mean_f1(build_fn, X, y, cols, cv) -> float:
    _, f1, _ = oof_predict(build_fn, X, y, cols, cv)
    return f1


def feature_ranking(X, y, cols, cv, est: str = "HGB", params: dict | None = None,
                    seed: int = C.SEED) -> list:
    """Mengurutkan fitur dengan permutation importance pada fold validasi CV.

    Model mengikuti est dan params (default HGB generik).
    2026-10-05: versi lama selalu memakai HGB generik walau model pemenang P1
    berbeda (misalnya XGB); pola importance tiap algoritma dapat berbeda.
    Sebelumnya importance dihitung in-sample, kini di fold validasi held-out.
    """
    params = params or {}
    y = np.asarray(y)
    Xd = X[cols] if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X), columns=list(cols))
    num_c = [c for c in cols if pd.api.types.is_numeric_dtype(Xd[c])]
    cat_c = [c for c in cols if c not in num_c]

    importances = np.zeros(len(cols))
    n_folds = 0
    for tr_idx, va_idx in cv.split(Xd, y):
        Xtr_f, Xva_f = Xd.iloc[tr_idx], Xd.iloc[va_idx]
        ytr_f, yva_f = y[tr_idx], y[va_idx]
        pre = make_preprocessor(num_c, cat_c)
        Ztr = pre.fit_transform(Xtr_f, ytr_f)
        Zva = pre.transform(Xva_f)
        model = build_estimator(est, params, len(ytr_f), final=False)
        model.fit(Ztr, ytr_f)
        r = permutation_importance(model, Zva, yva_f, n_repeats=5, random_state=seed,
                                   scoring=C.pi_scoring())
        importances += r.importances_mean
        n_folds += 1
    importances /= max(n_folds, 1)
    order = np.argsort(-importances)
    return [cols[i] for i in order]


def hgb_ranking(X, y, cols, cv, seed: int = C.SEED) -> list:
    """Pembungkus lama untuk ranking dengan HGB generik. Lihat feature_ranking()."""
    return feature_ranking(X, y, cols, cv, est="HGB", params=None, seed=seed)

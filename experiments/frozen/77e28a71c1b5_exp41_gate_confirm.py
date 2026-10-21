"""Eksperimen 41: konfirmasi prospektif kebijakan gate + margin 0.007 (SPEK_EXP41).

Salinan exp40_margin_gate.py (yang sendiri salinan exp39_transfer_gate.py);
satu-satunya perubahan algoritmik: ukuran referensi evaluasi (n_ref) lebih besar
untuk dataset besar. Catatan exp40: Algoritme (gate, split, model, kandidat, urutan
arm, RNG) IDENTIK; yang berubah hanya plumbing (dataset/stratum/seed/path),
guard target biner, dan instrumentasi keluaran (8-counts CONFIRM/ref, ci_hi99,
CI referensi, label baseline di NPZ) yang ditulis SETELAH draw gate sehingga
konsumsi RNG tidak berubah. Margin 0.002 TIDAK diterapkan di sini (analisis).

Sel uji: 2 model (M1=HGB jangkar, M2=KNN) x 4 intervensi (I1 threshold,
I2 tuning kasar, I3 ensemble lintas-model, I4 fitur-F2) = 8 sel, semua pada
split & CONFIRM yang sama (berpasangan). Kontrol: C+ (25% label-flip,
kandidat=bersih) & Cseed (rebag, diagnostik tanpa cap), per model.
Helper gate/split/RNG (load_openml_binary s.d. counts8) IDENTIK exp37/38
(verifikasi via diff); yang baru hanya konstruksi kandidat multi-arm.
R dibaca dari exp39_manifest/config.json (ditetapkan dari pengukuran §5).
"""
from __future__ import annotations

import csv
import os
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.datasets import fetch_openml

import data
import models
from exp15_stabilized_candidate import B_BAGS, SUBSAMPLE_FRAC
from exp22_candidate_construction_test import (
    GREEDY_CV_SPLITS, leave_one_out_fallback,
)

SCHEMA_VERSION = "exp41-v1"
MANIFEST_DIR = Path(__file__).resolve().parent / "exp41_manifest"
LOCKED_REF_SEED = 20261125  # SPEK_EXP41 §4; non-overlap semua seed referensi lama
OFFSETS = {"utama": 98_000_000}
MODELS = {"M1": ("HGB", {}), "M2": ("KNN", {"n_neighbors": 5})}
I2_GRIDS = {"M1": [{"learning_rate": 0.05, "max_leaf_nodes": 15},
                   {"learning_rate": 0.05, "max_leaf_nodes": 63},
                   {"learning_rate": 0.2, "max_leaf_nodes": 15},
                   {"learning_rate": 0.2, "max_leaf_nodes": 63}],
            "M2": [{"n_neighbors": 15}, {"n_neighbors": 31},
                   {"n_neighbors": 61}]}
I3_WEIGHTS = [0.25, 0.5, 0.75]
FLIP_RATE = 0.25  # C+ (kalibrasi REVIEW_SPEK_EXP39)
SEARCH_SIZE = 630
POOL_SIZE = 1800
TUNE_SIZE = 600
CONFIRM_SIZE = 1200
TOTAL_TRAIN = SEARCH_SIZE + POOL_SIZE  # 2430
N_BOOT = 20000
GRID = sorted(np.arange(1, 20) / 20, key=lambda t: abs(t - 0.5))
MIN_MINORITY_CONFIRM = 5

CODES8 = ["000", "001", "010", "011", "100", "101", "110", "111"]  # (y,b,c)
FIELDS = ["stratum", "data_id", "cluster", "rep", "dataset", "seed", "status",
          "search_size", "tune_size",
          "confirm_size", "n_features_total", "n_bags_effective", "model",
          "arm", "arm_ordinal", "config", "tune_gain", "n_confirm_disagreements",
          "ci_lo99", "ci_hi99", "decision99", "ref_delta", "ref_ambiguous_95",
          "ref_ci_lo95", "ref_ci_hi95"] + [f"c{k}" for k in CODES8] + \
         [f"r{k}" for k in CODES8] + ["schema_version", "trial_sec"]


def load_openml_binary(data_id):
    d = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    X = d.data.copy()
    y = d.target
    if y.isna().any():
        raise ValueError(f"data_id {data_id}: target mengandung missing")
    cats = sorted(y.astype(str).unique().tolist())
    if len(cats) != 2:
        raise ValueError(f"data_id {data_id}: target bukan biner ({len(cats)} nilai)")
    y = pd.Series(pd.Categorical(y.astype(str), categories=cats).codes, name="target")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"data_id {data_id}: kode target bukan {{0,1}}")
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            col_str = X[col].astype(object).where(X[col].notna(), "__MISSING__").astype(str)
            c2 = sorted(col_str.unique().tolist())
            X[col] = pd.Categorical(col_str, categories=c2).codes
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.reset_index(drop=True), y.reset_index(drop=True)


def ref_size(n):
    """SPEK_EXP41 §4.2: referensi besar untuk dataset besar (presisi evaluasi)."""
    if n >= 40000:
        return min(n - 20000, 100000)
    return min(8000, int(0.3 * n))


def make_dev_and_locked_reference(data_id):
    X, y = load_openml_binary(data_id)
    n_ref = ref_size(len(X))
    X_dev, X_ref, y_dev, y_ref = train_test_split(
        X, y, test_size=n_ref, stratify=y, random_state=LOCKED_REF_SEED)
    return (X_dev.reset_index(drop=True), y_dev.reset_index(drop=True),
            X_ref.reset_index(drop=True), y_ref.reset_index(drop=True))


def delta_from_counts(c):
    c = np.asarray(c, dtype=float).reshape(c.shape[:-1] + (2, 2, 2))

    def f(m):
        tn, fp = m[..., 0, 0], m[..., 0, 1]
        fn, tp = m[..., 1, 0], m[..., 1, 1]
        total = tn + fp + fn + tp
        d0, d1 = 2 * tn + fp + fn, 2 * tp + fp + fn
        f0 = np.divide(2 * tn, d0, out=np.zeros_like(total, dtype=float), where=d0 > 0)
        f1 = np.divide(2 * tp, d1, out=np.zeros_like(total, dtype=float), where=d1 > 0)
        return ((tn + fp) * f0 + (tp + fn) * f1) / np.maximum(total, 1)
    return f(c.sum(axis=-2)) - f(c.sum(axis=-1))


def counts8(y, b, c):
    return np.bincount(4 * np.asarray(y, dtype=int) + 2 * np.asarray(b, dtype=int)
                       + np.asarray(c, dtype=int), minlength=8)


def fit_proba_m(model, params, Xfit, yfit, feat_cols):
    num_c, cat_c = data.column_kinds(Xfit[feat_cols])
    pre = data.make_preprocessor(num_c, cat_c)
    Z = pre.fit_transform(Xfit[feat_cols], yfit)
    est = models.build_estimator(model, params, len(yfit), final=True)
    est.fit(Z, yfit)
    pos_idx = list(est.classes_).index(1) if 1 in getattr(est, "classes_", [0, 1]) else 1

    def predict_proba(Xnew):
        return est.predict_proba(pre.transform(Xnew[feat_cols]))[:, pos_idx]
    return predict_proba


def cfg_name(model, params):
    if model == "HGB":
        return (f"lr{params.get('learning_rate', 0.1)}"
                f"_leaf{params.get('max_leaf_nodes', 31)}")
    return f"k{params.get('n_neighbors', 5)}"


def run_trial(dataset, X_dev, y_dev, X_ref, y_ref, seed, writer, pred_dir, meta=None):
    meta = meta or {}
    t00 = time.time()
    try:
        Xtr_full, _, ytr_full, _ = train_test_split(
            X_dev, y_dev, train_size=TOTAL_TRAIN, stratify=y_dev, random_state=seed)
    except ValueError:
        writer.writerow({**meta, "dataset": dataset, "seed": seed, "status": "skip_stratify_fail",
                         "search_size": SEARCH_SIZE, "tune_size": TUNE_SIZE,
                         "confirm_size": CONFIRM_SIZE})
        return
    Xtr_full = Xtr_full.reset_index(drop=True); ytr_full = ytr_full.reset_index(drop=True)
    all_cols = list(Xtr_full.columns)
    mc, _ = data.minority_info(ytr_full)
    if mc < 10:
        writer.writerow({**meta, "dataset": dataset, "seed": seed, "status": "skip_minority_train",
                         "search_size": SEARCH_SIZE, "tune_size": TUNE_SIZE,
                         "confirm_size": CONFIRM_SIZE})
        return
    Xsearch, Xpool, ysearch, ypool = train_test_split(
        Xtr_full, ytr_full, train_size=SEARCH_SIZE, stratify=ytr_full, random_state=seed)
    Xsearch = Xsearch.reset_index(drop=True); ysearch = ysearch.reset_index(drop=True)
    Xpool = Xpool.reset_index(drop=True); ypool = ypool.reset_index(drop=True)

    # --- indeks bag DIBAGI semua model/arm; reshuffle + flip mask tetap/trial ---
    rng_bag = np.random.RandomState((seed * 777013) % (2**32 - 1))
    n_search = len(ysearch)
    sub_n = int(SUBSAMPLE_FRAC * n_search)
    bag_idx = [rng_bag.choice(n_search, size=sub_n, replace=False)
               for _ in range(B_BAGS)]
    rng_bag2 = np.random.RandomState((seed * 777013 + 777001) % (2**32 - 1))
    bag_idx2 = [rng_bag2.choice(n_search, size=sub_n, replace=False)
                for _ in range(B_BAGS)]
    rng_flip = np.random.RandomState((seed * 31 + 555) % (2**32 - 1))
    flip_order = rng_flip.permutation(n_search)
    ysearch_np = np.asarray(ysearch)
    n_flip = int(FLIP_RATE * n_search)
    y_flip = ysearch_np.copy()
    y_flip[flip_order[:n_flip]] = 1 - y_flip[flip_order[:n_flip]]

    def fit_ens(model, params, y_fit, idx_list):
        fns = []
        for ib, idx_b in enumerate(idx_list):
            Xb = Xsearch.iloc[idx_b].reset_index(drop=True)
            yb = pd.Series(np.asarray(y_fit)[idx_b]).reset_index(drop=True)
            mc_b, _ = data.minority_info(yb)
            if mc_b < 10:
                continue
            fns.append(fit_proba_m(model, params, Xb, yb, all_cols))
        return fns

    def fit_f2(model, params, y_fit, idx_list):
        fns = []
        for ib, idx_b in enumerate(idx_list):
            Xb = Xsearch.iloc[idx_b].reset_index(drop=True)
            yb = pd.Series(np.asarray(y_fit)[idx_b]).reset_index(drop=True)
            mc_b, _ = data.minority_info(yb)
            if mc_b < 10:
                continue
            cv_b = StratifiedKFold(n_splits=GREEDY_CV_SPLITS, shuffle=True,
                                   random_state=(seed * 31 + ib * 97) % (2**32 - 1))
            sel = leave_one_out_fallback(Xb, yb, all_cols, cv_b, model)
            fns.append(fit_proba_m(model, params, Xb, yb, sel)
                       if set(sel) != set(all_cols) else fit_proba_m(
                           model, params, Xb, yb, all_cols))
        return fns

    def ens(fns, Xeval):
        return np.column_stack([fn(Xeval) for fn in fns]).mean(axis=1)

    # --- fit semua ensemble (baseline, I2, F2, broken, reshuffle) ---
    P = {}
    nb_eff = {}
    ok_trial = True
    for mkey, (model, params) in MODELS.items():
        P[(mkey, "base")] = fit_ens(model, params, ysearch_np, bag_idx)
        P[(mkey, "resh")] = fit_ens(model, params, ysearch_np, bag_idx2)
        P[(mkey, "brok")] = fit_ens(model, params, y_flip, bag_idx)
        P[(mkey, "f2")] = fit_f2(model, params, ysearch_np, bag_idx)
        for ci, cp in enumerate(I2_GRIDS[mkey]):
            P[(mkey, f"i2_{ci}")] = fit_ens(model, cp, ysearch_np, bag_idx)
        for key in ["base", "resh", "brok", "f2"] + \
                [f"i2_{ci}" for ci in range(len(I2_GRIDS[mkey]))]:
            nb_eff[(mkey, key)] = len(P[(mkey, key)])
            if len(P[(mkey, key)]) < 2:
                ok_trial = False
    if not ok_trial:
        writer.writerow({**meta, "dataset": dataset, "seed": seed, "status": "skip_bags_insufficient",
                         "search_size": SEARCH_SIZE, "tune_size": TUNE_SIZE,
                         "confirm_size": CONFIRM_SIZE})
        return

    ypool_np = np.asarray(ypool)
    Q = {k: ens(fns, Xpool) for k, fns in P.items()}
    Qref = {k: ens(fns, X_ref) for k, fns in P.items()}

    # --- split TUNE/CONFIRM (skema probe persis) ---
    rng_split = np.random.default_rng(seed + 20261014)
    i0 = rng_split.permutation(np.flatnonzero(ypool_np == 0))
    i1 = rng_split.permutation(np.flatnonzero(ypool_np == 1))
    n1 = int(round(TUNE_SIZE * np.mean(ypool_np)))
    n0 = TUNE_SIZE - n1
    if n0 > len(i0) or n1 > len(i1) or min(np.bincount(ypool_np)) < MIN_MINORITY_CONFIRM:
        writer.writerow({**meta, "dataset": dataset, "seed": seed, "status": "skip_minority_confirm",
                         "search_size": SEARCH_SIZE, "tune_size": TUNE_SIZE,
                         "confirm_size": CONFIRM_SIZE})
        return
    tune = np.r_[i0[:n0], i1[:n1]]
    confirm = np.r_[i0[n0:], i1[n1:]]
    y_tune, y_c = ypool_np[tune], ypool_np[confirm]
    y_ref_np = np.asarray(y_ref)

    def tune_delta(base_probs, cand_probs):
        base_t = (base_probs[tune] >= 0.5).astype(int)
        return float(delta_from_counts(counts8(
            y_tune, base_t, (cand_probs[tune] >= 0.5).astype(int))))

    # --- pilih varian di TUNE (strictly-greater, default baseline) ---
    sel = {}
    for mkey, (model, params) in MODELS.items():
        pb = Q[(mkey, "base")]
        thr, tg = 0.5, 0.0
        base_tune = (pb[tune] >= 0.5).astype(int)
        for tt in GRID:
            g = float(delta_from_counts(counts8(
                y_tune, base_tune, (pb[tune] >= float(tt)).astype(int))))
            if g > tg + 1e-12:
                thr, tg = float(tt), g
        sel[(mkey, "I1")] = (thr, tg,
                             f"thr{thr}" if thr != 0.5 else "base-t0.5")
        best_ci, best_g = -1, 0.0
        for ci in range(len(I2_GRIDS[mkey])):
            g = tune_delta(pb, Q[(mkey, f"i2_{ci}")])
            if g > best_g + 1e-12:
                best_ci, best_g = ci, g
        sel[(mkey, "I2")] = (best_ci, best_g,
                             cfg_name(model, I2_GRIDS[mkey][best_ci])
                             if best_ci >= 0 else cfg_name(model, params) + "-base")
        pM1, pM2 = Q[("M1", "base")], Q[("M2", "base")]
        own = pM1 if mkey == "M1" else pM2
        best_w, best_gw, bw_name = None, 0.0, "base"
        for w in I3_WEIGHTS:
            mix = w * pM1 + (1 - w) * pM2
            g = tune_delta(own, mix)
            if g > best_gw + 1e-12:
                best_w, best_gw, bw_name = w, g, f"w{w}"
        sel[(mkey, "I3")] = (best_w, best_gw, bw_name)

    def cand_probs(mkey, arm):
        if arm == "I1":
            thr, _, _ = sel[(mkey, "I1")]
            return Q[(mkey, "base")], thr
        if arm == "I2":
            ci, _, _ = sel[(mkey, "I2")]
            return (Q[(mkey, f"i2_{ci}")] if ci >= 0
                    else Q[(mkey, "base")]), 0.5
        if arm == "I3":
            w, _, _ = sel[(mkey, "I3")]
            if w is None:
                return Q[(mkey, "base")], 0.5
            return (w * Q[("M1", "base")] + (1 - w) * Q[("M2", "base")]), 0.5
        if arm == "I4":
            return Q[(mkey, "f2")], 0.5
        if arm == "Cplus":
            return Q[(mkey, "base")], 0.5
        if arm == "Cseed":
            return Q[(mkey, "resh")], 0.5
        raise ValueError(arm)

    def cand_ref(mkey, arm):
        if arm == "I1":
            thr, _, _ = sel[(mkey, "I1")]
            return Qref[(mkey, "base")], thr
        if arm == "I2":
            ci, _, _ = sel[(mkey, "I2")]
            return (Qref[(mkey, f"i2_{ci}")] if ci >= 0
                    else Qref[(mkey, "base")]), 0.5
        if arm == "I3":
            w, _, _ = sel[(mkey, "I3")]
            if w is None:
                return Qref[(mkey, "base")], 0.5
            return (w * Qref[("M1", "base")] + (1 - w) * Qref[("M2", "base")]), 0.5
        if arm == "I4":
            return Qref[(mkey, "f2")], 0.5
        if arm == "Cplus":
            return Qref[(mkey, "base")], 0.5
        if arm == "Cseed":
            return Qref[(mkey, "resh")], 0.5
        raise ValueError(arm)

    # base ref per model untuk kandidat Cplus (broken) & I4/F2-vs-base:
    # gate selalu (y, baseline_model, kandidat); ref pun begitu.
    arms = []
    for mkey in ["M1", "M2"]:
        arms += [(mkey, "I1"), (mkey, "I2"), (mkey, "I3"), (mkey, "I4"),
                 (mkey, "Cplus"), (mkey, "Cseed")]
    rng_gate = np.random.default_rng(seed + 82736)
    npz = {"y_c": y_c, "confirm_idx": confirm, "tune_idx": tune,
           "y_ref": y_ref_np}
    for ordinal, (mkey, arm) in enumerate(arms):
        if arm == "Cplus":
            pb_c = Q[(mkey, "brok")]
            pb_r = Qref[(mkey, "brok")]
            pc_c, tc = cand_probs(mkey, arm)
            pc_r, tr = cand_ref(mkey, arm)
            cfg, tg = "flip0.25-vs-clean", ""
        else:
            pb_c = Q[(mkey, "base")]
            pb_r = Qref[(mkey, "base")]
            pc_c, tc = cand_probs(mkey, arm)
            pc_r, tr = cand_ref(mkey, arm)
            if arm in ("I1", "I2", "I3"):
                _, tg, cfg = sel[(mkey, arm)]
            elif arm == "I4":
                cfg, tg = "F2", ""
            else:
                cfg, tg = "rebag", ""
        lbl_c = (pc_c[confirm] >= tc).astype(int)
        lbl_r = (pc_r >= tr).astype(int)
        base_c = (pb_c[confirm] >= 0.5).astype(int)
        base_r = (pb_r >= 0.5).astype(int)
        c = counts8(y_c, base_c, lbl_c)
        boot = delta_from_counts(rng_gate.multinomial(CONFIRM_SIZE, c / CONFIRM_SIZE, size=N_BOOT))
        lo99 = float(np.quantile(boot, 0.01))
        cr = counts8(y_ref_np, base_r, lbl_r)
        ref_delta = float(delta_from_counts(cr))
        ref_draws = delta_from_counts(np.random.default_rng(seed + 987163).dirichlet(cr + 0.5, size=N_BOOT))
        rlo, rhi = float(np.quantile(ref_draws, 0.025)), float(np.quantile(ref_draws, 0.975))
        hi99 = float(np.quantile(boot, 0.99))
        writer.writerow({
            **meta, "arm_ordinal": ordinal, "ci_hi99": hi99,
            "ref_ci_lo95": rlo, "ref_ci_hi95": rhi,
            **{f"c{k}": int(v) for k, v in zip(CODES8, c)},
            **{f"r{k}": int(v) for k, v in zip(CODES8, cr)},
            "schema_version": SCHEMA_VERSION,
            "dataset": dataset, "seed": seed, "status": "ok",
            "search_size": SEARCH_SIZE, "tune_size": TUNE_SIZE, "confirm_size": CONFIRM_SIZE,
            "n_features_total": len(all_cols),
            "n_bags_effective": min(nb_eff[(mkey, "base")], nb_eff[(mkey, "f2")]),
            "model": mkey, "arm": arm, "config": cfg, "tune_gain": tg,
            "n_confirm_disagreements": int(c[[1, 2, 5, 6]].sum()),
            "ci_lo99": lo99,
            "decision99": "accept" if lo99 > 1e-14 else ("reject" if float(np.quantile(boot, 0.99)) < 0 else "abstain"),
            "ref_delta": ref_delta,
            "ref_ambiguous_95": bool(rlo <= 0 <= rhi),
            "trial_sec": round(time.time() - t00, 1),
        })
        npz[f"{mkey}_{arm}_c"] = lbl_c
        npz[f"{mkey}_{arm}_r"] = lbl_r
    for mkey in MODELS:
        npz[f"{mkey}_base_c"] = (Q[(mkey, "base")][confirm] >= 0.5).astype(int)
        npz[f"{mkey}_base_r"] = (Qref[(mkey, "base")] >= 0.5).astype(int)
        npz[f"{mkey}_brok_c"] = (Q[(mkey, "brok")][confirm] >= 0.5).astype(int)
        npz[f"{mkey}_brok_r"] = (Qref[(mkey, "brok")] >= 0.5).astype(int)
    if pred_dir is not None:
        np.savez_compressed(pred_dir / f"{dataset}_{seed}.npz", **npz)


def load_config():
    with open(MANIFEST_DIR / "config.json") as f:
        cfg = json.load(f)
    if cfg.get("experiment") != "exp41":
        raise RuntimeError("config bukan exp41")
    if {k: int(v) for k, v in cfg["OFFSETS"].items()} != OFFSETS:
        raise RuntimeError("OFFSETS config != kode")
    if int(cfg["LOCKED_REF_SEED"]) != LOCKED_REF_SEED:
        raise RuntimeError("LOCKED_REF_SEED config != kode")
    return cfg


class _Buffer:
    """Penampung baris satu trial; ditulis ke CSV hanya setelah trial selesai."""
    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(row)


def completed_trials(out_dir):
    """(dataset, seed) yang sudah utuh (12 baris ok atau 1 baris skip) di CSV exp41."""
    import glob
    cnt, skip = {}, set()
    for fp in glob.glob(str(Path(out_dir) / "exp41_*_w*.csv")):
        with open(fp, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["dataset"], int(r["seed"]))
                if r["status"] == "ok":
                    cnt[key] = cnt.get(key, 0) + 1
                else:
                    skip.add(key)
    return {k for k, v in cnt.items() if v == 12} | skip


def trial_seed(stratum, ds_idx, rep):
    """Seed trial unik per (dataset, rep): offset + 1000*indeks_manifest + rep."""
    assert 0 <= rep < 1000
    return OFFSETS[stratum] + 1000 * ds_idx + rep


def check_reference(data_id, y_dev, y_ref):
    """Fail-closed: ukuran & komposisi referensi harus sama dgn manifest (audit #4)."""
    m = pd.read_csv(MANIFEST_DIR / "datasets.csv").set_index("data_id").loc[data_id]
    got = (len(y_dev) + len(y_ref), len(y_ref), int(np.sum(y_ref)))
    want = (int(m.N), int(m.n_ref), int(m.ref_pos))
    if got != want or len(y_ref) != ref_size(got[0]):
        raise RuntimeError(f"data_id {data_id}: (N, n_ref, ref_pos) {got} != manifest {want}")


def load_datasets():
    """(stratum, dataset, data_id, cluster) dari exp40_manifest/datasets.csv."""
    ds = pd.read_csv(MANIFEST_DIR / "datasets.csv")
    out = []
    for r in ds.itertuples():
        if r.stratum not in OFFSETS:
            raise RuntimeError(f"stratum tak dikenal: {r.stratum}")
        out.append((r.stratum, r.dataset, int(r.data_id), r.cluster))
    if len({d for _, d, _, _ in out}) != len(out):
        raise RuntimeError("nama dataset duplikat di manifest")
    return out


def main():
    cfg = load_config()
    reps = {k: int(v) for k, v in cfg["R"].items()}
    worker_id, n_workers = 0, 1
    if len(sys.argv) >= 3:
        worker_id, n_workers = int(sys.argv[1]), int(sys.argv[2])
    only = sys.argv[3] if len(sys.argv) >= 4 else None  # smoke: satu dataset
    out_dir = Path(sys.argv[4]) if len(sys.argv) >= 5 else \
        Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "exp41_preds"
    pred_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"exp41_{stamp}_w{worker_id}.csv"
    jobs = []  # (stratum, dataset, data_id, cluster, rep, seed)
    for ds_idx, (stratum, dataset, data_id, cluster) in enumerate(load_datasets()):
        if only is not None and dataset != only:
            continue
        for rep in range(reps[stratum]):  # seed unik per dataset (exp41 #2)
            jobs.append((stratum, dataset, data_id, cluster, rep,
                         trial_seed(stratum, ds_idx, rep)))
    mine = [j for k, j in enumerate(jobs) if k % n_workers == worker_id]
    done = completed_trials(out_dir)  # RESUME: lewati trial yg sudah tertulis utuh
    mine = [j for j in mine if (j[1], j[5]) not in done]
    print(f"[worker {worker_id}] {len(mine)} trial tersisa (resume lewati {len(done)} trial selesai global)",
          flush=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        f.flush()
        loaded = {}
        for stratum, dataset, data_id, cluster, rep, seed in mine:
            if dataset not in loaded:
                loaded.clear()
                print(f"[worker {worker_id}] Memuat {dataset}...", flush=True)
                loaded[dataset] = make_dev_and_locked_reference(data_id)
                check_reference(data_id, loaded[dataset][1], loaded[dataset][3])
            X_dev, y_dev, X_ref, y_ref = loaded[dataset]
            meta = {"stratum": stratum, "data_id": data_id, "cluster": cluster,
                    "rep": rep, "schema_version": SCHEMA_VERSION}
            buf = _Buffer()  # tulis SATU trial utuh sekaligus (aman terhadap crash)
            run_trial(dataset, X_dev, y_dev, X_ref, y_ref, seed, buf, pred_dir, meta)
            for row in buf.rows:
                writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
    print(f"[worker {worker_id}] selesai {len(mine)} trial -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

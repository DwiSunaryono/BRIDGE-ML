"""Eksperimen 22: isolasi cara konstruksi kandidat, bukan gate.

Latar: greedy_forward_select (exp01) tidak pernah membandingkan subset akhir
dengan baseline semua-fitur, sehingga dapat mengembalikan subset yang lebih
buruk. Di exp21 pada data real, tiap bag rata-rata mempertahankan 20-48%
fitur dan 537/600 kandidat gabungan punya ref_delta <= 0.

Tiga arm pencarian kandidat. Bagging, gate, SEARCH-CONFIRM, model, dan metrik
dibekukan. Bag dan fold sama antar-arm (idx_b dan cv_b dibagikan), sehingga
perbedaan hasil berasal dari metode seleksi:
  F0: forward lama (greedy_forward_select, tidak diubah).
  F1: forward dengan fallback. Skor OOF subset dibandingkan dengan skor OOF
      semua-fitur; bila subset tidak lebih baik, dipakai semua-fitur.
  F2: hapus maksimal satu fitur. Evaluasi baseline dan tiap leave-one-out
      (p+1 evaluasi OOF, sebanding satu lintasan forward), lalu pilih removal
      terbaik hanya bila mengalahkan baseline.

Catatan implementasi:
- Imputasi tidak lagi memakai fillna global sebelum split (exp21 py:95).
  Enam dataset ini tidak punya NaN numerik; kategorikal ditangani lewat kode
  kategori bersih. Imputer median di-fit per partisi lewat make_preprocessor.
- Ketidakpastian referensi dihitung per pasangan seperti exp17/20. Prediksi
  pasangan disimpan dan bootstrap dijalankan di referensi untuk tiap keputusan
  accept. Kasus ambigu ditandai, tidak dipaksa benar atau salah.
- Enam dataset exp21 dipakai sebagai DEV, bukan klaim generalitas independen.
  Seed baru dari dataset lama tidak cukup untuk menyebut validasi independen
  setelah dipakai mengarahkan desain. Dipakai 30 seed baru (exp21 memakai 50)
  per (dataset, N) untuk ketiga arm.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import config as C
import data
import models
from exp01_naive_selection import greedy_forward_select
from exp15_stabilized_candidate import fit_proba, B_BAGS, SUBSAMPLE_FRAC, CONFIRM_FRAC, DELTA_MIN, MODEL_NAME
from exp16_confirmatory_arm_e import paired_bootstrap_diffs_from_proba, decide, N_BOOT

DATASETS = {
    "adult": 1590, "bank-marketing": 1461, "MagicTelescope": 1120,
    "eeg-eye-state": 1471, "PhishingWebsites": 4534, "spambase": 44,
}
N_TRAIN_LIST = (600, 900)
REPS = 30
NEW_SEED_OFFSET = 50_000_000  # Tidak tumpang tindih dengan 50 seed exp21
LOCKED_REF_SEED = 20261007  # Sama dengan exp21 agar dev_pool dan referensi konsisten
GREEDY_CV_SPLITS = 3
REF_BOOT = 150

ARMS = ("F0_forward", "F1_forward_fallback", "F2_leave1out_fallback")
FIELDS = ["dataset", "n_train", "seed", "arm", "status", "n_features_total",
          "n_bags_effective", "mean_n_features_bags", "frac_noop_bags",
          "ci_lo", "ci_hi", "decision", "ref_delta", "ref_se_pair",
          "ref_ambiguous", "decision_correct_vs_ref"]


def load_openml_binary(data_id):
    """Muat dataset biner OpenML tanpa fillna global. Imputasi dilakukan di make_preprocessor per partisi saat fit_proba."""
    d = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    X = d.data.copy()
    y = d.target
    cats = sorted(y.astype(str).unique().tolist())
    y = pd.Series(pd.Categorical(y.astype(str), categories=cats).codes, name="target")
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            col_str = X[col].astype(object).where(X[col].notna(), "__MISSING__").astype(str)
            c2 = sorted(col_str.unique().tolist())
            X[col] = pd.Categorical(col_str, categories=c2).codes
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.reset_index(drop=True), y.reset_index(drop=True)


def make_dev_and_locked_reference(data_id):
    X, y = load_openml_binary(data_id)
    n_ref = min(8000, int(0.3 * len(X)))
    X_dev, X_ref, y_dev, y_ref = train_test_split(
        X, y, test_size=n_ref, stratify=y, random_state=LOCKED_REF_SEED)
    return X_dev.reset_index(drop=True), y_dev.reset_index(drop=True), \
        X_ref.reset_index(drop=True), y_ref.reset_index(drop=True)


def gt_f1(y_true, p):
    return f1_score(y_true, (p >= 0.5).astype(int), average="weighted", zero_division=0)


def forward_with_fallback(Xb, yb, cols, cv, model_name, seed):
    def build_fn(n):
        return models.build_estimator(model_name, {}, n, final=False)
    _, full_score, _ = models.oof_predict(build_fn, Xb, yb, cols, cv)
    sel, _, sel_score = greedy_forward_select(Xb, yb, cols, cv, model_name, seed)
    return sel if sel_score > full_score else list(cols)


def leave_one_out_fallback(Xb, yb, cols, cv, model_name):
    def build_fn(n):
        return models.build_estimator(model_name, {}, n, final=False)
    _, full_score, _ = models.oof_predict(build_fn, Xb, yb, cols, cv)
    best_cols, best_score = list(cols), full_score
    for feat in cols:
        trial_cols = [c for c in cols if c != feat]
        if not trial_cols:
            continue
        _, trial_score, _ = models.oof_predict(build_fn, Xb, yb, trial_cols, cv)
        if trial_score > best_score:
            best_cols, best_score = trial_cols, trial_score
    return best_cols


def run_trial(dataset, X_dev, y_dev, X_ref, y_ref, n_train, seed, writer):
    try:
        Xtr_full, _, ytr_full, _ = train_test_split(
            X_dev, y_dev, train_size=n_train, stratify=y_dev, random_state=seed)
    except ValueError:
        for arm in ARMS:
            writer.writerow({"dataset": dataset, "n_train": n_train, "seed": seed,
                              "arm": arm, "status": "skip_stratify_fail"})
        return []
    Xtr_full = Xtr_full.reset_index(drop=True); ytr_full = ytr_full.reset_index(drop=True)
    all_cols = list(Xtr_full.columns)

    mc, _ = data.minority_info(ytr_full)
    if mc < C.MIN_MINORITY_OUTER_TRAIN:
        for arm in ARMS:
            writer.writerow({"dataset": dataset, "n_train": n_train, "seed": seed,
                              "arm": arm, "status": "skip_minority"})
        return []

    Xsearch, Xconfirm, ysearch, yconfirm = train_test_split(
        Xtr_full, ytr_full, test_size=CONFIRM_FRAC, stratify=ytr_full, random_state=seed)
    Xsearch = Xsearch.reset_index(drop=True); ysearch = ysearch.reset_index(drop=True)
    Xconfirm = Xconfirm.reset_index(drop=True); yconfirm = yconfirm.reset_index(drop=True)

    # Bag dan fold dibagikan antar-arm (idx_b dan cv_b sama)
    rng_bag = np.random.RandomState((seed * 777013) % (2**32 - 1))
    n_search = len(ysearch)
    sub_n = int(SUBSAMPLE_FRAC * n_search)
    base_fns = []
    cand_fns = {arm: [] for arm in ARMS}
    n_feat_bags = {arm: [] for arm in ARMS}
    noop_bags = {arm: [] for arm in ARMS}

    for b in range(B_BAGS):
        idx_b = rng_bag.choice(n_search, size=sub_n, replace=False)
        Xb = Xsearch.iloc[idx_b].reset_index(drop=True)
        yb = ysearch.iloc[idx_b].reset_index(drop=True)
        mc_b, _ = data.minority_info(yb)
        if mc_b < C.MIN_MINORITY_OUTER_TRAIN:
            continue
        cv_b = StratifiedKFold(n_splits=GREEDY_CV_SPLITS, shuffle=True,
                                random_state=(seed * 31 + b * 97) % (2**32 - 1))
        bag_seed = (seed * 31 + b * 97) % (2**32 - 1)

        base_fns.append(fit_proba(Xb, yb, all_cols))

        sel_f0, _, _ = greedy_forward_select(Xb, yb, all_cols, cv_b, MODEL_NAME, bag_seed)
        sel_f1 = forward_with_fallback(Xb, yb, all_cols, cv_b, MODEL_NAME, bag_seed)
        sel_f2 = leave_one_out_fallback(Xb, yb, all_cols, cv_b, MODEL_NAME)

        for arm, sel in zip(ARMS, (sel_f0, sel_f1, sel_f2)):
            cand_fns[arm].append(fit_proba(Xb, yb, sel))
            n_feat_bags[arm].append(len(sel))
            noop_bags[arm].append(set(sel) == set(all_cols))

    n_bags_eff = len(base_fns)
    if n_bags_eff < 2:
        for arm in ARMS:
            writer.writerow({"dataset": dataset, "n_train": n_train, "seed": seed,
                              "arm": arm, "status": "skip_bags_insufficient", "n_bags_effective": n_bags_eff})
        return []

    def ens_predict(fns, Xeval):
        return np.column_stack([fn(Xeval) for fn in fns]).mean(axis=1)

    p_base_confirm = ens_predict(base_fns, Xconfirm)
    p_base_ref = ens_predict(base_fns, X_ref)

    out_rows = []
    rng_boot = np.random.RandomState(seed * 104729 % (2**32 - 1))
    for arm in ARMS:
        p_cand_confirm = ens_predict(cand_fns[arm], Xconfirm)
        diffs = paired_bootstrap_diffs_from_proba(yconfirm, p_base_confirm, p_cand_confirm, N_BOOT, rng_boot)
        ci_lo, ci_hi, decision = decide(diffs)

        p_cand_ref = ens_predict(cand_fns[arm], X_ref)
        ref_delta = gt_f1(y_ref, p_cand_ref) - gt_f1(y_ref, p_base_ref)

        # Ketidakpastian referensi per pasangan: bootstrap prediksi pasangan ini
        # di referensi, bukan nilai presisi yang diasumsikan.
        rng_ref = np.random.RandomState((seed * 13 + hash(arm) % 97) % (2**32 - 1))
        ref_diffs = paired_bootstrap_diffs_from_proba(y_ref, p_base_ref, p_cand_ref, REF_BOOT, rng_ref)
        ref_se_pair = float(np.std(ref_diffs))
        ref_ambiguous = abs(ref_delta) < ref_se_pair

        decision_correct = None
        if not ref_ambiguous:
            if decision == "accept":
                decision_correct = ref_delta > DELTA_MIN
            elif decision == "reject":
                decision_correct = ref_delta < DELTA_MIN

        row = {
            "dataset": dataset, "n_train": n_train, "seed": seed, "arm": arm, "status": "ok",
            "n_features_total": len(all_cols), "n_bags_effective": n_bags_eff,
            "mean_n_features_bags": float(np.mean(n_feat_bags[arm])),
            "frac_noop_bags": float(np.mean(noop_bags[arm])),
            "ci_lo": ci_lo, "ci_hi": ci_hi, "decision": decision,
            "ref_delta": ref_delta, "ref_se_pair": ref_se_pair,
            "ref_ambiguous": ref_ambiguous, "decision_correct_vs_ref": decision_correct,
        }
        writer.writerow(row)
        out_rows.append(row)
    return out_rows


def main():
    worker_id, n_workers = 0, 1
    if len(sys.argv) >= 3:
        worker_id, n_workers = int(sys.argv[1]), int(sys.argv[2])

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_w{worker_id}" if n_workers > 1 else ""
    out_path = out_dir / f"exp22_{stamp}{suffix}.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for dataset, data_id in DATASETS.items():
            print(f"[worker {worker_id}] Memuat {dataset}...")
            X_dev, y_dev, X_ref, y_ref = make_dev_and_locked_reference(data_id)
            for n_train in N_TRAIN_LIST:
                my_seeds = [s for s in range(REPS) if s % n_workers == worker_id]
                t0 = time.time()
                for seed in my_seeds:
                    run_trial(dataset, X_dev, y_dev, X_ref, y_ref, n_train, NEW_SEED_OFFSET + seed, writer)
                    f.flush()
                print(f"[worker {worker_id}][progress] {dataset} N={n_train} selesai "
                      f"({len(my_seeds)}/{REPS}, {time.time()-t0:.0f}s)")

    print(f"[worker {worker_id}] Log: {out_path}")


if __name__ == "__main__":
    main()

"""Eksperimen 9: memisahkan efek ukuran TRAIN dan ukuran TEST pada exp07.

exp07 menaikkan N_TRAIN dan N_TEST bersama pada split 80/20, sehingga perbaikan
tidak dapat diatribusikan ke pencarian yang lebih baik (TRAIN lebih besar),
evaluasi yang lebih stabil (TEST lebih besar), atau keduanya.

N_TRAIN dan N_TEST divariasikan secara independen. Distribusi generatif
ditetapkan sekali sebagai satu pool 150_000 baris dengan random_state tetap.
TRAIN, TEST_kecil, dan TEST_referensi tiap trial diambil sebagai subset tanpa
tumpang tindih dari pool yang sama, bukan dari make_classification dengan seed
berbeda yang dapat mengubah struktur soal. TEST_referensi (20_000 baris,
disjoint dari TRAIN dan TEST_kecil trial itu) menjadi pendekatan delta populasi
dan hanya dipakai sebagai instrumen riset, bukan data untuk mekanisme keputusan.

Tiap (N_TRAIN, N_TEST_kecil, seed): seleksi greedy_forward_select dari exp01
hanya memakai TRAIN untuk oof_delta; baseline dan kandidat di-fit pada TRAIN
penuh lalu dievaluasi pada TEST_kecil dan TEST_referensi; hasil dibandingkan
sebagai sign(oof_delta) vs sign(test_delta_kecil), sign(oof_delta) vs
sign(test_delta_ref) untuk kualitas seleksi, dan sign(test_delta_kecil) vs
sign(test_delta_ref) untuk noise TEST kecil.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

import config as C
import data
import models
from exp01_naive_selection import fit_eval_test, greedy_forward_select

POOL_SIZE = 150_000
N_INFORMATIVE = 5
N_NOISE = 25
N_REF = 20_000
N_TRAIN_LIST = (300, 800, 3000, 8000)
N_TEST_LIST = (200, 1000, 10000)
SEEDS = (0, 1, 2, 3, 4)
MODEL_NAME = "HGB"

FIELDS = ["n_train", "n_test", "seed", "status", "n_features_total",
          "oof_delta", "test_delta_small", "test_delta_ref",
          "match_oof_vs_small", "match_oof_vs_ref", "match_small_vs_ref"]


def build_pool():
    """Membangun satu pool populasi 150_000 baris dengan seed tetap.

    Semua trial menarik subset independen dari pool ini, bukan memanggil
    make_classification ulang dengan seed berbeda.
    """
    n_features = N_INFORMATIVE + N_NOISE
    X, y = make_classification(
        n_samples=POOL_SIZE, n_features=n_features, n_informative=N_INFORMATIVE,
        n_redundant=0, n_repeated=0, n_clusters_per_class=2, class_sep=1.0,
        flip_y=0.03, random_state=42)
    cols = [f"inf_{i}" for i in range(N_INFORMATIVE)] + [f"noise_{i}" for i in range(N_NOISE)]
    Xdf = pd.DataFrame(X, columns=cols)
    yser = pd.Series(y, name="target")
    return Xdf, yser, cols


def draw_disjoint(Xpool, ypool, n_train, n_test, seed):
    rng = np.random.RandomState(seed * 7919 + n_train * 13 + n_test)
    perm = rng.permutation(POOL_SIZE)
    train_idx = perm[:n_train]
    rest = perm[n_train:]
    test_idx = rest[:n_test]
    ref_idx = rest[n_test:n_test + N_REF]
    Xtr = Xpool.iloc[train_idx].reset_index(drop=True)
    ytr = ypool.iloc[train_idx].reset_index(drop=True)
    Xte = Xpool.iloc[test_idx].reset_index(drop=True)
    yte = ypool.iloc[test_idx].reset_index(drop=True)
    Xref = Xpool.iloc[ref_idx].reset_index(drop=True)
    yref = ypool.iloc[ref_idx].reset_index(drop=True)
    return Xtr, ytr, Xte, yte, Xref, yref


def run_combo(Xpool, ypool, cols, n_train, n_test, seed, writer):
    Xtr, ytr, Xte, yte, Xref, yref = draw_disjoint(Xpool, ypool, n_train, n_test, seed)
    mc, ratio = data.minority_info(ytr)
    if mc < C.MIN_MINORITY_OUTER_TRAIN:
        writer.writerow({"n_train": n_train, "n_test": n_test, "seed": seed,
                          "status": "skip_minority"})
        return None

    cv_full = StratifiedKFold(n_splits=C.INNER_CV_SPLITS, shuffle=True, random_state=seed)

    def build_fn(n_fold):
        return models.build_estimator(MODEL_NAME, {}, n_fold, final=False)

    _, oof_base_f1, _ = models.oof_predict(build_fn, Xtr, ytr, cols, cv_full)
    sel_cols, _, oof_sel_f1 = greedy_forward_select(Xtr, ytr, cols, cv_full, MODEL_NAME, seed)
    oof_delta = oof_sel_f1 - oof_base_f1

    # TEST_kecil dan TEST_referensi berasal dari fit yang sama (TRAIN penuh) dan
    # hanya dipakai di sini sebagai instrumen pengukuran riset, bukan oleh
    # mekanisme keputusan.
    test_base_small, _ = fit_eval_test(Xtr, ytr, Xte, yte, cols, MODEL_NAME)
    test_sel_small, _ = fit_eval_test(Xtr, ytr, Xte, yte, sel_cols, MODEL_NAME)
    test_delta_small = test_sel_small - test_base_small

    test_base_ref, _ = fit_eval_test(Xtr, ytr, Xref, yref, cols, MODEL_NAME)
    test_sel_ref, _ = fit_eval_test(Xtr, ytr, Xref, yref, sel_cols, MODEL_NAME)
    test_delta_ref = test_sel_ref - test_base_ref

    row = {
        "n_train": n_train, "n_test": n_test, "seed": seed, "status": "ok",
        "n_features_total": len(cols),
        "oof_delta": oof_delta, "test_delta_small": test_delta_small, "test_delta_ref": test_delta_ref,
        "match_oof_vs_small": (oof_delta > 0) == (test_delta_small > 0),
        "match_oof_vs_ref": (oof_delta > 0) == (test_delta_ref > 0),
        "match_small_vs_ref": (test_delta_small > 0) == (test_delta_ref > 0),
    }
    writer.writerow(row)
    print(f"[ok] n_train={n_train} n_test={n_test} seed={seed}: oof={oof_delta:+.4f} "
          f"small={test_delta_small:+.4f} ref={test_delta_ref:+.4f} "
          f"match(oof/small={row['match_oof_vs_small']} oof/ref={row['match_oof_vs_ref']} "
          f"small/ref={row['match_small_vs_ref']})")
    return row


def print_summary(rows):
    df = pd.DataFrame(rows)
    print("\n=== Efek N_TRAIN (dirata-ratakan lintas semua N_TEST) -- apakah SELEKSI makin benar? ===")
    for n_train, g in df.groupby("n_train"):
        print(f"  N_TRAIN={n_train:5d}: match_oof_vs_ref={100*g.match_oof_vs_ref.mean():.0f}%  "
              f"(n={len(g)}, ini ukuran KUALITAS SELEKSI murni, independen dr TEST kecil)")
    print("\n=== Efek N_TEST (dirata-ratakan lintas semua N_TRAIN) -- apakah EVALUASI makin stabil? ===")
    for n_test, g in df.groupby("n_test"):
        print(f"  N_TEST={n_test:6d}: match_small_vs_ref={100*g.match_small_vs_ref.mean():.0f}%  "
              f"(n={len(g)}, ini ukuran NOISE TEST KECIL sendiri, independen dr mekanisme seleksi)")
    print("\n=== Tabel lengkap N_TRAIN x N_TEST: match_oof_vs_small (yg selama ini kita ukur di exp01-08) ===")
    pivot = df.pivot_table(index="n_train", columns="n_test", values="match_oof_vs_small", aggfunc="mean") * 100
    print(pivot.round(0))
    print("\n=== Tabel lengkap N_TRAIN x N_TEST: match_oof_vs_ref (kualitas SELEKSI sebenarnya) ===")
    pivot2 = df.pivot_table(index="n_train", columns="n_test", values="match_oof_vs_ref", aggfunc="mean") * 100
    print(pivot2.round(0))
    print("\nDetail (ringkas, per n_train x n_test, rata2 lintas seed):")
    summ = df.groupby(["n_train", "n_test"]).agg(
        match_oof_small=("match_oof_vs_small", "mean"),
        match_oof_ref=("match_oof_vs_ref", "mean"),
        match_small_ref=("match_small_vs_ref", "mean"),
    ).round(2)
    print(summ.to_string())


def main():
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"exp09_{stamp}.csv"

    print("Membangun pool populasi (150_000 baris, distribusi tetap)...")
    Xpool, ypool, cols = build_pool()

    rows = []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for n_train in N_TRAIN_LIST:
            for n_test in N_TEST_LIST:
                for seed in SEEDS:
                    row = run_combo(Xpool, ypool, cols, n_train, n_test, seed, writer)
                    f.flush()
                    if row:
                        rows.append(row)

    print(f"\nLog lengkap: {out_path}")
    if rows:
        print_summary(rows)


if __name__ == "__main__":
    main()

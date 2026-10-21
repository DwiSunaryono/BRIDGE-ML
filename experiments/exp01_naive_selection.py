"""Eksperimen 1: bukti dasar bahwa kenaikan di OOF-TRAIN tidak selalu diikuti
kenaikan di TEST (README.md temuan #1).

Kode dibangun ulang; angka dan kesimpulan lama di README/config.py tidak
dipakai, hanya idenya (greedy pemilihan fitur).

Desain sengaja naif dan murah: seleksi fitur hanya memakai skor OOF-TRAIN
(5-fold) tanpa proteksi SEL/BLIND, sehingga pola false-positive terlihat tanpa
gate yang menutupinya. Greedy forward single-pass, bukan exhaustive 2^F: fitur
diurutkan dengan permutation importance (models.feature_ranking), lalu ditambah
ke himpunan terpilih hanya bila menaikkan skor OOF-TRAIN saat itu.

TEST dikunci dan dibaca tepat sekali per run setelah seleksi selesai. Disiplin
ditegakkan oleh data.lock_test, yang melempar RuntimeError bila TEST dibaca
dua kali.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

import config as C
import data
import models

DATASETS = ("pc4", "diabetes")
MODEL_POOL = ("HGB", "DT")
SEEDS = (0, 1, 2, 3, 4)
EPS = 1e-9  # ambang kenaikan OOF-TRAIN saat greedy, hanya meredam noise float

FIELDS = ["dataset", "model", "seed", "status", "minority_count", "minority_ratio",
          "n_features_total", "n_features_selected",
          "oof_base_wf1", "oof_sel_wf1", "oof_delta_wf1",
          "oof_base_bal", "oof_sel_bal", "oof_delta_bal",
          "test_base_wf1", "test_sel_wf1", "test_delta_wf1",
          "test_base_bal", "test_sel_bal", "test_delta_bal"]


def greedy_forward_select(Xtr, ytr, cols, cv, model_name, seed):
    """Single-pass greedy forward murni OOF-TRAIN. Return
    (selected_cols, oof_preds_selected, oof_wf1_selected)."""
    ranking = models.feature_ranking(Xtr, ytr, cols, cv, est=model_name,
                                      params=None, seed=seed)

    def build_fn(n_fold_train_size):
        return models.build_estimator(model_name, {}, n_fold_train_size, final=False)

    selected = [ranking[0]]
    best_preds, best_score, _ = models.oof_predict(build_fn, Xtr, ytr, selected, cv)
    for feat in ranking[1:]:
        trial_cols = selected + [feat]
        trial_preds, trial_score, _ = models.oof_predict(build_fn, Xtr, ytr, trial_cols, cv)
        if trial_score > best_score + EPS:
            selected, best_score, best_preds = trial_cols, trial_score, trial_preds
    return selected, best_preds, best_score


def fit_eval_test(Xtr, ytr, Xte, yte, cols, model_name):
    num_c, cat_c = data.column_kinds(Xtr[cols])
    pre = data.make_preprocessor(num_c, cat_c)
    Ztr, Zte = pre.fit_transform(Xtr[cols], ytr), pre.transform(Xte[cols])
    est = models.build_estimator(model_name, {}, len(ytr), final=False)
    est.fit(Ztr, ytr)
    pred = est.predict(Zte)
    return C.score_metric(yte, pred), balanced_accuracy_score(yte, pred)


def run_one(X, y, dataset, model_name, seed, writer):
    # Counter TEST di data.py bersifat global per proses dan dirancang untuk satu
    # outer-split per run pipeline. Tiap (dataset, model, seed) di sini adalah run
    # independen dengan outer-split berbeda, jadi counter direset agar tiap run
    # tetap membaca TEST satu kali, bukan untuk melewati disiplin blind.
    data.reset_test_counter()
    Xtr, Xte, ytr, yte = data.outer_split(X, y, seed=seed)
    mc, ratio = data.minority_info(ytr)
    if mc < C.MIN_MINORITY_OUTER_TRAIN:
        writer.writerow({"dataset": dataset, "model": model_name, "seed": seed,
                          "status": "skip_minority", "minority_count": mc,
                          "minority_ratio": round(ratio, 4)})
        print(f"[skip] {dataset}/{model_name}/seed{seed}: minority_count={mc} < "
              f"{C.MIN_MINORITY_OUTER_TRAIN}")
        return None

    cols = list(Xtr.columns)
    cv = StratifiedKFold(n_splits=C.INNER_CV_SPLITS, shuffle=True, random_state=seed)

    def build_fn(n_fold_train_size):
        return models.build_estimator(model_name, {}, n_fold_train_size, final=False)

    oof_base_preds, oof_base_f1, _ = models.oof_predict(build_fn, Xtr, ytr, cols, cv)
    oof_base_bal = balanced_accuracy_score(ytr, oof_base_preds)

    sel_cols, oof_sel_preds, oof_sel_f1 = greedy_forward_select(
        Xtr, ytr, cols, cv, model_name, seed)
    oof_sel_bal = balanced_accuracy_score(ytr, oof_sel_preds)

    # TEST dikunci dan dibaca satu kali setelah seleksi selesai.
    Xte, yte = data.lock_test(Xte, yte)
    test_base_f1, test_base_bal = fit_eval_test(Xtr, ytr, Xte, yte, cols, model_name)
    test_sel_f1, test_sel_bal = fit_eval_test(Xtr, ytr, Xte, yte, sel_cols, model_name)

    row = {
        "dataset": dataset, "model": model_name, "seed": seed, "status": "ok",
        "minority_count": mc, "minority_ratio": round(ratio, 4),
        "n_features_total": len(cols), "n_features_selected": len(sel_cols),
        "oof_base_wf1": oof_base_f1, "oof_sel_wf1": oof_sel_f1,
        "oof_delta_wf1": oof_sel_f1 - oof_base_f1,
        "oof_base_bal": oof_base_bal, "oof_sel_bal": oof_sel_bal,
        "oof_delta_bal": oof_sel_bal - oof_base_bal,
        "test_base_wf1": test_base_f1, "test_sel_wf1": test_sel_f1,
        "test_delta_wf1": test_sel_f1 - test_base_f1,
        "test_base_bal": test_base_bal, "test_sel_bal": test_sel_bal,
        "test_delta_bal": test_sel_bal - test_base_bal,
    }
    writer.writerow(row)
    print(f"[ok] {dataset}/{model_name}/seed{seed}: "
          f"oof_delta_wf1={row['oof_delta_wf1']:+.4f} "
          f"test_delta_wf1={row['test_delta_wf1']:+.4f}")
    return row


def main():
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"exp01_{stamp}.csv"

    rows = []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for dataset in DATASETS:
            try:
                X, y = data.load_dataset(dataset)
            except Exception as exc:  # fetch_openml membutuhkan internet, jadi kegagalan load dilewati
                print(f"[skip-dataset] {dataset}: gagal load ({exc!r})")
                writer.writerow({"dataset": dataset, "model": "", "seed": "",
                                  "status": f"skip_load_error:{exc!r}"})
                continue
            for model_name in MODEL_POOL:
                for seed in SEEDS:
                    row = run_one(X, y, dataset, model_name, seed, writer)
                    f.flush()
                    if row:
                        rows.append(row)

    print(f"\nLog lengkap: {out_path}")
    n_ok = len(rows)
    n_fp = sum(1 for r in rows if r["oof_delta_wf1"] > 0 and r["test_delta_wf1"] < 0)
    n_tp = sum(1 for r in rows if r["oof_delta_wf1"] > 0 and r["test_delta_wf1"] > 0)
    n_noop = sum(1 for r in rows if r["oof_delta_wf1"] <= 0)
    print(f"Ringkasan ({n_ok} run sukses):")
    print(f"  OOF naik tapi TEST turun (pola false-positive): {n_fp}")
    print(f"  OOF naik dan TEST ikut naik (klop):              {n_tp}")
    print(f"  OOF tidak naik sama sekali (tak ada fitur nambah): {n_noop}")


if __name__ == "__main__":
    main()

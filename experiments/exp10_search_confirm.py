"""Eksperimen 10: implementasi mekanisme SEARCH/CONFIRM dengan target keputusan berisiko terukur,
boleh abstain bila bukti belum cukup. Target sebelumnya, menebak tanda delta di satu TEST kecil,
dibuktikan exp09 punya batas noise sekitar 55% di N_TEST=200 dan tidak terhindarkan secanggih apa pun
mekanismenya.

Prosedur:
1. TRAIN dibagi SEARCH (70%, cari dan pilih kandidat via greedy_forward_select, fungsi sama dari exp01
   yang dibekukan) dan CONFIRM (30%, dipakai sekali).
2. Baseline dan kandidat hanya di-fit di SEARCH, keduanya diprediksi di CONFIRM.
3. Paired bootstrap (resample baris CONFIRM, hitung ulang selisih weighted-F1) menghasilkan interval
   kepercayaan 90% untuk delta.
4. Keputusan: batas bawah > delta_min berarti terima, batas atas < 0 berarti tolak, selain itu abstain.

Validasi memakai simulasi dengan pool tetap sama persis dengan exp09 dan distribusi generatif
ditetapkan sekali, karena perlu ground-truth delta populasi dari TEST referensi besar (instrumen riset,
bukan data yang dipakai mekanisme) untuk mengukur risiko keputusan salah yang sesungguhnya. exp09
menunjukkan risiko itu tidak bisa diukur jujur dari satu TEST real kecil.
Interval dari pasangan SEARCH-fit tidak otomatis berlaku untuk pasangan hasil refit di seluruh TRAIN,
jadi dicatat dua ground-truth terpisah (searchfit dan fulltrainfit).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import config as C
import data
import models
from exp01_naive_selection import fit_eval_test, greedy_forward_select
from exp09_isolate_train_test_noise import build_pool, POOL_SIZE

N_TRAIN_LIST = (600, 900, 3000)
REPS = 30
N_REF = 20_000
CONFIRM_FRAC = 0.3
N_BOOT = 1000
ALPHA = 0.10  # interval 90%
DELTA_MIN = 0.0
MODEL_NAME = "HGB"

FIELDS = ["n_train", "seed", "status", "n_search", "n_confirm",
          "ci_lo", "ci_hi", "decision",
          "true_delta_searchfit_ref", "true_delta_fulltrainfit_ref",
          "n_features_selected"]


def paired_bootstrap_ci(y_true, pred_base, pred_cand, n_boot, alpha, rng):
    y_true = np.asarray(y_true)
    pred_base = np.asarray(pred_base)
    pred_cand = np.asarray(pred_cand)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        f1_b = f1_score(y_true[idx], pred_base[idx], average="weighted", zero_division=0)
        f1_c = f1_score(y_true[idx], pred_cand[idx], average="weighted", zero_division=0)
        diffs[b] = f1_c - f1_b
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def run_trial(Xpool, ypool, cols, n_train, seed, writer):
    rng_np = np.random.RandomState(seed * 7919 + n_train * 13)
    perm = rng_np.permutation(POOL_SIZE)
    train_idx = perm[:n_train]
    ref_idx = perm[n_train:n_train + N_REF]
    Xtr_full = Xpool.iloc[train_idx].reset_index(drop=True)
    ytr_full = ypool.iloc[train_idx].reset_index(drop=True)
    Xref = Xpool.iloc[ref_idx].reset_index(drop=True)
    yref = ypool.iloc[ref_idx].reset_index(drop=True)

    mc, _ = data.minority_info(ytr_full)
    if mc < C.MIN_MINORITY_OUTER_TRAIN:
        writer.writerow({"n_train": n_train, "seed": seed, "status": "skip_minority"})
        return None

    Xsearch, Xconfirm, ysearch, yconfirm = train_test_split(
        Xtr_full, ytr_full, test_size=CONFIRM_FRAC, stratify=ytr_full, random_state=seed)
    Xsearch = Xsearch.reset_index(drop=True); ysearch = ysearch.reset_index(drop=True)
    Xconfirm = Xconfirm.reset_index(drop=True); yconfirm = yconfirm.reset_index(drop=True)

    mc_s, _ = data.minority_info(ysearch)
    if mc_s < C.MIN_MINORITY_OUTER_TRAIN:
        writer.writerow({"n_train": n_train, "seed": seed, "status": "skip_minority_search"})
        return None

    cv_search = StratifiedKFold(n_splits=C.INNER_CV_SPLITS, shuffle=True, random_state=seed)
    sel_cols, _, _ = greedy_forward_select(Xsearch, ysearch, cols, cv_search, MODEL_NAME, seed)

    def fit_predict(Xfit, yfit, Xpred, feat_cols):
        num_c, cat_c = data.column_kinds(Xfit[feat_cols])
        pre = data.make_preprocessor(num_c, cat_c)
        Zfit = pre.fit_transform(Xfit[feat_cols], yfit)
        Zpred = pre.transform(Xpred[feat_cols])
        est = models.build_estimator(MODEL_NAME, {}, len(yfit), final=False)
        est.fit(Zfit, yfit)
        return est.predict(Zpred)

    # Fase CONFIRM: prediksi dari model fit SEARCH saja, dipakai sekali
    pred_base_confirm = fit_predict(Xsearch, ysearch, Xconfirm, cols)
    pred_cand_confirm = fit_predict(Xsearch, ysearch, Xconfirm, sel_cols)
    rng = np.random.RandomState(seed * 104729)
    ci_lo, ci_hi = paired_bootstrap_ci(yconfirm, pred_base_confirm, pred_cand_confirm,
                                        N_BOOT, ALPHA, rng)
    if ci_lo > DELTA_MIN:
        decision = "accept"
    elif ci_hi < 0:
        decision = "reject"
    else:
        decision = "abstain"

    # Ground truth (instrumen riset, tidak dipakai mekanisme keputusan)
    true_delta_searchfit, _ = fit_eval_test(Xsearch, ysearch, Xref, yref, sel_cols, MODEL_NAME)
    true_base_searchfit, _ = fit_eval_test(Xsearch, ysearch, Xref, yref, cols, MODEL_NAME)
    true_delta_searchfit = true_delta_searchfit - true_base_searchfit

    true_delta_fulltrain, _ = fit_eval_test(Xtr_full, ytr_full, Xref, yref, sel_cols, MODEL_NAME)
    true_base_fulltrain, _ = fit_eval_test(Xtr_full, ytr_full, Xref, yref, cols, MODEL_NAME)
    true_delta_fulltrain = true_delta_fulltrain - true_base_fulltrain

    row = {
        "n_train": n_train, "seed": seed, "status": "ok",
        "n_search": len(ysearch), "n_confirm": len(yconfirm),
        "ci_lo": ci_lo, "ci_hi": ci_hi, "decision": decision,
        "true_delta_searchfit_ref": true_delta_searchfit,
        "true_delta_fulltrainfit_ref": true_delta_fulltrain,
        "n_features_selected": len(sel_cols),
    }
    writer.writerow(row)
    print(f"[ok] n_train={n_train} seed={seed}: CI=[{ci_lo:+.4f},{ci_hi:+.4f}] "
          f"decision={decision:7s} true_delta(searchfit)={true_delta_searchfit:+.4f} "
          f"true_delta(fulltrain)={true_delta_fulltrain:+.4f}")
    return row


def print_summary(rows):
    df = pd.DataFrame(rows)
    print("\n=== Ringkasan per N_TRAIN ===")
    for n_train, g in df.groupby("n_train"):
        n = len(g)
        acc = g[g.decision == "accept"]
        rej = g[g.decision == "reject"]
        abst = g[g.decision == "abstain"]
        wrong_among_accept = (acc.true_delta_searchfit_ref <= DELTA_MIN).mean() if len(acc) else float("nan")
        wrong_among_reject = (rej.true_delta_searchfit_ref > DELTA_MIN).mean() if len(rej) else float("nan")
        print(f"N_TRAIN={n_train:5d} (n={n}): accept={len(acc)} ({100*len(acc)/n:.0f}%) "
              f"reject={len(rej)} ({100*len(rej)/n:.0f}%) abstain={len(abst)} ({100*len(abst)/n:.0f}%)")
        print(f"             -> di antara ACCEPT: salah (true_delta<=0) = {100*wrong_among_accept:.0f}% "
              f"(target reviu internal: <=10%)")
        print(f"             -> di antara REJECT: sebenarnya untung (true_delta>0) = {100*wrong_among_reject:.0f}%")
    print("\nDetail:")
    print(df[["n_train", "seed", "ci_lo", "ci_hi", "decision",
              "true_delta_searchfit_ref", "true_delta_fulltrainfit_ref"]].to_string(index=False))


def main():
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"exp10_{stamp}.csv"

    print("Membangun pool populasi (sama dgn exp09, 150_000 baris, distribusi tetap)...")
    Xpool, ypool, cols = build_pool()

    rows = []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for n_train in N_TRAIN_LIST:
            for seed in range(REPS):
                row = run_trial(Xpool, ypool, cols, n_train, seed, writer)
                f.flush()
                if row:
                    rows.append(row)

    print(f"\nLog lengkap: {out_path}")
    if rows:
        print_summary(rows)


if __name__ == "__main__":
    main()

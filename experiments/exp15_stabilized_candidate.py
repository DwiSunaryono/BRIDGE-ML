"""Eksperimen 15: perubahan cara membangun kandidat, bukan sweep ambang.
Rujukan: Soloff/Barber/Willett (bagging stability), Sun (feature bagging),
Adrian/Soloff/Willett (inflated argmax), Bayle/Janson/Mackey (relative
instability of model comparison). Arm A sampai E dibandingkan pada baseline p0
tunggal yang sama (fit sekali di SEARCH dengan fitur baseline lengkap),
memakai CONFIRM/CI/delta_min yang sama dengan exp14 (one-sided 95%, ALPHA
dua sisi 0.10), sehingga efek perubahan kandidat terpisah dari perubahan
ambang. Kandidat tetap pre-registered dari exp12/14. Ground truth per
repetisi diambil dari objek prediksi yang sama dengan yang dikonfirmasi.
B=32, subsample 0.80, lambda 0.5, tau 0.80 adalah titik awal engineering,
bukan konstanta optimal. No-op dicatat terpisah, bukan dihitung menang.
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
from sklearn.model_selection import train_test_split

import config as C
import data
import models
from exp12_calibration_validation import SCENARIOS, sample_dgp, scenario_feature_sets, make_reference, clopper_pearson

N_TRAIN_LIST = (600, 900)
REPS_BY_SCENARIO = {"negative": 30, "nearzero": 80, "positive": 80}
CONFIRM_FRAC = 0.3
N_BOOT = 1000
ALPHA = 0.10  # one-sided 95%, level dasar exp14
DELTA_MIN = 0.0
MODEL_NAME = "HGB"

B_BAGS = 32
SUBSAMPLE_FRAC = 0.80
LAMBDA_ = 0.5
TAU = 0.80

ARMS = ("A_single", "B_bagged_candidate_vs_p0_CONFOUNDED", "C_anchored_vs_bagged_baseline",
        "D_bagged_baseline_diag", "E_bagged_candidate_vs_bagged_baseline_FAIR")
FIELDS = ["scenario", "n_train", "seed", "arm", "status", "n_bags_effective",
          "is_noop", "ci_lo", "ci_hi", "decision", "gt_delta_rep", "gt_in_ci", "decision_correct_vs_gt"]


def fit_proba(Xfit, yfit, feat_cols):
    num_c, cat_c = data.column_kinds(Xfit[feat_cols])
    pre = data.make_preprocessor(num_c, cat_c)
    Z = pre.fit_transform(Xfit[feat_cols], yfit)
    est = models.build_estimator(MODEL_NAME, {}, len(yfit), final=True)
    est.fit(Z, yfit)
    pos_idx = list(est.classes_).index(1) if 1 in getattr(est, "classes_", [0, 1]) else 1

    def predict_proba(Xnew):
        return est.predict_proba(pre.transform(Xnew[feat_cols]))[:, pos_idx]
    return predict_proba


def paired_bootstrap_diffs_from_proba(y_true, p_base, p_cand, n_boot, rng):
    y_true = np.asarray(y_true)
    lbl_base = (p_base >= 0.5).astype(int)
    lbl_cand = (p_cand >= 0.5).astype(int)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        f1b = f1_score(y_true[idx], lbl_base[idx], average="weighted", zero_division=0)
        f1c = f1_score(y_true[idx], lbl_cand[idx], average="weighted", zero_division=0)
        diffs[b] = f1c - f1b
    return diffs


def decide(diffs, alpha=ALPHA, delta_min=DELTA_MIN):
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if lo > delta_min:
        return lo, hi, "accept"
    if hi < 0:
        return lo, hi, "reject"
    return lo, hi, "abstain"


def gt_f1(y_true, p):
    return f1_score(y_true, (p >= 0.5).astype(int), average="weighted", zero_division=0)


def run_trial(scenario, cfg, n_train, seed, Xref, yref, writer):
    Xtr_full, ytr_full, cols = sample_dgp(n_train, cfg["beta_switch"], cfg["beta_extra"],
                                           seed=(seed * 131071 + n_train) % (2**32 - 1))
    baseline_cols, candidate_cols = scenario_feature_sets(cols, cfg)

    mc, _ = data.minority_info(ytr_full)
    if mc < C.MIN_MINORITY_OUTER_TRAIN:
        for arm in ARMS:
            writer.writerow({"scenario": scenario, "n_train": n_train, "seed": seed,
                              "arm": arm, "status": "skip_minority"})
        return []

    Xsearch, Xconfirm, ysearch, yconfirm = train_test_split(
        Xtr_full, ytr_full, test_size=CONFIRM_FRAC, stratify=ytr_full, random_state=seed)
    Xsearch = Xsearch.reset_index(drop=True); ysearch = ysearch.reset_index(drop=True)
    Xconfirm = Xconfirm.reset_index(drop=True); yconfirm = yconfirm.reset_index(drop=True)

    # p0: baseline tunggal, dipakai sebagai pembanding di semua arm
    p0_fn = fit_proba(Xsearch, ysearch, baseline_cols)
    p0_confirm = p0_fn(Xconfirm)
    p0_ref = p0_fn(Xref)

    # Arm A: kandidat lama, satu kali fit, kontrol historis
    pcand_single_fn = fit_proba(Xsearch, ysearch, candidate_cols)
    pcand_single_confirm = pcand_single_fn(Xconfirm)
    pcand_single_ref = pcand_single_fn(Xref)

    # 32 bag: subsample 80% SEARCH tanpa penggantian, fit baseline_b dan candidate_b
    rng_bag = np.random.RandomState((seed * 999983 + n_train) % (2**32 - 1))
    n_search = len(ysearch)
    sub_n = int(SUBSAMPLE_FRAC * n_search)
    baseline_b_fns, candidate_b_fns = [], []
    for b in range(B_BAGS):
        idx_b = rng_bag.choice(n_search, size=sub_n, replace=False)
        Xb = Xsearch.iloc[idx_b].reset_index(drop=True)
        yb = ysearch.iloc[idx_b].reset_index(drop=True)
        mc_b, _ = data.minority_info(yb)
        if mc_b < C.MIN_MINORITY_OUTER_TRAIN:
            continue
        baseline_b_fns.append(fit_proba(Xb, yb, baseline_cols))
        candidate_b_fns.append(fit_proba(Xb, yb, candidate_cols))
    n_bags_eff = len(baseline_b_fns)
    if n_bags_eff < 2:
        for arm in ARMS:
            writer.writerow({"scenario": scenario, "n_train": n_train, "seed": seed,
                              "arm": arm, "status": "skip_bags_insufficient",
                              "n_bags_effective": n_bags_eff})
        return []

    def bag_matrix(fns, Xeval):
        return np.column_stack([fn(Xeval) for fn in fns])

    cand_b_confirm = bag_matrix(candidate_b_fns, Xconfirm)
    cand_b_ref = bag_matrix(candidate_b_fns, Xref)
    base_b_confirm = bag_matrix(baseline_b_fns, Xconfirm)
    base_b_ref = bag_matrix(baseline_b_fns, Xref)

    # Arm B: rata-rata proba 32 fit kandidat terhadap p0 tunggal. Perbandingan
    # ini tercampur manfaat bagging generik dan dipertahankan untuk mengukur
    # besarnya confound (bandingkan dengan arm E).
    pcand_bag_confirm = cand_b_confirm.mean(axis=1)
    pcand_bag_ref = cand_b_ref.mean(axis=1)

    # Arm D: rata-rata proba 32 fit baseline terhadap p0 tunggal, mengukur
    # manfaat bagging generik tanpa perubahan fitur.
    pbase_bag_confirm = base_b_confirm.mean(axis=1)
    pbase_bag_ref = base_b_ref.mean(axis=1)

    # Arm C: anchored, d_b = cand_b - base_b pada pasangan bag yang sama
    def anchored(p0_eval, cand_b, base_b):
        d = cand_b - base_b
        mean_d = d.mean(axis=1)
        frac_pos = (d > 0).mean(axis=1)
        frac_neg = (d < 0).mean(axis=1)
        s = np.maximum(frac_pos, frac_neg)
        mask = (s >= TAU).astype(float)
        p_new = np.clip(p0_eval + LAMBDA_ * mean_d * mask, 0.0, 1.0)
        is_noop_row = mask == 0.0
        return p_new, is_noop_row

    # Anchor C memakai bagged_baseline, bukan p0 tunggal, agar C mewarisi
    # manfaat bagging yang sama dengan D. Versi awal yang anchor ke p0
    # ternyata lebih buruk dari D murni.
    panchor_confirm, noop_confirm = anchored(pbase_bag_confirm, cand_b_confirm, base_b_confirm)
    panchor_ref, noop_ref = anchored(pbase_bag_ref, cand_b_ref, base_b_ref)
    is_noop_anchor = bool(np.all(noop_confirm) and np.all(noop_ref))

    # urutan: baseline confirm, baseline ref, kandidat confirm, kandidat ref, is_noop
    variants = {
        "A_single": (p0_confirm, p0_ref, pcand_single_confirm, pcand_single_ref, False),
        "B_bagged_candidate_vs_p0_CONFOUNDED": (p0_confirm, p0_ref, pcand_bag_confirm, pcand_bag_ref, False),
        "D_bagged_baseline_diag": (p0_confirm, p0_ref, pbase_bag_confirm, pbase_bag_ref, False),
        "E_bagged_candidate_vs_bagged_baseline_FAIR": (pbase_bag_confirm, pbase_bag_ref, pcand_bag_confirm, pcand_bag_ref, False),
        "C_anchored_vs_bagged_baseline": (pbase_bag_confirm, pbase_bag_ref, panchor_confirm, panchor_ref, is_noop_anchor),
    }

    rows = []
    rng_boot = np.random.RandomState((seed * 104729 + n_train) % (2**32 - 1))
    for arm, (p_base_confirm, p_base_ref, p_cand_confirm, p_cand_ref, is_noop) in variants.items():
        diffs = paired_bootstrap_diffs_from_proba(yconfirm, p_base_confirm, p_cand_confirm, N_BOOT, rng_boot)
        ci_lo, ci_hi, decision = decide(diffs)
        gt_delta_rep = gt_f1(yref, p_cand_ref) - gt_f1(yref, p_base_ref)
        gt_in_ci = ci_lo <= gt_delta_rep <= ci_hi
        if decision == "accept":
            decision_correct = gt_delta_rep > DELTA_MIN
        elif decision == "reject":
            decision_correct = gt_delta_rep < DELTA_MIN
        else:
            decision_correct = None
        row = {
            "scenario": scenario, "n_train": n_train, "seed": seed, "arm": arm, "status": "ok",
            "n_bags_effective": n_bags_eff, "is_noop": is_noop,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "decision": decision,
            "gt_delta_rep": gt_delta_rep, "gt_in_ci": gt_in_ci, "decision_correct_vs_gt": decision_correct,
        }
        writer.writerow(row)
        rows.append(row)
    return rows


def print_report(df):
    print("\n########## RINGKASAN PER ARM ##########")
    for (scenario, n_train, arm), g in df.groupby(["scenario", "n_train", "arm"]):
        n = len(g)
        n_noop = int(g.is_noop.sum())
        acc = g[g.decision == "accept"]
        n_acc = len(acc)
        n_wrong = int((acc.decision_correct_vs_gt == False).sum()) if n_acc else 0
        risk_lo, risk_hi = clopper_pearson(n_wrong, n_acc, alpha=0.10) if n_acc else (float("nan"), float("nan"))
        mean_gt_pos = g[g.gt_delta_rep > 0].shape[0] / n if n else float("nan")
        print(f"  {scenario:9s} N={n_train:4d} {arm:24s}: n={n} no_op={n_noop} | "
              f"accept={n_acc} ({100*n_acc/n:.1f}%) salah={n_wrong} "
              f"risiko={100*n_wrong/n_acc if n_acc else float('nan'):.1f}%[{100*risk_lo:.1f},{100*risk_hi:.1f}] | "
              f"mean_gt_delta={g.gt_delta_rep.mean():+.5f} frac_gt_positif={100*mean_gt_pos:.0f}%")


def main():
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"exp15_{stamp}.csv"

    all_rows = []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for scenario, cfg in SCENARIOS.items():
            print(f"Membuat referensi besar utk {scenario}...")
            Xref, yref = make_reference(cfg)
            reps = REPS_BY_SCENARIO[scenario]
            for n_train in N_TRAIN_LIST:
                for seed in range(reps):
                    rows = run_trial(scenario, cfg, n_train, seed, Xref, yref, writer)
                    f.flush()
                    all_rows.extend(rows)
                print(f"[progress] {scenario} N={n_train} selesai ({reps} repetisi)")

    print(f"\nLog lengkap: {out_path}")
    df = pd.DataFrame(all_rows)
    if len(df):
        print_report(df)


if __name__ == "__main__":
    main()

"""Eksperimen 16: validasi konfirmatori Arm E (bagged_candidate vs
bagged_baseline). Desain dibekukan dari exp15 (B=32, subsample=0.80,
SEARCH/CONFIRM=70/30, threshold=0.5, aturan accept=CI one-sided 95%,
delta_min=0). Semua hasil exp15 diperlakukan sebagai DEV; di sini dipakai
seed validasi baru yang tidak beririsan dengan seed DEV.

Perubahan dari exp15, hanya untuk efisiensi komputasi dan tanpa mengubah
mekanisme atau aturan keputusan:
- Hanya Arm A (kontrol lama) dan Arm E (target utama, fair) yang dihitung
  CI-nya via bootstrap. Arm B/C/D exp15 sudah menjawab pertanyaan
  diagnostiknya (confound bagging), sehingga tidak diulang di sini.
- Arm E_accept_all (baru): varian Arm E tanpa gate CI, selalu "accept" apa
  pun hasil bootstrap. Varian ini memisahkan manfaat konstruksi kandidat
  (bagging) dari gate (CI), atau keduanya. Tidak butuh bootstrap karena
  gt_delta_rep sudah tersedia dari komputasi Arm E.
- N_BOOT diturunkan dari 1000 ke 500. Profiling menunjukkan bootstrap base
  cost dominan, sekitar 0.92s/1000rep menjadi sekitar 0.46s/500rep per arm.
  Resolusi persentil ke-5 dari 500 sampel masih cukup, sekitar 25 nilai
  terkecil. Yang dikorbankan hanya presisi estimasi CI, bukan aturan
  keputusan.
- Ketidakpastian referensi dicek eksplisit, tidak diasumsikan presisi.
  Nilainya dihitung sekali per skenario via bootstrap terhadap prediksi satu
  pasangan model representatif pada Xref, lalu dilaporkan sebagai deadband.
  Kasus dengan |gt_delta_rep| di bawah deadband ditandai ambigu, bukan
  otomatis dianggap benar.
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
from exp15_stabilized_candidate import (
    fit_proba, B_BAGS, SUBSAMPLE_FRAC, CONFIRM_FRAC, DELTA_MIN, MODEL_NAME, ALPHA,
)

N_BOOT = 500  # turun dari 1000 (exp15), hanya presisi estimasi CI, bukan aturan keputusan
VAL_SEED_OFFSET = 5_000_000  # tidak beririsan dengan seed DEV (exp15 pakai 0..159)
REF_UNCERTAINTY_BOOT = 300

FIELDS = ["scenario", "n_train", "seed", "arm", "status", "n_bags_effective",
          "ci_lo", "ci_hi", "decision", "gt_delta_rep", "gt_in_ci", "decision_correct_vs_gt",
          "gt_ambiguous_ref_noise"]


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


def estimate_reference_noise(cfg, Xref, yref, seed_rep=777001):
    """Hitung sekali per skenario: fit satu pasangan model representatif pada
    900 baris, yaitu skala N_TRAIN gabungan dan rentang yang diuji, lalu
    bootstrap prediksinya pada Xref. Hasilnya SE delta akibat finite N_REF
    saja, bukan akibat variasi fit SEARCH kecil. Nilai ini dipakai sebagai
    deadband untuk menandai gt_delta_rep yang ambigu, karena delta dekat nol
    tidak otomatis dihitung benar.
    """
    baseline_cols, candidate_cols = scenario_feature_sets(
        sample_dgp(10, cfg["beta_switch"], cfg["beta_extra"], seed=1)[2], cfg)
    Xrep, yrep, _ = sample_dgp(900, cfg["beta_switch"], cfg["beta_extra"], seed=seed_rep)
    p_base_fn = fit_proba(Xrep, yrep, baseline_cols)
    p_cand_fn = fit_proba(Xrep, yrep, candidate_cols)
    p_base_ref = p_base_fn(Xref)
    p_cand_ref = p_cand_fn(Xref)
    rng = np.random.RandomState(seed_rep)
    diffs = paired_bootstrap_diffs_from_proba(yref, p_base_ref, p_cand_ref, REF_UNCERTAINTY_BOOT, rng)
    se = float(np.std(diffs))
    print(f"  [ketidakpastian referensi] SE(delta) akibat N_REF saja ~= {se:.5f} "
          f"(dipakai sbg deadband ambigu)")
    return se


def run_trial(scenario, cfg, n_train, seed, Xref, yref, ref_se, writer):
    Xtr_full, ytr_full, cols = sample_dgp(n_train, cfg["beta_switch"], cfg["beta_extra"],
                                           seed=(seed * 131071 + n_train) % (2**32 - 1))
    baseline_cols, candidate_cols = scenario_feature_sets(cols, cfg)

    mc, _ = data.minority_info(ytr_full)
    if mc < C.MIN_MINORITY_OUTER_TRAIN:
        for arm in ("A_single", "E_fair", "E_accept_all"):
            writer.writerow({"scenario": scenario, "n_train": n_train, "seed": seed,
                              "arm": arm, "status": "skip_minority"})
        return []

    Xsearch, Xconfirm, ysearch, yconfirm = train_test_split(
        Xtr_full, ytr_full, test_size=CONFIRM_FRAC, stratify=ytr_full, random_state=seed)
    Xsearch = Xsearch.reset_index(drop=True); ysearch = ysearch.reset_index(drop=True)
    Xconfirm = Xconfirm.reset_index(drop=True); yconfirm = yconfirm.reset_index(drop=True)

    p0_fn = fit_proba(Xsearch, ysearch, baseline_cols)
    p0_confirm = p0_fn(Xconfirm); p0_ref = p0_fn(Xref)
    pcand_single_fn = fit_proba(Xsearch, ysearch, candidate_cols)
    pcand_single_confirm = pcand_single_fn(Xconfirm); pcand_single_ref = pcand_single_fn(Xref)

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
        for arm in ("A_single", "E_fair", "E_accept_all"):
            writer.writerow({"scenario": scenario, "n_train": n_train, "seed": seed,
                              "arm": arm, "status": "skip_bags_insufficient", "n_bags_effective": n_bags_eff})
        return []

    def bag_matrix(fns, Xeval):
        return np.column_stack([fn(Xeval) for fn in fns])

    pcand_bag_confirm = bag_matrix(candidate_b_fns, Xconfirm).mean(axis=1)
    pcand_bag_ref = bag_matrix(candidate_b_fns, Xref).mean(axis=1)
    pbase_bag_confirm = bag_matrix(baseline_b_fns, Xconfirm).mean(axis=1)
    pbase_bag_ref = bag_matrix(baseline_b_fns, Xref).mean(axis=1)

    rows = []
    rng_boot = np.random.RandomState((seed * 104729 + n_train) % (2**32 - 1))

    # Arm A: kandidat lama vs p0 tunggal (kontrol)
    diffs_a = paired_bootstrap_diffs_from_proba(yconfirm, p0_confirm, pcand_single_confirm, N_BOOT, rng_boot)
    ci_lo_a, ci_hi_a, dec_a = decide(diffs_a)
    gt_a = gt_f1(yref, pcand_single_ref) - gt_f1(yref, p0_ref)
    rows.append(("A_single", ci_lo_a, ci_hi_a, dec_a, gt_a))

    # Arm E: bagged_candidate vs bagged_baseline (fair, target utama)
    diffs_e = paired_bootstrap_diffs_from_proba(yconfirm, pbase_bag_confirm, pcand_bag_confirm, N_BOOT, rng_boot)
    ci_lo_e, ci_hi_e, dec_e = decide(diffs_e)
    gt_e = gt_f1(yref, pcand_bag_ref) - gt_f1(yref, pbase_bag_ref)
    rows.append(("E_fair", ci_lo_e, ci_hi_e, dec_e, gt_e))

    # Arm E_accept_all: Arm E tanpa gate, selalu accept, untuk diagnostik
    rows.append(("E_accept_all", ci_lo_e, ci_hi_e, "accept", gt_e))

    out_rows = []
    for arm, ci_lo, ci_hi, decision, gt_delta_rep in rows:
        gt_in_ci = ci_lo <= gt_delta_rep <= ci_hi
        gt_ambiguous = abs(gt_delta_rep) < ref_se
        if decision == "accept":
            decision_correct = (gt_delta_rep > DELTA_MIN) if not gt_ambiguous else None
        elif decision == "reject":
            decision_correct = (gt_delta_rep < DELTA_MIN) if not gt_ambiguous else None
        else:
            decision_correct = None
        row = {
            "scenario": scenario, "n_train": n_train, "seed": seed, "arm": arm, "status": "ok",
            "n_bags_effective": n_bags_eff, "ci_lo": ci_lo, "ci_hi": ci_hi, "decision": decision,
            "gt_delta_rep": gt_delta_rep, "gt_in_ci": gt_in_ci,
            "decision_correct_vs_gt": decision_correct, "gt_ambiguous_ref_noise": gt_ambiguous,
        }
        writer.writerow(row)
        out_rows.append(row)
    return out_rows


def print_report(df):
    print("\n########## VALIDASI KONFIRMATORI -- RINGKASAN PER ARM ##########")
    for (scenario, n_train, arm), g in df.groupby(["scenario", "n_train", "arm"]):
        n = len(g)
        n_ambig = int(g.gt_ambiguous_ref_noise.sum())
        acc = g[g.decision == "accept"]
        n_acc = len(acc)
        acc_nonambig = acc[~acc.gt_ambiguous_ref_noise]
        n_wrong = int((acc_nonambig.decision_correct_vs_gt == False).sum())
        n_eval = len(acc_nonambig)
        risk_lo, risk_hi = clopper_pearson(n_wrong, n_eval, alpha=0.10) if n_eval else (float("nan"), float("nan"))
        true_accepts_per_100 = 100 * (n_eval - n_wrong) / n if n else float("nan")
        frac_pos = 100 * (g.gt_delta_rep > 0).mean() if n else float("nan")
        print(f"  {scenario:9s} N={n_train:4d} {arm:14s}: n={n} (ambigu={n_ambig}) | "
              f"accept={n_acc} ({100*n_acc/n:.1f}%) | di antara accept non-ambigu: n={n_eval} salah={n_wrong} "
              f"risiko_1sisi95%_upper={100*risk_hi:.1f}% | "
              f"frac_gt_positif={frac_pos:.0f}% | penerimaan_benar_per_100_percobaan={true_accepts_per_100:.1f}")


def main():
    # Eksekusi paralel: python exp16_confirmatory_arm_e.py <worker_id> <n_workers>
    # Tiap worker mengerjakan seed lokal dengan (seed_lokal % n_workers == worker_id).
    # Partisi genap dan independen antar worker. Xref dibangun ulang secara
    # deterministik di tiap worker, tidak dibagi via file atau IPC.
    worker_id, n_workers = 0, 1
    if len(sys.argv) >= 3:
        worker_id, n_workers = int(sys.argv[1]), int(sys.argv[2])

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_w{worker_id}" if n_workers > 1 else ""
    out_path = out_dir / f"exp16_{stamp}{suffix}.csv"

    import json
    reps_path = Path(__file__).resolve().parent / "exp16_reps_budget.json"
    reps_budget = json.loads(reps_path.read_text()) if reps_path.exists() else None
    if reps_budget is None:
        raise SystemExit("exp16_reps_budget.json tidak ditemukan -- tentukan REPS dulu sesuai budget waktu.")

    all_rows = []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for scenario, cfg in SCENARIOS.items():
            print(f"[worker {worker_id}] Membuat referensi besar utk {scenario}...")
            Xref, yref = make_reference(cfg)
            ref_se = estimate_reference_noise(cfg, Xref, yref)
            for n_train in (600, 900):
                reps = reps_budget[scenario][str(n_train)]
                if reps == 0:
                    continue
                my_seeds = [s for s in range(reps) if s % n_workers == worker_id]
                t_start = time.time()
                for seed in my_seeds:
                    rows = run_trial(scenario, cfg, n_train, VAL_SEED_OFFSET + seed, Xref, yref, ref_se, writer)
                    f.flush()
                    all_rows.extend(rows)
                print(f"[worker {worker_id}][progress] {scenario} N={n_train} selesai "
                      f"({len(my_seeds)} dari {reps} repetisi jatah worker ini, {time.time()-t_start:.0f}s)")

    print(f"\n[worker {worker_id}] Log: {out_path}")
    df = pd.DataFrame(all_rows)
    if len(df) and n_workers == 1:
        print_report(df)


if __name__ == "__main__":
    main()

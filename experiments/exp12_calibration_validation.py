"""Eksperimen 12: validasi kalibrasi SEARCH/CONFIRM, bukan pencarian mekanisme baru.

Versi sebelumnya menghitung ground truth dengan satu pasangan model
representatif pada sampel 20_000 lalu memakai angka itu untuk semua repetisi.
Target CI CONFIRM adalah delta performa pasangan model hasil SEARCH repetisi
itu sendiri, yang dapat berbeda tanda antar repetisi karena SEARCH hanya
memakai sampel kecil. Bila ground truth kebetulan positif, semua accept
tercatat benar. Sekarang referensi besar (Xref, yref) dibuat sekali per
skenario dan dibagi bersama untuk menghemat komputasi, tetapi prediksi dan
delta ground truth dihitung per repetisi dari objek model yang sama dengan
yang dipakai CONFIRM, bukan model terpisah.

Desain: kandidat tetap (pre-registered, bukan greedy adaptif), 3 skenario
delta populasi (negative/nearzero/positive), keputusan 3 arah
accept/reject/abstain lewat paired bootstrap CI pada CONFIRM. Cakupan CI dan
risiko salah dicatat dengan interval Clopper-Pearson exact, bukan titik
estimasi.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import config as C
import data
import models
from exp10_search_confirm import paired_bootstrap_ci

SCENARIOS = {
    "negative": dict(beta_switch=2.0, beta_extra=0.0, drop_switch=True),
    "nearzero": dict(beta_switch=0.0, beta_extra=0.0, drop_switch=True),
    "positive": dict(beta_switch=0.0, beta_extra=0.3, drop_switch=False),
}
N_TRAIN_LIST = (600, 900, 3000)
REPS_BY_SCENARIO = {"negative": 100, "nearzero": 500, "positive": 500}
CONFIRM_FRAC = 0.3
N_BOOT = 500
ALPHA = 0.10  # CI 90%
DELTA_MIN = 0.0
MODEL_NAME = "HGB"
N_REF = 30_000  # Referensi besar, dibagi bersama per skenario dan tidak dilatih ulang

FIELDS = ["scenario", "n_train", "seed", "status", "n_search", "n_confirm",
          "ci_lo", "ci_hi", "decision",
          "gt_delta_rep", "gt_in_ci", "decision_correct_vs_gt",
          "n_features_baseline", "n_features_candidate"]


def sample_dgp(n, beta_switch, beta_extra, seed):
    rng = np.random.RandomState(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x_switch = rng.normal(size=n)
    x_extra = rng.normal(size=n)
    noise = rng.normal(size=(n, 10))
    logits = 1.5 * x1 + 1.5 * x2 + beta_switch * x_switch + beta_extra * x_extra
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)
    cols = ["x1", "x2", "x_switch", "x_extra"] + [f"noise_{i}" for i in range(10)]
    X = pd.DataFrame(np.column_stack([x1, x2, x_switch, x_extra, noise]), columns=cols)
    yser = pd.Series(y, name="target")
    return X, yser, cols


def scenario_feature_sets(cols, cfg):
    if cfg["drop_switch"]:
        baseline_cols = [c for c in cols if c != "x_extra"]
        candidate_cols = [c for c in baseline_cols if c != "x_switch"]
    else:
        baseline_cols = [c for c in cols if c not in ("x_switch", "x_extra")]
        candidate_cols = baseline_cols + ["x_extra"]
    return baseline_cols, candidate_cols


def make_reference(cfg, seed_eval=999002):
    """Buat referensi besar sekali per skenario agar dapat dibagi antar repetisi.
    Referensi tidak dipakai untuk fit model, hanya untuk diprediksi model hasil SEARCH tiap repetisi."""
    Xref, yref, cols = sample_dgp(N_REF, cfg["beta_switch"], cfg["beta_extra"], seed_eval)
    return Xref, yref


def fit_and_predict(Xsearch, ysearch, Xconfirm, Xref, feat_cols):
    """Fit satu model di SEARCH, lalu prediksi di CONFIRM (untuk CI) dan di referensi besar
    (untuk ground truth repetisi ini). Model yang dipakai sama, bukan model terpisah."""
    num_c, cat_c = data.column_kinds(Xsearch[feat_cols])
    pre = data.make_preprocessor(num_c, cat_c)
    Zs = pre.fit_transform(Xsearch[feat_cols], ysearch)
    est = models.build_estimator(MODEL_NAME, {}, len(ysearch), final=False)
    est.fit(Zs, ysearch)
    pred_confirm = est.predict(pre.transform(Xconfirm[feat_cols]))
    pred_ref = est.predict(pre.transform(Xref[feat_cols]))
    return pred_confirm, pred_ref


def run_trial(scenario, cfg, n_train, seed, Xref, yref, writer):
    Xtr_full, ytr_full, cols = sample_dgp(n_train, cfg["beta_switch"], cfg["beta_extra"],
                                           seed=seed * 131071 + n_train)
    baseline_cols, candidate_cols = scenario_feature_sets(cols, cfg)

    mc, _ = data.minority_info(ytr_full)
    if mc < C.MIN_MINORITY_OUTER_TRAIN:
        writer.writerow({"scenario": scenario, "n_train": n_train, "seed": seed, "status": "skip_minority"})
        return None

    Xsearch, Xconfirm, ysearch, yconfirm = train_test_split(
        Xtr_full, ytr_full, test_size=CONFIRM_FRAC, stratify=ytr_full, random_state=seed)
    Xsearch = Xsearch.reset_index(drop=True); ysearch = ysearch.reset_index(drop=True)
    Xconfirm = Xconfirm.reset_index(drop=True); yconfirm = yconfirm.reset_index(drop=True)

    pred_base_confirm, pred_base_ref = fit_and_predict(Xsearch, ysearch, Xconfirm, Xref, baseline_cols)
    pred_cand_confirm, pred_cand_ref = fit_and_predict(Xsearch, ysearch, Xconfirm, Xref, candidate_cols)

    rng = np.random.RandomState(seed * 104729 + n_train)
    ci_lo, ci_hi = paired_bootstrap_ci(yconfirm, pred_base_confirm, pred_cand_confirm, N_BOOT, ALPHA, rng)

    if ci_lo > DELTA_MIN:
        decision = "accept"
    elif ci_hi < 0:
        decision = "reject"
    else:
        decision = "abstain"

    # Ground truth khusus repetisi ini: pasangan model yang sama dengan CI,
    # dievaluasi di referensi besar yang sama, bukan dilatih ulang
    f1_base_ref = f1_score(yref, pred_base_ref, average="weighted", zero_division=0)
    f1_cand_ref = f1_score(yref, pred_cand_ref, average="weighted", zero_division=0)
    gt_delta_rep = f1_cand_ref - f1_base_ref

    gt_in_ci = ci_lo <= gt_delta_rep <= ci_hi
    if decision == "accept":
        decision_correct = gt_delta_rep > DELTA_MIN
    elif decision == "reject":
        decision_correct = gt_delta_rep < DELTA_MIN
    else:
        decision_correct = None

    row = {
        "scenario": scenario, "n_train": n_train, "seed": seed, "status": "ok",
        "n_search": len(ysearch), "n_confirm": len(yconfirm),
        "ci_lo": ci_lo, "ci_hi": ci_hi, "decision": decision,
        "gt_delta_rep": gt_delta_rep, "gt_in_ci": gt_in_ci,
        "decision_correct_vs_gt": decision_correct,
        "n_features_baseline": len(baseline_cols), "n_features_candidate": len(candidate_cols),
    }
    writer.writerow(row)
    return row


def clopper_pearson(n_success, n_total, alpha=0.10):
    if n_total == 0:
        return float("nan"), float("nan")
    lo = stats.beta.ppf(alpha / 2, n_success, n_total - n_success + 1) if n_success > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, n_success + 1, n_total - n_success) if n_success < n_total else 1.0
    return float(lo), float(hi)


def print_summary(rows):
    df = pd.DataFrame(rows)
    print("\n=== Distribusi gt_delta_rep per skenario (variasi akibat SEARCH sampel kecil) ===")
    for scenario, g in df.groupby("scenario"):
        print(f"  {scenario:9s}: mean={g.gt_delta_rep.mean():+.5f} std={g.gt_delta_rep.std():.5f} "
              f"min={g.gt_delta_rep.min():+.5f} max={g.gt_delta_rep.max():+.5f} "
              f"(n_positif={100*(g.gt_delta_rep>0).mean():.0f}%)")

    print("\n=== Cakupan CI (ground truth REPETISI INI ada di dalam CI?) target nominal 90% ===")
    for (scenario, n_train), g in df.groupby(["scenario", "n_train"]):
        n = len(g)
        n_cov = g.gt_in_ci.sum()
        lo, hi = clopper_pearson(n_cov, n, alpha=0.10)
        print(f"  {scenario:9s} N_TRAIN={n_train:5d}: cakupan={n_cov}/{n} ({100*n_cov/n:.1f}%)  "
              f"CI90%(Clopper-Pearson)=[{100*lo:.1f}%,{100*hi:.1f}%]")

    print("\n=== Risiko salah di antara ACCEPT (interval Clopper-Pearson presisi, bukan dibulatkan) ===")
    for (scenario, n_train), g in df.groupby(["scenario", "n_train"]):
        acc = g[g.decision == "accept"]
        n_acc = len(acc)
        n_wrong = int((acc.decision_correct_vs_gt == False).sum()) if n_acc else 0
        lo, hi = clopper_pearson(n_wrong, n_acc, alpha=0.10) if n_acc else (float("nan"), float("nan"))
        rej = g[g.decision == "reject"]
        abst = g[g.decision == "abstain"]
        print(f"  {scenario:9s} N_TRAIN={n_train:5d}: accept={n_acc}/{len(g)} salah={n_wrong} "
              f"risiko_titik={100*n_wrong/n_acc if n_acc else float('nan'):.1f}% "
              f"CI90%(Clopper-Pearson)=[{100*lo:.2f}%,{100*hi:.2f}%] | "
              f"reject={len(rej)} abstain={len(abst)}")


def main():
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"exp12fix_{stamp}.csv"

    rows = []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for scenario, cfg in SCENARIOS.items():
            print(f"Membuat referensi besar utk {scenario} (N={N_REF}, dibagi bersama antar repetisi)...")
            Xref, yref = make_reference(cfg)
            reps = REPS_BY_SCENARIO[scenario]
            for n_train in N_TRAIN_LIST:
                for seed in range(reps):
                    row = run_trial(scenario, cfg, n_train, seed, Xref, yref, writer)
                    f.flush()
                    if row:
                        rows.append(row)
                print(f"[progress] {scenario} N_TRAIN={n_train} selesai ({reps} repetisi)")

    print(f"\nLog lengkap: {out_path}")
    if rows:
        print_summary(rows)


if __name__ == "__main__":
    main()

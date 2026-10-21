"""Mini-FDE frozen config. SEMUA angka warisan — DILARANG diubah/di-tune ulang.

Spec beku: arsitektur tahap max-rule, M1 dua lapis, Optuna terkendala,
protokol data blind, logging wajib. Perubahan apapun = pelanggaran warisan.
"""
from __future__ import annotations

# ---- AMENDEMEN METRIK 2026-10-05 (disetujui user, EKSPERIMENTAL): opsi ganti
# metrik keputusan dari weighted-F1 ke balanced-accuracy. Alasan: weighted-F1
# condong ke kelas mayoritas (pc4 88%/12%) -- diduga ini sebab M1/P3/P4 tak
# pernah menang di dataset timpang. balanced-accuracy menilai tiap kelas setara.
# Ganti SCORING_METRIC di sini -> SEMUA tahap (P1/M1/P3/P4/verdict) ikut lewat
# score_metric()/pi_scoring(), TANPA perlu ubah run.py/models.py lagi.
SCORING_METRIC = "weighted_f1"  # "weighted_f1" (default/beku) | "balanced_accuracy"


def score_metric(y_true, y_pred) -> float:
    from sklearn.metrics import balanced_accuracy_score, f1_score
    if SCORING_METRIC == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def pi_scoring() -> str:
    """Nama scorer sklearn utk permutation_importance (models.hgb_ranking),
    konsisten dgn SCORING_METRIC."""
    return "balanced_accuracy" if SCORING_METRIC == "balanced_accuracy" else "f1_weighted"


# ---- Protokol data (D) ----
SEED = 42
STABILITY_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # dilaporkan untuk stabilitas (OOF-TRAIN saja, tanpa sentuh TEST)
OUTER_TEST_SIZE = 0.20
INNER_CV_SPLITS = 5                # OOF-5 DALAM TRAIN
MIN_MINORITY_OUTER_TRAIN = 10      # syarat jalan: minority outer-TRAIN >= 10

# ---- Gate (A) ----
WIN_EPS = 1e-12                    # "menang" = weighted-F1 OOF-TRAIN strictly greater + eps

# ---- Optuna terkendala (C) ----
T1 = 20                            # budget per estimator di P1
T2B = 20                           # budget retune B2 di atas masker terkunci
T3 = 300                           # budget total P3
OPTUNA_SEED = 42
EARLY_STOP_VALUE = 1.0             # early-stop study bila best weighted-F1 mencapai 1.0


def t2a_budget(n_features: int) -> int:
    """Budget evaluasi B1 (lapis masker). B1 greedy butuh <= 2F evaluasi; cap warisan."""
    return min(max(50, 30 * n_features), 300)


def p3_per_member_budget() -> int:
    """Budget tuning per anggota P3. Dipercepat: max(20, T3//5)."""
    return max(20, T3 // 5)


def svc_max_iter(n_samples: int) -> int:
    """SVC max_iter: 500 bila N>1000 else 2000. cache selalu 1000."""
    return 500 if n_samples > 1000 else 2000


SVC_CACHE_SIZE = 1000


def p3_allow_svc(n_samples: int, n_classes: int) -> bool:
    """P3 keluarkan SVC bila N>1000 atau C>5."""
    return not (n_samples > 1000 or n_classes > 5)


def pruner_name(n_samples: int, minority_ratio: float) -> str:
    """NopPruner bila N<500 ATAU rasio-minoritas<0.15 else MedianPruner (lebih cepat)."""
    if n_samples < 500 or minority_ratio < 0.15:
        return "nop"
    return "median"


# ---- Pool tahap (A) ----
P1_POOL = ("HGB",)                       # P1 hanya HGB (hemat waktu)
P3_POOL = ("SVC", "DT", "KNN", "HGB")    # pool anggota P3
P4_CORRECTOR = "HGB"                     # korektor greedy/residual di atas output incumbent

# ---- AMENDEMEN P3-robust 2026-10-04 (disetujui user, mekanisme
#      FEDSEL_multiplatform run_phase3; P1/M1/P4/gate/blind tetap beku) ----
# P3 lama (2 anggota global, vote hard-label rata-rata) diganti ensemble per-fold:
# tiap fold punya study Optuna sendiri (T_per_fold = p3_per_member_budget())
# yang mengusulkan M2(tipe+mask-boolean+param) + M3(...) + bobot w1/w2/w3
# (w1 in [0.1,1.0], w2/w3 in [0.0,1.0]) demi F1 GABUNGAN proba-berbobot di vl
# fold itu. Masker bebas (boleh sama/komplemen/overlap, TPE yang menentukan).
# Model per-fold memakai predict_proba (SVC probability=True) saat search;
# P1/M1 tetap probability=False saat search (Platt final saja).

# ---- AMENDEMEN SEL/BLIND + BRIDGE 2026-10-04 (disetujui user, mekanisme
#      FEDSEL_multiplatform run_fde_bridge_cascade) ----
# Search P3 (di atas) menilai di vl yang sama -> optimistis dalam-fold.
# Gate jujur: R rotasi SEL(4/5: search+fit) / BLIND(1/5: nilai saja, TRAIN-only).
# Study per rotasi (T_per_fold trial, objektif inner-CV-3 dalam SEL), refit
# terbaik di full-SEL, nilai gabungan di BLIND. Terima iff
# mean(diff) > SE(diff) + WIN_EPS (paired 1-SE, TRAIN-only, tanpa TEST).
# FIX 2026-10-05: rotasi=1 bikin SE(diff) = std(ddof=1) atas 1 angka = NaN
# (RuntimeWarning "Degrees of freedom <= 0") -> _gain > NaN selalu False ->
# gate P3 buntu PERMANEN (tak pernah bisa diterima apapun hasilnya). Minimal
# 2 rotasi utk SE terdefinisi; 3 dipakai sbg kompromi minimal-tambahan-biaya.
# FIX audit 2026-10-05 v5 (disetujui user): 3 rotasi TERBUKTI kesedikitan --
# gain/se goyang, 1xSE sempat meloloskan false-positif nyata (P3 pc4 kalah di
# SEMUA metrik TEST padahal lolos gain>se). Akar masalah = jumlah sampel, bukan
# ambang. Naikkan ke 15 rotasi (bukan ubah ambang lagi) biar gain/se dihitung
# dari lebih banyak titik data, estimasi lebih stabil dipercaya.
P3_BLIND_ROTATIONS = 15
P3_BLIND_SEED_OFFSET = 1000

# ---- AMENDEMEN M1-BRIDGE 2026-10-05 (disetujui user) ----
# Guard M1 lama (margin + stabilitas-lintas-10-seed) terbukti TIDAK CUKUP: semua
# re-split di situ masih dari TRAIN yang sama (cuma reshuffle fold), tak pernah
# ada data benar2 "asing". Bukti nyata: diabetes_hgb_seed42 -> M1 accept (lolos
# margin+stabilitas) tapi TEST malah turun -2.14pp; iris -> M1 accept, TEST
# identik/tak naik sama sekali. M1 kini dipakaikan BRIDGE SEL/BLIND spt P3:
# cari mask(B1)+param(B2) di SEL(4/5) pakai inner-CV, nilai P1 vs M1-kandidat
# di BLIND(1/5) yg tak pernah dilihat proses seleksi. Offset seed beda dari
# P3 (1000) biar rotasi SEL/BLIND-nya tidak sama persis.
M1_BLIND_ROTATIONS = 15  # FIX v5: naikkan dari 3, lihat alasan di P3_BLIND_ROTATIONS
M1_BLIND_SEED_OFFSET = 2000

# ---- FIX audit 2026-10-05 (disetujui user): B1 (pilih mask) lama SELALU pakai
# hyperparameter P1 yang dibekukan (hasil tuning utk fitur PENUH) utk menilai
# SEMUA kandidat mask -- padahal hyperparameter optimal bisa beda per ukuran
# subset fitur, dan B2 (retune) baru jalan SETELAH mask dikunci, tanpa iterasi
# balik. Sekarang B1<->B2 diulang M1_COORD_ROUNDS kali (B1 pakai param ronde
# sebelumnya -> B2 retune mask baru -> ulang), berhenti lebih awal kalau mask
# sudah tidak berubah (konvergen). 2 ronde = kompromi (cukup utk koreksi 1x,
# tanpa biaya meledak).
M1_COORD_ROUNDS = 2

# ---- FIX audit 2026-10-05 v2 (disetujui user): SEMUA BRIDGE (M1/P3/P4) fit
# model di SEL (dulu 4/5=80% TRAIN), padahal yang beneran dipakai saat deploy
# di-fit ke TRAIN PENUH (100%) -- utk model fragile (lihat temuan pc4: SVC+poly
# di data timpang, BRIDGE blind ~0.88 tapi TEST asli jatuh ke 0.36), beda
# ukuran fit ini bisa menyembunyikan ketidakstabilan. SEL_BLIND_SPLITS dinaikkan
# dari 5 (SEL=80%) ke 10 (SEL=90%) -- ukuran fit BRIDGE lebih dekat ke deployment,
# trade-off: BLIND jadi lebih kecil/lebih berisik per rotasi.
SEL_BLIND_SPLITS = 10

# ---- AMENDEMEN P4-BRIDGE 2026-10-05 (disetujui user, pola sama dgn M1) ----
# P4 BRIDGE cuma didukung utk incumbent single-model (P1/M1_sel) -- kasus plg
# umum krn P3 hampir selalu direvert oleh BRIDGE-nya sendiri. Utk incumbent
# kind=p3_robust (refit-ulang-di-SEL-arbitrer belum ada abstraksinya), run.py
# fallback ke guard lama (margin+stabilitas) khusus kasus itu, ditandai jelas
# di decision_p4.json -> guard.bridge_supported=False.
P4_BLIND_ROTATIONS = 15  # FIX v5: naikkan dari 3, lihat alasan di P3_BLIND_ROTATIONS
P4_BLIND_SEED_OFFSET = 3000
P3_INNER_FOLDS = 3

# ---- AMENDEMEN GUARD 2026-10-04 (disetujui user; diagnosis murni TRAIN:
#      gate eps-murni menerima kemenangan mikro noise, mis. skulls +0.001) ----
# Semua guard TRAIN-only (tanpa TEST). Nilai default konservatif-prinsip
# (BUKAN disetel dari angka TEST): margin 0.2pp membunuh debu; 2-SE standar;
# konfirmasi stabilitas memakai STABILITY_SEEDS yang memang sudah dihitung.
# CATAT: karena ambang ini dimotivasi hasil TEST 11 set klaim, hasil guard
# pada 11 set itu = pilot/replay, BUKAN bukti klaim; bukti butuh data segar.
GATE_M1_MARGIN = 0.001    # M1 accept iff win_main > margin
# FIX 2026-10-05: floor lama (1.0) = nonaktifkan P4 PERMANEN apapun hasilnya --
# itu cuma tambalan krn guard lama (margin+stabilitas) gampang ketipu overfitting.
# Sekarang P4 dipakaikan BRIDGE SEL/BLIND spt M1/P3 (lihat run.py), jadi floor
# diturunkan ke level sama dgn M1/P3 -- proteksi overfitting pindah ke BRIDGE,
# bukan lagi ke floor buta.
GATE_P4_MARGIN = 0.001    # P4 accept iff gain > floor (BRIDGE SEL/BLIND)
GATE_P3_MARGIN = 0.001    # P3 accept iff gain > margin (selain SE)
# FIX audit 2026-10-05 v4 (disetujui user, DICOBA & DIBATALKAN): sumber asli
# (FEDSEL_multiplatform/fde_bridge_cascade.py) pakai gain>se+EPS (1xSE). Dicoba
# di pc4+XGB -- P3 LOLOS (gain=0.0184>se=0.0119) TAPI TEST ASLI KALAH DI SEMUA
# METRIK (wF1 0.90 vs baseline 0.92, bal_acc 0.73 vs 0.78) -- FALSE POSITIVE
# nyata, terverifikasi. Akar masalah bukan di angka ambang, tapi di JUMLAH
# ROTASI BLIND (cuma 3) yg bikin gain/se sendiri goyang -- dikembalikan ke 2.0.
GATE_SE_MULT = 2.0        # BRIDGE: gain > SE_MULT * se + eps
GATE_REQ_STABILITY = True  # M1/P4: kandidat wajib menang mean across seeds

# ---- Dataset registry (F) ----
# Dev (BUKAN bukti): iris, wine (sklearn built-in).
# Uji klaim via fetch_openml(data_id=...). Syarat jalan: minority outer-TRAIN>=10 + numerik.
OPENML_IDS = {
    "diabetes": 37,
    "pc4": 1049,
}
DEV_DATASETS = ("iris", "wine")  # FIX audit 2026-10-05: "wine" kelupaan (README sebut dev=iris+wine)

# ---- Ruang search Optuna (beku, bagian dari paket C) ----
# Batas-batas di bawah adalah warisan implementasi; jangan diganti tanpa
# membatalkan klaim keterbandingan. TPE hanya untuk ruang kontinu-halus (param),
# TIDAK PERNAH untuk masker diskret 2^F (lihat M1/B1).
SEARCH_SPACES = {
    "SVC": {
        "C": ("logfloat", 1e-3, 1e3),
        "kernel": ("categorical", ["rbf", "linear", "poly"]),
        "gamma": ("categorical", ["scale", "auto"]),
        "degree": ("int", 2, 4),       # hanya dipakai bila kernel=poly
        "coef0": ("float", 0.0, 1.0),   # hanya dipakai bila kernel=poly
    },
    "KNN": {
        "n_neighbors": ("int", 1, 31),
        "weights": ("categorical", ["uniform", "distance"]),
        "p": ("categorical", [1, 2]),
    },
    "HGB": {
        "learning_rate": ("logfloat", 0.01, 0.3),
        "max_iter": ("int", 50, 300),
        "max_leaf_nodes": ("int", 3, 63),
        "min_samples_leaf": ("int", 1, 50),
        "l2_regularization": ("logfloat", 1e-8, 10.0),
    },
    "DT": {
        "max_depth": ("int", 2, 30),    # None ditangani via flag terpisah (none_prob=0.25)
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 20),
        "criterion": ("categorical", ["gini", "entropy"]),
    },
    "XGB": {
        "learning_rate": ("logfloat", 0.01, 0.3),
        "max_iter": ("int", 50, 300),
        "max_depth": ("int", 2, 10),
        "min_child_weight": ("float", 1.0, 10.0),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
        "reg_lambda": ("logfloat", 1e-3, 10.0),
    },
}

"""Konfigurasi beku Mini-FDE; semua angka warisan dan tidak boleh diubah atau di-tune ulang.

Spec beku: arsitektur tahap max-rule, M1 dua lapis, Optuna terkendala,
protokol data blind, logging wajib. Perubahan angka dianggap pelanggaran warisan.
"""
from __future__ import annotations

# Amendemen metrik 2026-10-05 (eksperimental): opsi ganti metrik keputusan
# dari weighted-F1 ke balanced-accuracy. Alasan: weighted-F1 condong ke kelas
# mayoritas (pc4 88%/12%), diduga sebab M1/P3/P4 tidak pernah menang di dataset
# timpang. balanced-accuracy menilai tiap kelas setara. Mengganti SCORING_METRIC
# di sini membuat P1/M1/P3/P4/verdict ikut lewat score_metric()/pi_scoring(),
# tanpa perlu mengubah run.py/models.py.
SCORING_METRIC = "weighted_f1"  # "weighted_f1" (default/beku) | "balanced_accuracy"


def score_metric(y_true, y_pred) -> float:
    from sklearn.metrics import balanced_accuracy_score, f1_score
    if SCORING_METRIC == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def pi_scoring() -> str:
    """Nama scorer sklearn untuk permutation_importance (models.hgb_ranking), konsisten dengan SCORING_METRIC."""
    return "balanced_accuracy" if SCORING_METRIC == "balanced_accuracy" else "f1_weighted"


# Protokol data (D)
SEED = 42
STABILITY_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # dilaporkan untuk stabilitas (OOF-TRAIN saja, tanpa sentuh TEST)
OUTER_TEST_SIZE = 0.20
INNER_CV_SPLITS = 5                # OOF-5 di dalam TRAIN
MIN_MINORITY_OUTER_TRAIN = 10      # syarat jalan: minority outer-TRAIN >= 10

# Gate (A)
WIN_EPS = 1e-12                    # "menang" = weighted-F1 OOF-TRAIN strictly greater + eps

# Optuna terkendala (C)
T1 = 20                            # budget per estimator di P1
T2B = 20                           # budget retune B2 di atas masker terkunci
T3 = 300                           # budget total P3
OPTUNA_SEED = 42
EARLY_STOP_VALUE = 1.0             # early-stop study bila best weighted-F1 mencapai 1.0


def t2a_budget(n_features: int) -> int:
    """Budget evaluasi B1 (lapis masker). B1 greedy butuh <= 2F evaluasi; cap warisan."""
    return min(max(50, 30 * n_features), 300)


def p3_per_member_budget() -> int:
    """Budget tuning per anggota P3, max(20, T3//5)."""
    return max(20, T3 // 5)


def svc_max_iter(n_samples: int) -> int:
    """SVC max_iter: 500 bila N>1000 else 2000. cache selalu 1000."""
    return 500 if n_samples > 1000 else 2000


SVC_CACHE_SIZE = 1000


def p3_allow_svc(n_samples: int, n_classes: int) -> bool:
    """P3 mengeluarkan SVC bila N>1000 atau C>5."""
    return not (n_samples > 1000 or n_classes > 5)


def pruner_name(n_samples: int, minority_ratio: float) -> str:
    """NopPruner bila N<500 atau rasio minoritas < 0.15, selain itu MedianPruner (lebih cepat)."""
    if n_samples < 500 or minority_ratio < 0.15:
        return "nop"
    return "median"


# Pool tahap (A)
P1_POOL = ("HGB",)                       # P1 hanya HGB (hemat waktu)
P3_POOL = ("SVC", "DT", "KNN", "HGB")    # pool anggota P3
P4_CORRECTOR = "HGB"                     # korektor greedy/residual di atas output incumbent

# Amendemen P3-robust 2026-10-04 (mekanisme FEDSEL_multiplatform run_phase3;
# P1/M1/P4/gate/blind tetap beku).
# P3 lama (2 anggota global, vote hard-label rata-rata) diganti ensemble per-fold:
# tiap fold punya study Optuna sendiri (T_per_fold = p3_per_member_budget())
# yang mengusulkan M2 (tipe + mask boolean + param) + M3 (...) + bobot w1/w2/w3
# (w1 in [0.1,1.0], w2/w3 in [0.0,1.0]) demi F1 gabungan proba berbobot di vl
# fold itu. Masker bebas (boleh sama, komplemen, atau overlap; TPE yang menentukan).
# Model per-fold memakai predict_proba (SVC probability=True) saat search;
# P1/M1 tetap probability=False saat search (Platt hanya di final).

# Amendemen SEL/BLIND dan BRIDGE 2026-10-04 (mekanisme
# FEDSEL_multiplatform run_fde_bridge_cascade).
# Search P3 di atas menilai di vl yang sama, sehingga optimistis dalam-fold.
# Gate jujur: R rotasi SEL (4/5: search+fit) / BLIND (1/5: nilai saja, TRAIN-only).
# Study per rotasi (T_per_fold trial, objektif inner-CV-3 dalam SEL), refit
# terbaik di full-SEL, nilai gabungan di BLIND. Terima bila
# mean(diff) > SE(diff) + WIN_EPS (paired 1-SE, TRAIN-only, tanpa TEST).
# 2026-10-05: rotasi=1 membuat SE(diff) = std(ddof=1) atas 1 angka = NaN
# (RuntimeWarning "Degrees of freedom <= 0"), _gain > NaN selalu False, gate P3
# buntu permanen. Minimal 2 rotasi agar SE terdefinisi; 3 dipakai sebagai
# kompromi biaya minimal.
# 2026-10-05 v5: 3 rotasi terlalu sedikit. gain/se bergoyang dan 1xSE sempat
# meloloskan false positive nyata (P3 pc4 kalah di semua metrik TEST padahal
# lolos gain>se). Akar masalah adalah jumlah sampel, bukan ambang. Rotasi
# dinaikkan ke 15 agar gain/se dihitung dari lebih banyak titik data.
P3_BLIND_ROTATIONS = 15
P3_BLIND_SEED_OFFSET = 1000

# Amendemen M1-BRIDGE 2026-10-05.
# Guard M1 lama (margin + stabilitas lintas 10 seed) tidak cukup: semua re-split
# masih dari TRAIN yang sama (hanya reshuffle fold), tidak pernah ada data
# benar-benar asing. Bukti: diabetes_hgb_seed42 lolos M1 (margin+stabilitas)
# tetapi TEST turun -2.14pp; iris lolos M1, TEST identik atau tidak naik. M1 kini
# memakai BRIDGE SEL/BLIND seperti P3: cari mask (B1) + param (B2) di SEL (4/5)
# dengan inner-CV, nilai P1 vs kandidat M1 di BLIND (1/5) yang tidak pernah
# dilihat proses seleksi. Offset seed dibedakan dari P3 (1000) agar rotasi
# SEL/BLIND tidak sama persis.
M1_BLIND_ROTATIONS = 15  # 2026-10-05 v5: naik dari 3, alasan sama dengan P3_BLIND_ROTATIONS
M1_BLIND_SEED_OFFSET = 2000

# 2026-10-05: B1 (pilih mask) lama selalu memakai hyperparameter P1 yang
# dibekukan (hasil tuning untuk fitur penuh) untuk menilai semua kandidat mask,
# padahal hyperparameter optimal bisa berbeda per ukuran subset fitur, dan B2
# (retune) baru jalan setelah mask dikunci tanpa iterasi balik. Sekarang B1 dan
# B2 diulang M1_COORD_ROUNDS kali (B1 pakai param iterasi sebelumnya, B2 retune
# mask baru, ulangi) dan berhenti lebih awal bila mask tidak berubah. 2 iterasi
# dipilih sebagai kompromi agar koreksi tetap satu kali tanpa biaya meledak.
M1_COORD_ROUNDS = 2

# 2026-10-05 v2: semua BRIDGE (M1/P3/P4) dulu fit model di SEL (4/5 = 80%
# TRAIN), padahal saat deploy model di-fit ke TRAIN penuh (100%). Untuk model
# fragile (pc4: SVC+poly di data timpang, BRIDGE blind ~0.88 tetapi TEST asli
# jatuh ke 0.36), beda ukuran fit ini bisa menyembunyikan ketidakstabilan.
# SEL_BLIND_SPLITS dinaikkan dari 5 (SEL=80%) ke 10 (SEL=90%) agar ukuran fit
# BRIDGE lebih dekat ke deployment; konsekuensinya BLIND lebih kecil dan lebih
# berisik per rotasi.
SEL_BLIND_SPLITS = 10

# Amendemen P4-BRIDGE 2026-10-05 (pola sama dengan M1).
# P4 BRIDGE hanya didukung untuk incumbent single-model (P1/M1_sel), kasus paling
# umum karena P3 hampir selalu direvert oleh BRIDGE-nya sendiri. Untuk incumbent
# kind=p3_robust (refit ulang di SEL arbitrer belum ada abstraksinya), run.py
# memakai guard lama (margin+stabilitas) khusus kasus itu, ditandai di
# decision_p4.json -> guard.bridge_supported=False.
P4_BLIND_ROTATIONS = 15  # 2026-10-05 v5: naik dari 3, alasan sama dengan P3_BLIND_ROTATIONS
P4_BLIND_SEED_OFFSET = 3000
P3_INNER_FOLDS = 3

# Amendemen guard 2026-10-04 (diagnosis murni TRAIN: gate eps-murni menerima
# kemenangan mikro dari noise, mis. skulls +0.001).
# Semua guard TRAIN-only (tanpa TEST). Nilai default konservatif dan tidak
# disetel dari angka TEST: margin 0.2pp membunuh debu; 2-SE standar; konfirmasi
# stabilitas memakai STABILITY_SEEDS yang sudah dihitung. Karena ambang ini
# dimotivasi hasil TEST 11 set klaim, hasil guard pada 11 set itu bersifat
# pilot/replay, bukan bukti klaim; bukti membutuhkan data segar.
GATE_M1_MARGIN = 0.001    # M1 accept iff win_main > margin
# 2026-10-05: floor lama (1.0) menonaktifkan P4 permanen. Itu tambalan karena
# guard lama (margin+stabilitas) mudah tertipu overfitting. P4 kini memakai
# BRIDGE SEL/BLIND seperti M1/P3 (lihat run.py), jadi floor diturunkan ke level
# yang sama dan proteksi overfitting pindah ke BRIDGE.
GATE_P4_MARGIN = 0.001    # P4 accept iff gain > floor (BRIDGE SEL/BLIND)
GATE_P3_MARGIN = 0.001    # P3 accept iff gain > margin (selain SE)
# 2026-10-05 v4 (dicoba lalu dibatalkan): sumber asli
# (FEDSEL_multiplatform/fde_bridge_cascade.py) memakai gain>se+EPS (1xSE).
# Dicoba di pc4+XGB: P3 lolos (gain=0.0184>se=0.0119) tetapi TEST asli kalah di
# semua metrik (wF1 0.90 vs baseline 0.92, bal_acc 0.73 vs 0.78), false positive
# yang terverifikasi. Akar masalah ada di jumlah rotasi BLIND (3) yang membuat
# gain/se bergoyang, bukan di angka ambang; multiplier dikembalikan ke 2.0.
GATE_SE_MULT = 2.0        # BRIDGE: gain > SE_MULT * se + eps
GATE_REQ_STABILITY = True  # M1/P4: kandidat wajib menang mean across seeds

# Registry dataset (F).
# Dev (bukan bukti): iris, wine (sklearn built-in).
# Uji klaim via fetch_openml(data_id=...). Syarat jalan: minority outer-TRAIN>=10 + numerik.
OPENML_IDS = {
    "diabetes": 37,
    "pc4": 1049,
}
DEV_DATASETS = ("iris", "wine")  # 2026-10-05: "wine" sempat terlewat (README menyebut dev=iris+wine)

# Ruang search Optuna (beku, bagian dari paket C).
# Batas di bawah adalah warisan implementasi dan tidak diganti tanpa membatalkan
# klaim keterbandingan. TPE hanya untuk ruang kontinu-halus (param), bukan untuk
# masker diskret 2^F (lihat M1/B1).
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

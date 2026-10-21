"""Verifier exp41 (fail-closed). SPEK_EXP41 §5-§6. Turunan verify_exp40.py (diaudit
Kimi+GPT). Batas risiko PRIMER (v3) = rasio Chernoff-KL (U_W/L_A, alpha .025+.025), skala 8 pooled
dan skala 1 per sel. CPMAX, CPCLUS, CP naif, bootstrap trial HANYA pembanding TANPA
jaminan umum (CPMAX/CPCLUS punya counterexample; bootstrap anti-konservatif).

Pakai:  python verify_exp40.py --csv <file> [--csv <file> ...] [--json out.json]
Input HARUS eksplisit. Validasi (skema, manifest, seed, kelengkapan, duplikat,
boolean) dijalankan SEBELUM statistik; pelanggaran apa pun -> exit 2.

Prosedur bootstrap (dibekukan; verify_exp40_independent.py mengikuti persis):
- unit = trial (dataset, rep); semua arm satu trial ikut bersama.
- dataset diurut menurut nama; untuk tiap dataset berurutan:
  idx = rng.integers(0, R, size=(B, R)); total = sum per dataset.
- rasio = total_salah / total_accept; replikasi dgn total_accept == 0 -> 1.0.
- batas atas = np.quantile(rasio, 0.95).
- seed RNG: pooled BOOT_SEED; LOCO klaster ke-j (urut nama klaster)
  BOOT_SEED + 1 + j; sel ke-i (urut CELLS) BOOT_SEED + 1000 + i.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
SCHEMA = "exp41-v1"
CELLS = [(m, a) for m in ("M1", "M2") for a in ("I1", "I2", "I3", "I4")]
ARMS12 = [(m, a) for m in ("M1", "M2")
          for a in ("I1", "I2", "I3", "I4", "Cplus", "Cseed")]
DECISIONS = {"accept", "reject", "abstain"}
SKIPS = {"skip_stratify_fail", "skip_minority_train", "skip_bags_insufficient",
         "skip_minority_confirm"}  # whitelist = alasan yang ditulis kode trial
CODES8 = ["000", "001", "010", "011", "100", "101", "110", "111"]
CONFIRM_SIZE = 1200


class Invalid(Exception):
    pass


# ---------------------------------------------------------------- statistik
def cp_up(k, n):
    if n == 0:
        return None
    if k >= n:
        return 1.0
    return float(beta.ppf(0.95, k + 1, n - k))


def cp_lo(k, n):
    if n == 0:
        return None
    if k == 0:
        return 0.0
    return float(beta.ppf(0.05, k, n - k + 1))


ALPHA_HALF = 0.025


def _kl(p, q):
    t1 = p * np.log(p / q) if p > 0 else 0.0
    t2 = (1 - p) * np.log((1 - p) / (1 - q)) if p < 1 else 0.0
    return t1 + t2


def kl_upper(xbar, N, a=ALPHA_HALF):
    """Batas atas Chernoff-KL untuk mean variabel independen di [0,1]."""
    from scipy.optimize import brentq
    if xbar >= 1:
        return 1.0
    f = lambda q: N * _kl(xbar, q) - np.log(1 / a)
    hi = 1 - 1e-15
    return 1.0 if f(hi) <= 0 else float(brentq(f, max(xbar, 0) + 1e-18, hi, xtol=1e-15))


def kl_lower(xbar, N, a=ALPHA_HALF):
    from scipy.optimize import brentq
    if xbar <= 0:
        return 0.0
    f = lambda q: N * _kl(xbar, q) - np.log(1 / a)
    lo = 1e-15
    return 0.0 if f(lo) <= 0 else float(brentq(f, lo, min(xbar, 1) - 1e-18, xtol=1e-15))


def kl_ratio_up(k, n, N, b):
    """Batas atas rasio E[W]/E[A] (audit GPT #1): U_W / L_A, union bound alpha .05."""
    la = kl_lower(n / (b * N), N)
    return 1.0 if la <= 0 else min(1.0, kl_upper(k / (b * N), N) / la)


def kl_ratio_lo(k, n, N, b):
    ua = kl_upper(n / (b * N), N)
    return 0.0 if k == 0 else kl_lower(k / (b * N), N) / ua


def cpmax_up(acc_rows, n_trial):
    """PEMBANDING saja: min(1, 8*CP(ceil(k/8), N)*N/n). TIDAK valid secara umum
    (counterexample audit GPT exp41 §1)."""
    n = len(acc_rows)
    if n == 0:
        return None
    k = float(acc_rows["_w"].sum())
    return min(1.0, 8 * cp_up(int(np.ceil(k / 8 - 1e-12)), n_trial) * n_trial / n)


def cpclus_up(acc_rows):
    """CPCLUS95: n_t = #trial (dataset, rep) ber-accept; m = n / n_t;
    k' = ceil(k / m); CP satu-sisi 95% atas (k', n_t). None bila n = 0."""
    n = len(acc_rows)
    if n == 0:
        return None, 0
    k = float(acc_rows["_w"].sum())
    nt = acc_rows.groupby(["dataset", "rep"]).ngroups
    kk = int(np.ceil(k / (n / nt) - 1e-12))
    return cp_up(kk, nt), nt


def boot_upper(per_ds, B, seed):
    """per_ds: list (urut nama dataset) of (acc_t, wrong_t) array panjang R."""
    rng = np.random.default_rng(seed)
    tot_a = np.zeros(B)
    tot_w = np.zeros(B)
    for acc_t, wr_t in per_ds:
        R = len(acc_t)
        idx = rng.integers(0, R, size=(B, R))
        tot_a += acc_t[idx].sum(axis=1)
        tot_w += wr_t[idx].sum(axis=1)
    ratio = np.where(tot_a > 0, tot_w / np.maximum(tot_a, 1), 1.0)
    return float(np.quantile(ratio, 0.95))


# ---------------------------------------------------------------- validasi
def load_and_validate(paths, manifest, cfg):
    names = [Path(p).name for p in paths]
    if len(paths) > 1 and any("COMBINED" in n for n in names):
        raise Invalid("COMBINED tidak boleh dicampur dengan file lain")
    if len(set(map(str, paths))) != len(paths):
        raise Invalid("path input ganda")
    df = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False)
                    for p in paths], ignore_index=True)
    return validate(df, manifest, cfg)


def validate(df, manifest, cfg):
    df = df.copy()
    if (df.schema_version != SCHEMA).any():
        raise Invalid("schema_version salah/kosong")
    man = {r.dataset: r for r in manifest.itertuples()}
    unknown = set(df.dataset) - set(man)
    if unknown:
        raise Invalid(f"dataset di luar manifest: {sorted(unknown)}")
    for col in ("data_id", "rep", "seed"):
        if not df[col].str.fullmatch(r"-?\d+").all():
            raise Invalid(f"{col} bukan integer")
        df[col] = df[col].astype(int)
    for ds, g in df.groupby("dataset"):
        m = man[ds]
        if (g.stratum != m.stratum).any() or (g.cluster != m.cluster).any() \
                or (g.data_id != int(m.data_id)).any():
            raise Invalid(f"{ds}: stratum/cluster/data_id != manifest")
    off = {k: int(v) for k, v in cfg["OFFSETS"].items()}
    R = {k: int(v) for k, v in cfg["R"].items()}
    idx = {d: i for i, d in enumerate(manifest.dataset)}
    if ((df.seed - df.stratum.map(off) - 1000 * df.dataset.map(idx)) != df.rep).any():
        raise Invalid("seed != offset + 1000*indeks_manifest + rep")
    if ((df.rep < 0) | (df.rep >= df.stratum.map(R))).any():
        raise Invalid("rep di luar rentang R")
    ok = df[df.status == "ok"].copy()
    skip = df[df.status != "ok"]
    if not skip.status.isin(SKIPS).all():
        raise Invalid("status skip tak dikenal (bukan whitelist)")
    if ok.duplicated(["dataset", "seed", "model", "arm"]).any():
        raise Invalid("duplikat (dataset,seed,model,arm)")
    if skip.duplicated(["dataset", "seed"]).any():
        raise Invalid("duplikat baris skip")
    # kelengkapan
    want = set(ARMS12)
    for ds, m in man.items():
        for rep in range(R[m.stratum]):
            g = ok[(ok.dataset == ds) & (ok.rep == rep)]
            s = skip[(skip.dataset == ds) & (skip.rep == rep)]
            if len(s) == 1 and len(g) == 0:
                continue
            if len(s) == 0 and len(g) == 12 and \
                    set(zip(g.model, g.arm)) == want:
                continue
            raise Invalid(f"{ds} rep {rep}: tidak lengkap ({len(g)} ok, {len(s)} skip)")
    # boolean & numerik eksplisit
    if not ok.ref_ambiguous_95.isin(["True", "False"]).all():
        raise Invalid("ref_ambiguous_95 bukan 'True'/'False'")
    ok["amb"] = ok.ref_ambiguous_95 == "True"
    if not (ok.model.isin(["M1", "M2"]) & ok.arm.isin(
            ["I1", "I2", "I3", "I4", "Cplus", "Cseed"])).all():
        raise Invalid("model/arm tak dikenal")
    if not ok.decision99.isin(DECISIONS).all():
        raise Invalid("decision99 tak dikenal")
    for col in ("ci_lo99", "ref_delta"):
        try:
            ok[col] = ok[col].astype(float)
        except ValueError:
            raise Invalid(f"{col} bukan angka")
        if not np.isfinite(ok[col]).all():  # NaN dan +-inf (audit Kimi B1/B2)
            raise Invalid(f"{col} tidak finite")
    check_evidence(ok)
    check_reference_counts(ok, manifest)
    return ok, skip


def check_reference_counts(ok, manifest):
    """Audit GPT #4: referensi = ukuran & komposisi terkunci di manifest; marginal y
    CONFIRM sama di 12 arm satu trial."""
    m = manifest.set_index("dataset")
    rsum = ok[[f"r{k}" for k in CODES8]].astype(int).sum(axis=1)
    rpos = ok[[f"r1{k}" for k in ("00", "01", "10", "11")]].astype(int).sum(axis=1)
    if (rsum.values != ok.dataset.map(m.n_ref).values).any() or (rsum <= 0).any():
        raise Invalid("ukuran referensi != n_ref manifest")
    if (rpos.values != ok.dataset.map(m.ref_pos).values).any():
        raise Invalid("jumlah positif referensi != ref_pos manifest")
    cpos = ok[[f"c1{k}" for k in ("00", "01", "10", "11")]].astype(int).sum(axis=1)
    if (cpos.groupby([ok.dataset.values, ok.rep.values]).nunique() > 1).any():
        raise Invalid("marginal y CONFIRM berbeda antar-arm dalam satu trial")


def wf1_delta(c):
    """Δ weighted-F1 (kandidat - baseline) dari 8-counts (y,b,c); implementasi
    sendiri (bukan impor kode trial)."""
    c = np.asarray(c, dtype=float).reshape(-1, 2, 2, 2)  # [y,b,c]

    def wf1(tn, fp, fn, tp):
        n0, n1 = tn + fp, fn + tp
        f0 = np.where(2 * tn + fp + fn > 0, 2 * tn / np.maximum(2 * tn + fp + fn, 1e-300), 0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-300), 0)
        return (n0 * f0 + n1 * f1) / np.maximum(n0 + n1, 1)
    cb = c.sum(axis=3)  # [y,b]
    cc = c.sum(axis=2)  # [y,c]
    return (wf1(cc[:, 0, 0], cc[:, 0, 1], cc[:, 1, 0], cc[:, 1, 1])
            - wf1(cb[:, 0, 0], cb[:, 0, 1], cb[:, 1, 0], cb[:, 1, 1]))


def check_evidence(ok):
    """G1 (audit GPT): ringkasan harus konsisten dengan bukti tersimpan."""
    need = ["arm_ordinal", "ci_hi99", "ref_ci_lo95", "ref_ci_hi95",
            "n_confirm_disagreements"] + [f"c{k}" for k in CODES8] + \
        [f"r{k}" for k in CODES8]
    for col in need:
        if col not in ok.columns or (ok[col] == "").any():
            raise Invalid(f"kolom bukti kosong/hilang: {col}")
    ints = ["arm_ordinal", "n_confirm_disagreements"] + [f"c{k}" for k in CODES8] + \
        [f"r{k}" for k in CODES8]
    for col in ints:
        if not ok[col].str.fullmatch(r"\d+").all():
            raise Invalid(f"{col} bukan integer nonnegatif")
    C = ok[[f"c{k}" for k in CODES8]].astype(int).values
    Rr = ok[[f"r{k}" for k in CODES8]].astype(int).values
    hi = ok.ci_hi99.astype(float).values
    rlo, rhi = ok.ref_ci_lo95.astype(float).values, ok.ref_ci_hi95.astype(float).values
    lo, ref = ok.ci_lo99.values, ok.ref_delta.values
    if not (np.isfinite(hi).all() and np.isfinite(rlo).all() and np.isfinite(rhi).all()):
        raise Invalid("CI tidak finite")
    for v in (lo, hi, ref, rlo, rhi):
        if (np.abs(v) > 1).any():
            raise Invalid("nilai delta/CI di luar [-1,1]")
    if (lo > hi).any() or (rlo > rhi).any():
        raise Invalid("batas bawah > batas atas")
    if (C.sum(axis=1) != CONFIRM_SIZE).any():
        raise Invalid("counts CONFIRM tidak berjumlah 1200")
    if (C[:, [1, 2, 5, 6]].sum(axis=1) != ok.n_confirm_disagreements.astype(int).values).any():
        raise Invalid("D != counts")
    nref = pd.Series(Rr.sum(axis=1)).groupby(ok.dataset.values).nunique()
    if (nref > 1).any():
        raise Invalid("ukuran referensi tidak konsisten dalam dataset")
    if (np.abs(wf1_delta(Rr) - ref) > 1e-9).any():
        raise Invalid("ref_delta != delta(counts referensi)")
    if ((ok.amb.values) != ((rlo <= 0) & (0 <= rhi))).any():
        raise Invalid("flag ambigu != CI referensi")
    dec = np.where(lo > 1e-14, "accept", np.where(hi < 0, "reject", "abstain"))
    if (dec != ok.decision99.values).any():
        raise Invalid("decision99 != aturan gate dari ci_lo99/ci_hi99")
    want = {a: i for i, a in enumerate(ARMS12)}
    if (ok.arm_ordinal.astype(int).values !=
            np.array([want[(m, a)] for m, a in zip(ok.model, ok.arm)])).any():
        raise Invalid("arm_ordinal != urutan arm beku")


# ---------------------------------------------------------------- analisis
def per_trial(rows, datasets, R, flag_col):
    """list (urut nama) of (acc_t, wrong_t) dari baris accept bertanda."""
    out = []
    for ds in datasets:
        acc_t = np.zeros(R[ds])
        wr_t = np.zeros(R[ds])
        g = rows[rows.dataset == ds]
        np.add.at(acc_t, g.rep.values, 1)
        np.add.at(wr_t, g.rep.values, g[flag_col].values.astype(float))
        out.append((acc_t, wr_t))
    return out


def risk_block(acc, datasets, clus_of, R, B, seed0, flag_col):
    a = acc.assign(_w=acc[flag_col].astype(float))
    n, k = len(a), int(a._w.sum())
    by_c = a.groupby(a.dataset.map(clus_of)).size()
    contrib = sorted(by_c.index)
    Rd = R[datasets[0]]
    r1 = kl_ratio_up(k, n, Rd * len(datasets), 8) if n else None
    all_clusters = sorted(set(clus_of[d] for d in datasets))
    loco, power = {}, {}
    for c in all_clusters:  # SEMUA klaster
        keep = [d for d in datasets if clus_of[d] != c]
        sub = a[a.dataset.map(clus_of) != c]
        ns, ks = len(sub), float(sub._w.sum())
        loco[c] = kl_ratio_up(ks, ns, Rd * len(keep), 8) if ns else None
        power[c] = kl_ratio_up(0, ns, Rd * len(keep), 8) if ns else None  # daya: k=0
    r0 = len(contrib) >= 5 and all(v is not None and v < 0.10 for v in power.values())
    r2 = bool(loco) and all(v is not None and v < 0.10 for v in loco.values())
    boot = boot_upper(per_trial(acc, datasets, R, flag_col), B, seed0) if n else None
    clus, _ = cpclus_up(a)
    return {"n": n, "k": k, "naive_cp_up": cp_up(k, n), "R0": bool(r0),
            "R1_upper": r1, "R1": r1 is not None and r1 < 0.10,
            "R2": bool(r2), "loco": loco, "pembanding_boot_up": boot,
            "pembanding_cpclus_up": clus, "pembanding_cpmax_up": cpmax_up(a, Rd * len(datasets))}


def cell_status(acc_cell, datasets, R, B, seed, flag_col):
    """Per sel: batas rasio KL skala 1 (audit GPT #2: CP pooled lintas dataset tidak otomatis valid)."""
    n, k = len(acc_cell), int(acc_cell[flag_col].astype(float).sum())
    if n == 0:
        return {"n": 0, "k": 0, "status": "tanpa-accept"}
    N = R[datasets[0]] * len(datasets)  # trial terjadwal; <=1 accept/trial/sel
    lo, up = kl_ratio_lo(k, n, N, 1), kl_ratio_up(k, n, N, 1)
    if lo > 0.10 or (n >= 10 and k / n > 0.15):
        st = "buruk"
    elif up < 0.10:
        st = "baik"
    else:
        st = "belum teratasi"
    return {"n": n, "k": k, "cp_lo": lo, "upper": up, "status": st}


def overall(block, cells):
    if not block["R0"]:
        return "tidak dapat dinilai"
    if not (block["R1"] and block["R2"]):
        return "tidak memenuhi kriteria risiko"
    if any(c["status"] == "buruk" for c in cells.values()):
        return "ada sel buruk: klaim kebijakan tidak boleh"
    return "memenuhi kriteria risiko"


def analyze_stratum(ok, manifest, cfg, stratum, margin):
    man = manifest[manifest.stratum == stratum]
    datasets = sorted(man.dataset)
    clus_of = dict(zip(man.dataset, man.cluster))
    R = {d: int(cfg["R"][stratum]) for d in datasets}
    B, s0 = int(cfg["N_BOOT_CLUSTER"]), int(cfg["BOOT_SEED"])
    o = ok[ok.dataset.isin(datasets)].copy()
    o["wrong"] = o.ref_delta <= 0
    o["wrongA"] = o.wrong | o.amb
    o["acc"] = (o.decision99 == "accept") & \
        ((o.ci_lo99 > margin) if margin is not None else True)
    in_cells = o.set_index(["model", "arm"]).index.isin(CELLS)
    acc = o[o.acc & in_cells]
    res = {}
    for tag, col in (("titik", "wrong"), ("A", "wrongA")):
        blk = risk_block(acc, datasets, clus_of, R, B, s0, col)
        cells = {f"{m}-{a}": cell_status(
            acc[(acc.model == m) & (acc.arm == a)], datasets, R, B,
            s0 + 1000 + i, col) for i, (m, a) in enumerate(CELLS)}
        res[tag] = {**blk, "cells": cells, "status": overall(blk, cells)}
    nd = len(datasets)
    Rs = int(cfg["R"][stratum])
    good = acc[acc.ref_delta > 0]
    y_ds = {d: int((good.dataset == d).sum()) / (Rs * 8) for d in datasets}
    yp = len(good) / (Rs * nd * 8)
    res["U"] = {"yield_pooled": yp, "n_ds_y5": sum(v >= 0.05 for v in y_ds.values()),
                "U": yp >= 0.05 and sum(v >= 0.05 for v in y_ds.values()) >= 3,
                "yield_per_ds": y_ds}
    cp = {}
    for m in ("M1", "M2"):
        c = o[(o.model == m) & (o.arm == "Cplus") & o.acc & (o.ref_delta > 0)]
        y = len(c) / (Rs * nd)
        cp[m] = {"yield": y, "ok": y >= 0.20}
    res["Cplus"] = cp
    cs = {}
    for m in ("M1", "M2"):
        c = o[(o.model == m) & (o.arm == "Cseed") & o.acc]
        cs[m] = {"accept": len(c), "ref_le0": int((c.ref_delta <= 0).sum()),
                 "pos_amb": int(((c.ref_delta > 0) & c.amb).sum()),
                 "pos_nonamb": int(((c.ref_delta > 0) & ~c.amb).sum()),
                 "per_trial": len(c) / (Rs * nd)}
    res["Cseed"] = cs
    return res


def analyze(ok, manifest, cfg):
    out = {}
    for stratum in ("utama",):
        out[stratum] = {
            "primer": analyze_stratum(ok, manifest, cfg, stratum, float(cfg["MARGIN"])),
            "pembanding_0.002": analyze_stratum(ok, manifest, cfg, stratum,
                                                float(cfg["MARGIN_PEMBANDING"])),
            "tanpa_margin": analyze_stratum(ok, manifest, cfg, stratum, None)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--manifest", default=str(HERE / "exp41_manifest"))
    ap.add_argument("--json")
    a = ap.parse_args()
    manifest = pd.read_csv(Path(a.manifest) / "datasets.csv")
    cfg = json.load(open(Path(a.manifest) / "config.json"))
    try:
        ok, skip = load_and_validate(a.csv, manifest, cfg)
    except (Invalid, ValueError, KeyError) as e:
        print(f"VALIDASI GAGAL: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
    import replay_exp41_gate as RP
    raw = pd.concat([pd.read_csv(p, float_precision="round_trip") for p in a.csv],
                    ignore_index=True)
    n_rep, bad = RP.replay(raw)
    if bad or n_rep != len(ok):
        print(f"REPLAY GAGAL: {len(bad)} selisih, {n_rep} vs {len(ok)} baris", file=sys.stderr)
        sys.exit(2)
    print(f"validasi + replay lulus: {len(ok)} baris ok, {len(skip)} trial skip")
    res = analyze(ok, manifest, cfg)
    txt = json.dumps(res, indent=1, default=float)
    if a.json:
        open(a.json, "w").write(txt)
    for st, r in res.items():
        for rule, x in r.items():
            print(f"[{st} | {rule}] titik: {x['titik']['status']} "
                  f"(n={x['titik']['n']} k={x['titik']['k']} R1={x['titik']['R1_upper']}) | "
                  f"A: {x['A']['status']} | U={x['U']['U']} y={x['U']['yield_pooled']:.3f}")


if __name__ == "__main__":
    main()

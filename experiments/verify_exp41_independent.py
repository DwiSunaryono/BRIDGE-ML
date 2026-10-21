"""Verifier exp41 kedua (implementasi terpisah; tidak mengimpor verify_exp41/40).
Batas primer (v3) = rasio Chernoff-KL, diimplementasikan ulang di sini (bisection);
CPCLUS/CPMAX hanya pembanding tanpa jaminan umum.

Membaca CSV mentah dengan modul csv + dict murni. Prosedur bootstrap mengikuti definisi beku di
docstring verify_exp40.py (urutan RNG sama) sehingga angka risiko harus identik. Kontrak kesamaan
bersifat semantik: n, k, R1, LOCO, status sel, status keseluruhan, yield pooled, U, yield C+
(struktur JSON beda). Validasi bukti (audit G1) diimplementasikan ulang dengan Python murni.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

CELLS = ["M1-I1", "M1-I2", "M1-I3", "M1-I4", "M2-I1", "M2-I2", "M2-I3", "M2-I4"]
NEED = {f"{m}-{a}" for m in ("M1", "M2")
        for a in ("I1", "I2", "I3", "I4", "Cplus", "Cseed")}


SKIPS = ("skip_stratify_fail", "skip_minority_train", "skip_bags_insufficient",
         "skip_minority_confirm")
ORDER = [f"{m}-{a}" for m in ("M1", "M2") for a in ("I1", "I2", "I3", "I4", "Cplus", "Cseed")]
K8 = ["000", "001", "010", "011", "100", "101", "110", "111"]


def fail(msg):
    raise ValueError(msg)


def as_int(s, what):
    if s != s.strip():
        fail(f"{what} berspasi: {s!r}")  # kontrak sama dengan V1
    if not (s.isdigit() or (s.startswith("-") and s[1:].isdigit())):
        fail(f"{what} bukan integer: {s!r}")
    return int(s)


def wf1_of(cnt, pred_axis):
    """weighted-F1 dari dict (y,b,c)->n untuk prediksi pada axis 1 (b) atau 2 (c)."""
    m = {(y, p): 0 for y in (0, 1) for p in (0, 1)}
    for (y, b, c), n in cnt.items():
        m[(y, b if pred_axis == 1 else c)] += n
    tot = sum(m.values())
    s = 0.0
    for k in (0, 1):
        tpk = m[(k, k)]
        fpk = m[(1 - k, k)]
        fnk = m[(k, 1 - k)]
        den = 2 * tpk + fpk + fnk
        f = 2 * tpk / den if den > 0 else 0.0
        s += (m[(k, 0)] + m[(k, 1)]) * f
    return s / max(tot, 1)


def evidence(r):
    """G1: cek konsistensi ringkasan terhadap bukti pada satu baris ok."""
    for col in ["arm_ordinal", "ci_hi99", "ref_ci_lo95", "ref_ci_hi95",
                "n_confirm_disagreements"] + ["c" + k for k in K8] + ["r" + k for k in K8]:
        if r.get(col, "") in ("", None):
            fail(f"bukti kosong {col}")
    cc = {tuple(int(ch) for ch in k): as_int(r["c" + k], "c") for k in K8}
    rr = {tuple(int(ch) for ch in k): as_int(r["r" + k], "r") for k in K8}
    if min(cc.values()) < 0 or min(rr.values()) < 0:
        fail("count negatif")
    if sum(cc.values()) != 1200:
        fail("sum CONFIRM")
    D = sum(n for (y, b, c), n in cc.items() if b != c)
    if D != as_int(r["n_confirm_disagreements"], "D"):
        fail("D")
    lo, hi = float(r["ci_lo99"]), float(r["ci_hi99"])
    rlo, rhi, ref = float(r["ref_ci_lo95"]), float(r["ref_ci_hi95"]), float(r["ref_delta"])
    for v in (lo, hi, rlo, rhi, ref):
        if not math.isfinite(v) or abs(v) > 1:
            fail("domain")
    if lo > hi or rlo > rhi:
        fail("lo>hi")
    if abs((wf1_of(rr, 2) - wf1_of(rr, 1)) - ref) > 1e-9:
        fail("ref_delta vs counts")
    if (r["ref_ambiguous_95"] == "True") != (rlo <= 0 <= rhi):
        fail("ambigu vs CI")
    want = "accept" if lo > 1e-14 else ("reject" if hi < 0 else "abstain")
    if r["decision99"] != want:
        fail("decision vs CI gate")
    if as_int(r["arm_ordinal"], "ord") != ORDER.index(f"{r['model']}-{r['arm']}"):
        fail("arm_ordinal")
    return sum(rr.values())


def read_rows(paths):
    if len(paths) > 1 and any("COMBINED" in Path(p).name for p in paths):
        fail("COMBINED dicampur")
    if len(set(paths)) != len(paths):
        fail("path ganda")
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def check(rows, man, cfg):
    off, Rc = cfg["OFFSETS"], cfg["R"]
    seen, skip, ok, nrefs = set(), set(), [], {}
    order = {d: i for i, d in enumerate(man)}
    cpos = {}
    for r in rows:
        if r["schema_version"] != "exp41-v1":
            fail("schema")
        ds = r["dataset"]
        if ds not in man:
            fail("dataset asing")
        m = man[ds]
        if r["stratum"] != m["stratum"] or r["cluster"] != m["cluster"] \
                or int(r["data_id"]) != int(m["data_id"]):
            fail("manifest mismatch")
        rep, seed = as_int(r["rep"], "rep"), as_int(r["seed"], "seed")
        if as_int(r["data_id"], "data_id") != int(m["data_id"]):
            fail("data_id")
        if seed != int(off[m["stratum"]]) + 1000 * order[ds] + rep or not 0 <= rep < int(Rc[m["stratum"]]):
            fail("seed/rep")
        if r["status"] == "ok":
            if f"{r['model']}-{r['arm']}" not in ORDER:
                fail("model/arm")
            key = (ds, rep, f"{r['model']}-{r['arm']}")
            if key in seen:
                fail("duplikat")
            seen.add(key)
            if r["ref_ambiguous_95"] not in ("True", "False"):
                fail("boolean")
            if r["decision99"] not in ("accept", "reject", "abstain"):
                fail("decision")
            try:
                lo, ref = float(r["ci_lo99"]), float(r["ref_delta"])
            except ValueError:
                fail("numerik")
            if not (math.isfinite(lo) and math.isfinite(ref)):  # kasus B1/B2
                fail("tidak finite")
            nref = evidence(r)
            if nref != int(m["n_ref"]) or nref <= 0:
                fail("n_ref != manifest")
            if sum(as_int(r["r1" + t], "r") for t in ("00", "01", "10", "11")) != int(m["ref_pos"]):
                fail("ref_pos != manifest")
            cp_ = sum(as_int(r["c1" + t], "c") for t in ("00", "01", "10", "11"))
            if cpos.setdefault((ds, rep), cp_) != cp_:
                fail("marginal y CONFIRM beda antar-arm")
            ok.append({"ds": ds, "rep": rep, "cell": f"{r['model']}-{r['arm']}",
                       "dec": r["decision99"], "lo": lo,
                       "ref": ref,
                       "amb": r["ref_ambiguous_95"] == "True"})
        elif r["status"] in SKIPS:
            if (ds, rep) in skip:
                fail("skip ganda")
            skip.add((ds, rep))
        else:
            fail("status")
    for ds, m in man.items():
        for rep in range(int(Rc[m["stratum"]])):
            cells = {c for (d, rp, c) in seen if d == ds and rp == rep}
            if (ds, rep) in skip:
                if cells:
                    fail("skip + ok")
            elif cells != NEED:
                fail(f"tidak lengkap {ds} {rep}")
    return ok


def cpu(k, n):
    return None if n == 0 else (1.0 if k >= n else float(beta.ppf(0.95, k + 1, n - k)))


def cpl(k, n):
    return None if n == 0 else (0.0 if k == 0 else float(beta.ppf(0.05, k, n - k + 1)))


def cpclus(xs, wk):
    """CPCLUS95 versi kedua: hitung trial unik dengan set, k' lewat pembagian bulat ke atas."""
    n = len(xs)
    if n == 0:
        return None, 0
    k = sum(x[wk] for x in xs)
    trials = {(x["ds"], x["rep"]) for x in xs}
    nt = len(trials)
    kk = -int(-(k * nt) // n) if k > 0 else 0          # ceil(k / (n/nt)) = ceil(k*nt/n)
    return cpu(kk, nt), nt


def _klb(p, q):
    a = p * math.log(p / q) if p > 0 else 0.0
    b = (1 - p) * math.log((1 - p) / (1 - q)) if p < 1 else 0.0
    return a + b


def klu(x, N, al=0.025):
    """Batas atas KL lewat bisection (implementasi kedua)."""
    if x >= 1:
        return 1.0
    lo, hi, t = max(x, 0.0), 1.0, math.log(1 / al)
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid >= 1 or N * _klb(x, mid) > t:
            hi = mid
        else:
            lo = mid
    return hi


def kll(x, N, al=0.025):
    if x <= 0:
        return 0.0
    lo, hi, t = 0.0, min(x, 1.0), math.log(1 / al)
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid <= 0 or N * _klb(x, mid) > t:
            lo = mid
        else:
            hi = mid
    return lo


def ratio_up(k, n, N, b):
    la = kll(n / (b * N), N)
    return 1.0 if la <= 0 else min(1.0, klu(k / (b * N), N) / la)


def ratio_lo(k, n, N, b):
    return 0.0 if k == 0 else kll(k / (b * N), N) / klu(n / (b * N), N)


def cpmax(xs, wk, n_trial):
    """CPMAX versi kedua: ceil(k/8) lewat aritmetika integer."""
    n = len(xs)
    if n == 0:
        return None
    k = int(round(sum(x[wk] for x in xs)))
    kk = (k + 7) // 8
    return min(1.0, 8 * cpu(kk, n_trial) * n_trial / n)


def boot(acc, dss, R, B, seed, wkey):
    rng = np.random.default_rng(seed)
    A = np.zeros(B)
    W = np.zeros(B)
    for ds in dss:  # urut nama
        a = np.zeros(R)
        w = np.zeros(R)
        for x in acc:
            if x["ds"] == ds:
                a[x["rep"]] += 1
                w[x["rep"]] += x[wkey]
        idx = rng.integers(0, R, size=(B, R))
        A += np.take(a, idx).sum(axis=1)
        W += np.take(w, idx).sum(axis=1)
    r = np.ones(B)
    pos = A > 0
    r[pos] = W[pos] / A[pos]
    return float(np.quantile(r, 0.95))


def stratum_result(ok, man, cfg, stratum, margin):
    dss = sorted(d for d, m in man.items() if m["stratum"] == stratum)
    cl = {d: man[d]["cluster"] for d in dss}
    R, B, s0 = int(cfg["R"][stratum]), int(cfg["N_BOOT_CLUSTER"]), int(cfg["BOOT_SEED"])
    rows = [dict(x, w=float(x["ref"] <= 0), wA=float(x["ref"] <= 0 or x["amb"]))
            for x in ok if x["ds"] in cl]
    is_acc = (lambda x: x["dec"] == "accept" and (margin is None or x["lo"] > margin))
    acc = [x for x in rows if is_acc(x) and x["cell"] in CELLS]
    out = {}
    for tag, wk in (("titik", "w"), ("A", "wA")):
        n, k = len(acc), int(sum(x[wk] for x in acc))
        cnt = {}
        for x in acc:
            cnt[cl[x["ds"]]] = cnt.get(cl[x["ds"]], 0) + 1
        r1 = ratio_up(k, n, R * len(dss), 8) if n else None
        allc = sorted(set(cl.values()))
        loco, pw = {}, {}
        for c in allc:
            sub = [x for x in acc if cl[x["ds"]] != c]
            nd = sum(1 for d in dss if cl[d] != c)
            ks = int(round(sum(x[wk] for x in sub)))
            loco[c] = ratio_up(ks, len(sub), R * nd, 8) if sub else None
            pw[c] = ratio_up(0, len(sub), R * nd, 8) if sub else None
        r0 = len(cnt) >= 5 and all(v is not None and v < 0.10 for v in pw.values())
        r2 = bool(loco) and all(v is not None and v < 0.10 for v in loco.values())
        cells = {}
        for c in CELLS:
            xs = [x for x in acc if x["cell"] == c]
            nn, kk = len(xs), int(sum(x[wk] for x in xs))
            if nn == 0:
                cells[c] = "tanpa-accept"
                continue
            up = ratio_up(kk, nn, R * len(dss), 1)  # per sel: KL skala 1
            if ratio_lo(kk, nn, R * len(dss), 1) > 0.10 or (nn >= 10 and kk / nn > 0.15):
                cells[c] = "buruk"
            elif up < 0.10:
                cells[c] = "baik"
            else:
                cells[c] = "belum teratasi"
        if not r0:
            st = "tidak dapat dinilai"
        elif not (r1 is not None and r1 < 0.10 and r2):
            st = "tidak memenuhi kriteria risiko"
        elif "buruk" in cells.values():
            st = "ada sel buruk: klaim kebijakan tidak boleh"
        else:
            st = "memenuhi kriteria risiko"
        out[tag] = {"n": n, "k": k, "R1_upper": r1, "loco": loco, "cells": cells, "status": st}
    good = [x for x in acc if x["ref"] > 0]
    yds = {d: sum(1 for x in good if x["ds"] == d) / (R * 8) for d in dss}
    yp = len(good) / (R * len(dss) * 8)
    out["U"] = yp >= 0.05 and sum(v >= 0.05 for v in yds.values()) >= 3
    out["yield_pooled"] = yp
    out["Cplus"] = {m: sum(1 for x in rows if x["cell"] == f"{m}-Cplus" and is_acc(x)
                           and x["ref"] > 0) / (R * len(dss)) for m in ("M1", "M2")}
    out["Cseed_accept"] = {m: sum(1 for x in rows if x["cell"] == f"{m}-Cseed" and is_acc(x))
                           for m in ("M1", "M2")}
    return out


def analyze(paths, manifest_dir, replay=True):
    man = {}
    with open(Path(manifest_dir) / "datasets.csv", newline="") as f:
        for r in csv.DictReader(f):
            man[r["dataset"]] = r
    cfg = json.load(open(Path(manifest_dir) / "config.json"))
    ok = check(read_rows(paths), man, cfg)
    if not replay:  # hanya untuk uji logika sintetis; CLI selalu replay
        return _analyze_ok(ok, man, cfg)
    import pandas as _pd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import replay_exp41_gate as RP
    n_rep, bad = RP.replay(_pd.concat([_pd.read_csv(p, float_precision="round_trip")
                                       for p in paths], ignore_index=True))
    if bad or n_rep != len(ok):
        fail(f"replay gagal: {len(bad)} selisih, {n_rep} vs {len(ok)}")
    return _analyze_ok(ok, man, cfg)


def _analyze_ok(ok, man, cfg):
    res = {}
    res["utama"] = {"primer": stratum_result(ok, man, cfg, "utama", float(cfg["MARGIN"])),
                    "pembanding_0.002": stratum_result(ok, man, cfg, "utama",
                                                       float(cfg["MARGIN_PEMBANDING"])),
                    "tanpa_margin": stratum_result(ok, man, cfg, "utama", None)}
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    mdir = str(Path(__file__).resolve().parent / "exp41_manifest")
    if "--manifest" in args:
        i = args.index("--manifest"); mdir = args[i + 1]; del args[i:i + 2]
    try:
        print(json.dumps(analyze(args, mdir), indent=1))
    except ValueError as e:
        print(f"VALIDASI GAGAL: {e}", file=sys.stderr)
        sys.exit(2)

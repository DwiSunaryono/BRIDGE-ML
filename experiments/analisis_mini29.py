"""Analisis praregistrasi MINI-29 (PRAREG_MINI29.md): kalibrasi t* di dev (TOTAL 2.430, wF1),
evaluasi pada 6 setting."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import brentq
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_exp41 import kl_ratio_up as up

O = HERE / "out"
rd = lambda f: pd.read_csv(O / f, keep_default_na=False, float_precision="round_trip",
                           na_values={c: ["nan", "NaN"] for c in ("lo95", "ref", "dpool")})
GRID = [round(.005 * i, 3) for i in range(1, 11)]


def kl(p, q):
    q = min(max(q, 1e-15), 1 - 1e-15)
    return (p * np.log(p / q) if p > 0 else 0.0) + ((1 - p) * np.log((1 - p) / (1 - q)) if p < 1 else 0.0)


def prop1(S, M, d=.05):
    if M == 0:
        return 1.0
    ph = S / M; f = lambda u: M * kl(ph, u) - np.log(1 / d)
    return 1.0 if f(1 - 1e-12) <= 0 else brentq(f, ph if ph >= 1 else ph + 1e-15, 1 - 1e-12)


def stats(s, acc, loco_by="dataset", use_prop1=False):
    NT = len(s); a = s[acc]; n, k = len(a), int((a.ref <= 0).sum())
    if use_prop1:
        b, lo = prop1(k, n), np.nan
    else:
        b = up(k, n, NT, 1); per = s.groupby(loco_by).size()
        lo = max(up(int((a[a[loco_by] != q].ref <= 0).sum()), int((a[loco_by] != q).sum()), NT - per[q], 1)
                 if (a[loco_by] != q).sum() else 1.0 for q in per.index)
    return dict(terima=n, salah=k, batas=round(100 * b, 2), LOCO=round(100 * lo, 2) if lo == lo else "-",
                yield_=round(100 * n / NT, 1), totalD=round(a.ref.sum(), 2),
                lulus=bool(b < .1 and (lo != lo or lo < .1) and 100 * n / NT >= 5))


# data ukuran (MINI-20)
u = pd.concat([rd(f"mini29_ukuran_{i}.csv") for i in range(3)], ignore_index=True)
u20 = pd.concat([rd("mini20_size_a.csv"), rd("mini20_size_b.csv")], ignore_index=True)
U = u20.merge(u[["dataset", "TOTAL", "rep", "cfg", "dpool"]], on=["dataset", "TOTAL", "rep"], suffixes=("", "_29"))
U = U[U.kelas != "skip"]
print("ukuran: baris", len(U), "cfg identik:", bool((U.cfg == U.cfg_29).all()), "| error:", int(u.cfg.astype(str).str.startswith("ERROR").sum()))
# data metrik (MINI-21)
m = rd("mini29_metrik.csv")
m21 = pd.concat([rd(f"mini21_m{i}.csv") for i in (1, 2, 3)], ignore_index=True)
m21 = m21[m21.design.str.startswith("union_")].copy(); m21["metric"] = m21.design.str.replace("union_", "")
Mt = m21.merge(m[["dataset", "rep", "metric", "cfg", "dpool"]], on=["dataset", "rep", "metric"], suffixes=("", "_29"))
print("metrik: baris", len(Mt), "cfg identik:", bool((Mt.cfg == Mt.cfg_29).all()))
# data Q (MINI-25)
q = rd("mini29_Q.csv"); q25 = pd.concat([rd(f"mini25_m{i}.csv") for i in (1, 2, 3)], ignore_index=True)
Q = q25.merge(q[["dataset", "cfg", "dpool"]], on="dataset", suffixes=("", "_29"))
print("Q: baris", len(Q), "cfg identik:", bool((Q.cfg == Q.cfg_29).all()))

# kalibrasi di dev, TOTAL 2.430
cal = U[U.TOTAL == 2430]; b_bridge = stats(cal, cal.lo95 > .005)
print("\nKALIBRASI (dev, 2.430 baris, wF1): BRIDGE", b_bridge)
tstar = None
for t in GRID:
    s = stats(cal, (cal.cfg != "None") & (cal.dpool > t))
    print(f"  VALMARGIN t={t:.3f}:", s)
    if tstar is None and s["batas"] <= b_bridge["batas"]:
        tstar = t
tstar = tstar if tstar is not None else .05
print(f"==> t* = {tstar}  (terkecil dgn batas <= BRIDGE {b_bridge['batas']}%)")

# evaluasi dengan t* tetap
print("\nEVALUASI (t* tidak diubah):")
rows = []
for nm, s, lb, p1 in (("dev 1.000 baris wF1", U[U.TOTAL == 1000], "dataset", False),
                      ("dev 6.000 baris wF1", U[U.TOTAL == 6000], "dataset", False),
                      ("baru wF1", Mt[Mt.metric == "wF1"], "dataset", False),
                      ("baru balanced accuracy", Mt[Mt.metric == "bacc"], "dataset", False),
                      ("baru macro-F1", Mt[Mt.metric == "macroF1"], "dataset", False),
                      ("populasi Q wF1 (Prop.1)", Q, "dataset", True)):
    vb = stats(s, s.lo95 > .005, lb, p1); vv = stats(s, (s.cfg != "None") & (s.dpool > tstar), lb, p1)
    rows.append(dict(setting=nm, aturan="BRIDGE", **vb)); rows.append(dict(setting=nm, aturan=f"VALMARGIN t*={tstar}", **vv))
pd.set_option("display.width", 220)
print(pd.DataFrame(rows).to_string(index=False))

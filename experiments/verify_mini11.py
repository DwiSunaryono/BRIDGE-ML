"""Verifikasi penuh MINI-11 (PRAREG_MINI11.md).
1. Kelengkapan dan duplikat. 2. Kriteria primer (R0,R1,R2,U) dengan dua implementasi KL
(brentq verify_exp41 vs bisection verify_exp41_independent), harus identik.
3. Replay determinisme 30 trial acak (seed 777), semua kolom identik dengan CSV.
4. Delta wF1 referensi untuk kandidat union dihitung ulang dengan sklearn.f1_score (label mentah).
Keluar non-nol bila ada yang gagal. Mencetak status praregistrasi."""
import ast, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
sys.argv = [sys.argv[0], "1", "1", "x.csv", "1003000000"] + sys.argv[1:]
import verify_exp41 as V1
import verify_exp41_independent as V2
import mini11_confirm as M
import exp41_gate_confirm as G
FAILS = []


def ok(n, c, i=""):
    print(("OK  " if c else "GAGAL ") + n + (f"  [{i}]" if i else ""))
    if not c:
        FAILS.append(n)


def crit(d, up):
    dss = list(d.dataset.unique()); T = 100; NT = T * len(dss)
    a = d[d.lo95 > 0.005]; n, k = len(a), int((a.ref <= 0).sum())
    r1 = up(k, n, NT, 1)
    loco = {x: up(int((a[a.dataset != x].ref <= 0).sum()), int((a.dataset != x).sum()), NT - T, 1) for x in dss}
    pw = {x: up(0, int((a.dataset != x).sum()), NT - T, 1) for x in dss}
    yds = a[a.ref > 0].groupby("dataset").size() / T
    return dict(n=n, k=k, R1=r1, loco=loco, R0=a.dataset.nunique() >= 5 and max(pw.values()) < .1,
                U=(a.ref > 0).sum() / NT >= .05 and int((yds >= .05).sum()) >= 3,
                yield_=(a.ref > 0).sum() / NT, nds5=int((yds >= .05).sum()), contrib=a.dataset.nunique())


def main():
    d = pd.read_csv(HERE / "out" / "mini11_confirm.csv", float_precision="round_trip",
                    keep_default_na=False, na_values={c: ["nan", "NaN"] for c in ("lo99", "lo95", "ref")})
    ok("kelengkapan 34 ds x 100 x 5 desain, tanpa duplikat",
       len(d) == 17000 and not d.duplicated(["dataset", "rep", "design"]).any())
    u = d[d.design == "union"]
    c1, c2 = crit(u, V1.kl_ratio_up), crit(u, V2.ratio_up)
    ok("brentq == bisection", abs(c1["R1"] - c2["R1"]) < 1e-9 and all(abs(c1["loco"][x] - c2["loco"][x]) < 1e-9 for x in c1["loco"]))
    w = max(c1["loco"], key=c1["loco"].get)
    print(f"PRIMER P_union: n={c1['n']} k={c1['k']} | R0={c1['R0']} | R1 {c1['R1']*100:.2f}% | LOCO maks "
          f"{c1['loco'][w]*100:.2f}% ({w}) | U={c1['U']} yield {c1['yield_']*100:.1f}/100, ds>=5: {c1['nds5']}, ds menyumbang {c1['contrib']}")
    passed = c1["R0"] and c1["R1"] < .1 and c1["loco"][w] < .1 and c1["U"]
    print("STATUS PRAREGISTRASI:", "MEMENUHI SEMUA KRITERIA" if passed else "TIDAK MEMENUHI")
    for des in ("thr_HGB", "ens_DT", "ensthr_DT", "ens3_SVC"):
        s = d[(d.design == des) & (d.lo95 > .005)]; n, k = len(s), int((s.ref <= 0).sum())
        print(f"  deskriptif {des}: {n}/{k} salah, KL {V1.kl_ratio_up(k, n, 3400, 1)*100:.1f}%, yield {(s.ref>0).sum()/3400*100:.1f}")
    trials = d[["dataset", "rep"]].drop_duplicates().sample(30, random_state=777).values
    mism = f1bad = 0
    for ds, rep in trials:
        rows = {r["design"]: r for r in M.trial((ds, M.DS[ds], int(rep)))}
        for des, r in rows.items():
            c = d[(d.dataset == ds) & (d.rep == rep) & (d.design == des)].iloc[0]
            for col in ("D", "lo99", "lo95", "ref", "kelas", "cfg"):
                a_, b_ = c[col], r[col]
                if not (a_ == b_ or (isinstance(b_, float) and abs(a_ - b_) < 1e-12)):
                    mism += 1
        cfg = rows["union"]["cfg"]
        if cfg != "None":
            Xd, yd, Xr, yr = M.data(M.DS[ds]); seed = M.OFF + 1000 * list(M.DS).index(ds) + int(rep)
            Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=G.TOTAL_TRAIN, stratify=yd, random_state=seed)
            Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=G.SEARCH_SIZE, stratify=ytr, random_state=seed)
            Xs, ys = Xs.reset_index(drop=True), np.asarray(ys)
            rr = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
            fn = {m: [] for m in M.MODELS}
            for _ in range(G.B_BAGS):
                idx = rr.choice(len(ys), size=sub, replace=False)
                if np.bincount(ys[idx], minlength=2).min() < 10:
                    continue
                for m in M.MODELS:
                    fn[m].append(G.fit_proba_m(m, {}, Xs.iloc[idx].reset_index(drop=True), ys[idx], list(Xs.columns)))
            Pr = {m: np.column_stack([f(Xr) for f in fs]).mean(1) for m, fs in fn.items()}
            w_, t_ = eval(cfg.replace("np.float64", ""))
            pc = sum(wi * Pr[m] for wi, m in zip(w_, M.MODELS)); yr_ = np.asarray(yr)
            dsk = f1_score(yr_, (pc >= t_).astype(int), average="weighted") - f1_score(yr_, (Pr["HGB"] >= .5).astype(int), average="weighted")
            f1bad += abs(dsk - rows["union"]["ref"]) > 1e-12
    ok("replay 30 trial x 5 desain identik", mism == 0, f"{mism} selisih")
    ok("Δ wF1 referensi union == sklearn", f1bad == 0, f"{f1bad} beda")
    print("\n" + ("SEMUA CEK LULUS" if not FAILS else f"{len(FAILS)} GAGAL: {FAILS}"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

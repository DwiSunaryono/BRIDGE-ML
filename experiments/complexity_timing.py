"""Table 8 (complexity): wall-clock cost of BRIDGE against naive selection, and inference latency of the deployed predictor.
Inference latency belongs to the deployed predictor, not to BRIDGE; the gate adds no inference cost.
Reported setting: one process, OMP_NUM_THREADS=1, batch of 2,000 reference rows, CPU model printed below.
Run on an otherwise idle machine: python complexity_timing.py -> out/complexity_timing.csv + ../paper_assets/tab8_complexity.tex
Measured per trial (single core, OMP_NUM_THREADS=1): fitting the bagged base models on SEARCH, predicting POOL, naive
selection over the 173 candidates, the BRIDGE gate (3-fold cross-fitted selection + 20,000 bootstrap replicates), and
inference latency per row for the baseline (32 HGB bags) and for a full three-model ensemble (96 bagged models).
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
_argv = sys.argv; sys.argv = sys.argv[:1]
import mini21_final as F
sys.argv = _argv
import exp41_gate_confirm as G, bridge_metrics as BM
from mini12_optuna import union, lab

U = union(("HGB", "DT", "SVC")); rows = []
import platform, subprocess
print('CPU:', subprocess.run(['sh', '-c', "lscpu | grep 'Model name'"], capture_output=True, text=True).stdout.strip() or platform.processor())
for ds in ("MIC", "Marketing_Campaign", "Census-Income-KDD", "Homesite", "Santander2016"):
    for rep in range(3):
        seed = F.OFF + 1000 * list(F.DS).index(ds) + rep
        Xd, yd, Xr, yr = F.data(ds)
        TOT, SS = (G.TOTAL_TRAIN, G.SEARCH_SIZE) if len(Xd) >= G.TOTAL_TRAIN + 50 else (1000, 259)
        Xtr, _, ytr, _ = train_test_split(Xd, yd, train_size=TOT, stratify=yd, random_state=seed)
        Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
        Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)
        rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(G.SUBSAMPLE_FRAC * len(ys))
        bags = [i for i in (rng.choice(len(ys), size=sub, replace=False) for _ in range(G.B_BAGS)) if np.bincount(ys[i], minlength=2).min() >= 10]
        t = time.perf_counter(); fns = {m: [G.fit_proba_m(m, {}, Xs.iloc[i].reset_index(drop=True), ys[i], list(Xs.columns)) for i in bags] for m in ("HGB", "DT", "SVC")}; t_fit = time.perf_counter() - t
        t = time.perf_counter(); P = {m: np.column_stack([f(Xp) for f in fs]).mean(1) for m, fs in fns.items()}; t_pool = time.perf_counter() - t
        base = (P["HGB"] >= .5).astype(int)
        t = time.perf_counter(); F.pick(yp, P, np.arange(len(yp)), U, base, "wF1"); t_naive = time.perf_counter() - t
        t = time.perf_counter(); cnt = np.zeros(8, int)
        for a, b in StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp):
            c = F.pick(yp, P, a, U, base, "wF1"); cnt += G.counts8(yp[b], base[b], base[b] if c is None else lab(P, c, b))
        t_cf = time.perf_counter() - t; t = time.perf_counter()
        np.quantile(BM.delta(np.random.default_rng(seed + 61).multinomial(len(yp), cnt / len(yp), size=20000), "wF1"), .05)
        t_boot = time.perf_counter() - t; t_gate = t_cf + t_boot
        Xb = Xr.iloc[:2000]
        t = time.perf_counter(); np.column_stack([f(Xb) for f in fns["HGB"]]).mean(1); t_inf_b = time.perf_counter() - t
        t = time.perf_counter(); [np.column_stack([f(Xb) for f in fs]).mean(1) for fs in fns.values()]; t_inf_e = time.perf_counter() - t
        rows.append(dict(dataset=ds, rep=rep, n_search=len(ys), n_pool=len(yp), n_features=Xs.shape[1], bags=len(bags), fit_s=t_fit, predict_pool_s=t_pool,
                         naive_select_s=t_naive, crossfit_select_s=t_cf, bootstrap_s=t_boot, bridge_gate_s=t_gate, infer_baseline_ms_per_row=1000 * t_inf_b / len(Xb),
                         infer_ensemble_ms_per_row=1000 * t_inf_e / len(Xb)))
        print(rows[-1], flush=True)
d = pd.DataFrame(rows); d.to_csv(HERE / "out" / "complexity_timing.csv", index=False)
g = d.groupby("dataset", sort=False).agg(n_pool=("n_pool", "first"), features=("n_features", "first"), fit=("fit_s", "mean"), naive=("naive_select_s", "mean"),
                                         cf=("crossfit_select_s", "mean"), boot=("bootstrap_s", "mean"),
                                         gate=("bridge_gate_s", "mean"), inf_b=("infer_baseline_ms_per_row", "mean"), inf_e=("infer_ensemble_ms_per_row", "mean")).reset_index()
g["gate % of fit"] = 100 * g.gate / g.fit
g.columns = ["Dataset", "POOL rows", "Features", "Fit models (s)", "Naive selection (s)", "Cross-fitted selection (s)", "Bootstrap (s)", "BRIDGE gate total (s)", "Inference, baseline: 32 HGB bags (ms/row)", "Inference, 3-model ensemble: 96 bagged models (ms/row)", "Gate / fit (%)"]
sys.path.insert(0, str(HERE)); import paper_assets as PA  # LaTeX writer
PA.tex(g.round(3), "tab8_complexity", "Wall-clock cost (mean of 3 trials, one CPU core): model fitting, naive selection, the BRIDGE gate and inference latency.", "tab:complexity")
print(g.round(3).to_string(index=False))

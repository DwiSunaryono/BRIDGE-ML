"""Table C (PRAREG_TABEL_C.md): the bridge-gate program, default procedure, run once per new OpenML dataset.

Stages:
  screen <ids_file> <out.csv> [n_proc]
      downloads each candidate, codes it as the paper's loaders do, and applies the paper's data-based exclusion rules:
      duplicate rows > 25%, class ratio exactly 50% (two classes), one feature that almost perfectly predicts the target
      (AUC >= 0.99, one-vs-rest for more classes), and coverage (two classes: >= 1,050 rows after the reference; more
      classes: >= 2,480 and K <= 26). No model is fitted.
  run <frozen_list.csv> <out.csv> <n_proc> [i/n]
      for every dataset of the frozen list (column data_id, in list order): locked reference by the paper's rule and
      seed, then `python -m bridge_gate --dataset dev.pkl --reference ref.pkl --seed S --out ...` in its own process.
      One row per dataset is appended to out.csv as soon as it is known; a rerun skips finished datasets.
Seed of the dataset at position p of the frozen list: 1_043_000_000 + 1000 * p."""
import csv, json, os, subprocess, sys, tempfile, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
LOCKED_REF_SEED, OFF = 20261125, 1_043_000_000
CACHE = Path.home() / "tabelC_cache"


def ref_size(n):                               # the paper's rule (exp41_gate_confirm.ref_size)
    return min(n - 20000, 100000) if n >= 40000 else min(8000, int(0.3 * n))


def load(did):
    """The paper's coding: target -> codes of the sorted labels; text columns -> codes of the sorted values (missing is
    its own value); every column numeric. Rows with a missing target are dropped (counted)."""
    CACHE.mkdir(exist_ok=True); p = CACHE / f"{did}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=int(did), as_frame=True, parser="auto")
    X, y = d.data.copy(), d.target
    if isinstance(y, pd.DataFrame):
        raise ValueError("more than one target column")
    keep = y.notna().to_numpy(); n_missing_y = int((~keep).sum())
    X, y = X[keep].reset_index(drop=True), y[keep].astype(str).reset_index(drop=True)
    y = pd.Series(pd.Categorical(y, categories=sorted(y.unique())).codes, name="target")
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            s = X[c].astype(object).where(X[c].notna(), "__MISSING__").astype(str)
            X[c] = pd.Categorical(s, categories=sorted(s.unique())).codes
        X[c] = pd.to_numeric(X[c], errors="coerce")
    out = (X, y, n_missing_y); pd.to_pickle(out, p)
    return out


def screen_one(did):
    from sklearn.metrics import roc_auc_score
    try:
        X, y, nmy = load(did)
    except Exception as e:
        return dict(data_id=did, status="ERROR load: " + str(e)[:120])
    n, K = len(y), int(y.max()) + 1; cc = np.bincount(y, minlength=K)
    dup = float(pd.concat([X, y], axis=1).duplicated().mean())
    left = n - ref_size(n)
    best_auc = 0.0; best_col = ""
    Z = X.fillna(X.median(numeric_only=True)).fillna(0)
    rng = np.random.default_rng(0); idx = rng.choice(n, size=min(n, 50000), replace=False)   # AUC on a fixed sample
    for c in Z.columns:
        v = Z[c].to_numpy()[idx]; t = y.to_numpy()[idx]
        if np.unique(v).size < 2:
            continue
        try:
            if K == 2:
                a = roc_auc_score(t, v); a = max(a, 1 - a)
            else:
                a = float(np.mean([max(u, 1 - u) for u in (roc_auc_score(t == k, v) for k in range(K) if 0 < (t == k).sum() < len(t))]))
        except ValueError:
            continue
        if a > best_auc:
            best_auc, best_col = a, str(c)
    reasons = []
    if dup > 0.25:
        reasons.append(f"duplicates {dup:.1%} > 25%")
    if K == 2 and cc[0] == cc[1]:
        reasons.append("class ratio exactly 50%")
    if best_auc >= 0.99:
        reasons.append(f"one feature predicts the target (AUC {best_auc:.3f}, {best_col})")
    if K < 2 or cc.min() == 0:
        reasons.append("fewer than two classes")
    if K == 2 and left < 1050:
        reasons.append(f"outside the coverage: {left} rows after the reference (< 1,050)")
    if K > 2 and left < 2480:
        reasons.append(f"outside the coverage: {left} rows after the reference (< 2,480)")
    if K > 26:
        reasons.append(f"outside the coverage: {K} classes (> 26)")
    return dict(data_id=did, status="KEEP" if not reasons else "OUT", reasons="; ".join(reasons), N=n, K=K, F=X.shape[1],
                minority=float(cc.min() / n), duplicates=round(dup, 4), max_single_auc=round(best_auc, 4), missing_target=nmy)


def run_one(job):
    pos, did = job; seed = OFF + 1000 * pos; t0 = time.time()
    try:
        from sklearn.model_selection import train_test_split
        X, y, _ = load(did)
        Xd, Xr, yd, yr = train_test_split(X, y, test_size=ref_size(len(y)), stratify=y, random_state=LOCKED_REF_SEED)
        with tempfile.TemporaryDirectory() as tmp:
            for nm, A, b in (("dev", Xd, yd), ("ref", Xr, yr)):
                df = A.reset_index(drop=True).copy(); df["__target__"] = np.asarray(b); df.to_pickle(f"{tmp}/{nm}.pkl")
            r = subprocess.run([sys.executable, "-m", "bridge_gate", "--dataset", f"{tmp}/dev.pkl", "--reference", f"{tmp}/ref.pkl",
                                "--target", "__target__", "--seed", str(seed), "--metric", "wF1", "--no-progress",
                                "--out", f"{tmp}/res"], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1][:200])
            j = json.load(open(f"{tmp}/res.json"))
        return dict(pos=pos, data_id=did, seed=seed, status="ok", K=j["K"], rows=j["rows"], total=j["total"], bags=j["bags"],
                    validated=j["validated"], chosen=j["chosen_config"], accept=j["accept"], naive_accept=j["naive_accept"],
                    lower_bound=repr(j["lower_bound"]), gain_pool=repr(j["gain_pool_chosen"]), gain_ref=repr(j["gain_ref"]),
                    ref_lo=repr(j["ref_interval"][0]), ref_hi=repr(j["ref_interval"][1]), ref_class=j["ref_class"],
                    ref_rows=j["reference_rows"], seconds=round(time.time() - t0, 1))
    except Exception as e:
        return dict(pos=pos, data_id=did, seed=seed, status="ERROR " + str(e)[:200], seconds=round(time.time() - t0, 1))


COLS = ["pos", "data_id", "seed", "status", "K", "rows", "total", "bags", "validated", "chosen", "accept", "naive_accept",
        "lower_bound", "gain_pool", "gain_ref", "ref_lo", "ref_hi", "ref_class", "ref_rows", "seconds"]

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if sys.argv[1] == "screen":
        ids = [int(x) for x in open(sys.argv[2]).read().split()]; n_proc = int(sys.argv[4]) if len(sys.argv) > 4 else 4
        with Pool(n_proc) as pool:
            rows = list(pool.imap(screen_one, ids))
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(open(sys.argv[3], "w", newline=""), fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"screened {len(rows)}: keep {sum(r['status'] == 'KEEP' for r in rows)}")
    elif sys.argv[1] == "run":
        lst = [int(r["data_id"]) for r in csv.DictReader(open(sys.argv[2]))]
        out, n_proc = Path(sys.argv[3]), int(sys.argv[4])
        jobs = list(enumerate(lst))
        if len(sys.argv) > 5:
            i, n = map(int, sys.argv[5].split("/")); jobs = jobs[i - 1::n]
        done = {int(r["data_id"]) for r in csv.DictReader(open(out))} if out.exists() else set()
        todo = [j for j in jobs if j[1] not in done]
        print(f"{len(jobs)} datasets in this part, {len(jobs) - len(todo)} already done", flush=True)
        new = not out.exists()
        with open(out, "a", newline="") as f, Pool(n_proc, maxtasksperchild=1) as pool:
            w = csv.DictWriter(f, fieldnames=COLS, restval="");
            if new:
                w.writeheader()
            for r in pool.imap_unordered(run_one, todo):
                w.writerow(r); f.flush(); print(f"{r['data_id']}: {r['status']} {r.get('accept', '')} {r['seconds']} s", flush=True)

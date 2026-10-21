"""Appendix table: exact data partition per dataset/task.
Uses the same loaders and size rules as the experiments. Output: out/dataset_splits.csv (+ LaTeX via paper_assets in the repo).
    python dataset_splits.py
Columns: group, dataset, source id, N (after loader cleaning), features, minority %, reference rows (locked), development rows,
training rows used (SEARCH + POOL), SEARCH rows, POOL rows, trials."""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
_argv = sys.argv; sys.argv = sys.argv[:1]
import exp41_gate_confirm as G
import mini21_final as F
import mini18_multiclass as MC
import health_data as H, health_data_b as HB, health_data_c as HC
sys.argv = _argv

rows = []


def add(group, name, src, X, y, trials):
    n = len(X); nref = G.ref_size(n); ndev = n - nref
    tot, ss = (G.TOTAL_TRAIN, G.SEARCH_SIZE) if ndev >= G.TOTAL_TRAIN + 50 else (1000, 259)
    vc = pd.Series(y).value_counts()
    rows.append(dict(group=group, dataset=name, source=src, N=n, features=X.shape[1], minority_pct=round(100 * vc.min() / n, 2),
                     reference=nref, development=ndev, train_used=tot, search=ss, pool=tot - ss, trials=trials))
    print(rows[-1], flush=True)


man = pd.concat([pd.read_csv(HERE / f) for f in ("exp40_manifest/datasets.csv", "exp41_manifest/datasets.csv")]).drop_duplicates("dataset")
for r in man.itertuples():
    X, y = G.load_openml_binary(int(r.data_id)); add("confirmation (34)", r.dataset, f"OpenML {r.data_id}", X, y, 100)
for name, (src, i, drop) in F.DS.items():
    X, y = F.load(src, i, drop); add("new binary (11)", name, f"{'OpenML' if src == 'openml' else 'UCI'} {i}", X, y, 100)
for name, did in {"Fashion-MNIST": 40996, "DryBean": "uci602", "USPS": 41082, "mnist_784": 554, "Kuzushiji-MNIST": 41982}.items():
    if isinstance(did, str):
        df = pd.read_pickle(Path.home() / "uci_cache" / "uci_602.pkl"); y = df.pop("target"); X = df
    else:
        X, y = MC.load(did)
    add("new multiclass (5)", name, str(did), X, y, 60)
for name in ["mammography", "actg175"] + list(H.NH_TGT) + list(H.BR_TGT):
    X, y = H.load(name); add("clinical (12 tasks)", name, H.KLASTER[name], X, y, 200)
X, y = HB.load("covid_mx"); add("clinical (12 tasks)", "covid_mx", "covid_mx", X, y, 200)
X, y = HC.load("wids_icu"); add("clinical (12 tasks)", "wids_icu", "wids", X, y, 200)
pd.DataFrame(rows).to_csv(HERE / "out" / "dataset_splits.csv", index=False)
print("tersimpan out/dataset_splits.csv", len(rows), "baris")

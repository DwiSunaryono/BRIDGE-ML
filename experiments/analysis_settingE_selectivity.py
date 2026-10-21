"""Random-selection control and gain-loss reanalysis for the OpenML collection (E), same definitions as
Table selectivity (post hoc, exploratory). Input: settingE_laptop.csv + settingE_part2_laptop.csv."""
import sys, numpy as np, pandas as pd
from scipy.stats import hypergeom
base = sys.argv[1] if len(sys.argv) > 1 else "."
d = pd.concat([pd.read_csv(f"{base}/settingE_laptop.csv"), pd.read_csv(f"{base}/settingE_part2_laptop.csv")]).drop_duplicates(["data_id", "rep"])
d = d[d.status == "ok"].copy()
d["h"] = d.gain_ref <= 0; d["na"] = d.naive_accept.astype(bool); d["ba"] = d.accept.astype(bool)
def stats(x):
    g = x.gain_ref
    net_n = g[x.na].sum(); net_b = g[x.ba].sum()
    loss_n = -g[x.na & (g < 0)].sum(); loss_b = -g[x.ba & (g < 0)].sum()
    harm_n = (x.na & x.h).sum(); harm_b = (x.ba & x.h).sum()
    return dict(kept=100 * net_b / net_n, harm_removed=100 * (1 - harm_b / harm_n) if harm_n else np.nan,
                loss_removed=100 * (1 - loss_b / loss_n) if loss_n else np.nan,
                gpu_b=g[x.ba].mean(), gpu_n=g[x.na].mean())
s = stats(d)
N, K, n, k = d.na.sum(), (d.na & d.h).sum(), d.ba.sum(), (d.ba & d.h).sum()
print(f"decisions {len(d)}  naive {N} ({100*K/N:.1f})  BRIDGE {n} ({100*k/n:.1f})")
print(f"random draw: expected {n*K/N:.1f} vs {k}, p = {hypergeom.cdf(k, N, K, n):.2e}")
# within each dataset: draw BRIDGE's per-dataset count from that dataset's naive acceptances
rng = np.random.default_rng(20261020); sims = np.zeros(100000)
exp_w = 0.0
for _, x in d.groupby("data_id"):
    Nd, Kd, nd = x.na.sum(), (x.na & x.h).sum(), x.ba.sum()
    if nd == 0 or Nd == 0: continue
    exp_w += nd * Kd / Nd; sims += rng.hypergeometric(Kd, Nd - Kd, nd, size=sims.size)
print(f"within datasets: expected {exp_w:.1f} vs {k}, p = {(sims <= k).mean():.4f}")
ids = d.data_id.unique(); B = []
for _ in range(10000):
    smp = pd.concat([d[d.data_id == i] for i in rng.choice(ids, len(ids))]); B.append(stats(smp))
B = pd.DataFrame(B)
for c in ["kept", "harm_removed", "loss_removed"]:
    lo, hi = np.nanpercentile(B[c], [2.5, 97.5]); print(f"{c}: {s[c]:.0f} [{lo:.0f}; {hi:.0f}]")
print(f"gain per update BRIDGE/naive: {s['gpu_b']:.4f} / {s['gpu_n']:.4f}  ratio {s['gpu_b']/s['gpu_n']:.1f}")

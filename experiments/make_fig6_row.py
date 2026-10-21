"""Candidate-space figure as one row of five panels ((E) from MINI-52) for a full-width (figure*) placement; same data and styling as paper_assets.py."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os
HERE = Path(__file__).resolve().parent; OUT = Path(os.environ.get("BRIDGE_ASSETS_OUT", HERE.parent / "paper_assets"))
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9, "axes.labelsize": 8})
def wilson(k, n, z=1.96):
    """Same Wilson 95% interval as paper_assets.py, in percent."""
    k, n = np.asarray(k, float), np.asarray(n, float); p = np.where(n > 0, k / np.maximum(n, 1), 0.0)
    den = 1 + z ** 2 / np.maximum(n, 1); c = (p + z ** 2 / (2 * np.maximum(n, 1))) / den
    h = z * np.sqrt(p * (1 - p) / np.maximum(n, 1) + z ** 2 / (4 * np.maximum(n, 1) ** 2)) / den
    return 100 * (c - h), 100 * (c + h)
k = pd.read_csv(HERE / "out" / "analisis_mini35.csv"); k51 = pd.concat([pd.read_csv(HERE / "out" / "analysis_mini51.csv"), pd.read_csv(HERE / "out" / "analysis_mini52.csv")], ignore_index=True)
fig, axs = plt.subplots(1, 5, figsize=(7.2, 2.1), sharey=False)
style = [("Naive (point, same POOL)", "#999999", 1), ("Bootstrap + margin, same POOL", "#e69138", 1), ("BRIDGE (cross-fitted)", "#1f4e79", 1.6)]
for ax, (d, nm) in zip(axs[:2], (("nyata", "(A) 11 new"), ("Q", "(B) Synthetic"))):
    for rule, (lbl, c, lw) in zip(("naif", "POOL-boot 0,005", "BRIDGE"), style):
        s = k[(k.data == d) & (k.aturan == rule)].sort_values("nK"); lo, hi = wilson(s.salah, s.terima)
        ax.errorbar(s.nK, s.pct_salah, yerr=[s.pct_salah - lo, hi - s.pct_salah], marker="o", ms=3, label=lbl, color=c, lw=lw, capsize=2, elinewidth=.6)
    ax.set_title(nm)
for ax, (st, nm) in zip(axs[2:], (("C", "(C) 12 clinical"), ("D", "(D) 34 binary"), ("E", "(E) 11 binary"))):
    for rule, (lbl, c, lw) in zip(("naive", "same_pool_boot_005", "BRIDGE"), style):
        s = k51[(k51.setting == st) & (k51.rule == rule)].sort_values("nK")
        ax.errorbar(s.nK, s.harm_pct, yerr=[s.harm_pct - s.harm_ci_lo, s.harm_ci_hi - s.harm_pct], marker="o", ms=3, label=lbl, color=c, lw=lw, capsize=2, elinewidth=.6)
    ax.set_title(nm)
for i, ax in enumerate(axs):
    ax.axhline(10, color="r", lw=.6, ls="--"); ax.set_xscale("log"); ax.set_xlabel("$|\\mathcal{C}|$" if i != 2 else "Candidate-space size $|\\mathcal{C}|$")
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7)
axs[0].set_ylabel("False improvements (%)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(OUT / "fig6_candidate_space_size.pdf", bbox_inches="tight"); print("saved")

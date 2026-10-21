"""Download the two UCI datasets used in the paper (Rice, Dry Bean) into ~/uci_cache/ as pickles.

The experiment scripts read them from there (mini21_final.load, mini22_mc_final.load).
Usage: python fetch_uci.py
"""
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

OUT = Path.home() / "uci_cache"
OUT.mkdir(exist_ok=True)
for uci_id in (545, 602):  # 545 = Rice (Cammeo and Osmancik), 602 = Dry Bean
    d = fetch_ucirepo(id=uci_id)
    df = pd.concat([d.data.features, d.data.targets.iloc[:, 0].astype(str).rename("target")], axis=1)
    df.to_pickle(OUT / f"uci_{uci_id}.pkl")
    print(f"saved {OUT / f'uci_{uci_id}.pkl'}  ({len(df)} rows)")

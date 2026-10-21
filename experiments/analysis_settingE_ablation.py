"""Ablation cells computable for the OpenML collection (E) from the stored decision statistics (post hoc).
gain_pool = same-POOL point estimate; lower_bound = cross-fitted bootstrap lower bound (5%); deployed configuration fixed.
Cells needing a same-POOL bootstrap or a cross-fitted point estimate are not stored by the released program."""
import numpy as np, pandas as pd
import analysis_setting_e as E
import analysis_mini44 as A44
d = E.read_parts("settingE_").reset_index(drop=True)
harm = (d.gain_ref <= 0).to_numpy(); ds = d.data_id.astype(str).to_numpy()
cells = {"Naive": d.gain_pool > 0, "Fixed margin": d.gain_pool > 0.005,
         "BRIDGE without margin": d.lower_bound > 0, "BRIDGE": d.lower_bound > 0.005}
assert ((d.lower_bound > 0.005) == d.accept.astype(bool)).all() and ((d.gain_pool > 0) == d.naive_accept.astype(bool)).all()
for k, a in cells.items():
    a = a.to_numpy(); c = A44.crit(a, harm, ds)
    print(f"{k:22s} accepted {a.sum():5d} harmful {(a & harm).sum():4d} ({100*(a&harm).sum()/max(a.sum(),1):5.2f}%)  "
          f"bound {c['kl']:.2f}  LOCO {c['loco']:.2f}  meets {c['meets']}")

"""Two-sided paired cluster-bootstrap p for BRIDGE minus the naive rule on (E) (14 datasets): the same resamples as
analysis_setting_e.py (analysis_mini44.boot, 10,000 resamples, seed 20261024); p = 2 x min(share >= 0, share <= 0),
as in analysis_paired_ci.py. Single comparison, so no Holm adjustment."""
import numpy as np
import analysis_setting_e as E
import analysis_mini44 as A44
x = E.read_parts("settingE_").merge(E.metadata("settingE_daftar_beku.csv"), on=["data_id", "K"], validate="many_to_one")
x["dataset"] = x.data_id.astype(str) + ":" + x.name
x = x.reset_index(drop=True)
acc, nv, harm, _ = E.basic_counts(x)
est, lo, hi, out = A44.boot(x, lambda s: A44.hp(acc, harm)(s) - A44.hp(nv, harm)(s))
p = min(1.0, 2 * min((out >= 0).mean(), (out <= 0).mean()))
print(f"E BRIDGE minus naive: {est:.2f} pp [{lo:.2f}; {hi:.2f}], resamples {len(out)}, p = {p:.4f}")

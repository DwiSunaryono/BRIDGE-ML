"""Analysis of the second OpenML collection (E2) and the Volkert supplement (Supplementary Appendix G).

E2 is the frozen list of 19 datasets (settingE2_daftar_beku.csv) after amendment 1, which replaces shuttle (60 trials
that stop with a structural error) by pendigits (settingE2_amendment1_daftar_beku.csv); 100 trials per binary dataset
and 60 per multiclass dataset, 1,700 decisions. The Volkert supplement adds one multiclass dataset with 60 trials.
Inputs (per-trial outputs, merged from the shards of the distributed run; completeness in the audit files):
  out/settingE2_FINAL_AMENDED_1700.csv, out/settingE2_FINAL_AUDIT.json  (E2 after the amendment)
  out/settingE2_tabular_hasil.csv, out/settingE2_tabular_AUDIT.json     (Volkert)
  out/settingE_laptop.csv, out/settingE_part2_laptop.csv                (collection E, for the pooled E+E2 rows)
Each group is analysed with analysis_setting_e.analyze_group (criterion, bound, LOCO, yield, cluster bootstrap).
Usage: python analysis_setting_e2.py -> out/analysis_settingE2.csv, out/analysis_settingE2_trial0.csv"""
import json
from pathlib import Path

import pandas as pd

import analysis_setting_e as ASE

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def audit_ready(path, key=None):
    obj = json.loads(path.read_text())
    if key:
        obj = obj[key]
    return obj.get("analysis_ready") is True


def prepare(path, prefix):
    d = pd.read_csv(path, float_precision="round_trip")
    if not (d.status == "ok").all():
        raise RuntimeError(f"non-ok row in {path}")
    if d.duplicated(["source_id", "rep"]).any():
        raise RuntimeError(f"duplicate source_id/rep in {path}")
    d["dataset"] = prefix + ":" + d.source_id.astype(str) + ":" + d.name.astype(str)
    d["K"] = d.K.astype(int)
    return d


def prepare_e():
    parts = [OUT / "settingE_laptop.csv", OUT / "settingE_part2_laptop.csv"]
    d = pd.concat([pd.read_csv(p, float_precision="round_trip") for p in parts], ignore_index=True)
    if len(d) != 1280 or not (d.status == "ok").all() or d.duplicated(["data_id", "rep"]).any():
        raise RuntimeError("collection E is not the frozen complete 1,280-row set")
    d["dataset"] = "E:" + d.data_id.astype(str)
    d["K"] = d.K.astype(int)
    return d


def groups(d, stem):
    return [ASE.analyze_group(d, f"{stem}_all"),
            ASE.analyze_group(d[d.K == 2], f"{stem}_binary"),
            ASE.analyze_group(d[d.K > 2], f"{stem}_multiclass")]


def main():
    if not audit_ready(OUT / "settingE2_FINAL_AUDIT.json", "amended_replacing_shuttle_with_pendigits"):
        raise SystemExit("E2 audit is not analysis-ready")
    if not audit_ready(OUT / "settingE2_tabular_AUDIT.json"):
        raise SystemExit("Volkert audit is not analysis-ready")
    e = prepare_e()
    e2 = prepare(OUT / "settingE2_FINAL_AMENDED_1700.csv", "E2")
    volkert = prepare(OUT / "settingE2_tabular_hasil.csv", "E2TAB")
    if len(e2) != 1700 or e2.dataset.nunique() != 19:
        raise RuntimeError("E2 cardinality mismatch")
    if len(volkert) != 60 or volkert.dataset.nunique() != 1:
        raise RuntimeError("Volkert cardinality mismatch")

    confirm = pd.concat([e, e2], ignore_index=True)
    exploratory = pd.concat([e, e2, volkert], ignore_index=True)
    no_pendigits = confirm[confirm.name.fillna("") != "pendigits"].copy()
    no_pendigits_svhn = no_pendigits[no_pendigits.name.fillna("") != "SVHN_small"].copy()
    rows = groups(e2, "E2") + groups(confirm, "E_plus_E2")
    rows += [ASE.analyze_group(no_pendigits, "E_plus_E2_without_pendigits_all")]
    rows += [ASE.analyze_group(no_pendigits[no_pendigits.K > 2], "E_plus_E2_without_pendigits_multiclass")]
    rows += [ASE.analyze_group(no_pendigits_svhn[no_pendigits_svhn.K > 2], "E_plus_E2_without_pendigits_SVHN_multiclass")]
    rows += [ASE.analyze_group(volkert, "Volkert_only", criterion=False)]
    rows += groups(exploratory, "E_plus_E2_plus_Volkert_exploratory")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "analysis_settingE2.csv", index=False)

    trial0 = pd.DataFrame([ASE.trial_zero(e2, "E2_trial0"), ASE.trial_zero(confirm, "E_plus_E2_trial0"),
                           ASE.trial_zero(volkert, "Volkert_trial0"),
                           ASE.trial_zero(exploratory, "E_plus_E2_plus_Volkert_trial0_exploratory")])
    trial0.to_csv(OUT / "analysis_settingE2_trial0.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[["group", "datasets", "decisions", "bridge_accepted", "bridge_harmful", "bridge_kl", "bridge_loco",
               "bridge_meets"]].to_string(index=False))


if __name__ == "__main__":
    main()

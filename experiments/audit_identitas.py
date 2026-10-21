"""Audit identitas per sumber klinis untuk memeriksa kebocoran per pasien. Melaporkan kunci identitas yang tersedia
di berkas sumber, jumlah baris, jumlah kunci unik, dan duplikat. Dedup dan filter dilakukan di loader sebelum
pembagian data."""
from pathlib import Path
import pandas as pd
H = Path.home(); rows = []
def add(src, key, n, nu, note): rows.append(dict(source=src, identity_key=key, rows=n, unique_keys=nu, duplicates=n - nu if nu != "" else "", note=note))
w = pd.read_csv(H / "health_cache/kaggle/wids/training_v2.csv", usecols=["patient_id", "encounter_id", "hospital_id"])
add("WiDS 2020 (GOSSIS)", "patient_id", len(w), w.patient_id.nunique(), f"one row per patient; unique encounter_id: {w.encounter_id.nunique():,}; hospitals: {w.hospital_id.nunique()}")
c = pd.read_csv(H / "health_cache/kaggle/covid_tanmoy/covid.csv", usecols=["id", "covid_res"])
add("COVID-19 Mexico", "id (record id)", len(c), c.id.nunique(), f"Duplicates = excess rows; ids occurring more than once: {c.id[c.id.duplicated(keep=False)].nunique():,}; all {int(c.id.duplicated(keep=False).sum()):,} rows with such ids removed before filtering confirmed cases; confirmed cases kept: {(c[~c.id.duplicated(keep=False)].covid_res == 1).sum():,}")
d = pd.read_sas(H / "health_cache/DEMO_J.XPT", format="xport", columns=["SEQN"]) if False else pd.read_sas(H / "health_cache/DEMO_J.XPT", format="xport")
add("NHANES 2017-2018", "SEQN", len(d), d.SEQN.nunique(), "one row per respondent (survey design); the 4 outcomes share respondents and form one LOCO cluster")
a = pd.read_pickle(H / "uci_cache/uci_890.pkl"); k = [x for x in a.columns if x.lower() in ("pidnum", "pid")]
add("ACTG 175 (UCI 890)", k[0] if k else "(not available in the UCI copy)", len(a), a[k[0]].nunique() if k else "", "randomized trial, one row per participant (source documentation)")
add("BRFSS 2015", "(not available; SEQNO is state-specific)", "", "", "one record per interviewed respondent (survey design); the 4 outcomes form one LOCO cluster")
add("Mammography (OpenML 310)", "(not available)", "", "", "identical (X, y) rows kept once; identical X with conflicting labels removed; patient independence cannot be verified")
df = pd.DataFrame(rows); df.to_csv(Path(__file__).resolve().parent / "out" / "audit_identitas.csv", index=False); print(df.to_string(index=False))

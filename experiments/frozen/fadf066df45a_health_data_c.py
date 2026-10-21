"""Loader WiDS Datathon 2020 (GOSSIS, data ICU asli) untuk MINI-28c — spesifikasi DITETAPKAN sebelum hasil apa pun.
Satu baris per pasien (patient_id unik). Target: hospital_death. Dibuang: encounter_id, patient_id, hospital_id, icu_id
(identitas), readmission_status (konstan), apache_4a_hospital_death_prob & apache_4a_icu_death_prob (PREDIKSI model lain,
bukan pengukuran pasien). Kolom teks -> kode kategori. Berkas: ~/health_cache/kaggle/wids/training_v2.csv. Klaster: wids."""
from pathlib import Path
import pandas as pd

F = Path.home() / "health_cache" / "kaggle" / "wids" / "training_v2.csv"
DROP = ["encounter_id", "patient_id", "hospital_id", "icu_id", "readmission_status",
        "apache_4a_hospital_death_prob", "apache_4a_icu_death_prob"]
KLASTER = {"wids_icu": "wids"}


def load(name):
    assert name == "wids_icu"
    d = pd.read_csv(F); y = d.pop("hospital_death").astype(int); d = d.drop(columns=DROP)
    for c in d.columns:
        if not pd.api.types.is_numeric_dtype(d[c]):
            s = d[c].astype(object).where(d[c].notna(), "__MISSING__").astype(str)
            d[c] = pd.Categorical(s, categories=sorted(s.unique())).codes
    return d.apply(pd.to_numeric, errors="coerce").reset_index(drop=True), pd.Series(y.values, name="target")

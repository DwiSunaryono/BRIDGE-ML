"""Loader dataset KESEHATAN tambahan untuk MINI-28b (spesifikasi DITETAPKAN sebelum melihat hasil apa pun).
  covid_mx : COVID-19 Meksiko (data terbuka Secretaria de Salud; salinan Kaggle tanmoyx/covid19-patient-precondition-dataset).
             Populasi: pasien TERKONFIRMASI positif (covid_res == 1). Target: meninggal (date_died != '9999-99-99').
             Fitur saat masuk saja: sex, patient_type (rawat jalan/inap saat triase), pneumonia, age, pregnancy, diabetes,
             copd, asthma, inmsupr, hypertension, other_disease, cardiovascular, obesity, renal_chronic, tobacco,
             contact_other_covid. Dibuang: id, semua tanggal, date_died (target), intubed & icu (pasca-masuk),
             covid_res (konstan setelah filter). Identitas rekaman = id: id yang muncul > 1 kali dibuang SEMUA.
Berkas: ~/health_cache/kaggle/covid_tanmoy/covid.csv. Klaster LOCO: covid_mx (sumber sendiri)."""
from pathlib import Path
import pandas as pd

HC = Path.home() / "health_cache" / "kaggle"
COVID_FEAT = ["sex", "patient_type", "pneumonia", "age", "pregnancy", "diabetes", "copd", "asthma", "inmsupr",
              "hypertension", "other_disease", "cardiovascular", "obesity", "renal_chronic", "tobacco", "contact_other_covid"]
KLASTER = {"covid_mx": "covid_mx"}


def load(name):
    assert name == "covid_mx"
    d = pd.read_csv(HC / "covid_tanmoy" / "covid.csv")
    d = d[~d.id.duplicated(keep=False)]
    d = d[d.covid_res == 1]
    y = (d.date_died.astype(str) != "9999-99-99").astype(int)
    X = d[COVID_FEAT].apply(pd.to_numeric, errors="coerce")
    return X.reset_index(drop=True), pd.Series(y.values, name="target")

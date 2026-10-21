"""Loader dataset KESEHATAN baru untuk MINI-28 (spesifikasi DITETAPKAN sebelum melihat hasil apa pun).

Aturan umum: target = diagnosis/kejadian yang dilaporkan (ya/tidak); jawaban "tidak tahu"/"menolak"/borderline
dikeluarkan (baris dibuang, bukan digabung); fitur hanya variabel yang tersedia SEBELUM/terpisah dari diagnosis
(demografi, antropometri, tanda vital, gaya hidup). Pertanyaan tentang obat, perawatan, atau diagnosis yang sama
tidak dipakai sebagai fitur.
  mammography   : OpenML 310. Baris duplikat (X,y) identik -> sisakan satu; X kembar dgn label bertentangan -> semua dibuang.
  actg175       : UCI 890 (AIDS Clinical Trials Group 175). Dibuang (tercatat setelah awal studi): time, cd420, cd820, offtrt.
  nhanes_*      : NHANES 2017-2018 (CDC), dewasa >= 20 th. Fitur: DEMO, BMX, BPX, SMQ020, ALQ111/121, PAQ.
                  Target: hipertensi BPQ020, diabetes DIQ010 (1/2 saja), asma MCQ010, PJK MCQ160C.
  brfss_*       : BRFSS 2015 (CDC). Fitur: demografi, BMI, merokok, alkohol, olahraga, kesehatan umum, akses layanan.
                  Target: stroke CVDSTRK3, asma ASTHMA3, ginjal CHCKIDNY, depresi ADDEPEV2.
Klaster (utk LOCO): mammography | actg175 | nhanes (4 target, orang yang sama) | brfss (4 target, orang yang sama).
Berkas sumber di ~/health_cache (NHANES *.XPT, BRFSS LLCP2015XPT.zip) dan ~/uci_cache/uci_890.pkl."""
from pathlib import Path
import numpy as np, pandas as pd

HC = Path.home() / "health_cache"
NH_FEAT = {"DEMO_J": ["RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "DMDMARTL", "INDFMPIR"],
           "BMX_J": ["BMXWT", "BMXHT", "BMXBMI", "BMXWAIST"],
           "BPX_J": ["BPXPLS", "BPXSY1", "BPXDI1", "BPXSY2", "BPXDI2"],
           "SMQ_J": ["SMQ020"], "ALQ_J": ["ALQ111", "ALQ121"],
           "PAQ_J": ["PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665", "PAD680"]}
NH_TGT = {"nhanes_hipertensi": ("BPQ_J", "BPQ020"), "nhanes_diabetes": ("DIQ_J", "DIQ010"),
          "nhanes_asma": ("MCQ_J", "MCQ010"), "nhanes_pjk": ("MCQ_J", "MCQ160C")}
BR_FEAT = ["_AGEG5YR", "SEX", "_RACE", "EDUCA", "INCOME2", "MARITAL", "EMPLOY1", "_BMI5", "_SMOKER3", "_RFDRHV5",
           "EXERANY2", "GENHLTH", "PHYSHLTH", "MENTHLTH", "HLTHPLN1", "MEDCOST", "CHECKUP1"]  # SLEPTIM1 tak ada di inti 2015
BR_TGT = {"brfss_stroke": "CVDSTRK3", "brfss_asma": "ASTHMA3", "brfss_ginjal": "CHCKIDNY", "brfss_depresi": "ADDEPEV2"}
KLASTER = {"mammography": "mammography", "actg175": "actg175", **{k: "nhanes" for k in NH_TGT}, **{k: "brfss" for k in BR_TGT}}


def _yn(s):
    return s.where(s.isin([1, 2])).map({1: 1, 2: 0})


def _nhanes(name):
    df = None
    for f, cols in NH_FEAT.items():
        d = pd.read_sas(HC / f"{f}.XPT", format="xport")[["SEQN"] + cols]
        df = d if df is None else df.merge(d, on="SEQN", how="left")
    df = df[df.RIDAGEYR >= 20]
    f, c = NH_TGT[name]
    t = pd.read_sas(HC / f"{f}.XPT", format="xport")[["SEQN", c]]
    df = df.merge(t, on="SEQN", how="inner"); y = _yn(df.pop(c)); df = df.drop(columns="SEQN")
    ok = y.notna(); return df[ok].reset_index(drop=True), y[ok].astype(int).reset_index(drop=True)


def _brfss_all():
    p = HC / "brfss2015_sub.pkl"
    if not p.exists():
        import zipfile
        with zipfile.ZipFile(HC / "LLCP2015XPT.zip") as z:
            nm = z.namelist()[0]; xp = Path(z.extract(nm, HC))  # nama anggota zip berakhiran spasi
        d = pd.read_sas(xp, format="xport")
        d[BR_FEAT + list(BR_TGT.values())].to_pickle(p)
        xp.unlink()
    return pd.read_pickle(p)


def _brfss(name):
    d = _brfss_all(); y = _yn(d[BR_TGT[name]]); X = d[BR_FEAT]
    ok = y.notna(); return X[ok].reset_index(drop=True), y[ok].astype(int).reset_index(drop=True)


def _mammo():
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=310, as_frame=True, parser="auto")
    X = d.data.astype(float); y = (d.target.astype(str) == "1").astype(int)
    df = X.assign(_y=y.values).drop_duplicates()
    konflik = df.duplicated(subset=list(X.columns), keep=False)
    df = df[~konflik]; y = df.pop("_y")
    return df.reset_index(drop=True), y.reset_index(drop=True)


def _actg():
    df = pd.read_pickle(Path.home() / "uci_cache" / "uci_890.pkl")
    y = df.pop("target").astype(int)
    df = df.drop(columns=[c for c in ("time", "cd420", "cd820", "offtrt") if c in df.columns])
    return df.reset_index(drop=True), y.reset_index(drop=True)


def load(name):
    if name == "mammography":
        X, y = _mammo()
    elif name == "actg175":
        X, y = _actg()
    elif name.startswith("nhanes_"):
        X, y = _nhanes(name)
    else:
        X, y = _brfss(name)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, pd.Series(np.asarray(y), name="target")

"""Metrik klinis dari 8-counts (y, baseline, kandidat); kelas positif = 1 (penyakit/kejadian, kelas jarang).
Delta = metrik(kandidat) - metrik(baseline), tervektorisasi. Memperluas bridge_metrics tanpa mengubahnya (manifest lama aman).
  bacc   : balanced accuracy = (sensitivitas + spesifisitas) / 2
  nb10   : net benefit (Vickers & Elkin 2006) pada ambang probabilitas p_t = 0,1: TP/n - FP/n * p_t/(1-p_t)
           (melewatkan 1 positif ~ 9x lebih buruk dari 1 alarm palsu)
  nb20   : net benefit pada p_t = 0,2 (4x)
  F2     : F-beta (beta = 2) kelas positif: 5 TP / (5 TP + 4 FN + FP)
  sens, spes : sensitivitas, spesifisitas (deskriptif saja, tidak dipakai sebagai target gate)"""
import numpy as np
import bridge_metrics as BM


def _m(m, name):
    tn, fp, fn, tp = m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1]
    n = tn + fp + fn + tp
    if name in ("bacc",):
        return BM._metric(m, "bacc")
    if name == "sens":
        return BM._div(tp, tp + fn)
    if name == "spes":
        return BM._div(tn, tn + fp)
    if name in ("nb10", "nb20"):
        pt = .1 if name == "nb10" else .2
        return BM._div(tp, n) - BM._div(fp, n) * pt / (1 - pt)
    if name == "F2":
        return BM._div(5 * tp, 5 * tp + 4 * fn + fp)
    return BM._metric(m, name)


def delta(c, name):
    mb, mc = BM._conf(c)
    return _m(mc, name) - _m(mb, name)


METRICS = ("bacc", "nb10", "nb20", "F2")

"""Metrik dari 8-counts (y, baseline, kandidat), tervektorisasi atas dimensi depan.
Delta = metrik(kandidat) - metrik(baseline). Dipakai gate (bootstrap multinomial) dan kebenaran
referensi (Dirichlet), identik dengan delta_from_counts exp41 untuk wF1."""
import numpy as np


def _conf(c):
    c = np.asarray(c, dtype=float).reshape(np.shape(c)[:-1] + (2, 2, 2))  # [y,b,c]
    return c.sum(axis=-1), c.sum(axis=-2)  # baseline [y,b], kandidat [y,c]


def _div(a, b):
    return np.divide(a, b, out=np.zeros_like(np.asarray(a, dtype=float)), where=np.asarray(b) > 0)


def _metric(m, name):
    tn, fp, fn, tp = m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1]
    tot = tn + fp + fn + tp
    if name == "acc":
        return _div(tn + tp, tot)
    if name == "bacc":
        return 0.5 * (_div(tp, tp + fn) + _div(tn, tn + fp))
    f0 = _div(2 * tn, 2 * tn + fp + fn); f1 = _div(2 * tp, 2 * tp + fp + fn)
    if name == "macroF1":
        return 0.5 * (f0 + f1)
    if name == "wF1":
        return _div((tn + fp) * f0 + (tp + fn) * f1, tot)
    raise ValueError(name)


def delta(c, name):
    mb, mc = _conf(c)
    return _metric(mc, name) - _metric(mb, name)


METRICS = ("wF1", "acc", "bacc", "macroF1")

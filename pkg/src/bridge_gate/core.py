"""Core of the BRIDGE gate.

The arithmetic and random streams match the reference experiments (binary:
bridge_metrics.py and MINI-32; multiclass: mini18_multiclass.py). For the same
inputs and seeds, the package returns the same decision as the reference runs.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import hypergeom
from sklearn.model_selection import StratifiedKFold, train_test_split

MARGIN, QUANTILE, REPLICATES, FOLDS = 0.005, 0.05, 20_000, 3
SEARCH_FRACTION = 630 / 2430      # SEARCH share of the training rows (630 of 2,430 in the reference runs)
BAGS, BAG_FRACTION = 32, 0.8      # bagging used in the reference runs (for the requirement check)
MIN_PER_CLASS_IN_BAG = 10         # binary bag rule


# Evidence: counts of (true label, current prediction, candidate prediction).
def counts(y, current, candidate, K=None):
    """K^3 counts of (y, current, candidate). For K=2 this is the 8-count vector indexed by 4*y + 2*current + candidate."""
    y, b, c = (np.asarray(v, dtype=np.int64) for v in (y, current, candidate))
    K = int(max(y.max(), b.max(), c.max()) + 1) if K is None else int(K)
    K = max(K, 2)
    return np.bincount((y * K + b) * K + c, minlength=K ** 3)


def _div(a, b):
    return np.divide(a, b, out=np.zeros_like(np.asarray(a, dtype=float)), where=np.asarray(b) > 0)


def _binary_metric(m, name):
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
    if name in ("nb10", "nb20"):      # net benefit (Vickers & Elkin) at threshold probability 0.1 or 0.2; positive class = 1
        pt = .1 if name == "nb10" else .2
        return _div(tp, tot) - _div(fp, tot) * pt / (1 - pt)
    if name == "F2":                  # F-beta with beta = 2 for the positive class
        return _div(5 * tp, 5 * tp + 4 * fn + fp)
    raise ValueError(f"unknown metric {name!r}")


def _multi_wf1(M):
    tp = np.diagonal(M, axis1=-2, axis2=-1); row = M.sum(-1); col = M.sum(-2)
    f1 = np.divide(2 * tp, row + col, out=np.zeros_like(tp), where=(row + col) > 0)
    return (row * f1).sum(-1) / np.maximum(row.sum(-1), 1e-300)


def delta(c, K=2, metric="wF1"):
    """Gain of the candidate over the current model, computed from counts. Extra leading axes are vectorized."""
    c = np.asarray(c, dtype=float)
    if K == 2:
        t = c.reshape(c.shape[:-1] + (2, 2, 2))
        return _binary_metric(t.sum(axis=-2), metric) - _binary_metric(t.sum(axis=-1), metric)
    if metric != "wF1":
        raise NotImplementedError("for more than two classes the validated metric is weighted-F1 ('wF1')")
    T = c.reshape(c.shape[:-1] + (K, K, K))
    return _multi_wf1(T.sum(-2)) - _multi_wf1(T.sum(-1))


def score(y, pred, metric="wF1", K=None):
    """Metric of one set of predicted labels, computed with the same formulas as `delta`."""
    y, p = np.asarray(y, dtype=np.int64), np.asarray(pred, dtype=np.int64); K = _K(y, p) if K is None else int(K)
    c = counts(y, p, p, K).astype(float)
    if K == 2:
        return float(_binary_metric(c.reshape(2, 2, 2).sum(-1), metric))
    if metric != "wF1":
        raise NotImplementedError("for more than two classes the validated metric is weighted-F1 ('wF1')")
    return float(_multi_wf1(c.reshape(K, K, K).sum(-1)))


def lower_bound(c, K=2, metric="wF1", seed=0, replicates=REPLICATES, quantile=QUANTILE):
    """Lower bootstrap quantile of the gain. Counts are resampled from a multinomial distribution."""
    c = np.asarray(c); rng = np.random.default_rng(seed); n = int(c.sum())
    if K == 2:
        return float(np.quantile(delta(rng.multinomial(n, c / n, size=replicates), 2, metric), quantile))
    nz = np.flatnonzero(c); out = []
    for s in range(0, replicates, 2500):          # same chunking and random stream as the multiclass runs
        m = min(2500, replicates - s)
        W = np.zeros((m, K ** 3)); W[:, nz] = rng.multinomial(n, c[nz] / n, size=m); out.append(delta(W, K, metric))
    return float(np.quantile(np.concatenate(out), quantile))


# Decisions.
@dataclass
class Decision:
    accept: bool
    lower_bound: float
    gain_pool: float                 # gain of the selected candidate on all POOL rows; the naive rule accepts when > 0
    chosen: object = None            # selected candidate name, or None when no candidate improved
    n: int = 0
    K: int = 2
    metric: str = "wF1"
    counts: np.ndarray = field(default=None, repr=False)
    warnings: list = field(default_factory=list)
    n_tried: int = 1                 # number of candidates judged
    score_current: float = None      # metric of the current model on all POOL rows
    score_candidate: float = None    # metric of the selected candidate on all POOL rows (None when none improved)
    right_current: int = None        # POOL rows predicted correctly by the current model
    right_candidate: int = None

    @property
    def naive_accept(self):
        return self.gain_pool > 0

    def explain(self):
        """The decision as a plain-language result card, with the tested conditions it does not meet."""
        from .report import card, pool_conditions, UNCHECKABLE
        smallest = int(np.asarray(self.counts).reshape((self.K,) * 3).sum((1, 2)).min())   # rows of the smallest class
        return card(self.accept, self.n, self.metric, "current", self.chosen, self.score_current, self.score_candidate,
                    self.right_current, self.right_candidate, None if self.chosen is None else self.gain_pool,
                    self.lower_bound, self.n_tried, conditions=pool_conditions(self.n, self.K, self.metric, smallest),
                    uncheckable=UNCHECKABLE)

    def __str__(self):
        verdict = f"INSTALL {self.chosen}" if self.accept else "KEEP the current model"
        return (f"BRIDGE decision: {verdict} (gain on {self.n} judging rows {self.gain_pool:+.4f}; "
                f"unlucky-case gain {self.lower_bound:+.4f}, needs > {MARGIN:+.4f}). Use .explain() for the full card.")


def _K(*arrays):
    return max(2, int(max(np.max(a) for a in arrays)) + 1)


def gate(y_pool, pred_current, pred_candidate, metric="wF1", seed=0, margin=MARGIN, K=None):
    """Decide on one candidate fixed in advance, for example a model tuned on other data. Uses all POOL rows."""
    y, b, c = (np.asarray(v) for v in (y_pool, pred_current, pred_candidate)); K = _K(y, b, c) if K is None else int(K)
    cnt = counts(y, b, c, K); lb = lower_bound(cnt, K, metric, seed)
    return Decision(accept=lb > margin, lower_bound=lb, gain_pool=float(delta(cnt, K, metric)), chosen="candidate",
                    n=len(y), K=K, metric=metric, counts=cnt, warnings=_pool_warnings(y, K),
                    score_current=score(y, b, metric, K), score_candidate=score(y, c, metric, K),
                    right_current=int((b == y).sum()), right_candidate=int((c == y).sum()))


def gate_select(y_pool, pred_current, candidates, metric="wF1", fold_seed=0, boot_seed=0, margin=MARGIN, folds=FOLDS, K=None):
    """Decide among several candidates with cross-fitting.

    In each of `folds` POOL folds, the candidate selected on the other folds is
    scored on the held-out fold. The gate accepts when the lower bootstrap
    quantile of the pooled cross-fitted gain exceeds the margin.
    `candidates` maps a name to predicted labels on POOL. On ties, the first
    name wins. `K` is read from the labels unless given; pass it when a class
    may be absent from POOL. The candidate to deploy is selected on all POOL.
    """
    y, b = np.asarray(y_pool), np.asarray(pred_current)
    names = list(candidates); P = [np.asarray(candidates[k]) for k in names]; K = _K(y, b, *P) if K is None else int(K)

    def pick(idx):
        best, g0 = None, 0.0
        for j, p in enumerate(P):
            g = float(delta(counts(y[idx], b[idx], p[idx], K), K, metric))
            if g > g0 + 1e-12:
                best, g0 = j, g
        return best
    cnt = np.zeros(K ** 3, int)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=fold_seed % (2**32 - 1)).split(y, y):
        j = pick(tr); cnt += counts(y[te], b[te], b[te] if j is None else P[j][te], K)
    lb = lower_bound(cnt, K, metric, boot_seed)
    j = pick(np.arange(len(y))); g = 0.0 if j is None else float(delta(counts(y, b, P[j], K), K, metric))
    return Decision(accept=(lb > margin) and j is not None, lower_bound=lb, gain_pool=g, chosen=None if j is None else names[j],
                    n=len(y), K=K, metric=metric, counts=cnt, warnings=_pool_warnings(y, K), n_tried=len(names),
                    score_current=score(y, b, metric, K), score_candidate=None if j is None else score(y, P[j], metric, K),
                    right_current=int((b == y).sum()), right_candidate=None if j is None else int((P[j] == y).sum()))


# Data split and input requirements.
def split(y, search_fraction=SEARCH_FRACTION, seed=0):
    """Return the SEARCH and POOL indices. Fit and tune on SEARCH; judge on POOL only."""
    idx = np.arange(len(y))
    s, p = train_test_split(idx, train_size=search_fraction, stratify=np.asarray(y), random_state=seed)
    return np.sort(s), np.sort(p)


def _pool_warnings(y, K):
    w = []; cc = np.bincount(np.asarray(y, int), minlength=K)
    if cc.min() < 30:
        w.append(f"POOL has only {int(cc.min())} examples of its smallest class: only large improvements can be accepted")
    if K > 2:
        w.append("more than two classes: evidence is spread over K^3 cells; acceptance needs more POOL data")
    return w


def check(y_search, y_pool, bagged=False):
    """Report input notes. The gate itself needs only POOL.

    With `bagged=True`, also check the bagged-training requirement: 32 bags of
    80% of SEARCH, each with at least 10 examples per class. Returns a pair
    (ok, messages).
    """
    ys, yp = np.asarray(y_search, int), np.asarray(y_pool, int); K = _K(ys, yp); msg = []
    cp = np.bincount(yp, minlength=K); ok = bool(cp.min() > 0)
    msg.append(f"POOL: {len(yp)} rows, {K} classes, smallest class {int(cp.min())} rows"
               + ("" if cp.min() > 0 else " -- a class is absent from POOL: the gate cannot judge it"))
    if bagged:
        cs = np.bincount(ys, minlength=K); m = int(cs.min()); sub = int(BAG_FRACTION * len(ys))
        p_bag = float(hypergeom(len(ys), m, sub).sf(MIN_PER_CLASS_IN_BAG - 1)) if m else 0.0
        okA = m >= 16; ok = ok and okA
        msg.append(f"bagged training: smallest SEARCH class {m} rows; {'meets' if okA else 'BELOW'} requirement A (>= 16); "
                   f"expected usable bags {BAGS * p_bag:.1f} of {BAGS}")
    msg += _pool_warnings(yp, K)
    return ok, msg

"""Reference procedure of the paper, exposed as one function.

Given labeled rows and a seed, `run` reproduces one trial of the confirmation
runs (binary: MINI-21 and MINI-32; multiclass: MINI-18 and MINI-22):
  1. draw 2,430 training rows (630 SEARCH, 1,800 POOL), or 1,000 (259/741)
     when fewer than 2,480 rows are given; stratified by the seed
  2. fit 32 bags of 80% of SEARCH; skip bags with too few examples per class;
     fit HGB, DT and SVC on each bag and average class probabilities per learner
  3. current model: HGB at threshold 0.5 (binary) or argmax (multiclass)
     binary candidates: HGB threshold, HGB+DT weights, HGB+DT+SVC weights,
     each with a threshold; multiclass candidates: HGB/DT/SVC weights and
     class scales of HGB or HGB+DT
  4. gate: 3-fold cross-fitting on POOL; the 5% bootstrap quantile (20,000
     draws) of the gain must exceed 0.005
  5. optional: check the decision on rows the procedure never saw
     (`X_ref`, `y_ref`), as in the locked-reference check
Same inputs and seed produce the same decision, configuration and bound as the
trial code.
"""
import itertools
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from . import core

TOTAL_TRAIN, SEARCH_SIZE = 2430, 630          # training rows; SEARCH rows; the rest of the training rows is POOL
SMALL_TOTAL, SMALL_SEARCH = 1000, 259         # used when fewer than TOTAL_TRAIN + 50 rows are given
BAGS, BAG_FRACTION = 32, 0.80
MODEL_SEED = 42                               # random_state used by every learner
THRESHOLDS = [round(t, 2) for t in np.arange(.05, .96, .05)]
SIMPLEX = [w for w in itertools.product((0, .25, .5, .75, 1), repeat=3) if abs(sum(w) - 1) < 1e-9 and w != (1, 0, 0)]
SCALES = (.5, .7, 1.4, 2)
METRICS_BINARY = ("wF1", "bacc", "macroF1")   # validated for two classes (MINI-16, MINI-21); more classes: core.METRICS_MULTI
CONFIRMED_K = (7, 10)                         # numbers of classes in the multiclass confirmation run (MINI-22)
DEVELOPMENT_K = 26                            # largest number of classes in the multiclass development run (MINI-18)
VALIDATED_MAX_ROWS = 6000                     # largest training size in the development runs (MINI-17)
REF_SEED_OFFSET, BOOT_SEED_OFFSET_BIN, BOOT_SEED_OFFSET_MC = 987163, 61, 71


# Data coding, following the loaders used in the runs.
def encode(X):
    """Encode text columns as integer codes of their sorted values. Missing text gets its own code. All columns numeric."""
    X = X.copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            s = X[col].astype(object).where(X[col].notna(), "__MISSING__").astype(str)
            X[col] = pd.Categorical(s, categories=sorted(s.unique())).codes
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.reset_index(drop=True)


# Learners with the fixed settings used in the runs.
def _learner(name, n):
    if name == "HGB":
        return HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20,
                                              l2_regularization=0.0, early_stopping=True, n_iter_no_change=10,
                                              random_state=MODEL_SEED)
    if name == "DT":
        return DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion="gini",
                                      random_state=MODEL_SEED)
    if name == "SVC":
        return SVC(C=1.0, kernel="rbf", gamma="scale", max_iter=500 if n > 1000 else 2000, cache_size=1000,
                   probability=True, random_state=MODEL_SEED)
    raise ValueError(name)


def _preprocessor(X):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]; cat = [c for c in X.columns if c not in num]
    parts = []
    if num:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat))
    return ColumnTransformer(parts, remainder="drop")


class _Fitted:
    """One learner fitted on one bag. proba(X) returns (n, K) class probabilities in class-code order."""

    def __init__(self, name, X, y, K):
        self.cols, self.K = list(X.columns), K
        self.pre = _preprocessor(X); Z = self.pre.fit_transform(X, y)
        self.est = _learner(name, len(y)); self.est.fit(Z, y); self.cl = np.asarray(self.est.classes_, int)

    def proba(self, X):
        p = self.est.predict_proba(self.pre.transform(X[self.cols]))
        if self.K == 2:     # the binary runs keep the probability of class 1 only
            return p[:, list(self.cl).index(1) if 1 in self.cl else 1]
        out = np.zeros((len(X), self.K)); out[:, self.cl] = p; return out


# Candidate spaces.
def candidates(K):
    """Return the candidate configurations.

    Binary: (weights of HGB/DT/SVC, threshold). Multiclass: (weights,
    (class, scale) or None). Order matters; on ties the first candidate wins.
    """
    if K == 2:
        thr = [({"HGB": 1}, t) for t in THRESHOLDS if t != .5]
        e2 = [({"HGB": w, "DT": 1 - w}, t) for w in (.75, .5, .25) for t in THRESHOLDS]
        e3 = [({"HGB": a, "DT": b, "SVC": c}, t) for a, b, c in SIMPLEX for t in (.3, .35, .4, .45, .5, .55, .6)]
        return thr + e2 + e3
    e = [({"HGB": a, "DT": b, "SVC": c}, None) for a, b, c in SIMPLEX]
    s = [({"HGB": 1}, (k, v)) for k in range(K) for v in SCALES]
    es = [({"HGB": .5, "DT": .5}, (k, v)) for k in range(K) for v in SCALES]
    return e + s + es


def labels(P, cfg, K, idx=None):
    """Predicted labels for one configuration. `P` maps learner name to POOL probabilities."""
    w, post = cfg
    p = sum(wi * P[m] for m, wi in w.items() if wi)
    if K == 2:
        return ((p if idx is None else p[idx]) >= post).astype(int)
    if post is not None:
        p = p.copy(); p[:, post[0]] *= post[1]
    return (p if idx is None else p[idx]).argmax(1)


def name_of(cfg, K, class_names=None):
    """Readable name, e.g. 'HGB@thr0.3', 'HGB .5 + DT .5 @thr0.45', 'HGB@scale<class>x1.4'."""
    if cfg is None:
        return None
    w, post = cfg; parts = [m if wi == 1 else f"{m} {wi:g}" for m, wi in w.items() if wi]
    left = parts[0] if len(parts) == 1 else " + ".join(parts) + " "
    if K == 2:
        return f"{left}@thr{float(post):g}"
    if post is None:
        return left.strip()
    cls = class_names[post[0]] if class_names is not None else post[0]
    return f"{left}@scale[{cls}]x{post[1]:g}"


def _pick(y, b, P, idx, C, K, metric):
    best, g0 = None, 0.0
    for cfg in C:
        g = float(core.delta(core.counts(y[idx], b[idx], labels(P, cfg, K, idx), K), K, metric))
        if g > g0 + 1e-12:
            best, g0 = cfg, g
    return best


def _dirichlet_interval(cr, K, metric, seed):
    """Return the 95% interval of the gain on the reference rows (clearly better, clearly worse, or uncertain)."""
    rng = np.random.default_rng(seed)
    if K == 2:
        d = core.delta(rng.dirichlet(cr + .5, size=20000), 2, metric)
    else:
        nz = np.flatnonzero(cr); out = []
        for s in range(0, 20000, 2500):
            m = min(2500, 20000 - s); W = np.zeros((m, K ** 3)); W[:, nz] = rng.dirichlet(cr[nz], size=m)
            out.append(core.delta(W, K, metric))
        d = np.concatenate(out)
    lo, hi = np.quantile(d, [.025, .975])
    return float(lo), float(hi)


@dataclass
class Result:
    accept: bool                      # install the chosen configuration
    lower_bound: float                # 5% bootstrap quantile of the cross-fitted gain
    chosen: object                    # configuration selected on all POOL rows (None when no candidate scored higher)
    gain_pool: float                  # gain of `chosen` on all POOL rows; the naive rule installs when > 0
    K: int
    metric: str
    total: int                        # training rows used
    search: int
    pool: int
    bags: int                         # usable bags
    validated: bool                   # inside the conditions covered by the runs
    conditions: list = field(default_factory=list)      # conditions not met (empty when validated)
    gain_ref: float = None            # gain of `chosen` on the reference rows, when given
    ref_interval: tuple = None        # 95% interval of that gain
    ref_class: str = None             # 'better', 'worse' or 'uncertain'
    counts: np.ndarray = field(default=None, repr=False)
    models: dict = field(default=None, repr=False)     # learner name -> list of fitted bags
    pool_labels: np.ndarray = field(default=None, repr=False)      # true classes of the POOL rows
    pool_current: np.ndarray = field(default=None, repr=False)     # current model's predictions on POOL
    pool_chosen: np.ndarray = field(default=None, repr=False)      # chosen configuration's predictions on POOL

    @property
    def naive_accept(self):
        return self.chosen is not None and self.gain_pool > 0

    def name(self, class_names=None):
        return name_of(self.chosen, self.K, class_names)


def conditions(n_rows, K, metric):
    """The conditions covered by the confirmation runs: at least TOTAL_TRAIN + 50 rows (or SMALL_TOTAL + 50 for two
    classes), 7-10 classes, and the metrics listed above. Returns the conditions that are not met."""
    c = []
    if K == 2 and n_rows < SMALL_TOTAL + 50:
        c.append(f"{n_rows} rows: the reference runs use at least {SMALL_TOTAL + 50} rows for two classes")
    if K > 2 and n_rows < TOTAL_TRAIN + 50:
        c.append(f"{n_rows} rows: the reference runs with more classes use at least {TOTAL_TRAIN + 50} rows")
    if K > 2 and not CONFIRMED_K[0] <= K <= CONFIRMED_K[1]:
        c.append(f"{K} classes: the multiclass confirmation covers {CONFIRMED_K[0]}-{CONFIRMED_K[1]} classes "
                 f"(the development runs went up to {DEVELOPMENT_K})")
    if (K == 2 and metric not in METRICS_BINARY) or (K > 2 and metric not in core.METRICS_MULTI):
        c.append(f"metric {metric}: validated metrics are {', '.join(METRICS_BINARY)} (two classes) and "
                 f"{', '.join(core.METRICS_MULTI)} (more classes)")
    return c


def run(X, y, seed, metric="wF1", X_ref=None, y_ref=None, K=None, progress=None):
    """One decision with the reference procedure.

    X: DataFrame of labeled rows, encoded with `encode`.
    y: class codes 0..K-1.
    seed: trial seed; the training draw, bags, folds and bootstrap all derive
    from it.
    progress(done, total, what): optional callback while fitting.
    """
    return run_metrics(X, y, seed, (metric,), X_ref, y_ref, K, progress)[metric]


def run_metrics(X, y, seed, metrics, X_ref=None, y_ref=None, K=None, progress=None):
    """The decisions of `run` for several metrics, with the models fitted once; returns {metric: Result}."""
    y = np.asarray(y, int); K = int(y.max()) + 1 if K is None else int(K); K = max(K, 2)
    if len(y) >= TOTAL_TRAIN + 50:
        TOT, SS = TOTAL_TRAIN, SEARCH_SIZE
    elif len(y) >= SMALL_TOTAL + 50:
        TOT, SS = SMALL_TOTAL, SMALL_SEARCH
    else:
        TOT, SS = len(y), int(round(SEARCH_SIZE / TOTAL_TRAIN * len(y)))
    X = X.reset_index(drop=True)
    if TOT < len(y):
        Xtr, _, ytr, _ = train_test_split(X, pd.Series(y), train_size=TOT, stratify=y, random_state=seed)
    else:
        Xtr, ytr = X, pd.Series(y)
    Xs, Xp, ys, yp = train_test_split(Xtr, ytr, train_size=SS, stratify=ytr, random_state=seed)
    Xs, ys, Xp, yp = Xs.reset_index(drop=True), np.asarray(ys), Xp.reset_index(drop=True), np.asarray(yp)

    rng = np.random.RandomState((seed * 777013) % (2**32 - 1)); sub = int(BAG_FRACTION * len(ys)); bags = []
    for _ in range(BAGS):
        i = rng.choice(len(ys), size=sub, replace=False)
        cc = np.bincount(ys[i], minlength=K)
        if K == 2:
            ok = cc.min() >= 10                       # binary bag rule: >= 10 per class
        else:
            cc = cc[cc > 0]; ok = len(cc) >= 2 and cc.min() >= 2   # >= 2 per occurring class
        if ok:
            bags.append(i)
    if not bags:
        raise ValueError("no usable bag: too few examples of the smallest class in the training rows")
    fitted = {m: [] for m in ("HGB", "DT", "SVC")}; total, done = 3 * len(bags), 0
    for m in fitted:
        for i in bags:
            fitted[m].append(_Fitted(m, Xs.iloc[i].reset_index(drop=True), ys[i], K)); done += 1
            if progress:
                progress(done, total, m)

    def avg(Xe):
        if K == 2:
            return {m: np.column_stack([f.proba(Xe) for f in fs]).mean(1) for m, fs in fitted.items()}
        return {m: sum(f.proba(Xe) for f in fs) / len(fs) for m, fs in fitted.items()}
    P = avg(Xp)
    base = (P["HGB"] >= .5).astype(int) if K == 2 else P["HGB"].argmax(1)
    C = candidates(K)
    if X_ref is not None:
        Pr = avg(X_ref.reset_index(drop=True)); yr = np.asarray(y_ref, int)
        br = (Pr["HGB"] >= .5).astype(int) if K == 2 else Pr["HGB"].argmax(1)
    out = {}
    for metric in metrics:
        cond = conditions(len(y), K, metric); cnt = np.zeros(K ** 3, int)
        for a, b in StratifiedKFold(3, shuffle=True, random_state=seed % (2**32 - 1)).split(yp, yp):
            cfg = _pick(yp, base, P, a, C, K, metric)
            cnt += core.counts(yp[b], base[b], base[b] if cfg is None else labels(P, cfg, K, b), K)
        boot_seed = seed + (BOOT_SEED_OFFSET_BIN + METRICS_BINARY.index(metric) if K == 2
                            else BOOT_SEED_OFFSET_MC + core.METRICS_MULTI.index(metric))
        lb = core.lower_bound(cnt, K, metric, boot_seed)
        cfg = _pick(yp, base, P, np.arange(len(yp)), C, K, metric)
        g = 0.0 if cfg is None else float(core.delta(core.counts(yp, base, labels(P, cfg, K), K), K, metric))
        res = Result(accept=bool(lb > core.MARGIN) and cfg is not None, lower_bound=float(lb), chosen=cfg, gain_pool=g, K=K,
                     metric=metric, total=TOT, search=SS, pool=TOT - SS, bags=len(bags), validated=not cond, conditions=cond,
                     counts=cnt, models=fitted, pool_labels=yp, pool_current=base,
                     pool_chosen=None if cfg is None else labels(P, cfg, K))
        if X_ref is not None:
            cr = core.counts(yr, br, br if cfg is None else labels(Pr, cfg, K), K)
            res.gain_ref = float(core.delta(cr, K, metric))
            lo, hi = _dirichlet_interval(cr, K, metric, seed + REF_SEED_OFFSET)
            res.ref_interval = (lo, hi); res.ref_class = "better" if lo > 0 else "worse" if hi < 0 else "uncertain"
        out[metric] = res
    return out


class Model:
    """Model to use after a decision: the bagged learners as judged, without refitting, with the chosen configuration.
    When BRIDGE keeps the current model, the configuration is HGB at threshold 0.5 (or argmax). predict returns class
    names."""

    def __init__(self, res, class_names):
        self.models, self.K, self.classes_ = res.models, res.K, list(class_names)
        self.cfg = res.chosen if res.accept else ({"HGB": 1}, .5 if res.K == 2 else None)

    def predict_codes(self, X):
        X = X.reset_index(drop=True)
        if self.K == 2:
            P = {m: np.column_stack([f.proba(X) for f in fs]).mean(1) for m, fs in self.models.items()}
        else:
            P = {m: sum(f.proba(X) for f in fs) / len(fs) for m, fs in self.models.items()}
        return labels(P, self.cfg, self.K)

    def predict(self, X):
        return np.asarray(self.classes_)[self.predict_codes(X)]

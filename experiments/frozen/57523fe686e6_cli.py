"""Command-line program: decide whether a candidate model may replace the current model on your own data.

Default (no --candidates, no --auto): the paper's validated procedure, the one checked in the development runs and
confirmed on new data (current model HGB; candidates: threshold, HGB/DT/SVC ensembles, class scales; 32 bags):
    bridge-gate --dataset mango.csv --target grade
    bridge-gate --dataset data.csv --reference held_out.csv      (also checks the decision on rows it never used)

Your own candidates (outside the validated procedure; the output says so):
    bridge-gate --dataset iris --current rf --auto
    bridge-gate --dataset iris --current rf --auto combined
    bridge-gate --dataset mango.csv --target grade --auto --out result
    bridge-gate --dataset openml:61 --current hgb --candidates tuned_hgb,fs50_hgb,rf,ens_hgb+logreg

The program reads the data, splits it into SEARCH (all training, tuning and feature selection) and POOL (judging only),
builds the current model and the candidates on SEARCH, and lets BRIDGE choose among the candidates with cross-fitting
and decide (gate_select).

Candidate names (comma-separated in --candidates; --auto generates them). A candidate is built from components:
  base model     [tuned_][fs25_|fs50_]<learner>   learner: hgb, rf, dt, logreg, svc, knn
                 tuned_ = tuned on SEARCH (small grid, 3-fold CV); fs25_/fs50_ = top 25%/50% features (mutual information)
  ensemble       ens_<base>+<base>[+...]          average of the predicted probabilities
  post-step      <candidate>@thr<t>                two classes: decision threshold t (e.g. tuned_rf@thr0.35)
                 <candidate>@scale<k>x<v>          more classes: class k's probability times v (e.g. fs50_hgb@scale1x1.4)
  short forms    thr<t>, scale<k>x<v>             the step applied to the current model
"""
import argparse, json, sys, textwrap, time, warnings
from datetime import datetime
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
from . import core

BUILTIN = ("iris", "wine", "breast_cancer", "digits")
MODELS = {
    "hgb": lambda s: HistGradientBoostingClassifier(random_state=s),
    "rf": lambda s: RandomForestClassifier(n_estimators=300, random_state=s),
    "dt": lambda s: DecisionTreeClassifier(random_state=s),
    "logreg": lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "svc": lambda s: make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(random_state=s), ensemble=False)),
    "knn": lambda s: make_pipeline(StandardScaler(), KNeighborsClassifier()),
}
GRIDS = {   # small grids for tuned_<model> (tuned on SEARCH only, 3-fold CV)
    "hgb": {"learning_rate": [0.03, 0.1, 0.3], "max_leaf_nodes": [7, 31], "min_samples_leaf": [5, 20]},
    "rf": {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 3]},
    "dt": {"max_depth": [2, 3, 5, None], "min_samples_leaf": [1, 2, 5]},
    "logreg": {"logisticregression__C": [0.1, 1, 10]},
    "svc": {"calibratedclassifiercv__estimator__C": [0.3, 1, 3, 10]},
    "knn": {"kneighborsclassifier__n_neighbors": [3, 5, 9, 15]},
}
LEARNER_NAMES = {"hgb": "histogram gradient boosting", "rf": "random forest (300 trees)", "dt": "decision tree",
                 "logreg": "logistic regression", "svc": "support vector classifier (calibrated)", "knn": "k-nearest neighbors (k=5)"}
SMALL_SEARCH = 200   # hgb default: >= 20 rows per leaf, so fewer than 10 leaves below 200 SEARCH rows
THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05) if abs(t - 0.5) > 1e-9]   # as in the paper
SCALES = (0.5, 0.7, 1.4, 2.0)                                                              # as in the paper


def load(spec, target):
    """Returns X (DataFrame), y (array of class codes) and the class names."""
    if spec in BUILTIN:
        from sklearn import datasets
        d = getattr(datasets, f"load_{spec}")(as_frame=True)
        X, y = d.data, d.target.map(dict(enumerate(d.target_names)))
    elif spec.startswith("openml:"):
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=int(spec.split(":")[1]), as_frame=True, parser="auto"); X, y = d.data, d.target
    else:
        df = pd.read_pickle(spec) if spec.endswith(".pkl") else pd.read_csv(spec)
        t = target or df.columns[-1]
        if t not in df.columns:
            sys.exit(f"target column {t!r} not found; columns: {list(df.columns)}")
        X, y = df.drop(columns=[t]), df[t]
    y = pd.Series(y).astype(str); keep = (y.notna() & (y != "nan")).to_numpy()
    X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    names = sorted(y.unique())
    return X, y.map({n: i for i, n in enumerate(names)}).to_numpy(), names


def preprocessor(X):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]; cat = [c for c in X.columns if c not in num]
    parts = []
    if num:
        parts.append(("num", SimpleImputer(strategy="median"), num))
    if cat:
        parts.append(("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                          OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat))
    return ColumnTransformer(parts)


class Builder:
    """Fits the learners on SEARCH (once each) and returns POOL class probabilities, aligned to the K classes."""

    def __init__(self, Xs, ys, Xp, K, seed):
        self.Xs, self.ys, self.Xp, self.K, self.seed, self.cache, self.info, self.fitted = Xs, ys, Xp, K, seed, {}, {}, {}
        self.fit_seconds = 0.0

    def proba(self, name):
        if name not in self.cache:
            t0 = time.perf_counter(); self.cache[name] = self._fit(name); self.fit_seconds += time.perf_counter() - t0
        return self.cache[name]

    def _fit(self, name):
        tuned, fs, base = parse_base(name)
        steps = [("pre", preprocessor(self.Xs))]
        if fs:
            steps.append(("fs", SelectPercentile(partial(mutual_info_classif, random_state=self.seed), percentile=fs)))
        steps.append(("est", MODELS[base](self.seed))); pipe = Pipeline(steps)
        info = dict(learner=LEARNER_NAMES[base], settings="library defaults")
        if tuned:
            gs = GridSearchCV(pipe, {f"est__{k}": v for k, v in GRIDS[base].items()}, cv=3, n_jobs=1).fit(self.Xs, self.ys)
            pipe = gs.best_estimator_
            info["settings"] = "tuned on the training rows: " + ", ".join(f"{k.split('__')[-1]}={v}" for k, v in gs.best_params_.items())
        else:
            pipe.fit(self.Xs, self.ys)
        fn = [n.split("__", 1)[-1] for n in pipe.named_steps["pre"].get_feature_names_out()]; nf = len(fn)
        info["tuning"] = info["settings"] if tuned else "none (library defaults)"
        if fs:
            keep = [n for n, s in zip(fn, pipe.named_steps["fs"].get_support()) if s]
            info["features"] = f"{len(keep)} of {nf} (top {fs}% by mutual information): " + ", ".join(keep[:12]) + (" ..." if len(keep) > 12 else "")
        else:
            info["features"] = f"all {nf}" + (" (after encoding text columns)" if nf != self.Xs.shape[1] else "")
        self.info[name] = info; self.fitted[name] = pipe
        out = np.zeros((len(self.Xp), self.K)); out[:, np.asarray(pipe.classes_, int)] = pipe.predict_proba(self.Xp)
        return out


def parse_base(name):
    """'tuned_fs50_rf' -> (True, 50, 'rf'): the prefixes tuned_ and fs25_/fs50_ may be combined, in any order."""
    tuned, fs, rest = False, 0, name
    while True:
        if rest.startswith("tuned_"):
            tuned, rest = True, rest[6:]
        elif rest.startswith(("fs25_", "fs50_")):
            fs, rest = int(rest[2:4]), rest[5:]
        else:
            break
    if rest not in MODELS:
        sys.exit(f"unknown learner in {name!r}; learners: {sorted(MODELS)}")
    return tuned, fs, rest


def parse(name, current):
    """A candidate is one base model or an ensemble of base models, optionally followed by a post-processing step:
    'tuned_fs50_rf', 'ens_tuned_hgb+fs50_logreg', 'tuned_rf@thr0.35', 'fs25_hgb@scale1x1.4'.
    The short forms 'thr0.35' and 'scale1x1.4' apply the step to the current model."""
    if name.startswith(("thr", "scale")):
        name = f"{current}@{name}"
    left, _, post = name.partition("@")
    members = left[4:].split("+") if left.startswith("ens_") else [left]
    if post.startswith("thr"):
        post = ("thr", float(post[3:]))
    elif post.startswith("scale"):
        k, v = post[5:].split("x"); post = ("scale", int(k), float(v))
    elif post:
        sys.exit(f"unknown step {post!r} in {name!r}")
    for m in members:
        parse_base(m)
    return members, (post or None)


def apply_post(p, post):
    if post is None:
        return p
    if post[0] == "thr":
        if p.shape[1] != 2:
            sys.exit("thresholds apply to two classes; use scale<k>x<v> for more classes")
        return np.column_stack([1 - (p[:, 1] >= post[1]), p[:, 1] >= post[1]]).astype(float)
    p = p.copy(); p[:, post[1]] *= post[2]; return p


def candidate_proba(name, B, current, K):
    members, post = parse(name, current)
    return apply_post(sum(B.proba(m) for m in members) / len(members), post)


def auto_candidates(current, K, mode="single"):
    posts = [f"thr{t:.2f}" for t in THRESHOLDS] if K == 2 else [f"scale{k}x{v}" for k in range(K) for v in SCALES]
    if mode == "single":     # one kind of change at a time
        base = parse_base(current)[2]
        c = [m for m in MODELS if m != current] + [f"tuned_{m}" for m in MODELS] + [f"fs25_{base}", f"fs50_{base}"] + posts
        return c + [f"ens_{current}+{m}" for m in MODELS if m != current] + ["ens_" + "+".join(MODELS)]
    # combined: every learner x tuning x features, each with and without post-processing, plus ensembles of two
    bases = [f"{t}{f}{m}" for m in MODELS for t in ("", "tuned_") for f in ("", "fs25_", "fs50_")]
    c = [b + ("" if q is None else "@" + q) for b in bases for q in [None] + posts if not (b == current and q is None)]
    pairs = [f"{t}{m}" for m in MODELS for t in ("", "tuned_")]
    return c + [f"ens_{a}+{b}" for i, a in enumerate(pairs) for b in pairs[i + 1:]]


def kind_of(name, current="hgb"):
    """The kinds of change a candidate makes, e.g. 'tuned + feature selection + threshold'."""
    members, post = parse(name, current); parts = []
    if len(members) > 1:
        parts.append("ensemble")
    elif parse_base(members[0])[2] != parse_base(current)[2]:
        parts.append("other learner")
    if any(parse_base(m)[0] for m in members):
        parts.append("tuned")
    if any(parse_base(m)[1] for m in members):
        parts.append("feature selection")
    if post:
        parts.append("threshold" if post[0] == "thr" else "class scale")
    return " + ".join(parts) or "no change"


def spec(name, B, current, K, names=None):
    """The configuration as labeled lines: model, tuning, features, prediction rule."""
    members, post = parse(name, current); lines = []
    for j, m in enumerate(members):
        B.proba(m); i = B.info[m]; tag = f"member {j + 1}: " if len(members) > 1 else ""
        lines += [("model", tag + i["learner"]), ("tuning", tag + i["tuning"]), ("features", tag + i["features"])]
    if len(members) > 1:
        lines.insert(0, ("ensemble", f"average of the predicted probabilities of {len(members)} models"))
    if post is None:
        rule = "threshold 0.5 (unchanged)" if K == 2 else "most probable class (no class re-weighting)"
    elif post[0] == "thr":
        rule = f"threshold {post[1]} instead of 0.5"
    else:
        rule = f"most probable class after multiplying the probability of class '{names[post[1]] if names else post[1]}' by {post[2]}"
    return lines + [("prediction", rule)]


def print_spec(title, name, B, current, K, names):
    s = spec(name, B, current, K, names)
    print(f"  {title:13s}: {s[0][1] if s[0][0] in ('model', 'ensemble') else ''}")
    for k, v in s[1:]:
        print(f"  {'':13s}    {k:10s}: {v}")


def describe(name, B, current, K, names=None):
    """Plain description of a model or candidate."""
    members, post = parse(name, current)

    def one(m):
        B.proba(m); i = B.info[m]
        return f"{i['learner']}; {i['settings']}; features: {i['features']}"
    text = one(members[0]) if len(members) == 1 else "average of the probabilities of [" + "] and [".join(one(m) for m in members) + "]"
    if post is None:
        return text + ("; prediction: threshold 0.5" if K == 2 else "; prediction: most probable class")
    if post[0] == "thr":
        return text + f"; prediction: threshold {post[1]} instead of 0.5"
    cls = names[post[1]] if names else post[1]
    return text + f"; prediction: most probable class after multiplying the probability of class '{cls}' by {post[2]}"


class Deployed:
    """The model to use, refitted on all labeled data (SEARCH + POOL) with the configuration that was judged.
    predict(X) returns class names; predict_proba(X) returns probabilities in the order of `classes_`."""

    def __init__(self, parts, kind, param, classes):
        self.parts, self.kind, self.param, self.classes_ = parts, kind, param, list(classes)

    def predict_proba(self, X):
        K = len(self.classes_); P = []
        for m in self.parts:
            out = np.zeros((len(X), K)); out[:, np.asarray(m.classes_, int)] = m.predict_proba(X); P.append(out)
        p = sum(P) / len(P)
        if self.kind == "thr":
            p = np.column_stack([1 - (p[:, 1] >= self.param), p[:, 1] >= self.param]).astype(float)
        elif self.kind == "scale":
            p = p.copy(); p[:, self.param[0]] *= self.param[1]
        return p

    def predict(self, X):
        return np.asarray(self.classes_)[self.predict_proba(X).argmax(1)]


def refit(name, B, current, X, y, names):
    """Refit the configuration `name` on all labeled data (hyperparameters and features as chosen on SEARCH)."""
    from sklearn.base import clone
    members, post = parse(name, current)
    parts = []
    for m in members:
        B.proba(m); parts.append(clone(B.fitted[m]).fit(X, y))
    if post is None:
        return Deployed(parts, "plain", None, names)
    return Deployed(parts, post[0], post[1] if post[0] == "thr" else (post[1], post[2]), names)


class Progress:
    """A text progress bar on stderr, shown only on a terminal."""

    def __init__(self, total, label, enabled=True):
        self.total, self.label, self.t0 = max(total, 1), label, time.perf_counter()
        self.on = enabled and sys.stderr.isatty()

    def update(self, done, what=""):
        if not self.on:
            return
        el = time.perf_counter() - self.t0; frac = done / self.total; w = 24
        rem = el / frac - el if frac > 0 else 0
        fmt = lambda s: f"{int(s // 60)}:{int(s % 60):02d}"
        bar = "#" * int(w * frac) + "-" * (w - int(w * frac))
        sys.stderr.write(f"\r{self.label:15s} [{bar}] {done:>4d}/{self.total:<4d} {100 * frac:3.0f}%  elapsed {fmt(el)}"
                         f"  remaining ~{fmt(rem)}  {what[:28]:28s}")
        sys.stderr.flush()

    def close(self):
        if self.on:
            sys.stderr.write("\r" + " " * 118 + "\r"); sys.stderr.flush()


def scores(y, p):
    return dict(accuracy=accuracy_score(y, p), balanced_accuracy=balanced_accuracy_score(y, p),
                macro_F1=f1_score(y, p, average="macro"), weighted_F1=f1_score(y, p, average="weighted"))


def _scores_from_cm(C):
    """accuracy, balanced accuracy, macro-F1, weighted-F1 from confusion matrices C[..., true, pred] (vectorized)."""
    C = np.asarray(C, float); tp = np.diagonal(C, axis1=-2, axis2=-1); row = C.sum(-1); col = C.sum(-2); n = row.sum(-1)
    rec = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
    f1 = np.divide(2 * tp, row + col, out=np.zeros_like(tp), where=(row + col) > 0)
    present = row > 0
    return np.stack([tp.sum(-1) / n, (rec * present).sum(-1) / present.sum(-1),
                     (f1 * present).sum(-1) / present.sum(-1), (row * f1).sum(-1) / n], -1)


def ci(y, p, seed, B=2000):
    """95% bootstrap interval of each score over POOL rows (multinomial resampling of the confusion matrix)."""
    K = int(max(y.max(), p.max())) + 1; C = confusion_matrix(y, p, labels=range(K)).ravel(); n = int(C.sum())
    draws = np.random.default_rng(seed).multinomial(n, C / n, size=B).reshape(B, K, K)
    lo, hi = np.percentile(_scores_from_cm(draws), [2.5, 97.5], axis=0)
    return lo, hi


def print_cm(y, p, names, title):
    cm = confusion_matrix(y, p, labels=range(len(names))); w = max(8, max(len(n) for n in names) + 1)
    print(f"\n{title} (rows = true class, columns = predicted class)")
    print(" " * w + "".join(f"{n:>{w}}" for n in names))
    for n, r in zip(names, cm):
        print(f"{n:>{w}}" + "".join(f"{v:>{w}d}" for v in r))


def decide_from_predictions(a, t_start, t0):
    """The gate only: labels and predictions are given, nothing is trained."""
    df = pd.read_csv(a.predictions)
    if "y" not in df or "current" not in df:
        sys.exit("the predictions file needs the columns 'y' and 'current'")
    cand = [c for c in df.columns if c not in ("y", "current")]
    if not cand:
        sys.exit("the predictions file has no candidate column")
    labels = sorted(set(df["y"].astype(str)) | set(df["current"].astype(str)) | {v for c in cand for v in df[c].astype(str)},
                    key=lambda s: (len(s), s) if s.lstrip("-").isdigit() else (1e9, s))
    code = {s: i for i, s in enumerate(labels)}; enc = lambda col: df[col].astype(str).map(code).to_numpy()
    y, pc = enc("y"), enc("current"); preds = {c: enc(c) for c in cand}; K = len(labels)
    fs = a.seed if a.fold_seed is None else a.fold_seed; bs = a.seed if a.boot_seed is None else a.boot_seed
    tg = time.perf_counter()
    d = core.gate_select(y, pc, preds, a.metric, fold_seed=fs, boot_seed=bs, K=max(K, 2))
    gs = time.perf_counter() - tg
    print(f"Predictions: {a.predictions}: {len(y)} judging rows, {K} classes, {len(cand)} candidate{'s' if len(cand) > 1 else ''}")
    print(f"DECISION: {'INSTALL ' + str(d.chosen) if d.accept else 'KEEP the current model'}  "
          f"(best candidate {d.chosen}; {a.metric} gain {d.gain_pool:+.4f}; unlucky-case gain {d.lower_bound:+.4f}; limit {core.MARGIN:+.4f})")
    total = time.perf_counter() - t0
    print(f"Time: start {t_start:%Y-%m-%d %H:%M:%S}, total {total:.2f} s (BRIDGE decision {gs:.3f} s)")
    if a.out:
        res = dict(predictions=a.predictions, rows=len(y), classes=labels, metric=a.metric, n_candidates=len(cand),
                   chosen=d.chosen, accept=bool(d.accept), lower_bound=d.lower_bound, gain_pool_chosen=d.gain_pool,
                   naive_accept=bool(d.naive_accept), margin=core.MARGIN, fold_seed=fs, boot_seed=bs,
                   timing=dict(start=t_start.isoformat(timespec="seconds"), total_seconds=round(total, 3), bridge_decision_seconds=round(gs, 4)))
        Path(f"{a.out}.json").write_text(json.dumps(res, indent=2))
    return 0


def _frame(spec, target):
    """Raw rows of a dataset spec: X (DataFrame) and the labels as text."""
    if spec in BUILTIN:
        from sklearn import datasets
        d = getattr(datasets, f"load_{spec}")(as_frame=True); X, y = d.data, d.target.map(dict(enumerate(d.target_names)))
    elif spec.startswith("openml:"):
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=int(spec.split(":")[1]), as_frame=True, parser="auto"); X, y = d.data, d.target
    else:
        df = pd.read_pickle(spec) if spec.endswith(".pkl") else pd.read_csv(spec)
        t = target or df.columns[-1]
        if t not in df.columns:
            sys.exit(f"target column {t!r} not found; columns: {list(df.columns)}")
        X, y = df.drop(columns=[t]), df[t]
    y = pd.Series(y).astype(str); keep = (y.notna() & (y != "nan")).to_numpy()
    return X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)


def _plain(cfg, K, names):
    """A paper configuration in plain words."""
    w, post = cfg; used = [(m, wi) for m, wi in w.items() if wi]
    model = (f"{used[0][0]} (32 bags)" if len(used) == 1 else
             "average of the probabilities of " + ", ".join(f"{m} (weight {wi:g})" for m, wi in used) + " (each 32 bags)")
    if K == 2:
        return f"{model}; predicts class '{names[1]}' when its probability is at least {float(post):g} (instead of 0.5)"
    if post is None:
        return f"{model}; predicts the most probable class"
    return f"{model}; predicts the most probable class after multiplying the probability of '{names[post[0]]}' by {post[1]:g}"


def run_paper(a, t_start, t0):
    """The paper's validated procedure (protocol.run) on the user's data."""
    from . import protocol
    if not a.show_warnings:
        warnings.filterwarnings("ignore")
    if a.current not in (None, "hgb", "HGB"):
        sys.exit("the validated procedure uses HGB as the current model; for another current model give --candidates or --auto")
    X, ys = _frame(a.dataset, a.target); n_data = len(X)
    if a.reference:
        Xr, yrs = _frame(a.reference, a.target)
        if list(Xr.columns) != list(X.columns):
            sys.exit("the reference rows must have the same columns as the dataset")
        both = protocol.encode(pd.concat([X, Xr], ignore_index=True)); X, Xr = both.iloc[:n_data], both.iloc[n_data:]
    else:
        X = protocol.encode(X); yrs = pd.Series([], dtype=str)
    names = sorted(set(ys) | set(yrs)); code = {n: i for i, n in enumerate(names)}; K = len(names)
    y = ys.map(code).to_numpy()
    print(f"Data: {a.dataset}: {n_data} rows, {X.shape[1]} features, {K} classes ("
          + ", ".join(f"{n} {c}" for n, c in zip(names, np.bincount(y, minlength=K))) + ")")
    if K > 2 and a.metric != "wF1":
        sys.exit("for more than two classes use --metric wF1")
    bar = Progress(1, "Training models", not a.no_progress)
    try:
        r = protocol.run(X, y, a.seed, metric=a.metric, K=K,
                         X_ref=Xr.reset_index(drop=True) if a.reference else None,
                         y_ref=yrs.map(code).to_numpy() if a.reference else None,
                         progress=lambda d, t, w: (setattr(bar, "total", t), bar.update(d, w)))
    except ValueError as e:
        bar.close(); sys.exit(f"cannot run the procedure: {e}")
    bar.close()
    unused = n_data - r.total
    print(f"Procedure: the paper's procedure{' (validated in its runs)' if r.validated else ''}. {r.total} training rows"
          + (f" drawn from the {n_data}" if unused else "") + f": models are trained on {r.search} rows "
          f"({r.bags} bags of 80%; HGB, DT, SVC) and judged on the other {r.pool}, which no model saw in training.")
    if not r.validated:
        print("WARNING: outside the conditions of the paper's runs, so the paper's results do not cover this decision:")
        for c in r.conditions:
            print(f"  - {c}")
    cur = "HGB (32 bags), threshold 0.5" if K == 2 else "HGB (32 bags), most probable class"
    nc = len(protocol.candidates(K)); chosen = r.name(names)
    print(f"Current model: {cur}.  Candidates: {nc} (" + ("threshold; HGB+DT and HGB+DT+SVC weights x threshold" if K == 2
          else "HGB/DT/SVC weights; class scales of HGB and of HGB+DT") + ")")
    print("\n" + "=" * 78)
    if r.accept:
        print(f"  DECISION     : INSTALL {chosen}")
        print(f"  Model to use : {_plain(r.chosen, K, names)}")
    else:
        print(f"  DECISION     : KEEP the current model ({cur})")
    lim = f"To install, BRIDGE requires the gain to stay above {core.MARGIN:+.4f} in {a.metric} even in an unlucky case"
    if r.chosen is None:
        title, why = "Why keep?", f"No candidate scored higher than the current model on the {r.pool} judging rows."
    elif r.accept:
        title = "Why install?"
        why = (f"The best of {nc} candidates ({chosen}) gains {r.gain_pool:+.4f} in {a.metric} on the {r.pool} judging rows. "
               f"{lim}; here the unlucky-case gain is {r.lower_bound:+.4f}, above that limit.")
    else:
        title = "Why keep, although a candidate scored higher?"
        why = (f"The best of {nc} candidates ({chosen}) gains {r.gain_pool:+.4f} in {a.metric} on the {r.pool} judging rows, "
               f"but when {nc} candidates are tried some score higher by luck alone. {lim}; here the unlucky-case gain is "
               f"{r.lower_bound:+.4f}, below that limit, so it is not proof of a real improvement.")
    print(f"  {title}")
    for line in textwrap.wrap(why, 74):
        print(f"    {line}")
    print("=" * 78)
    if a.reference:
        lo, hi = r.ref_interval
        verdict = {"better": "clearly better", "worse": "clearly worse", "uncertain": "not clearly better or worse"}[r.ref_class]
        what = chosen if r.chosen is not None else "the current model"
        print(f"\nCheck on the {len(yrs)} reference rows (never used above): {what} vs the current model: {a.metric} gain "
              f"{r.gain_ref:+.4f} (95% interval [{lo:+.4f}, {hi:+.4f}]): {verdict}.")
        if r.chosen is not None:
            if r.accept:
                print("  BRIDGE installed it: " + ("a harmful install." if r.gain_ref <= 0 else "a useful install."))
            else:
                print("  BRIDGE kept the current model: " + ("a missed improvement." if r.gain_ref > 0 else "a harmful change avoided."))
    if a.detail:
        print(f"\nDETAILS\n  BRIDGE bound (5% bootstrap quantile, {core.REPLICATES} replicates, {core.FOLDS}-fold cross-fitting): "
              f"{r.lower_bound:+.6f}; limit {core.MARGIN}")
        print(f"  Naive rule (install if the best candidate scores higher on the judging rows): "
              + (f"would INSTALL {chosen}" if r.naive_accept else "would keep the current model"))
        print(f"  Seed {a.seed}: training draw, bags, folds and bootstrap all derive from it, as in the paper's trials.")
    if a.save_model:
        import joblib
        joblib.dump(protocol.Model(r, names), a.save_model)
        print(f"\nSaved the model to use ({chosen if r.accept else 'current model'}; the bagged models as judged) to {a.save_model}")
        print(f"  load: m = joblib.load('{a.save_model}'); m.predict(new_rows)   (returns class names; code text columns "
              f"with bridge_gate.protocol.encode first)")
    total = time.perf_counter() - t0
    print(f"\nTime: start {t_start:%Y-%m-%d %H:%M:%S}, total {total:.1f} s")
    if a.out:
        res = dict(procedure="paper", dataset=a.dataset, rows=n_data, classes=names, K=K, metric=a.metric, seed=a.seed,
                   total=r.total, search=r.search, pool=r.pool, bags=r.bags, validated=r.validated, conditions=r.conditions,
                   n_candidates=nc, chosen=chosen, chosen_config=str(r.chosen), accept=r.accept, lower_bound=r.lower_bound,
                   gain_pool_chosen=r.gain_pool, naive_accept=r.naive_accept, margin=core.MARGIN,
                   reference=a.reference, reference_rows=len(yrs) if a.reference else None, gain_ref=r.gain_ref,
                   ref_interval=r.ref_interval, ref_class=r.ref_class,
                   timing=dict(start=t_start.isoformat(timespec="seconds"), total_seconds=round(total, 2)))
        Path(f"{a.out}.json").write_text(json.dumps(res, indent=2))
        print(f"Saved: {a.out}.json")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bridge-gate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help=f"{', '.join(BUILTIN)}, a CSV file, or openml:<data id>")
    ap.add_argument("--predictions", help="decide from predictions you computed yourself: a CSV file with column 'y' (true labels "
                    "on POOL), column 'current' (the current model's predicted labels) and one column per candidate")
    ap.add_argument("--fold-seed", type=int, default=None, help="seed of the cross-fitting folds (default --seed)")
    ap.add_argument("--boot-seed", type=int, default=None, help="seed of the bootstrap (default --seed)")
    ap.add_argument("--target", help="target column of a CSV file (default: the last column)")
    ap.add_argument("--current", default=None, help="the model in use now; default hgb (the paper's base learner), "
                    "switched to rf when SEARCH has fewer than 200 rows (hgb needs >= 20 rows per leaf, so < 10 leaves)")
    ap.add_argument("--candidates", default=None, help="your own comma-separated candidate names (see above), e.g. "
                    "tuned_hgb,rf,logreg; without --candidates and --auto the paper's validated procedure is used")
    ap.add_argument("--reference", help="labeled rows the procedure never sees (same columns, CSV or .pkl): the decision "
                    "is checked on them, as the paper checks on its locked reference (validated procedure only)")
    ap.add_argument("--auto", nargs="?", const="single", choices=["single", "combined"],
                    help="generate the candidates: 'single' (default when given) changes one thing at a time (other learner, "
                         "tuning, feature selection, threshold or class scale, ensemble); 'combined' builds every combination "
                         "(learner x tuning x features x post-processing, plus ensembles of two)")
    ap.add_argument("--metric", default="wF1", choices=["wF1", "acc", "bacc", "macroF1", "nb10", "nb20", "F2"], help="acc, bacc, macroF1, net benefit nb10/nb20, F2: two classes only")
    ap.add_argument("--search-fraction", type=float, default=core.SEARCH_FRACTION,
                    help="share of the data used for training and tuning (default 0.259, as in the paper); the rest is POOL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=10, help="how many candidates to list (default 10)")
    ap.add_argument("--all", action="store_true", help="list every candidate (default: the --top best)")
    ap.add_argument("--cm", action="store_true", help="print the confusion matrices on POOL")
    ap.add_argument("--show-warnings", action="store_true", help="show library warnings (hidden by default)")
    ap.add_argument("--no-progress", action="store_true", help="do not show the progress bar")
    ap.add_argument("--detail", action="store_true", help="also print the technical details: 95% intervals, BRIDGE's bound, "
                    "the naive rule's choice, candidates by kind of change, notes about the data")
    ap.add_argument("--quiet", action="store_true", help="(kept for compatibility; notes are shown only with --detail)")
    ap.add_argument("--out", help="save the result as <out>.json and <out>_candidates.csv")
    ap.add_argument("--save-model", help="refit the model to use on all labeled data and save it (joblib file)")
    a = ap.parse_args(argv)
    t_start = datetime.now(); t0 = time.perf_counter()
    if a.predictions:
        return decide_from_predictions(a, t_start, t0)
    if not a.dataset:
        ap.error("give --dataset (or --predictions)")
    if not a.show_warnings:
        warnings.filterwarnings("ignore")

    if not (a.candidates or a.auto):
        return run_paper(a, t_start, t0)
    X, y, names = load(a.dataset, a.target); K = len(names)
    print(f"Data: {a.dataset}: {len(y)} rows, {X.shape[1]} features, {K} classes (" + ", ".join(f"{n} {c}" for n, c in zip(names, np.bincount(y))) + ")")
    if a.metric != "wF1" and K > 2:
        sys.exit("for more than two classes use --metric wF1")
    search, pool = core.split(y, search_fraction=a.search_fraction, seed=a.seed)
    Xs, ys, Xp, yp = X.iloc[search], y[search], X.iloc[pool], y[pool]
    ok, notes = core.check(ys, yp); auto_note = ""
    if a.current is None:
        a.current = "hgb" if len(search) >= SMALL_SEARCH else "rf"
        if a.current == "rf":
            auto_note = f" (chosen automatically: gradient boosting needs at least {SMALL_SEARCH} training rows)"
    elif a.current.split("_")[-1] == "hgb" and len(search) < SMALL_SEARCH:
        notes.append(f"only {len(search)} training rows: hgb (>= 20 rows per leaf) learns little; consider --current rf")
    print(f"Models are trained on {len(search)} rows and judged on the other {len(pool)} rows, which no model saw in training.")
    print(f"Current model: {a.current}{auto_note}")
    print("Note: your own candidates are outside the paper's validated procedure (run without --candidates and --auto "
          "for that procedure).")
    if a.detail:
        for n in notes:
            print("  note:", n)

    B = Builder(Xs, ys, Xp, K, a.seed)
    cands = auto_candidates(a.current, K, a.auto) if a.auto else [c.strip() for c in a.candidates.split(",") if c.strip()]
    needed = list(dict.fromkeys([a.current] + [m for c in cands for m in parse(c, a.current)[0]]))
    bar = Progress(len(needed), "Training models", not a.no_progress)
    for i, m in enumerate(needed):
        bar.update(i, m); B.proba(m)
    bar.update(len(needed), "done"); bar.close()
    pc = B.proba(a.current).argmax(1)
    preds = {c: candidate_proba(c, B, a.current, K).argmax(1) for c in cands}
    rows = sorted((dict(candidate=c, gain_pool=float(core.delta(core.counts(yp, pc, p, K), K, a.metric))) for c, p in preds.items()),
                  key=lambda r: -r["gain_pool"])
    tg = time.perf_counter(); d = core.gate_select(yp, pc, preds, a.metric, fold_seed=a.seed, boot_seed=a.seed, K=K); gate_seconds = time.perf_counter() - tg
    cf_gain = float(core.delta(d.counts, K, a.metric))
    use = d.chosen if d.accept else a.current
    sc_use = scores(yp, preds[use] if d.accept else pc); lo, hi = ci(yp, preds[use] if d.accept else pc, a.seed)
    sc_cur = scores(yp, pc); n, N = len(yp), len(cands)
    right = lambda p: p == yp
    fixed = lambda p: int((right(p) & ~right(pc)).sum()); broke = lambda p: int((~right(p) & right(pc)).sum())

    # 1. comparison table
    def sc(pred):
        C = confusion_matrix(yp, pred, labels=range(K)); base4 = _scores_from_cm(C)
        if K != 2:
            return np.concatenate([base4, [np.nan] * 3])
        tn, fp, fn, tp = C.ravel().astype(float); nn = C.sum()
        nb = lambda pt: tp / nn - fp / nn * pt / (1 - pt)
        f2 = 5 * tp / (5 * tp + 4 * fn + fp) if (5 * tp + 4 * fn + fp) > 0 else 0.0
        return np.concatenate([base4, [nb(.1), nb(.2), f2]])      # net benefit and F2 as in the paper's clinical metrics
    shown = rows if a.all else rows[:a.top]
    if d.chosen is not None and all(r["candidate"] != d.chosen for r in shown):
        shown = shown + [r for r in rows if r["candidate"] == d.chosen]
    wn = max(16, max(len(r["candidate"]) for r in shown) + 1); wk = max(16, max(len(kind_of(r["candidate"], a.current)) for r in shown) + 1)
    every = a.all or len(rows) <= a.top
    print(f"\nCOMPARISON on the {n} judging rows ("
          + (f"current model and all {N} candidate{'s' if N > 1 else ''})" if every else f"current model and the {len(shown)} best of {N} candidates; --all lists every one)"))
    # the paper's metrics: weighted-F1 (primary), balanced accuracy, macro-F1, accuracy; the decision metric first, marked *
    idx = {"wF1": 3, "bacc": 1, "macroF1": 2, "acc": 0, "nb10": 4, "nb20": 5, "F2": 6}
    head = {3: "w-F1", 1: "bal.acc", 2: "macro-F1", 0: "accuracy", 4: "nb10", 5: "nb20", 6: "F2"}
    cols = [idx[a.metric]] + [c for c in (3, 1, 2, 0) if c != idx[a.metric]]
    hdr = "".join(f"{head[c] + ('*' if j == 0 else ''):>10s}" for j, c in enumerate(cols))
    gl = f"{head[idx[a.metric]]}* current -> candidate"
    print(f"  {'model':{wn}s} {'change':{wk}s}{hdr}   {gl:27s} {'net rows':>8s}  result")
    s0 = sc(pc)
    print(f"  {a.current:{wn}s} {'(current model)':{wk}s}" + "".join(f"{s0[c]:10.3f}" for c in cols)
          + f"   {'':27s} {'':>8s}  {'REPLACED' if d.accept else 'KEPT'}")
    for r in shown:
        p = preds[r["candidate"]]; s1 = sc(p); is_best = r["candidate"] == d.chosen
        res = ("INSTALLED" if d.accept else "best, not installed") if is_best else ""
        print(f"  {r['candidate']:{wn}s} {kind_of(r['candidate'], a.current):{wk}s}" + "".join(f"{s1[c]:10.3f}" for c in cols)
              + f"   {s0[idx[a.metric]]:.3f} -> {s1[idx[a.metric]]:.3f} ({s1[idx[a.metric]] - s0[idx[a.metric]]:+.3f})   {fixed(p) - broke(p):+8d}  {res}")
    print(f"  * = the metric BRIDGE decides on ({head[idx[a.metric]]}); the arrow column shows it for the current model and the "
          "candidate, and the difference (the paper's delta).")
    print(f"  net rows = rows the candidate gets right that the current model gets wrong, minus the reverse.")

    # 2. decision, model to use, reason in plain words
    print("\n" + "=" * 78)
    if d.accept:
        print(f"  DECISION     : INSTALL the new model {d.chosen}  ({kind_of(d.chosen, a.current)})")
    else:
        print(f"  DECISION     : KEEP the current model {a.current}, unchanged")
    print_spec("Model to use", use, B, a.current, K, names)
    if d.accept:
        print_spec("Replaces", a.current, B, a.current, K, names)
    m0 = s0[idx[a.metric]]; mname = head[idx[a.metric]]
    if d.chosen is None:
        title = "Why keep?"
        why = f"No candidate scored higher than the current model on the {n} judging rows."
    else:
        pb = preds[d.chosen]; m1 = sc(pb)[idx[a.metric]]; net = fixed(pb) - broke(pb)
        diff = (f"{mname} {m0:.3f} -> {m1:.3f}; {abs(net)} {'more' if net >= 0 else 'fewer'} row{'s' if abs(net) != 1 else ''} "
                f"right out of {n}")
        if d.accept:
            title = "Why install?"
            why = (f"The new model scored higher ({diff}), and "
                   + (f"with only 1 candidate tried, a difference this size is very unlikely to be luck. " if N == 1 else
                      f"even after trying {N} candidates, a difference this size is very unlikely to be luck. ")
                   + f"To install, BRIDGE requires the gain to stay above {core.MARGIN:+.4f} in {mname} even in an unlucky case; "
                   f"here the unlucky-case gain is {d.lower_bound:+.4f}, above that limit.")
        else:
            title = "Why keep, although a candidate scored higher?"
            why = (f"The best candidate ({d.chosen}) scored higher ({diff}). "
                   + (f"But {N} candidates were tried, and when that many are tried, some score higher by luck alone. "
                      if N > 1 else f"But a difference of {abs(net)} row{'s' if abs(net) != 1 else ''} out of {n} could be luck. ")
                   + f"To install, BRIDGE requires the gain to stay above {core.MARGIN:+.4f} in {mname} even in an unlucky case; "
                   f"here the unlucky-case gain is {d.lower_bound:+.4f}, below that limit, so it is not proof of a real improvement.")
    print(f"  {title}")
    for line in textwrap.wrap(why, 74):
        print(f"    {line}")
    print("=" * 78)

    # 3. technical details only on request
    if a.detail:
        label = {"accuracy": "accuracy", "balanced_accuracy": "balanced accuracy", "macro_F1": "macro F1", "weighted_F1": "weighted F1"}
        print(f"\nDETAILS")
        print(f"  Expected performance of the model to use on new data (on the {n} judging rows; 95% interval):")
        for j, (k, v) in enumerate(sc_use.items()):
            print(f"    {label[k]:18s} {v:.3f}  [{lo[j]:.3f}, {hi[j]:.3f}]")
        if d.accept and N > 1:
            print(f"    (the new model was picked among {N} on these rows, so these values are slightly optimistic)")
        print(f"  BRIDGE: {a.metric} gain of {d.chosen}: on the judging rows {d.gain_pool:+.4f}; honest (cross-fitted) {cf_gain:+.4f}; "
              f"pessimistic bound {d.lower_bound:+.4f}; installs only if the bound exceeds {core.MARGIN}")
        print(f"  Naive rule (install the best judging score if higher): {'would INSTALL ' + str(d.chosen) if d.naive_accept else 'would keep the current model'}")
        kinds = {}
        for c in cands:
            kinds.setdefault(kind_of(c, a.current), []).append(c)
        order = sorted(kinds, key=lambda k: (k.count("+"), k)); w = max(len(k) for k in order) + 1; wc = max(14, max(len(c) for c in cands) + 1)
        print(f"  Candidates by kind of change ({a.metric} gain over the current model):")
        print(f"    {'kind of change':{w}s} {'tried':>5s}   {'best candidate':{wc}s} {'gain':>8s}   {'higher than current':>19s}")
        for k in order:
            rk = [r for r in rows if kind_of(r["candidate"], a.current) == k]
            print(f"    {k:{w}s} {len(rk):5d}   {rk[0]['candidate']:{wc}s} {rk[0]['gain_pool']:+8.4f}   {sum(r['gain_pool'] > 0 for r in rk):12d} of {len(rk)}")
        for nt in notes + [w_ for w_ in d.warnings if w_ not in notes]:
            print("  note:", nt)
    if a.cm:
        print_cm(yp, pc, names, f"Confusion matrix on the judging rows, current model ({a.current})")
        if d.chosen is not None:
            print_cm(yp, preds[d.chosen], names, f"Confusion matrix on the judging rows, best candidate ({d.chosen})")
    if a.save_model:
        import joblib
        joblib.dump(refit(use, B, a.current, X, y, names), a.save_model)
        print(f"\nSaved the model to use ({use}), refitted on all {len(y)} labeled rows, to {a.save_model}")
        print(f"  load: m = joblib.load('{a.save_model}'); m.predict(new_rows)   (returns class names)")

    t_end = datetime.now(); total = time.perf_counter() - t0
    timing = dict(start=t_start.isoformat(timespec="seconds"), end=t_end.isoformat(timespec="seconds"), total_seconds=round(total, 2),
                  training_seconds=round(B.fit_seconds, 2), bridge_decision_seconds=round(gate_seconds, 2))
    print(f"\nTime: start {t_start:%Y-%m-%d %H:%M:%S}, end {t_end:%H:%M:%S}, total {total:.1f} s "
          f"(training {B.fit_seconds:.1f} s, BRIDGE decision {gate_seconds:.2f} s, {N} candidate{'s' if N > 1 else ''})")
    if a.out:
        res = dict(dataset=a.dataset, rows=len(y), classes=names, search=len(search), pool=len(pool), metric=a.metric,
                   current=a.current, n_candidates=len(cands), chosen=d.chosen, accept=bool(d.accept), lower_bound=d.lower_bound,
                   gain_pool_chosen=d.gain_pool, gain_crossfitted=cf_gain, naive_accept=bool(d.naive_accept), margin=core.MARGIN, seed=a.seed,
                   model_to_use=use, model_to_use_description=describe(use, B, a.current, K, names),
                   expected_scores=sc_use, expected_scores_95=dict(zip(sc_use, zip(lo.tolist(), hi.tolist()))),
                   current_description=describe(a.current, B, a.current, K, names), current_scores_pool=sc_cur,
                   chosen_description=None if d.chosen is None else describe(d.chosen, B, a.current, K, names),
                   timing=timing, notes=notes + d.warnings)
        Path(f"{a.out}.json").write_text(json.dumps(res, indent=2))
        pd.DataFrame([dict(r, kind=kind_of(r["candidate"], a.current)) for r in rows]).to_csv(f"{a.out}_candidates.csv", index=False)
        print(f"\nSaved: {a.out}.json and {a.out}_candidates.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# bridge-gate: user manual

BRIDGE answers one question: should a new candidate model replace the current
model? It looks at labeled rows that neither model saw during training. The
candidate is accepted only when the lower bootstrap bound of its gain is above
a fixed margin. Otherwise the current model is kept.

## 1. Install

    pip install bridge-gate

From a local checkout:

    pip install ./pkg

Python 3.10 or later is required. NumPy, SciPy, pandas and scikit-learn are
installed automatically. The command line program is `bridge-gate`; the Python
package is `bridge_gate`.

## 2. Quick start: command line

### 2.1 Default: reference procedure

    bridge-gate --dataset mango.csv --target grade
    bridge-gate --dataset data.csv --target label --reference held_out.csv --out result

Without `--candidates` and `--auto`, the program runs the reference procedure.
The settings were fixed during development and then confirmed on new datasets.
For the same data and `--seed`, the decision matches the reference trial code.

| Step | Behavior |
|---|---|
| training rows | 2,430 rows drawn from the data (630 SEARCH, 1,800 POOL); 1,000 (259 / 741) when fewer than 2,480 rows are given |
| models | 32 bags of 80% of SEARCH; HGB, DT and SVC with fixed settings; probabilities averaged per learner |
| current model | HGB at threshold 0.5 (two classes) or argmax (more classes) |
| candidates | two classes: HGB threshold, HGB+DT weights and HGB+DT+SVC weights, each with a threshold (173 configurations); more classes: HGB/DT/SVC weights and class scales of HGB or HGB+DT |
| decision | BRIDGE with its frozen settings (section 9) |

Conditions covered by the reference procedure:

- two classes: at least 1,050 rows;
- more classes: at least 2,480 rows and 7 to 10 classes (development went up to
  26 classes);
- metric: `wF1`, `bacc` or `macroF1` for two classes; `wF1` for more classes.

Outside these conditions the program still runs and prints a warning. With
fewer rows it uses all available rows. The reference results do not cover such
decisions.

### 2.2 Reference check

`--reference` takes labeled rows that the procedure never sees, with the same
columns. The program reports the gain of the chosen candidate over the current
model on those rows, its 95% interval, and the outcome: useful install, harmful
install, missed improvement, or harmful change avoided.

### 2.3 Custom candidates

    bridge-gate --dataset iris --current rf --auto
    bridge-gate --dataset iris --current rf --auto combined
    bridge-gate --dataset mango.csv --target grade --current rf --auto --out result
    bridge-gate --dataset breast_cancer --current hgb --candidates tuned_hgb,fs50_hgb,thr0.35,ens_hgb+logreg

Custom candidates are outside the reference procedure. The output states this.

Data sources:

- built-in: `iris`, `wine`, `breast_cancer`, `digits`;
- a CSV file with `--target` (default: the last column);
- `openml:<id>`.

Numeric columns are filled with the median. Text columns are filled with the
most frequent value and one-hot encoded.

`--current` sets the model in use (default `hgb`). On very small data (SEARCH
below 200 rows) gradient boosting learns little, so the program warns. Give the
real model in that case, for example `--current rf`.

A candidate is built on SEARCH only from these components:

| Component | Names | Meaning |
|---|---|---|
| learner | `hgb`, `rf`, `dt`, `logreg`, `svc`, `knn` | model family |
| tuning | prefix `tuned_` | tuned on SEARCH with a small grid and 3-fold CV |
| features | prefix `fs25_` or `fs50_` | top 25% or 50% of features by mutual information on SEARCH |
| ensemble | `ens_<a>+<b>` | average of predicted probabilities |
| threshold (two classes) | suffix `@thr<t>` | decision threshold t from the reference grid |
| class scale (more classes) | suffix `@scale<k>x<v>` | multiply class k probability by v (0.5, 0.7, 1.4, 2) |

Examples: `tuned_fs50_rf@thr0.35`, `ens_tuned_hgb+fs50_logreg`,
`fs25_svc@scale1x1.4`. The short forms `thr<t>` and `scale<k>x<v>` apply the
step to the current model.

`--auto single` (or `--auto`) changes one thing at a time: about 30 to 40
candidates. `--auto combined` builds all combinations: 6 learners, with and
without tuning and feature selection, with and without a post-step, plus
ensembles of two. That is 749 candidates for two classes and 533 for three.
The full run took 23 to 30 seconds on Iris and breast cancer in the reference
environment.

Try few, well-motivated candidates on small data. The more candidates are
tried, the more likely the best one looks good by chance. Cross-fitting accounts
for this, so with many candidates and a small POOL the gate keeps the current
model more often. Example with the synthetic mango file (POOL 223 rows): with
three candidates, logistic regression (+0.042 on POOL) was installed; with 31
candidates (`--auto`), the same candidate was kept out.

### 2.4 What the program prints

1. COMPARISON: the current model and the candidates on the POOL rows. Columns
   are weighted-F1 (the decision metric, marked `*`), balanced accuracy,
   macro-F1 and accuracy. The arrow column traces the decision metric from the
   current model to the candidate, for example `0.946 -> 0.982 (+0.036)`. The
   difference is the reference delta. `net rows` counts rows the candidate gets
   right that the current model gets wrong, minus the reverse. The result column
   reads KEPT, REPLACED, INSTALLED, or "best, not installed".
2. DECISION: KEEP or INSTALL, followed by the full configuration to use:
   learner, tuning, features and prediction rule. For an installed model it
   also shows what is replaced.
3. Reason: why the gate kept or installed the candidate, with the gain on POOL
   and the lower bound used in the decision.
4. Time: start, end, total, and the split between training and the gate
   decision. `--out` also saves these values.

A progress bar is shown on a terminal while models are trained. It is not
written to redirected output. `--no-progress` turns it off.

`--detail` adds technical numbers: 95% intervals, the cross-fitted gain, the
lower bound, the naive rule's choice, candidates grouped by kind of change, and
notes about the data.

Scores can be lower than a report of 100%. Models are trained on SEARCH only
(by default 26% of the data), so that many rows remain for judging. Small test
sets cannot separate 93% from 100%. On Iris with 45 judging rows, for example,
the 95% interval is [0.93, 1.00].

`--save-model file.joblib` refits the model to use on all labeled rows and saves
it. Load it with `joblib.load` and call `predict`; the result contains class
names.

Useful options:

- `--cm`: print both confusion matrices.
- `--detail`: print the technical numbers.
- `--show-warnings`: show library warnings (hidden by default).
- `--out`: save `<out>.json` and `<out>_candidates.csv`.
- `--search-fraction`: set the SEARCH share (default 0.259).
- `python -m bridge_gate`: same as `bridge-gate`.

## 3. Python workflow

1. Split labeled data once into SEARCH and POOL:

        import bridge_gate as bridge
        search, pool = bridge.split(y, seed=0)

2. Build both models on SEARCH only. Tuning, feature selection and threshold
   choice also belong to SEARCH. Never use POOL for these steps.

3. Predict POOL with both models and ask the gate:

        d = bridge.gate(y[pool], current.predict(X[pool]), candidate.predict(X[pool]))
        if d.accept:
            deploy(candidate)

4. For several candidates, pass all of them. Do not pick the best one on POOL
   first, because that makes the evidence optimistic:

        d = bridge.gate_select(
            y[pool],
            current.predict(X[pool]),
            {"threshold 0.35": p1, "ensemble": p2, "tuned": p3},
        )
        d.chosen   # candidate to deploy when d.accept is True

## 4. What the gate returns

| Field | Meaning |
|---|---|
| `accept` | `True`: install the candidate. `False`: keep the current model. |
| `lower_bound` | Lower 5% bootstrap quantile of the gain. The gate accepts when it exceeds 0.005. |
| `gain_pool` | Gain on POOL. The common rule "install if validation improved" accepts when this is positive (`naive_accept`). |
| `chosen` | Selected candidate (`gate_select`). |
| `K`, `n` | Number of classes and POOL size. |
| `warnings` | Notes about the data, for example a class with very few rows. |

`bridge.check(y[search], y[pool])` lists the same notes before training.

## 5. Conditions of use

| # | Condition | Reason |
|---|---|---|
| 1 | POOL is never used for training, tuning or selection. | The evidence must come from rows that did not shape the models. |
| 2 | Candidates are fixed before POOL is examined. With several candidates, use `gate_select`. | Picking the best of many on the same rows inflates its apparent gain. |
| 3 | POOL and deployment data come from the same population. | The bound was evaluated on decisions from one population; it is not a guarantee for a single deployment. Distribution shift was not tested. |
| 4 | The metric is computed from predicted labels: weighted-F1 for any number of classes; accuracy, balanced accuracy and macro-F1 for two classes. | Probability metrics (Brier, log-loss) and ranking metrics (AUPRC) are not supported. |
| 5 | Every class appears in POOL. | A class absent from POOL cannot be judged. `check` reports it. |
| 6 | For imbalanced two-class data, prefer balanced accuracy or macro-F1. | Under weighted-F1 the rare class has little weight. |

## 6. Tested and not tested

Tested:

- two-class tasks on many datasets;
- multiclass tasks with 7 to 10 classes;
- training sets of 1,000 to 6,000 rows;
- minority classes down to 5% of the data;
- updates by threshold shift, ensemble weights, class reweighting, feature
  selection and hyperparameter tuning.

In the accompanying paper, the common rule accepted harmful updates about one
time in five. BRIDGE accepted about one time in a hundred on the same decisions.

Use with care:

- tuning updates: BRIDGE removed most harmful tuning results, but the formal
  risk criterion was not met for every learner;
- very small data (a few hundred rows): the gate runs, but only large, clear
  improvements can be accepted, and this range is still being evaluated.

Not tested:

- distribution shift over time;
- sequential chains of updates;
- deep or foundation models (the gate needs only predictions, but no such
  evidence exists yet);
- more than 30 classes (memory grows with the cube of the number of classes).

## 7. Reading a decision

`accept = False` is not a verdict that the candidate is worse. It means the
evidence is not strong enough. With small POOL data this is the common outcome.

The risk statement is about many decisions, not one. Over many accepted
updates, few turn out harmful. A single accepted update can still be harmful.

The margin is 0.005 on the chosen metric. The candidate must be better by more
than that under the pessimistic estimate.

## 8. Reproducing a stored result

    python -m bridge_gate --dataset <data.csv> --target <label> --seed <seed> --out <result>

The `--out` JSON contains the decision, the chosen configuration, the lower
bound, the gain on POOL and the timing. The same inputs and seed produce the
same decision as the reference trial code.

## 9. Frozen settings

Margin 0.005; lower bootstrap quantile 5%; 20,000 bootstrap replicates; 3
cross-fitting folds. These values were fixed during development and were not
tuned on the evaluation data. Do not change them per use.

## 10. License

MIT. See [LICENSE](https://github.com/DwiSunaryono/BRIDGE-ML/blob/main/LICENSE).

# bridge-gate

BRIDGE decides whether a candidate model may replace the current model. It
uses labeled rows that neither model saw during training (POOL) and the
predicted labels of both models on those rows. The number of classes is read
from the labels.

## Install

    pip install bridge-gate

From a local checkout:

    pip install ./pkg

Python 3.10 or later is required. NumPy, SciPy, pandas and scikit-learn are
installed automatically.

## Quick start

    pip install bridge-gate
    bridge-gate --dataset breast_cancer

The second command runs the gate on a built-in dataset in about five seconds
and ends with a result card:

    ==============================================================================
      RESULT: KEEP the current model
    ==============================================================================
                                Current model         Best candidate
      Model                     HGB, threshold 0.5    DT 0.25 + SVC 0.75 @thr0.55
      Score (weighted F1)       0.941                 0.955
      Rows predicted correctly  396 of 421            402 of 421
      ----------------------------------------------------------------------------
      Evidence check (both must pass to install)       value      needed   pass?
      1. Candidate scores higher on the judging rows  +0.0140     above 0     yes
      2. Still higher in an unlucky case (worst 5%)  -0.0071   > +0.0050      NO
      ----------------------------------------------------------------------------
      In plain words:
        The candidate scores higher on these rows, but 173 candidates were tried,
        and when many are tried some score higher by luck alone. In an unlucky
        case the gain falls to -0.0071, below the required +0.0050. The higher
        score is therefore not proof of a real improvement, and the current model
        stays.
      ----------------------------------------------------------------------------
      TESTED CONDITIONS: NOT MET (1)
        - 569 rows: the reference runs use at least 1050 rows for two
          classes
        The gate still decides, but the results of the paper (at most 1.7%
        false improvements among accepted updates) do not cover this decision.
        Treat it as a first screen.
      The program cannot check these; make sure that:
        - the judging rows come from the same population as the data the
          model will see
    ==============================================================================

How to read it:

- **RESULT** is the decision: install the candidate, or keep the current model.
- The first table compares the two models on the *judging rows*, labeled rows
  that no model saw in training.
- The second table holds the two checks. The usual practice stops at check 1
  ("the new model scores higher"). BRIDGE also requires check 2: the gain must
  stay above a small minimum (+0.005) even in an unlucky case, the lowest 5% of
  20,000 resamples of the judging rows. An update is installed only when both
  checks pass.
- **In plain words** explains the outcome in one paragraph.
- **TESTED CONDITIONS** says whether the decision lies inside the conditions
  tested in the paper. MET: the results of the paper cover it. NOT MET: each
  condition that is not met is listed; the gate still decides, but treat the
  result as a first screen. The card also lists the conditions of use that the
  program cannot check, such as that the judging rows come from the same
  population as future data.

More examples, including a data file inside the tested conditions, are in
[examples/](https://github.com/DwiSunaryono/BRIDGE-ML/tree/main/pkg/examples).

## Command line

    bridge-gate --dataset your.csv --target label

Without `--candidates` and `--auto`, the program runs the reference procedure
from the paper: HGB as the current model, threshold and ensemble candidates,
32 bags, and the frozen gate settings. For the same data and seed it returns
the same decision as the reference trial code.

    bridge-gate --dataset you.csv --target label --reference held_out.csv

`--reference` checks the decision on labeled rows that the procedure never saw.

Custom candidates are available with `--candidates` or `--auto`. These runs are
outside the reference procedure and the output says so.

    bridge-gate --dataset breast_cancer --current rf --auto
    bridge-gate --dataset breast_cancer --current rf --auto combined
    bridge-gate --dataset mango.csv --target grade --current rf \
        --candidates tuned_hgb,fs50_hgb,thr0.35,ens_hgb+logreg

Run `bridge-gate --help` for all options. The full user manual is in
[MANUAL.md](https://github.com/DwiSunaryono/BRIDGE-ML/blob/main/pkg/MANUAL.md).

## Python API

    import bridge_gate as bridge

    search, pool = bridge.split(y)          # fit and tune on SEARCH; judge on POOL
    # fit the current and candidate models on X[search], y[search]
    d = bridge.gate(y[pool], current.predict(X[pool]), candidate.predict(X[pool]))
    print(d)                # one sentence
    print(d.explain())      # the result card shown above
    d.accept, d.lower_bound, d.gain_pool, d.warnings   # the values themselves

If you already have the predicted labels of both models, the command line
needs no training: write a CSV file with the columns `y` (true labels),
`current`, and one column per candidate, then run

    bridge-gate --predictions my_predictions.csv

For several candidates, pass all of them and let the gate choose with
cross-fitting:

    d = bridge.gate_select(
        y[pool],
        current.predict(X[pool]),
        {"threshold 0.35": p1, "ensemble": p2},
    )
    print(d.chosen)

Frozen settings: margin 0.005, lower bootstrap quantile 5%, 20,000 bootstrap
replicates, 3 cross-fitting folds.

Metrics: weighted-F1 (`"wF1"`) for any number of classes. For two classes the
package also supports accuracy (`"acc"`), balanced accuracy (`"bacc"`),
macro-F1 (`"macroF1"`), net benefit at 0.1 or 0.2 (`"nb10"`, `"nb20"`) and F2
(`"F2"`).

## Before using the gate

1. Fit, tune and select on SEARCH only. Never use POOL for any of these steps.
2. Fix the candidates before looking at POOL. For several candidates, use
   `gate_select` instead of picking the best one yourself.
3. Keep POOL and deployment data from the same population. Distribution shift
   has not been tested.
4. Make sure every class is present in POOL. `bridge.check` reports this.
5. For imbalanced two-class data, prefer balanced accuracy or macro-F1.

## License

MIT. See [LICENSE](https://github.com/DwiSunaryono/BRIDGE-ML/blob/main/LICENSE).

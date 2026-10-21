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

    bridge-gate --dataset iris --current rf --auto
    bridge-gate --dataset iris --current rf --auto combined
    bridge-gate --dataset mango.csv --target grade --current rf \
        --candidates tuned_hgb,fs50_hgb,thr0.35,ens_hgb+logreg

Run `bridge-gate --help` for all options. The full user manual is in
[MANUAL.md](https://github.com/DwiSunaryono/BRIDGE-ML/blob/main/pkg/MANUAL.md).

## Python API

    import bridge_gate as bridge

    search, pool = bridge.split(y)          # fit and tune on SEARCH; judge on POOL
    # fit the current and candidate models on X[search], y[search]
    d = bridge.gate(y[pool], current.predict(X[pool]), candidate.predict(X[pool]))
    print(d.accept, d.lower_bound, d.gain_pool, d.warnings)

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

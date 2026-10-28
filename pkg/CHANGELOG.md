# Changelog

## 1.1.1

- Balanced accuracy (`bacc`) and macro-F1 (`macroF1`) now work with more than two classes, in the command line and
  in Python (`gate`, `gate_select`, `score`, `protocol.run`). They are the metrics of use when some classes are rare.
- Tested: on the eight multiclass datasets of the paper (480 decisions, MINI-54), BRIDGE accepts 253 updates with one
  false improvement under balanced accuracy and 248 updates with none under macro-F1 (bounds 2.9% and 1.7%, both
  within the pass criterion), so the result card marks these metrics as tested for more than two classes.
- `protocol.run_metrics` judges one fit under several metrics. Weighted-F1 decisions and all two-class decisions are
  unchanged: every multiclass trial of MINI-54 reproduces the stored weighted-F1 decision exactly.
- Accuracy, net benefit and F2 remain two-class metrics; asking for them with more classes stops with a message that
  names the metrics that apply.

## 1.1.0

- Every run ends with a result card: the current model and the best candidate side by side (score and rows
  predicted correctly), the two evidence checks with the value each needs and a pass column, and a short
  explanation in plain words. The command line in all modes and `Decision.explain()` in Python print the same card.
- The card ends with the tested conditions: MET, or NOT MET with each condition that the decision lies outside
  (rows, classes, metric, smallest class, custom candidates, tuning updates, a weak current model), followed by
  the conditions of use that the program cannot check. The warnings at the top of the output moved into the card.
- `examples/` holds a 2,000-row data file inside the tested conditions and a predictions file for `--predictions`.
- `print(decision)` gives one readable sentence instead of the raw data fields.
- `--predictions`: when no candidate scores higher, the card shows the closest candidate and its score instead of
  "best candidate None; gain +0.0000".
- Reference procedure: a warning when the current model (HGB) is trained on fewer than 100 rows, where large gains
  are expected and say little.
- `bridge-gate` without arguments prints three example commands instead of an error.
- New `bridge_gate.score` (metric of one set of predicted labels). Decisions and numbers are unchanged.

## 1.0.0

- First public release on PyPI.
- Package the gate core (`core.py`), the reference procedure (`protocol.py`) and
  the command line program (`cli.py`).
- Install with `pip install bridge-gate`.
- Fix packaging metadata and unify the version number.
- Rewrite comments and docstrings in the package.

## 0.8.1

- Add a warning for multiclass ranges outside the confirmed range.

## 0.8.0

- Reproduce the reference decisions in `protocol.run`.

## 0.7.0

- Initial wheel with the gate core and the command line program.

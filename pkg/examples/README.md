# Examples

Run these from this folder after `pip install bridge-gate`. Each run ends with a result card.

| Command | What it shows | Tested conditions |
|---|---|---|
| `bridge-gate` | three starter commands | - |
| `bridge-gate --dataset example_data.csv --target outcome` | reference procedure on 2,000 rows (two classes) | MET |
| `bridge-gate --dataset breast_cancer` | built-in data set with 569 rows | NOT MET: fewer rows than tested |
| `bridge-gate --dataset example_data.csv --target outcome --current rf --auto` | your own current model and generated candidates | NOT MET: custom candidates |
| `bridge-gate --predictions example_predictions.csv` | decision from stored predictions, no training | NOT MET: fewer judging rows than tested |

Files:

- `example_data.csv`: 2,000 synthetic rows, ten numeric features `x1`-`x10`, target `outcome` (`yes` 31%, `no` 69%).
- `example_predictions.csv`: 422 judging rows from the breast-cancer data; column `y` holds the true labels,
  `current` the predictions of a logistic regression, and `random_forest` and `threshold_0.3` two candidates.
  Your own file needs the same layout: `y`, `current`, and one column per candidate.

Add `--detail` for the technical numbers, `--cm` for the confusion matrices, or `--out result` to save the result as
JSON.

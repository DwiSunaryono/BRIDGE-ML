#!/usr/bin/env bash
# Recompute the numbers of the paper from the stored per-trial outputs in out/ (no model training, no new runs).
# Usage: cd experiments && bash reproduce_paper.sh
# Each step prints its results; the analysis steps rewrite their CSV in out/ with identical content.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
step() { echo; echo "=== $1"; shift; "$@"; }
step "Code identity: every run manifest against the released scripts" $PY verify_manifests.py
step "Number of trials per run (lower bound)" $PY count_trials.py
step "Main results, synthetic population, holdout, scope (Tables 7, 11, H1-H3)" $PY summarize_paper.py
step "Comparators on (C) and (D) (MINI-48)" $PY analysis_mini48.py
step "OpenML collection (E) (Tables 7, H1, H3)" $PY analysis_setting_e.py
step "OpenML collection (E2) and the Volkert supplement (Table G1)" $PY analysis_setting_e2.py
step "Random-selection control and false improvements removed (Table H4)" $PY analysis_selectivity.py
step "Accepted updates below and above the margin (Table H5, Table 10)" $PY analysis_false_improvement.py
step "Fixed margin chosen without the evaluated dataset (Table 10)" $PY analysis_margin_lodo.py
step "Cross-fit lower bound against the deployed configuration" $PY analysis_transfer.py
step "Random control and leave-one-setting-out fixed margin" $PY analysis_traceable_controls.py
echo; echo "Done. Compare the printed numbers with the manuscript tables named above."

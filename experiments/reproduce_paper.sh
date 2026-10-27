#!/usr/bin/env bash
# Recompute the numbers of the paper from the stored per-trial outputs in out/ (no model training, no new runs).
# Usage: cd experiments && bash reproduce_paper.sh            (one line per step, then the main result table)
#        VERBOSE=1 bash reproduce_paper.sh                     (also print every step's detailed tables)
# Each step writes its detailed tables to out/reproduce_log/<nn>_<script>.txt; the analysis steps rewrite their CSV
# in out/ with identical content. To use the gate on your own data, install the bridge-gate package (README).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
LOG=out/reproduce_log
mkdir -p "$LOG"; rm -f "$LOG"/*.txt
N=0
step() {   # step "title" <python> <script>
    local title=$1; shift; N=$((N + 1))
    local file; file=$(printf "%s/%02d_%s.txt" "$LOG" "$N" "${2%.py}")
    printf "%2d. %-68s" "$N" "$title"
    local t0=$SECONDS
    if "$@" > "$file" 2>&1; then
        printf "done %4ds\n" $((SECONDS - t0))
    else
        printf "FAILED\n"; cat "$file"; exit 1
    fi
    if [ "${VERBOSE:-0}" = 1 ]; then sed 's/^/    /' "$file"; fi
}
echo "Recomputing the paper's numbers from the stored outputs (about 3 minutes; no model is trained)."
echo
step "Code identity: every run manifest against the released scripts" $PY verify_manifests.py
step "Number of trials per run" $PY count_trials.py
step "Main results and holdout comparison (arXiv Tables 7, 11, H1-H3)" $PY summarize_paper.py
step "Comparison rules on collections (C) and (D)" $PY analysis_mini48.py
step "OpenML collection (E) (arXiv Tables 7, H1, H3)" $PY analysis_setting_e.py
step "OpenML collection (E2) and Volkert (arXiv Table G1)" $PY analysis_setting_e2.py
step "Random control, false improvements removed (arXiv Table H4)" $PY analysis_selectivity.py
step "Accepted updates below/above the margin (arXiv Tables H5, 10)" $PY analysis_false_improvement.py
step "Fixed margin chosen without the dataset (arXiv Table 10)" $PY analysis_margin_lodo.py
step "Cross-fit lower bound vs the deployed configuration" $PY analysis_transfer.py
step "Random control and leave-one-setting-out fixed margin" $PY analysis_traceable_controls.py
echo
if grep -q "all entries accounted for" "$LOG"/01_verify_manifests.txt; then
    echo "Code identity: every run manifest matches the released or archived scripts."
fi
grep -h "trials (lower bound)" "$LOG"/02_count_trials.txt | sed 's/^ */Trials stored: /'
echo
$PY paper_summary.py

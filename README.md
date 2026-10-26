# BRIDGE: An Acceptance Gate to Block False Improvements in Model Updates, with a Retrospective Clinical Evaluation

[![PyPI](https://img.shields.io/pypi/v/bridge-gate.svg)](https://pypi.org/project/bridge-gate/)
[![Python](https://img.shields.io/pypi/pyversions/bridge-gate.svg)](https://pypi.org/project/bridge-gate/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23523984.svg)](https://doi.org/10.5281/zenodo.23523984)

Code and experimental data for the BRIDGE paper, available as a preprint at [arXiv:2610.29515](https://arxiv.org/abs/2610.29515). Releases are archived on Zenodo under the concept DOI [10.5281/zenodo.23523984](https://doi.org/10.5281/zenodo.23523984), which resolves to the latest release (tag `paper-v1.1`); the first release, `paper-v1`, is doi:10.5281/zenodo.23523985.

## What BRIDGE is

Machine-learning models are revised often, whether by moving a decision threshold, adding an ensemble, changing the feature set, or retuning the model. A revision that improves the validation score without improving the score on new data is a **false improvement**; in a clinical setting, the revised model can then misguide decisions about patients. The naive rule accepts every positive validation gain, even though the same data both choose the revision and judge it.

BRIDGE is a model-agnostic acceptance gate that decides whether one update may replace the current model. It uses the predictions of the current and candidate models together with labels that are already available; it needs no new data, and its settings are fixed once:

1. **Cross-fitting:** each fold of the labeled data judges the update chosen on the other folds, so the data that choose an update do not judge it.
2. **Bootstrap lower bound:** the gate uses the lower 5% bootstrap quantile of the cross-fitted gain, not its point estimate.
3. **Margin:** the update is accepted only when that lower bound exceeds a minimum improvement of 0.005.

## Why it matters

- Across the five collections of the paper, **12.5% to 25.6%** of the updates accepted by the naive rule are false improvements; among the updates accepted by BRIDGE the rate is **at most 1.7%**.
- BRIDGE rejects **97% to 100%** of the false improvements accepted by the naive rule, and the mean reference gain per deployed update is **1.8 to 3.1 times** the naive value.
- It applies to any classifier that outputs labels, requires no additional labeled data, and keeps the same frozen settings for every task, so per-task tuning cannot reintroduce selection bias.
- In the retrospective clinical tasks under net benefit, BRIDGE accepted **none** of the false improvements that the naive rule accepted (0 of 1,199 accepted updates).

```python
import bridge_gate as bridge                      # pip install bridge-gate
search, pool = bridge.split(y)                     # fit and tune on SEARCH; judge on POOL
d = bridge.gate(y[pool], current.predict(X[pool]), candidate.predict(X[pool]))
print(d.accept, d.lower_bound)
```

## Design and evidence at a glance

**Scope of the evidence.** The gate was tested on 107 real datasets and clinical tasks and on 4,200 synthetic tasks, in at least 151,462 trials across 57 experiments: 53 numbered runs and four OpenML collections. Of these trials, at least 121,522 tested the frozen gate after development. Every trial has its own seed, assigned by a formula fixed in the run plan before the run, so no seed could be chosen after the results were seen.

Each component of the gate addresses a stated source of error, and the gate was frozen before it was tested:

- **Selection bias.** The candidate with the highest validation score among many tends to look better on the data that chose it, because the expected maximum of noisy gains exceeds the largest true gain. Cross-fitting therefore judges each candidate on data that did not choose it.
- **Sampling noise.** A positive point estimate can arise by chance, so the gate uses the lower bootstrap quantile of the cross-fitted gain.
- **Practical relevance.** A margin fixes the minimum improvement that justifies deployment.
- **Evaluation.** A finite-sample binomial bound (Proposition 1) and a KL-ratio bound estimate the share of false improvements among accepted updates.

**Development, then a frozen gate.** Nineteen development runs (MINI-01 to MINI-19, 29,940 trials) set the seven parameters and the candidate space, and the gate was frozen. The 34 later numbered runs (MINI-20 to MINI-53) and the OpenML collections, with 121,522 trials, test the same gate on new data, in sensitivity analyses, and under conditions of use. Variants are reported beside the frozen default, and none of these runs changes it.

| | |
|---|---|
| Real datasets and clinical tasks | **107** in total: **76** in the four collections of the paper and **31** further OpenML datasets in supplementary collections |
| Synthetic tasks | **4,200** distinct tasks from one generative population: two independent batches of 1,400 for the bound and 1,400 for the tuning study; no pilot task is reused |
| Experiments and trials | **57** experiments (53 numbered runs and four OpenML collections) with **151,462** trials: **29,940** in development and **121,522** on the frozen gate (lower bounds, counted from the stored outputs by `experiments/count_trials.py`) |
| Seeds | One seed per trial: run offset + 1,000 × dataset index + repetition, with a separate offset block for each run (for example 1,021,000,000 for MINI-21), fixed in the plan before the run. Seeds are reused on purpose when a run compares rules or variants on the identical decisions of a stored run (for example MINI-24, MINI-26, MINI-53). The locked reference of each task uses a pre-registered seed, and two further reference seeds leave the conclusion unchanged (MINI-38) |
| Gate decisions in the main collections | **9,880**, each with a fresh seed and a locked reference |
| Clinical sources | **6** (NHANES, BRFSS, COVID-19 Mexico, WiDS ICU, ACTG 175, mammography), **12** tasks |
| Acceptance rules compared on identical decisions | **10** (naive rule, test-statistic-based rules, honest holdouts, BRIDGE and its no-margin ablation), plus fixed margins and a random control |
| Conditions of use tested | metrics, training sizes (1,000 to 6,000 rows), 7 to 10 classes, base learners, ensemble members, candidate spaces (173 to 11,249 configurations), class imbalance, tuning |
| Pre-registration | internal plan per run, with **50** SHA-256 run manifests checked by `experiments/verify_manifests.py` |

False improvements among accepted updates (reference gain at most zero), naive rule against BRIDGE:

| Collection | Naive | BRIDGE |
|---|---|---|
| (A) 11 new real datasets | 19.1 % | 1.1 % |
| (B) 1,400 synthetic tasks | 20.7 % | 1.7 % |
| (C) 12 retrospective clinical tasks, net benefit | 20.0 % | 0.0 % |
| (D) 39 supporting datasets | 12.5 % | 0.5 % |
| (E) 14 OpenML datasets | 25.6 % | 1.0 % |

Install the gate:

```bash
pip install bridge-gate
```

The repository contains the method implementation, the SHA-256 manifests that froze the code, and the raw per-trial outputs behind every table in the paper. The published package is in `pkg/` and installs with `pip install bridge-gate`. The scripts under `experiments/` generated the reported numbers. `python experiments/verify_manifests.py` checks every run manifest against the released scripts. The pre-registration plans are internal working documents and are not part of this release; the SHA-256 hashes are listed in `experiments/PREREGISTRATION_INDEX.md`, and the plans are available to the editor and reviewers on request. Run logs are not included.

## How the gate works

A common workflow changes a trained model after reading validation scores: shift the decision threshold, build an ensemble, or select features. The change is deployed when the validation score rises. Because the same data select the change and then judge it, the improvement is optimistic (the winner's curse). On 11 previously untouched tabular datasets, the rule "accept if validation improves" accepted 1,064 changes, of which 203 (19.1%) were false improvements (gain at most zero on the locked reference).

BRIDGE accounts for selection without a separate holdout reserved for certification:

1. Cross-fitted selection: POOL, the evidence data, is split into three folds. The candidate configuration with the highest score is chosen on two folds, and its predictions are recorded on the third. Each recorded prediction therefore comes from a configuration that did not see that row during selection.
2. Paired bootstrap on counts: the out-of-fold (label, baseline, candidate) counts are resampled with a multinomial draw (20,000 draws) to obtain the distribution of the metric difference.
3. Margin: the change is accepted when the 5% quantile of that difference exceeds 0.005; otherwise the baseline is kept.

The gate applies to any metric computed from a confusion matrix (weighted-F1, balanced accuracy, macro-F1) and to K classes (K^3 counts). All parameters were frozen once and not retuned per dataset, change type, or metric.

The evaluation protocol measures the risk among accepted changes, defined as the fraction whose true effect is zero or negative, together with yield (accepts per 100 decisions). It uses finite-sample upper confidence bounds, leave-one-dataset-out (LOCO) checks, pre-registration, and cross-machine replay.

## Key results

The table compares acceptance rules on identical decisions (11 new real datasets, 1,100 decisions; every rule judges the same deployed configuration; MINI-23/24). The test-based rules apply classical test statistics to fixed predictions on POOL splits, without refitting the base models in each split:

| Rule | Accepted / false impr. | Upper bound on risk | LOCO |
|---|---|---|---|
| Naive (validation improves) | 1,064 / 203 | 22.9 % | 25.3 % |
| McNemar on the same POOL | 183 / 22 | 24.5 % | 30.7 % |
| 5x2cv paired t statistic (Dietterich 1998) | 461 / 29 | 11.0 % | 13.4 % |
| Corrected resampled t statistic (Nadeau & Bengio 2003) | 608 / 38 | 10.1 % | 12.2 % |
| Corrected t + per-split reselection | 349 / 8 | 5.9 % | 7.9 % |
| BRIDGE without margin (ablation) | 431 / 12 | 6.2 % | 7.9 % |
| BRIDGE | 284 / 3 | 4.2 % | 6.0 % |

At equal yield, defined as the top-N accepts by each rule's own statistic, BRIDGE has the fewest or equal false improvements (N = 150 / 284 / 400: 0 / 3 / 8). Across the five collections, it rejects 97 % to 100 % of the false improvements accepted by the naive rule (Supplementary Table H4), and its mean reference gain per deployed update is 1.8 to 3.1 times the naive value.

The bound is checked on an i.i.d. synthetic task population of 1,400 tasks, one trial per task, with truth from 100,000 reference rows (MINI-25). The bound is a finite-sample upper bound for the defined population (Proposition 1):

| Rule | Accepted / false impr. | 95 % bound on risk | Conservative bound* |
|---|---|---|---|
| Naive | 1,323 / 274 | 23.5 % | 33.2 % |
| 5x2cv paired t | 610 / 34 | 8.1 % | 12.5 % |
| Corrected resampled t | 774 / 44 | 8.0 % | 12.5 % |
| BRIDGE | 475 / 8 | 3.6 % | 4.7 % |

\*Every accepted change not confirmed as an improvement on the reference counts as a false improvement.

Cross-fitting is compared with an honest holdout at the same data budget (POOL split into selection and evaluation, same bootstrap and margin, and the configuration that passes the gate is the one deployed; MINI-26):

| Rule | Real data: accepted / false impr., bound, LOCO | Population Q: accepted / false impr., bound |
|---|---|---|
| BRIDGE (cross-fit) | 284 / 3, 4.2 %, 6.0 % | 475 / 8, 3.6 % |
| Holdout 2/3 - 1/3 | 153 / 2, 6.9 %, – | 261 / 5, 4.8 % |
| Holdout 1/2 - 1/2 | 190 / 6, 9.7 %, – | 339 / 7, 4.6 % |

Using the whole POOL as evidence through cross-fitting yields about 1.4 to 1.9 times more accepted changes than an honest holdout, with a lower risk bound. LOCO is not computed for the holdouts, as in the paper.

A fixed validation margin chosen without the evaluated dataset (post hoc; `experiments/analysis_margin_lodo.py`) does not meet the criterion on the 11 new datasets (bound 11.3 %, LOCO 15.1 %), and on Q its false-improvement rate is 4.2 % against 1.7 % for BRIDGE. The comparison is descriptive, without a test, because the margin of each held-out dataset is chosen on data that include the other evaluated datasets.

The clinical evaluation is retrospective: 12 health tasks from 6 independent sources (mammography, AIDS Clinical Trials Group 175, NHANES 2017-18 with 4 diagnoses, BRFSS 2015 with 4 diagnoses, COVID-19 Mexico mortality, and WiDS 2020 ICU mortality), with fresh seeds, 200 trials per task, and clinical metrics fixed in the pre-registration (MINI-30):

| Metric | BRIDGE: accepted / false impr., bound | Naive: accepted / false impr. |
|---|---|---|
| Net benefit, p_t = 0.2 (decision-curve analysis) | 1,199 / 0, 0.33 % | 2,266 / 453 (20.0 %), 202 clearly worse |
| Balanced accuracy | 2,273 / 8, 0.82 % | 2,400 / 8 |
| Weighted F1 | 435 / 9, 5.12 % (LOCO 7.84 %) | 2,288 / 539 (23.6 %) |
| Net benefit, p_t = 0.1 | 1,660 / 0, 0.23 % | 2,388 / 94 (3.9 %) |
| F2 (positive class) | 2,379 / 0, 0.16 % | 2,400 / 0 |

Under net benefit, the accepted changes raised sensitivity by 0.48 on average (specificity fell by 0.26), and 99.2% of them (1,190 of 1,199) also beat both simple strategies, treat-none (NB = 0) and treat-all. This is a retrospective evaluation on health data, not evidence that the model is fit for clinical deployment. NHANES and BRFSS are used without survey weights. Under balanced accuracy and F2 the naive rule rarely accepts a false improvement; recall-heavy metrics (net benefit 0.1, F2) often select thresholds near "flag everyone"; an earlier, smaller run with weighted F1 as the primary metric (MINI-28, 100 trials per task) did not meet the LOCO criterion (12.8%) and is reported as such.

Fixed validation-gain thresholds (MINI-29). A threshold calibrated on development data, as pre-registered, failed at 6,000 training rows and on the new datasets. An exploratory analysis shows that thresholds from 0.020 to 0.035 pass in all seven settings with comparable yield. BRIDGE matches the fixed threshold selected post hoc without per-task tuning and accounts for uncertainty that depends on sample size (6,000 rows: yield 31 against 14 per 100). When a fixed threshold is chosen on six settings and transferred to the seventh (exploratory), it passes in 6 of 7 settings with about twice the risk bound; BRIDGE passes in 7 of 7.

Scope (all results use the frozen gate):

| Claim | Data | Accepted / false impr. | Bound |
|---|---|---|---|
| Balanced accuracy | 11 new datasets | 848 / 1 | 0.81 % |
| Macro-F1 | 11 new datasets | 829 / 0 | 0.47 % |
| Training size 1,000 / 2,430 / 6,000 rows | 27 datasets, fresh seeds | 558/6, 688/1, 825/3 | 3.04, 1.05, 1.37 % |
| Multiclass (K = 7-10) | 5 new datasets | 197 / 0 | 2.11 % |
| Feature selection + threshold, KNN | 11 new datasets | 225 / 1 | 3.45 % |
| Union policy, confirmation | 34 datasets | 878 / 5 | 1.69 % |

Documented failures, reported as a map of where the gate should not be used: accuracy as the metric, feature selection with SVC, the union policy on very wide data with few rows, and feature selection for HGB.

## Repository contents

```
pkg/                                # published bridge-gate package (pip install ./pkg)
paper_assets/                       # final figures and LaTeX tables of the paper; TABLE_MAP.md maps each table to its script and CSV
config.py, data.py, models.py      # shared preprocessing and estimator construction
fetch_uci.py                        # downloads the two UCI datasets (Rice, Dry Bean) into ~/uci_cache/
experiments/
  exp41_gate_confirm.py             # data loading, locked reference split, counts, bagged models (+ import chain exp01..exp22)
  mini01_threshold_crossfit.py      # bootstrap quantiles of the gate statistic
  mini12_optuna.py                  # candidate space (union, 173 configurations), labelling, selection
  bridge_metrics.py                 # metric difference from 8 counts
  mini21_final.py                   # final validation on 11 new datasets (5 policies)
  mini24_standar.py                 # BRIDGE vs standard acceptance rules on identical decisions
  mini25_sintetis.py                # i.i.d. synthetic task population Q
  mini26_holdout.py, mini26_sintetis.py   # BRIDGE vs selection + honest holdout (same data budget)
  mini17_trainsize.py               # training-size study (dev + confirmation)
  mini18_multiclass.py, mini22_mc_final.py  # multiclass gate (dev + new datasets)
  mini11_confirm.py, mini16_metrics.py, mini19_fs_lebar.py, mini12..14  # earlier stages used in the paper
  verify_*.py                       # independent verification (two bound implementations, replay)
  verify_manifests.py               # resolves every manifest entry to the current file, the frozen copy, or a withheld record
  frozen/                           # exact run-time copies of scripts edited after the runs; WITHHELD.tsv
  PREREGISTRATION_INDEX.md          # per run: plan file, recorded time, SHA-256 of plan and manifest
  summarize_paper.py                # main results, synthetic population, holdout, and scope from out/*.csv
  analysis_review4.py               # clinical consequences per 1,000, paired rule differences with CI, gate vs deployed
  paper_assets.py                   # first versions of the figures and tables, into out/paper_assets_regenerated/ (see paper_assets/TABLE_MAP.md)
  reproduce_paper.sh                # recomputes the reported numbers from the stored per-trial outputs
  setting_e2*.py, analysis_setting_e2.py  # OpenML collection (E2) and the Volkert supplement
  out/                              # raw per-trial outputs + SHA-256 manifests
```

## Method implementation

| Paper concept | Where |
|---|---|
| Locked reference split, reference size rule | `exp41_gate_confirm.make_dev_and_locked_reference`, `ref_size` |
| Bagged base models (32 bags, 80 % of SEARCH) | `exp41_gate_confirm.fit_proba_m`, constants `B_BAGS`, `SUBSAMPLE_FRAC` |
| (label, baseline, candidate) counts | `exp41_gate_confirm.counts8`; multiclass `mini18_multiclass.cells` |
| Candidate space (threshold, HGB+DT, HGB+DT+SVC simplex × threshold) | `mini12_optuna.union`, `lab` |
| Cross-fitted selection + bootstrap + margin (the gate) | `mini21_final._trial` (`row`), `mini21_final.pick`; multiclass `mini18_multiclass._trial` |
| Metric differences from counts | `bridge_metrics.delta`; multiclass `mini18_multiclass.wf1_delta` |
| Standard rules (McNemar, 5×2cv, corrected t, reselection variant) | `mini24_standar._trial`, `mcn` |
| Honest-holdout comparator | `mini26_holdout._trial` (columns `ho23`, `ho12`) |
| Synthetic task population Q | `mini25_sintetis.param`, `gen` |
| KL-ratio bound (two independent implementations) | `verify_exp41.kl_ratio_up` (brentq), `verify_exp41_independent.ratio_up` (bisection) |
| Proposition 1 bound, conservative bound, dataset bootstrap, yield matching | `summarize_paper.py` |
| Clinical metrics (net benefit, F2, balanced accuracy, sensitivity/specificity) from counts | `bridge_metrics_klinis.delta` |
| Clinical tasks and the pre-specified features and targets | `health_data.py`, `health_data_b.py`, `health_data_c.py` |

## Data files and paper tables

Each row of a results CSV is one trial (dataset, replicate) and one policy or rule; `lo95` is the gate statistic, `ref` the true Δ on the locked reference, and `kelas` its 95 % reference-interval verdict (`membaik` = improves, `memburuk` = worsens, `tidak-pasti` = uncertain).

| File(s) | Paper table / section |
|---|---|
| `out/mini24_m{1,2,3}.csv` | Acceptance rules on identical decisions (11 new datasets), yield-matched comparison, dataset-level bootstrap |
| `out/mini25_m{1,2,3}.csv` | Synthetic population Q (Proposition 1 bound) |
| `out/mini26_nyata_m{1,2,3}.csv`, `out/mini26_q_m{1,2,3}.csv` | BRIDGE vs honest holdout (real data and population Q) |
| `out/mini21_m{1,2,3}.csv` | Generalization table: new datasets (wF1, balanced accuracy, macro-F1, FS+KNN, FS+SVC) |
| `out/mini20_size_{a,b}.csv`, `out/mini17_trainsize_dev.csv` | Generalization table: training size (confirmation, development) |
| `out/mini22_m*.csv`, `out/mini18_multiclass_dev.csv` | Generalization table: multiclass (new datasets, development) |
| `out/mini11_confirm.csv` | Generalization table: 34-dataset confirmation |
| `out/mini16_metrics_dev.csv`, `out/mini19*_dev.csv` | Failure map (accuracy, wide data, FS + SVC) |
| `out/mini12_optuna_dev.csv`, `out/mini13_fs_dev.csv`, `out/mini14_optuna_size.csv` | Side findings (Optuna on small data, feature selection) |
| `out/audit_baru.csv` | Selection of the 11 new datasets: duplicate share, conflicting labels, identifier and time columns, largest single-feature AUC |
| `exp38_manifest/kandidat_audit.csv`, `exp40_dataset_audit_final.csv`, `exp41_scan_79_keputusan.csv` | Candidate audits of the selection rounds behind the 34 phase-A datasets (final lists: `exp40_manifest/datasets.csv`, 25; `exp41_manifest/datasets.csv`, 9): decision code and reason per candidate (codes below) |
| `out/mini28_m*.csv`, `out/mini28b_covid.csv`, `out/mini28c_wids.csv`, `out/mini30_m*.csv` | Retrospective clinical evaluation (first run; confirmation with clinical metrics) |
| `out/analysis_selectivity.csv` (`experiments/analysis_selectivity.py`) | Appendix table: random-selection control and gain-loss trade-off against the naive rule (post hoc, exploratory) |
| `out/mini31_param_m*.csv` | Parameter sensitivity on Q (folds, lower-bound level, bootstrap replicates) |
| `out/mini31_xgb_m*.csv` | Base-learner study: project-default XGBoost on Q |
| `out/mini32_*.csv` | Factorial ablation (evidence source x statistic x margin) |
| `out/mini35_*.csv` (grids in `experiments/mini35_ruang_kandidat.py`, function `rapat`) | Candidate-space size 173 / 1,825 / 11,249 |
| `out/mini33*_m*.csv`, `out/mini34_m*.csv`, `out/mini36_m*.csv` | Base-learner study (XGBoost factors, MLP) and its audit |
| `out/mini37_*.csv` | Bag count, SEARCH size, bootstrap and reference precision |
| `out/mini38_*.csv` | Outer-split (reference seed) replication on the 11 new datasets |
| `out/mini29_*.csv` | Fixed validation-gain threshold comparison (calibration + sensitivity) |
| `out/mini27_m*.csv` | Figure 1: optimism gap, rank correlation, false improvements by size of validation gain |
| `experiments/mini52_ruang_kandidat_E.py`, `experiments/analisis_mini52.py`, `out/analysis_mini52_contrasts.csv` | Candidate-space size and ablation cells on the binary datasets of (E) (MINI-52) |
| `out/analisis_mini32.csv`, `out/analysis_mini48_ablation.csv`; `experiments/analysis_settingE_ablation.py` | Factorial ablation: counts, LOCO, and yield per cell; the four (E) cells from the released program |
| `experiments/mini50_all_families.py`, `experiments/analysis_mini50.py` → `out/analysis_mini50*.csv` | Joint candidate space of all model families on (A) and (D) (MINI-50) |
| `experiments/mini51_ruang_kandidat_CD.py`, `experiments/analisis_mini51.py` → `out/analysis_mini51*.csv` | Candidate-space size on (C) and (D) (MINI-51) |
| `experiments/mini53_anggota_learner_BCE.py`, `experiments/analisis_mini53.py` → `out/analysis_mini53.csv` | Ensemble members, base learners and the joint space on (B), (C) and (E) (MINI-53) |
| `experiments/mini53_diag_refuncertain.py` → `out/mini53_diag_refuncertain.csv` | Post hoc diagnostic: reference interval of the gain for every false improvement of the member pairs on (E) |
| `experiments/analysis_false_improvement.py` → `out/analysis_false_improvement.csv` | Post hoc: accepted updates gaining at most or more than the margin, per rule and collection, with paired cluster bootstrap and Holm (Table H5) |
| `experiments/analysis_transfer.py` → `out/analysis_transfer.csv` | Post hoc: among accepted updates, the cross-fit lower bound against the reference gain of the deployed configuration (share reached, shortfall, excess) |
| `experiments/count_trials.py` | Number of trials per run and in total, from the stored per-trial outputs |
| `experiments/setting_e2.py`, `setting_e2_amendment1.py`, `setting_e2_tabular_supplement.py`, `settingE2*_daftar_beku.csv` → `out/settingE2_FINAL_AMENDED_1700.csv`, `out/settingE2_tabular_hasil.csv`; `experiments/analysis_setting_e2.py` → `out/analysis_settingE2*.csv` | OpenML collection (E2, 19 datasets, 1,700 decisions, after amendment 1) and the Volkert supplement (60 trials); Supplementary Appendix G. The final CSVs merge the shards of the distributed run; completeness is recorded in `out/settingE2_FINAL_AUDIT.json` and `out/settingE2_tabular_AUDIT.json` |
| `paper_assets/TABLE_MAP.md` | Every table of the paper: file, how it was produced, and the script and CSV that recompute its numbers |
| `experiments/analysis_mini48.py` | Acceptance rules, holdouts and ablation cells on (C) and (D) (MINI-48) |
| `experiments/analysis_settingE_selectivity.py` | Random-selection control on (E) |
| `experiments/analysis_margin_sweep.py` → `out/analysis_margin_sweep.csv` | Fixed-margin sweep: trials, yield, LOCO0 power check, units with yield ≥ 5 |
| `experiments/analysis_traceable_controls.py` | Yield range of the fixed margins against BRIDGE (supplement, fixed margin) |
| `experiments/analysis_paired_ci.py` → `out/paired_ci.csv` | Paired cluster-bootstrap intervals, including the yield rows quoted in the supplement |
| `experiments/analysis_cited_runs.py` | Cited development runs table |
| `experiments/analysis_development_runs.py` | Development runs table |
| `experiments/analysis_review4.py` | Clinical consequences table (exploratory) |
| `experiments/mini25_sintetis.py` | Generator of the synthetic population Q and its task seeds |

Decision codes in the dataset-selection files (values are in Indonesian): `MASUK` = included; `CADANGAN` = reserve; `KELUAR-<reason>` = excluded, with reasons `touched` (family already used in an earlier experiment), `K` (same source family as another candidate; the smallest OpenML id is kept), `T2` (binary target obtained by thresholding or merging a continuous or multi-category variable), `T3` (provenance of the target not verifiable), `sintetis` (synthetic or simulated), `timeseries` (time-ordered rows or sessions), `grup` (several rows per subject or unit), `50%` (class ratio exactly 50%, i.e., artificially balanced), `regresi` (regression target), `leakage` (target leakage), `teknis` (the frozen loader could not read the file). Several codes may be combined with `/`.

## Installation

The published gate:

```bash
pip install bridge-gate
```

The research code and the pinned environment:

```bash
git clone https://github.com/DwiSunaryono/BRIDGE-ML.git
cd BRIDGE-ML
python3.12 -m venv .venv          # the pinned NumPy and SciPy need Python 3.12
source .venv/bin/activate
pip install -r requirements.txt          # pinned versions used for the runs
pip install ./pkg                        # bridge-gate from this checkout
python fetch_uci.py                      # Rice and Dry Bean into ~/uci_cache/
cd experiments && bash reproduce_paper.sh     # recomputes the reported numbers from the shipped CSVs
```

OpenML datasets are downloaded automatically by scikit-learn on first use.

## Reproducing the paper in one command

```bash
pip install -r requirements.txt          # pinned versions used for the paper
cd experiments && bash reproduce_paper.sh
```

The script checks every run manifest, counts the trials, and recomputes the main tables from the stored per-trial
outputs, without training any model (about 3 minutes on one CPU). Run it with Python 3.12 and the pinned versions. Each analysis rewrites its CSV in `out/` with
identical content, so `git status` stays clean after the run.

## Reproducing a run

```bash
cd experiments
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
python mini21_final.py 100 30 mini21_m1.csv semua            # final validation, 11 new datasets
python mini24_standar.py 100 30 mini24_m1.csv semua          # standard rules on identical decisions
python mini25_sintetis.py 0 1400 30 mini25_m1.csv            # synthetic task population
python mini26_holdout.py 100 30 mini26_nyata_m1.csv semua    # honest-holdout comparator
python mini17_trainsize.py 100 30 mini20_size.csv 1020000000 1000,2430,6000
python mini22_mc_final.py 60 30 mini22_m1.csv semua          # multiclass, new datasets
python verify_mini21.py 20 && python verify_mini20.py 20     # verification
python verify_manifests.py                                  # code identity, every run manifest
```

All seeds are fixed in the scripts. With the pinned versions, results were bit-identical across the tested Intel and AMD machines, except 17 trials with Optuna tuning or mutual-information feature selection (identical within one CPU family) and one MLP-KNN trial of MINI-53 (same decision, reference gain differs slightly between CPU families). The paper's runs were split across three machines by dataset, which is why most outputs come in `_m1/_m2/_m3` parts.

## Health data access

Raw health data are **not** redistributed in this repository. `health_data.py`, `health_data_b.py`, and `health_data_c.py` document the exact processing. Sources: NHANES 2017-2018 XPT files and BRFSS 2015 (`LLCP2015XPT.zip`) from cdc.gov into `~/health_cache/`; mammography (OpenML 310) is fetched automatically; ACTG175 via `ucimlrepo` (id 890); COVID-19 Mexico via Kaggle `tanmoyx/covid19-patient-precondition-dataset`; WiDS Datathon 2020 via Kaggle (competition rules must be accepted) into `~/health_cache/kaggle/`.

## Provenance notes

- Comments, docstrings, and printed messages of many scripts were edited after the runs (translation and wording). The manifests keep the run-time hashes, so `sha256sum -c` on a manifest reports the edited files as changed. `verify_manifests.py` resolves every manifest entry to one of three cases: (i) the current file; (ii) the exact run-time copy in `frozen/`, compared with the current file as a program (syntax tree without docstrings and comments); or (iii) for seven run-time Python scripts whose comments hold internal working notes, the SHA-256 of the program structure listed in `frozen/WITHHELD.tsv` (computed with Python 3.12). The run-time copy of one monitoring shell script (`watch_mini52.sh`, machine addresses) is also withheld. Plan and report documents listed in the manifests are withheld and available on request.
- Two scripts differ from the run-time versions in code: `pkg/src/bridge_gate/cli.py` renames `run_paper` to `run_reference` (decisions unchanged), and `analisis_mini39.py` reads an empty `error` column as text instead of as missing values; without this fix every row was filtered out. The MINI-39 fix was recorded as a deviation in the run report at the time.
- `mini17_trainsize.py` and `mini19_fs_lebar.py` gained optional command-line arguments (seed offset and training sizes; first replicate and dataset subset) after the development runs. Default behaviour is unchanged, so replays of development seeds are identical. The versions recorded in `out/mini17_MANIFEST.sha256` and `out/mini19_MANIFEST.sha256` were not retained; the confirmatory manifests (`mini20`, `mini19b`) match the copies in `frozen/`.
- `out/mini53_equnix.csv`: the home-directory prefix in the error messages of 260 failed rows was replaced by `~` before release; status and diagnostics are unchanged.
- The E2 amendment and the Volkert supplement list two dataset-screening scripts (`screen_e2_external.py`, `screen_e2_openml_custom.py`) that were edited after these runs; the run-time versions were not retained (`PREREGISTRATION_INDEX.md`, note 5).
- The synthetic population Q was designed with three pilot runs (documented in the internal pre-registration) to reach a realistic naive-rule error rate of at least 15%, without using BRIDGE results. Pilot tasks were excluded.
- Pre-registrations and some column values use Indonesian terms (`membaik` = improves, `memburuk` = worsens, `tidak-pasti` = uncertain, `naif` = naive, `semua` = all).

## Requirements

See `requirements.txt`. Experiments were run with Python 3.12, scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.6, and scipy 1.18.1, on three CPU machines (Intel Core i9-13900K, AMD Ryzen 9 9950X, Intel Core Ultra 9 285K).

## Author

Dwi Sunaryono, Department of Informatics, Institut Teknologi Sepuluh Nopember (ITS), Surabaya, Indonesia (dwi@its.ac.id).

## Citation

```bibtex
@misc{bridge2026,
  title         = {{BRIDGE}: An Acceptance Gate to Block False Improvements in Model Updates, with a Retrospective Clinical Evaluation},
  author        = {Sunaryono, Dwi},
  year          = {2026},
  eprint        = {2610.29515},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2610.29515}
}

@software{bridge2026code,
  title   = {{BRIDGE}: An Acceptance Gate to Block False Improvements in Model Updates, with a Retrospective Clinical Evaluation (code and data, release paper-v1.1)},
  author  = {Sunaryono, Dwi},
  year    = {2026},
  publisher = {Zenodo},
  version = {paper-v1.1},
  doi     = {10.5281/zenodo.23523984},
  url     = {https://doi.org/10.5281/zenodo.23523984}
}
```

## License

Both the code and the experimental data in this repository are released under the [MIT License](LICENSE). That license does not cover the manuscript text, which stays under copyright until peer review concludes.

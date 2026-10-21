# Table map: paper table → file → how it was produced → script and output that recompute its numbers

The files in this folder are the final tables of the manuscript and its Supplementary Material. Table numbers 1–13
are in the manuscript; numbers with a letter are appendix tables (B, C in the manuscript; D–H in the Supplementary
Material).

**How a table was produced:**
- **Generated, then edited.** `experiments/paper_assets.py` produced the first version. The final file was then edited
  by hand: terminology ("harmful" became "false improvement"), captions, notes, layout, and, where stated, added
  rows, columns, or collections.
- **Composed by hand.** The table was written from the outputs listed. The first line of each file says which.

`paper_assets.py` writes to `experiments/out/paper_assets_regenerated/` and does not reproduce the final files byte for
byte. The numbers in the final tables come from the analysis scripts and CSVs listed below, and
`experiments/reproduce_paper.sh` reruns those scripts on the stored per-trial outputs. All paths are relative to
`experiments/`.

## Manuscript

| Table | File | Produced | Numbers recomputed by → output |
|---|---|---|---|
| 2 | `tab1b_hyperparameters.tex` | generated, then edited | frozen settings in `exp41_gate_confirm.py`, `mini21_final.py`; no computed numbers |
| 7 | `tab0_headline.tex` | generated, then rebuilt by hand ((E) and the share of false improvements removed added) | `summarize_paper.py` (A: `out/mini24_m*.csv`; B: `out/mini25_m*.csv`), `analysis_mini48.py` (C, D: `out/mini48_m*.csv`), `analysis_setting_e.py` → `out/analysis_settingE.csv` (E), `analysis_selectivity.py` → `out/analysis_selectivity.csv` |
| 8 | `tab1d_hypothesis_tests.tex` | composed by hand | summary of Table H2 |
| 9 | `tab1c_rules_all.tex` | composed by hand | summary of Table H3 |
| 10 | `tab1e_strongest_comparators.tex` | composed by hand | Table H3; `analysis_false_improvement.py` → `out/analysis_false_improvement.csv`; `analysis_margin_lodo.py` → `out/analysis_margin_lodo.csv` |
| 11 | `tab2_synthetic_population.tex` | generated, then edited | `summarize_paper.py` (`out/mini25_m*.csv`); `analysis_mini48.py` → `out/analysis_mini48_rules.csv` |
| 12 | `tab5_clinical.tex` | generated, then edited | `analisis_mini30.py` (`out/mini30_m*.csv`); `analysis_mini48.py` |
| 13 | `tab9_ablation_factorial.tex` | generated, then edited (collections B–E added) | `analisis_mini32.py` → `out/analisis_mini32.csv`; `analysis_mini48.py` → `out/analysis_mini48_ablation.csv`; `analysis_settingE_ablation.py` (E) |
| B1 | `tabA_dataset_splits.tex` | generated, then rebuilt by hand ((E) added) | `dataset_splits.py`; `out/settingE_laptop.csv` (E) |
| B2 | `tab6_clinical_tasks.tex` | generated, then edited | descriptive; processing in `health_data.py`, `health_data_b.py`, `health_data_c.py` |
| C1 | `tabS12_cited_runs.tex` | merged by hand from two generated tables | `analysis_cited_runs.py` → `out/analysis_cited_runs.csv`; `analysis_development_runs.py` |

## Supplementary Material

| Table | File | Produced | Numbers recomputed by → output |
|---|---|---|---|
| D2 | `tabS13_hyper_basis.tex` | generated, then compacted by hand | descriptive; the runs named in each row (`out/mini31_param_m*.csv`, `analisis_mini37.py`, development runs) |
| D3 | `tabS6_margin_sweep.tex` | generated, then edited | `analysis_margin_sweep.py` → `out/analysis_margin_sweep.csv` |
| D4 | `tab16_margin_paired.tex` | generated, then set in one column | `analysis_margin_paired.py` → `out/analysis_margin_paired.csv` |
| D5 | `tab15_qeps_real.tex` | generated, then set in one column | `analysis_mini44.py` → `out/analysis_mini44_qeps.csv` |
| D6 | `tab14_tuning_tests.tex` | generated, then set in one column | `analysis_mini44.py` → `out/analysis_mini44_h1.csv`, `out/analysis_mini44_h2.csv` |
| D8 | `tab8_complexity.tex` | generated, then transposed by hand | `complexity_timing.py` → `out/complexity_timing.csv` |
| E1 | `tabS4_per_dataset.tex` | generated, then set in one column | `summarize_paper.py` (`out/mini24_m*.csv`) |
| E2 | `tabS7_paired_ci.tex` | generated, then set in one column | `analysis_paired_ci.py` → `out/paired_ci.csv` |
| E3 | `tabS2_base_learner_audit.tex` | generated, then set in one column | `analisis_mini34.py`, `analisis_mini36.py` → `out/analisis_mini34.csv`, `out/analisis_mini36.csv`; `out/mini33b_m*.csv` |
| F1 | `tabS3_identity_audit.tex` | generated, then set in one column | `audit_identitas.py` → `out/audit_identitas.csv` |
| F2 | `tabS5_clinical_by_source.tex` | generated, then set in one column | `analysis_clinical_by_source.py` → `out/analysis_clinical_by_source.csv` |
| G1 | `tabS19_openml_collections.tex` | composed by hand | `analysis_setting_e.py` → `out/analysis_settingE*.csv`; `analysis_setting_e2.py` → `out/analysis_settingE2*.csv` (E2, Volkert) |
| H1 | `tab0_summary.tex` | generated, then edited ((E) added) | `summarize_paper.py`; `analysis_setting_e.py`; `analysis_selectivity.py`; `analysis_paired_ci.py` |
| H2 | `tabS20_hypothesis_tests_detail.tex` | composed by hand | `analysis_paired_ci.py` → `out/paired_ci.csv` (A, B); `analysis_mini48.py` → `out/analysis_mini48_rules.csv` (C, D); `analysis_settingE_paired_p.py` (E); `analysis_selectivity.py` (random control) |
| H3 | `tabS21_rules_merged.tex` | composed by hand | `summarize_paper.py` (A, B); `analysis_mini48.py` (C, D); `analysis_setting_e.py` (E); `analysis_paired_ci.py` |
| H4 | `tabS8_selectivity.tex` | generated, then edited | `analysis_selectivity.py` → `out/analysis_selectivity.csv`; `analysis_settingE_selectivity.py` (E) |
| H5 | `tabS23_false_improvement.tex` | generated by `analysis_false_improvement.py --table` | `analysis_false_improvement.py` → `out/analysis_false_improvement.csv` |
| H6 | `tab5b_generalisation.tex` | generated, then edited | `summarize_paper.py` (`out/mini20_size_*.csv`, `out/mini21_m*.csv`, `out/mini22_m*.csv`); `analysis_mini48.py` |
| H7 | `tab10_candidate_space.tex` | generated, then compacted by hand | `analisis_mini35.py` → `out/analisis_mini35.csv`; `analisis_mini51.py` → `out/analysis_mini51*.csv`; `analisis_mini52.py` → `out/analysis_mini52_contrasts.csv` (E) |
| H9 | `tab13_imbalance.tex` | generated, then edited | `analisis_mini41.py` → `out/analisis_mini41.csv` |
| H10 | `tabS14_base_learner_full.tex` | generated, then edited | `analisis_mini34.py`, `analisis_mini36.py` |
| H11 | `tabS22_members_learners_all.tex` | composed by hand | `analysis_mini50.py`, `analisis_mini53.py` → `out/analysis_mini45.csv`, `out/analysis_mini46.csv`, `out/analysis_mini50.csv`, `out/analysis_mini53.csv` |
| H12 | `tabS18_absolute_quality.tex` | generated, then edited | `analysis_mini49.py` → `out/analysis_mini49_overall.csv`; `analysis_mini50.py` |
| H13 | `tab12_tuning_learners.tex` | generated, then edited | `analisis_mini39.py`, `analisis_mini40.py` → `out/analisis_mini39.csv`, `out/analisis_mini40.csv` |
| H14 | `tabS17_clinical_consequences.tex` | generated, then edited | `analysis_review4.py` (`out/mini30_m*.csv`) |

Files in this folder that the final manuscript does not use (earlier versions, replaced by the merged tables H3 and
H11): `tab1_rules_identical_decisions.tex`, `tab1b_rules_CD.tex`, `tab4_holdout.tex`, `tab11_base_learner.tex`,
`tabS9_development_runs.tex`, `tabS10_mini39_char_Q.tex`, `tabS11_mini39_char_R.tex`, `tabS15_member_families.tex`,
`tabS16_base_learners_real.tex`.

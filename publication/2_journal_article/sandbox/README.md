# Sandbox: development history and dated studies

Nothing under this folder is maintained, linted or cited by the paper. It is kept so that
every decision behind the method and the evaluation can be traced to the study that made it.
Each folder has its own README or results note; wording inside them reflects the project
stage they were written at.

| folder | date | what it is |
|---|---|---|
| `m9_pbm_pipeline_2026-07/` | June to July 2026 | the earlier journal pipeline around a nine-feature "physical bridge model" (M9-PBM), with its notebooks and outputs; superseded by the compact M9 |
| `m9_development/` | September 2026 | the development record of M9: the scorer with its research variants, the seven improvement rounds, weak-station diagnoses, run comparisons and decision notes |
| `2026-09-16_phase3_release/` | 16 September 2026 | the first locked station-held-out evaluation, in which M8 was trained on both datasets and M9 calibrated within each; superseded by the reference run in `../results/` |
| `2026-09-17_pooled_calibration/` | 17 September 2026 | what happens when M9's calibration is fitted on Alpha and Beta together |
| `2026-09-18_cross_cohort_calibration/` | 18 September 2026 | nested station-grouped search for a single cohort-blind calibration; none met the gate |
| `2026-09-18_intercept_transfer/` | 18 September 2026 | the slope carries over between populations, the intercept does not; about 27 reviewed days set an intercept |
| `2026-09-22_beta_only_eval/` | 22 September 2026 | the config and results note of the run that became `../results/` (every method fitted on Beta only) |
| `2026-09-22_bridge_anchors/` | 22 September 2026 | bridge anchors at the window edge for gap-adjacent days; not adopted |
| `2026-09-22_window_ranking/` | 22 September 2026 | how M9 ranks the labelled window among the 1,176 candidates; rank-1 quality and top-k oracles |
| `oracle_review/` | July 2026 | the Streamlit review tooling, per-reviewer annotation files and logs behind the Beta labels, with the tooling's tests |
| `dataset_processed/` | July 2026 | intermediate frames from which `../dataset/final/` was built |
| `notebook_planning/` | July 2026 | planning notes for the earlier notebook set |

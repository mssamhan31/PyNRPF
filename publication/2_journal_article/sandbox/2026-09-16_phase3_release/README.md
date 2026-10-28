# Phase 3 release (16 September 2026), kept unchanged

`outputs/` is the locked M7/M8/M9 station-held-out evaluation exactly as committed in
`ffa1d92` ("Phase 3: locked M7/M8/M9 station-held-out evaluation and operating-point study"),
moved here from `outputs/01_final_evaluation/` on 22 September 2026. Nothing inside has been
re-run or edited since; `tests/test_final_eval_release.py` verifies the manifests against it.

What it is: eighteen leave-one-station-out folds, M8 trained per fold on the other seventeen
stations of both cohorts, M9 calibrated per fold on the other stations of the same cohort,
`c = 0.7`, Beta sure days as the headline population, plus the Gamma forecasting case study
(stage 09) and the operating-point study (stage 10). Stage folders are numbered like the
notebooks; `manifests/` holds one hash manifest per stage with repository-relative paths.
M8 bundles (`02_bundles/*/bundle.pkl`) are regenerable and were never committed.

Superseded by the Beta-only evaluation of 22 September 2026
(`sandbox/2026-09-22_beta_only_eval/`), in which every method is trained or calibrated on
Beta only and Alpha is a comparison cohort. The two runs share `final_eval/` as the engine
and differ only in the fold rule set in their config.

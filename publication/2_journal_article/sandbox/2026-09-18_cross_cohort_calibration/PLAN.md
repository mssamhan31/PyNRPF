# Cross-cohort calibration for M9: experimental plan

Date: 18 September 2026. Status: approved with the decisions in section 9; implemented in `run.py`; results in `RESULTS.md`. Sandbox only; everything outside this folder is read-only.

Abbreviations: RPF, reverse power flow; LOSO, leave-one-station-out; IoU, intersection over union; ECE, expected calibration error.

## 1. Objective and target

Keep the frozen M9 window scorer and its evidence r unchanged, and find one calibration (one fitted mapping, one set of coefficients, one pair of thresholds at c = 0.7) that serves Alpha and Beta together. The mapping receives no cohort or station identifier. Held-out-station labels are never used for fitting, preprocessing, selection or thresholds. Any normalisation is estimated on outer-training stations only. Raw readings are untouched. Success is described as cross-cohort generalisation across Alpha and Beta, not universality.

## 2. What the exploration established (Phase 1)

- Evidence and its ingredients come from the committed `sandbox/2026-09-16_phase3_release/outputs/05_m9/scores_{alpha,beta}.parquet`: per site-day `r_best`, `best_score`, `runner_score`, `margin_window`, `best_start`, `best_end`, `sigma`, `n_admissible`, `input_ok`, plus labels `rpf`, `headline`, `confidence`. Energies per site-day come from `06_site_days/site_days.parquet` (method m9): `candidate_mwh`, `candidate_correct_mwh`, `required_mwh`. Folds from `01_folds/fold_manifest.csv`. No dataset and no scorer run is needed.
- The frozen calibration is `m9_scorer.Calibrator`: logistic regression of the label on z = sign(r)·log(1 + |r|), fitted per fold on the same-cohort stations' `input_ok & headline` days (`final_eval/m9.py::calibrate_fold`). The pooled and equally weighted variants are in `sandbox/2026-09-17_pooled_calibration/`.
- Where the cohorts differ: RPF days have similar evidence (median r 19 Alpha, 27 Beta); non-RPF days do not (95th percentile −0.8 on Alpha, +9.2 on Beta). A shared mapping must separate Beta's positive-evidence non-error days from true errors using something other than r alone, or it must move Beta's null distribution towards Alpha's with a label-free transformation.
- The round-5 diagnosis (`m9_dev/notes/round_5.md`) says what those Beta days are: long windows (median 21 to 26 slots) with modest evidence spread over many slots, on days where net load never nears zero. That is the mechanism the candidates below use.

## 3. Candidates

All candidates share: the frozen r; a logistic mapping to p; the same c = 0.7 for both cohorts; outcomes as in Phase 3 (no window → UNCERTAIN, `input_ok = False` → UNCERTAIN). Coefficient counts are given so complexity is visible.

| id | role | mapping | coefficients | cohort-specific? |
|---|---|---|---|---|
| R0 | reference | frozen within-cohort logistic on z (Phase 3 as committed) | 2 per cohort | yes (benchmark only) |
| B1 | benchmark | one pooled logistic on z, fitted on all 17 other stations | 2 | no |
| B2 | benchmark | pooled logistic on z with equal total weight per cohort | 2 | no (weighting uses cohort at fit time; diagnostic only) |
| B3 | diagnostic ceiling | pooled logistic on z plus a cohort indicator | 3 | yes (shows what the cohort gap is worth; not a target) |
| C1 | candidate: transformed evidence | one logistic on z′ = sign(r)·log(1 + |r| / L), the evidence per slot of the chosen window; L = best_end − best_start + 1 (L = length of the runner-up window when the null wins) | 2 | no |
| C2 | candidate: scorer-derived covariate | one logistic on (z, log L) | 3 | no |
| C3 | candidate: scorer-derived covariate | one logistic on (z, m), m = signed-log margin between the best window and the runner-up (`margin_window`; 0 when the null wins) | 3 | no |
| C4 | candidate: training-only normalisation | z standardised by the outer-training stations' pooled median and median absolute deviation of z among days where the null wins (r ≤ 0), a label-free null scale estimated once per outer fold; then one logistic | 2 (+2 fixed normalisation constants) | no |

C1 and C2 are the same idea in two forms: evidence per unit length. C1 keeps two coefficients and changes only the transform, which is the most elegant form if it works. C3 tests whether the alternative explanation (runner-up) carries information the frozen mapping ignores. C4 tests whether a label-free rescaling alone closes the gap; because the training stations mix both cohorts, its normalisation constants are pooled and cohort-blind, so it is expected to fail unless the null distributions are already close, which section 2 says they are not. It stays in the plan as the honest test of the "normalisation only" hypothesis.

Nothing beyond three coefficients is proposed. No tree model, no station-level statistic, no per-day covariate that is not already produced by the scorer.

## 4. Validation design

Nested, station-grouped.

- Outer loop: the 18 Phase 3 folds, one station held out; outer-training stations are the 17 others of both cohorts, headline days only, `input_ok` only.
- Inner loop, inside each outer fold: leave-one-station-out over the 17 outer-training stations. Each candidate (C1 to C4, and B1 as the plain baseline) is fitted on 16 and scored on the 17th; inner predictions are pooled over the 17 inner folds.
- Selection rule, fixed before any outer-test number is seen: pick the candidate with the lowest pooled inner log loss, subject to an inner Beta-station energy precision of at least 0.90 (computed on the inner held-out Beta stations of that outer fold); if no candidate meets the gate, pick the lowest log loss and record the gate failure. Log loss is a proper scoring rule and calibration is the object under test; the gate keeps the operational constraint in the selection.
- The selected candidate is refitted on all 17 outer-training stations and applied once to the held-out station. The pooled outer result is the nested result ("N").
- Every fixed candidate is also reported on the outer folds without selection, labelled as such, so the price of selection is visible; the winner is not chosen from that table.
- C4's normalisation constants are computed on the 16 inner-training stations in the inner loop and on the 17 outer-training stations in the outer refit; never on the held-out station.

## 5. Outputs

All under this folder. Reproducible scripts (`run.py` with fixed seeds; `README.md` with PowerShell commands); `predictions_outer.parquet` (per site-day: fold, candidate, p, outcome, energies); `fits.csv` (coefficients and normalisation constants per outer fold and candidate; inner selection record per outer fold); `metrics_pooled.csv` and `metrics_station.csv`; `reliability.csv` (Brier, log loss, ECE by cohort and combined); `bootstrap.csv` (station-level intervals for Energy IoU, energy precision, recall); `LEAKAGE_AUDIT.md` (what each outer fold could read); `RESULTS.md` (observations, then interpretations, separately).

Metrics: Beta Energy IoU, Beta energy precision against 0.90, Beta false corrections, Beta sure-day recall, Alpha Energy IoU, Alpha energy precision and recall, site-day precision, recall and F1 per cohort, Brier, log loss and ECE per cohort and combined, and per-station tables with station-bootstrap intervals (1,000 draws, seed 9). Metric implementations: `m9_metrics.summarise`, `m9_metrics.calibration_reliability`, `final_eval.metrics.bootstrap_stations`; log loss added locally.

Before any comparison: the sandbox must reproduce R0 to the committed `07_metrics/pooled.csv` (Beta 0.8945 / 0.9407, Alpha 0.8692 / 0.8886) and B1, B2 to the 17 September sandbox numbers.

## 6. Success criterion (proposed, to confirm)

A candidate succeeds only if, in the nested result, one shared mapping gives: Beta energy precision ≥ 0.90; Beta Energy IoU and Alpha Energy IoU each within 0.02 of R0 (0.895, 0.869); sure-day recall ≥ 0.85 on both cohorts; and no cohort-specific behaviour anywhere in the fitted mapping. Trade-offs are reported whatever the outcome.

## 7. Risks

- Correlated site-days: days within a station are not independent; all uncertainty is station-level (bootstrap over stations), and inner and outer splits are by station.
- Eight Beta stations: the inner gate on Beta precision is estimated on at most seven stations and is noisy; the gate may pass or fail on single days.
- Synthetic versus real labels: Alpha's planted errors have a clean null; a shared mapping tuned to satisfy Alpha's 7,000 null days may be dominated by Alpha (five times Beta's day count). B2 is kept to show that reweighting alone does not fix this.
- Class imbalance and prevalence: RPF prevalence is 32% on Alpha and 20% on Beta sure, and varies from 0% to 81% by station; the intercept absorbs prevalence, so a station with unusual prevalence will be miscalibrated by any pooled mapping.
- Selection from repeated experimentation: four candidates and one benchmark is the whole search; the selection rule is fixed here before any result; the fixed-candidate outer table is reported but not used for selection.
- The possibility that no shared mapping exists without changing the scorer: the round-5 diagnosis says the Beta false-evidence days are a scorer limitation (a smooth hump versus a straight bridge). If C1 to C4 all fail the gate, the finding is that the cohort gap lives in the evidence, not the calibration, and closing it would be an M9 revision 3, which is outside this sandbox.
- Multiple comparisons at the gate: a candidate passing 0.90 by a few MWh on eight stations is not a robust pass; the bootstrap interval on Beta precision is reported beside the point estimate.

## 8. Questions before implementation (answer by number)

1. Covariates. Are window length L and the best-versus-runner-up margin acceptable as label-free scorer-derived quantities for C2 and C3? Both are produced by the frozen scorer before any label; neither is a station or cohort identifier. If you want the strictest reading ("evidence only"), only C1 and C4 remain.
2. Per-station label-free normalisation. A deployment sees a station's whole unlabelled history, so one could standardise z by that station's own null-evidence scale at run time. It uses no labels and no identifier, but it is transductive (uses the held-out station's readings). Include as a diagnostic (D1), exclude, or include as a candidate? My recommendation: diagnostic only, because it blurs the "training-only estimation" condition.
3. Selection criterion. Pooled inner log loss with the inner Beta precision gate (as in section 4), or the operational rule alone (inner Beta precision ≥ 0.90, then maximum inner combined Energy IoU)? My recommendation: the first; the second selects on the metric being reported.
4. Success thresholds. Confirm the section 6 numbers (0.90 gate; IoU within 0.02 of R0 on both cohorts; recall ≥ 0.85), or set others.
5. Scope of the benchmark B3 (cohort indicator). It is useful as the ceiling that says how much the cohort gap is worth, but it is exactly what the target forbids. Include it labelled as a diagnostic, or omit it?


## 9. Decisions taken before implementation (18 September 2026)

1. Covariates: window length L and the best-versus-runner-up margin are admitted as label-free scorer outputs. C2 and C3 are described as an augmented fitting layer built on the frozen M9 scorer, because their covariates can change the ranking of days; C1 remains the simplest candidate (its risk: dividing by L may discard legitimate accumulated evidence from long windows).
2. D1 (per-station label-free normalisation) is a transductive diagnostic only: excluded from selection and from the success result; if it performed well that would point to a future station-adaptive method, not to a success of the shared calibration.
3. Selection criterion: station-macro inner log loss (each inner held-out station weighted equally) subject to the inner Beta energy-precision gate of 0.90; combined Energy IoU is reported as an operational outcome, not used to select. If no candidate meets the gate, the lowest station-macro log loss is selected as a documented fallback and the outer result is not described as a success.
4. Success thresholds (point estimates, non-inferiority): Beta energy precision ≥ 0.90 (hard gate); Beta Energy IoU ≥ 0.875; Alpha Energy IoU ≥ 0.849; sure-day recall ≥ 0.85 in each cohort; one shared mapping with no cohort-specific parameters or thresholds. Station tables and bootstrap intervals reported beside them; lower bounds not required to clear every threshold. Success must not be produced by a severe failure at one or two stations hidden by pooled energy totals.
5. B3 is renamed "cohort-indicator diagnostic" (a cohort indicator with a shared slope is not a ceiling, since R0 also has cohort-specific slopes); kept outside selection; its purpose is to measure how much comes from knowing the cohort.

### Deviation recorded at implementation

C4 as first written (z standardised by the training stations' pooled median and MAD, then a logistic) is an affine transform of a single feature, which a logistic regression absorbs exactly; it would have been identical to B1. It was replaced by a non-affine, still label-free and training-only form: the logit of the empirical null distribution of r on the outer-training stations (days where the null wins, r ≤ 0), with the evidence ordering kept above the null range, then one logistic. This is the honest "normalisation only" test the plan intended.

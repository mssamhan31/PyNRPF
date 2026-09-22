# Results: Beta-only-trained evaluation (22 September 2026)

Every fitted quantity in this run was learnt from Beta alone: M8 bundles and M9
calibration pairs come from the other seven Beta stations for a held-out Beta station
and from all eight Beta stations for every Alpha station. M7 fits nothing. Beta 'unsure'
days were never fitted on and never enter a headline number. The engine, thresholds,
hyperparameters, scorer, control c = 0.7, population rules and metric definitions are
Phase 3; the only changes are the fitting scopes, the M8 split and the output folder
(`README.md` lists them). Observations first, interpretations at the end.

Dataset ranges, read from the frozen parquet files: Beta spans 2023-10-01 00:00 to
2024-09-30 23:45 UTC for all eight stations (one year); Alpha spans 2021-11-01 to
2024-09-30 for all ten stations. The M8 split for this run lies inside the Beta year:
fit window 2023-10-01 to 2024-07-31, in-bundle validation window 2024-08-01 to
2024-09-30 (reported only; nothing is tuned). The Alpha range is irrelevant to fitting
here because nothing is fitted on Alpha.

Run: `folds`, `train-m8` (9 fits, 20 to 25 s each), `predict-m7`, `predict-m8`,
`predict-m9`, `outcomes`, `metrics`, `report`, `operating-points`; `gamma` was not run.
Population as in Phase 3: 12,730 headline site-days (Alpha 10,425 with 3,381 RPF days;
Beta sure 2,305 with 470 RPF days).

## 1. Results table

`outputs/08_tables/table10_pooled_macro_stations.{csv,md}`. Pooled columns weight every
site-day and every MWh equally (the Phase 3 headline definition); macro columns are the
unweighted mean over stations of the per-station values (`outputs/07_metrics/macro.csv`).
Station rows carry the station's own values in the pooled columns. Energy IoU and energy
precision are the reference-energy metrics; day metrics treat AUTO_CORRECT as the
positive prediction. Blank cells are undefined (no correction applied, or no RPF day).

| Method | Group | Stations | Site-days | RPF days | Pooled Energy IoU | Pooled Energy precision | Pooled Day F1 | Pooled Day precision | Pooled Day recall | Macro Energy IoU | Macro Energy precision | Macro Day F1 | Macro Day precision | Macro Day recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M7 | Overall | 18 | 12730 | 3851 | 0.299 | 0.318 | 0.603 | 0.437 | 0.970 | 0.272 | 0.311 | 0.375 | 0.302 | 0.905 |
| M7 | Alpha | 10 | 10425 | 3381 | 0.304 | 0.318 | 0.662 | 0.497 | 0.992 | 0.277 | 0.300 | 0.432 | 0.354 | 0.994 |
| M7 | Beta sure | 8 | 2305 | 470 | 0.277 | 0.320 | 0.336 | 0.212 | 0.806 | 0.266 | 0.324 | 0.303 | 0.238 | 0.817 |
| M7 | alpha_A | 1 | 1039 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  |
| M7 | alpha_B | 1 | 1055 | 27 | 0.003 | 0.003 | 0.067 | 0.034 | 1.000 |  |  |  |  |  |
| M7 | alpha_C | 1 | 1055 | 627 | 0.495 | 0.532 | 0.775 | 0.636 | 0.990 |  |  |  |  |  |
| M7 | alpha_D | 1 | 1035 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  |
| M7 | alpha_E | 1 | 1035 | 700 | 0.594 | 0.644 | 0.832 | 0.714 | 0.996 |  |  |  |  |  |
| M7 | alpha_F | 1 | 1033 | 842 | 0.730 | 0.816 | 0.918 | 0.855 | 0.990 |  |  |  |  |  |
| M7 | alpha_G | 1 | 1037 | 650 | 0.604 | 0.658 | 0.811 | 0.684 | 0.994 |  |  |  |  |  |
| M7 | alpha_H | 1 | 1039 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  |
| M7 | alpha_I | 1 | 1042 | 148 | 0.106 | 0.107 | 0.331 | 0.199 | 1.000 |  |  |  |  |  |
| M7 | alpha_J | 1 | 1055 | 387 | 0.237 | 0.244 | 0.589 | 0.419 | 0.987 |  |  |  |  |  |
| M7 | beta_A | 1 | 337 | 26 | 0.150 | 0.152 | 0.166 | 0.091 | 0.923 |  |  |  |  |  |
| M7 | beta_B | 1 | 231 | 125 | 0.376 | 0.734 | 0.532 | 0.562 | 0.504 |  |  |  |  |  |
| M7 | beta_C | 1 | 339 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  |
| M7 | beta_D | 1 | 188 | 45 | 0.257 | 0.301 | 0.327 | 0.235 | 0.533 |  |  |  |  |  |
| M7 | beta_E | 1 | 306 | 21 | 0.025 | 0.026 | 0.135 | 0.074 | 0.762 |  |  |  |  |  |
| M7 | beta_F | 1 | 286 | 164 | 0.770 | 0.809 | 0.749 | 0.601 | 0.994 |  |  |  |  |  |
| M7 | beta_G | 1 | 283 | 85 | 0.544 | 0.561 | 0.489 | 0.323 | 1.000 |  |  |  |  |  |
| M7 | beta_H | 1 | 335 | 4 | 0.006 | 0.006 | 0.029 | 0.015 | 1.000 |  |  |  |  |  |
| M8 | Overall | 18 | 12730 | 3851 | 0.662 | 0.928 | 0.688 | 0.963 | 0.535 | 0.459 | 0.763 | 0.498 | 0.787 | 0.571 |
| M8 | Alpha | 10 | 10425 | 3381 | 0.762 | 0.934 | 0.700 | 0.987 | 0.542 | 0.442 | 0.776 | 0.414 | 0.839 | 0.459 |
| M8 | Beta sure | 8 | 2305 | 470 | 0.333 | 0.888 | 0.608 | 0.809 | 0.487 | 0.480 | 0.752 | 0.602 | 0.741 | 0.684 |
| M8 | alpha_A | 1 | 1039 | 0 | 0.000 |  | 0.000 |  |  |  |  |  |  |  |
| M8 | alpha_B | 1 | 1055 | 27 | 0.000 |  | 0.000 |  | 0.000 |  |  |  |  |  |
| M8 | alpha_C | 1 | 1055 | 627 | 0.714 | 0.843 | 0.733 | 0.971 | 0.589 |  |  |  |  |  |
| M8 | alpha_D | 1 | 1035 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  |
| M8 | alpha_E | 1 | 1035 | 700 | 0.739 | 0.957 | 0.638 | 1.000 | 0.469 |  |  |  |  |  |
| M8 | alpha_F | 1 | 1033 | 842 | 0.808 | 0.962 | 0.771 | 1.000 | 0.627 |  |  |  |  |  |
| M8 | alpha_G | 1 | 1037 | 650 | 0.755 | 0.961 | 0.690 | 0.997 | 0.528 |  |  |  |  |  |
| M8 | alpha_H | 1 | 1039 | 0 | 0.000 |  | 0.000 |  |  |  |  |  |  |  |
| M8 | alpha_I | 1 | 1042 | 148 | 0.732 | 0.892 | 0.658 | 0.938 | 0.507 |  |  |  |  |  |
| M8 | alpha_J | 1 | 1055 | 387 | 0.671 | 0.817 | 0.651 | 0.964 | 0.491 |  |  |  |  |  |
| M8 | beta_A | 1 | 337 | 26 | 0.693 | 0.925 | 0.857 | 0.913 | 0.808 |  |  |  |  |  |
| M8 | beta_B | 1 | 231 | 125 | 0.012 | 1.000 | 0.047 | 1.000 | 0.024 |  |  |  |  |  |
| M8 | beta_C | 1 | 339 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  |
| M8 | beta_D | 1 | 188 | 45 | 0.717 | 0.833 | 0.747 | 0.739 | 0.756 |  |  |  |  |  |
| M8 | beta_E | 1 | 306 | 21 | 0.858 | 0.926 | 0.927 | 0.950 | 0.905 |  |  |  |  |  |
| M8 | beta_F | 1 | 286 | 164 | 0.278 | 0.911 | 0.595 | 0.796 | 0.476 |  |  |  |  |  |
| M8 | beta_G | 1 | 283 | 85 | 0.782 | 0.906 | 0.843 | 0.864 | 0.824 |  |  |  |  |  |
| M8 | beta_H | 1 | 335 | 4 | 0.503 | 0.518 | 0.800 | 0.667 | 1.000 |  |  |  |  |  |
| M9 | Overall | 18 | 12730 | 3851 | 0.879 | 0.936 | 0.794 | 0.991 | 0.662 | 0.643 | 0.925 | 0.613 | 0.982 | 0.692 |
| M9 | Alpha | 10 | 10425 | 3381 | 0.874 | 0.935 | 0.773 | 0.999 | 0.630 | 0.596 | 0.924 | 0.507 | 0.998 | 0.577 |
| M9 | Beta sure | 8 | 2305 | 470 | 0.895 | 0.941 | 0.923 | 0.952 | 0.896 | 0.702 | 0.926 | 0.744 | 0.965 | 0.806 |
| M9 | alpha_A | 1 | 1039 | 0 | 0.000 |  | 0.000 |  |  |  |  |  |  |  |
| M9 | alpha_B | 1 | 1055 | 27 | 0.839 | 0.948 | 0.500 | 1.000 | 0.333 |  |  |  |  |  |
| M9 | alpha_C | 1 | 1055 | 627 | 0.856 | 0.932 | 0.731 | 0.997 | 0.577 |  |  |  |  |  |
| M9 | alpha_D | 1 | 1035 | 0 | 0.000 |  | 0.000 |  |  |  |  |  |  |  |
| M9 | alpha_E | 1 | 1035 | 700 | 0.890 | 0.954 | 0.771 | 1.000 | 0.627 |  |  |  |  |  |
| M9 | alpha_F | 1 | 1033 | 842 | 0.892 | 0.933 | 0.836 | 1.000 | 0.719 |  |  |  |  |  |
| M9 | alpha_G | 1 | 1037 | 650 | 0.858 | 0.948 | 0.771 | 1.000 | 0.628 |  |  |  |  |  |
| M9 | alpha_H | 1 | 1039 | 0 | 0.000 |  | 0.000 |  |  |  |  |  |  |  |
| M9 | alpha_I | 1 | 1042 | 148 | 0.806 | 0.860 | 0.737 | 0.989 | 0.588 |  |  |  |  |  |
| M9 | alpha_J | 1 | 1055 | 387 | 0.817 | 0.893 | 0.725 | 1.000 | 0.568 |  |  |  |  |  |
| M9 | beta_A | 1 | 337 | 26 | 0.956 | 0.988 | 0.960 | 1.000 | 0.923 |  |  |  |  |  |
| M9 | beta_B | 1 | 231 | 125 | 0.889 | 0.948 | 0.927 | 1.000 | 0.864 |  |  |  |  |  |
| M9 | beta_C | 1 | 339 | 0 | 0.000 |  | 0.000 |  |  |  |  |  |  |  |
| M9 | beta_D | 1 | 188 | 45 | 0.902 | 0.966 | 0.850 | 0.971 | 0.756 |  |  |  |  |  |
| M9 | beta_E | 1 | 306 | 21 | 0.899 | 0.911 | 0.977 | 0.955 | 1.000 |  |  |  |  |  |
| M9 | beta_F | 1 | 286 | 164 | 0.930 | 0.950 | 0.952 | 0.946 | 0.957 |  |  |  |  |  |
| M9 | beta_G | 1 | 283 | 85 | 0.870 | 0.902 | 0.889 | 0.884 | 0.894 |  |  |  |  |  |
| M9 | beta_H | 1 | 335 | 4 | 0.173 | 0.818 | 0.400 | 1.000 | 0.250 |  |  |  |  |  |

Further observations from `outputs/07_metrics/`:

- The release gate (`gate.json`) reads "M9 is the default method": Beta sure energy
  precision 0.941 against the 0.90 gate, and M9 has the strongest overall Energy IoU.
- Confusion counts, headline days. M8: overall tp 2062, fp 79, fn 1789; Alpha tp 1833,
  fp 25, fn 1548; Beta sure tp 229, fp 54, fn 241. M9 Alpha: fp 2, fn 1251. M9 and M7
  counts are the Phase 3 counts (section 3).
- M8 applies 27,614 MWh on Alpha against 32,036 MWh required, and 3,878 MWh on Beta sure
  against 9,896 MWh required.
- Macro counts: 14 of the 18 stations have an RPF day (alpha_A, alpha_D, alpha_H and
  beta_C have none). M8 applies a correction on 15 stations, M9 on 14, M7 on all 18.
- Operating points (`outputs/10_operating_points/selections.csv`): the Alpha folds' thresholds
  were chosen on 2288 Beta calibration days each (the eligible days of all eight Beta
  stations); the Beta folds' on 1951 to 2105 days.

Headline figure: `outputs/08_figures/fig01_headline_metrics.png`
(`publication/2_journal_article/sandbox/2026-09-22_beta_only_eval/outputs/08_figures/fig01_headline_metrics.png`).

## 2. M8 in-bundle validation metrics

One row per distinct bundle, from `outputs/02_bundles/<fold_id>.json`. Validation rows
are the training stations' rows in 2024-08-01 to 2024-09-30 (a temporal split within the
training stations, as in Phase 3; the held-out station is never in them). Thresholds
are the frozen 0.585985 (day) and 0.892234 (interval). Reported only.

| bundle (fold) | shared by | training rows | training RPF days | day precision | day recall | day F1 | day positives | interval precision | interval recall | interval F1 | interval positives | elapsed s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alpha_alpha_A | alpha_alpha_B to alpha_alpha_J (9 folds) | 221,280 | 470 | 0.962 | 0.927 | 0.944 | 165 | 0.971 | 0.901 | 0.934 | 3569 | 20 |
| beta_beta_A | - | 188,928 | 444 | 0.941 | 0.966 | 0.954 | 149 | 0.972 | 0.915 | 0.942 | 3279 | 23 |
| beta_beta_B | - | 199,104 | 345 | 0.920 | 0.708 | 0.800 | 130 | 0.959 | 0.888 | 0.922 | 2663 | 21 |
| beta_beta_C | - | 188,736 | 470 | 0.948 | 0.879 | 0.912 | 165 | 0.971 | 0.904 | 0.936 | 3569 | 22 |
| beta_beta_D | - | 203,232 | 425 | 0.940 | 0.893 | 0.916 | 140 | 0.978 | 0.911 | 0.944 | 3241 | 21 |
| beta_beta_E | - | 191,904 | 449 | 0.946 | 0.979 | 0.962 | 144 | 0.975 | 0.907 | 0.940 | 3287 | 20 |
| beta_beta_F | - | 193,824 | 306 | 0.965 | 0.873 | 0.917 | 126 | 0.969 | 0.864 | 0.913 | 2607 | 14 |
| beta_beta_G | - | 194,112 | 385 | 0.976 | 0.891 | 0.931 | 137 | 0.968 | 0.882 | 0.923 | 2782 | 20 |
| beta_beta_H | - | 189,120 | 466 | 0.949 | 0.902 | 0.925 | 164 | 0.973 | 0.909 | 0.940 | 3555 | 25 |

## 3. Reproduction checks

Every metric column of `outputs/07_metrics/pooled.csv` was compared with the Phase 3
release (`sandbox/2026-09-16_phase3_release/outputs/07_metrics/pooled.csv`) for M9 Beta
sure and for M7 on all three groups, and with the intercept-transfer reference
(`sandbox/2026-09-18_intercept_transfer/curve.csv`, direction `beta->alpha`, level
`k0_source_pair`) for M9 Alpha. All 53 strict comparisons pass with a difference of
exactly zero. The rows below are the headline subset.

| check | metric | this run | reference | difference |
|---|---|---|---|---|
| M9 Beta sure vs Phase 3 | Energy IoU | 0.8945 | 0.8945 | 0 |
| M9 Beta sure vs Phase 3 | energy precision | 0.9407 | 0.9407 | 0 |
| M9 Beta sure vs Phase 3 | day F1 / day precision | 0.9232 / 0.9525 | 0.9232 / 0.9525 | 0 |
| M9 Beta sure vs Phase 3 | sure-day recall; fp; fn | 0.8957; 21; 49 | 0.8957; 21; 49 | 0 |
| M9 Alpha vs Beta-8 pair (intercept transfer) | Energy IoU | 0.8742 | 0.8742 | 0 |
| M9 Alpha vs Beta-8 pair (intercept transfer) | energy precision | 0.9347 | 0.9347 | 0 |
| M9 Alpha vs Beta-8 pair (intercept transfer) | sure-day recall | 0.6300 | 0.6300 | 0 |
| M9 Alpha vs Beta-8 pair (intercept transfer) | false-corrected days (fp); fn | 2; 1251 | 2; 1251 | 0 |
| M7 Alpha vs Phase 3 | Energy IoU / precision / day F1 | 0.3041 / 0.3175 / 0.6623 | same | 0 |
| M7 Beta sure vs Phase 3 | Energy IoU / precision / day F1 | 0.2775 / 0.3198 / 0.3360 | same | 0 |
| M7 combined vs Phase 3 | Energy IoU / precision / day F1 | 0.2986 / 0.3179 / 0.6028 | same | 0 |
| M8 combined vs Phase 3 (expected to differ) | Energy IoU / precision | 0.6618 / 0.9285 | 0.7322 / 0.9559 | -0.070 / -0.027 |
| M8 Alpha vs Phase 3 (expected to differ) | Energy IoU / precision / day recall | 0.7620 / 0.9342 / 0.5421 | 0.8521 / 0.9589 / 0.7894 | -0.090 / -0.025 / -0.247 |
| M8 Beta sure vs Phase 3 (expected to differ) | Energy IoU / precision / sure-day recall | 0.3333 / 0.8879 / 0.4872 | 0.3395 / 0.9319 / 0.5106 | -0.006 / -0.044 / -0.023 |

Two further identities hold. The M7 interval prediction table is row-for-row identical to
the Phase 3 table (1,281,312 rows, `pred_interval` and `pred_day`). The eight Beta M9
calibration pairs equal the Phase 3 pairs to machine precision, and the shared Alpha pair
(cal_intercept -4.590095, cal_slope 2.162080, fitted on 2288 eligible days of all eight
Beta stations, 470 RPF days) equals the intercept-transfer source pair to six decimals.

## 4. What changed versus Phase 3, and why

Only M8 changed, because it is the only method whose fit depends on which stations are in
the training set in a way this run alters. M9 on Beta uses the same seven-station pairs
as Phase 3 (the `beta_only` scope and the Phase 3 `other_stations_same_cohort` scope
coincide for a Beta fold), so it reproduces exactly; M9 on Alpha is, by construction, the
Beta-8 pair the intercept-transfer sandbox already applied, and reproduces that. M7 has
no fitted quantity. M8 in Phase 3 trained on 17 stations of both cohorts (about 1.1 to
1.2 million rows, three Alpha years plus the Beta year); here it trains on 7 or 8 Beta
stations (189 to 221 thousand rows, one year) with the fit window moved inside the Beta
year. The pooled M8 numbers move from 0.732 / 0.956 (overall Energy IoU / precision) to
0.662 / 0.928, from 0.852 / 0.959 to 0.762 / 0.934 on Alpha, and from 0.340 / 0.932 to
0.333 / 0.888 on Beta sure (sure-day recall 0.511 to 0.487). The gate decision and the
method ordering are unchanged.

## 5. Fold manifest summary (9 bundles)

`outputs/01_folds/fold_manifest.csv`. The training signature is the SHA-256 of the
sorted training-station set (first 12 characters shown); the ten Alpha folds share one,
so nine distinct signatures give nine bundles and eighteen kept bundle manifests, of
which nine name `alpha_alpha_A` in `shared_with`. Training days are complete headline
days of the training stations; M9 calibration days are the same population, of which
the eligible (`input_ok`) subset is fitted on.

| fold | held out | M8 training stations | training days | training RPF days | signature | M9 calibration days | calibration RPF days |
|---|---|---|---|---|---|---|---|
| alpha_alpha_A | alpha_A | beta_A to beta_H (all 8) | 2305 | 470 | 187be161c298 | 2305 | 470 |
| alpha_alpha_B | alpha_B | beta_A to beta_H (all 8) | 2305 | 470 | 187be161c298 | 2305 | 470 |
| alpha_alpha_C | alpha_C | beta_A to beta_H (all 8) | 2305 | 470 | 187be161c298 | 2305 | 470 |
| alpha_alpha_D | alpha_D | beta_A to beta_H (all 8) | 2305 | 470 | 187be161c298 | 2305 | 470 |
| alpha_alpha_E | alpha_E | beta_A to beta_H (all 8) | 2305 | 470 | 187be161c298 | 2305 | 470 |
| alpha_alpha_F | alpha_F | beta_A to beta_H (all 8) | 2305 | 470 | 187be161c298 | 2305 | 470 |
| alpha_alpha_G | alpha_G | beta_A to beta_H (all 8) | 2305 | 470 | 187be161c298 | 2305 | 470 |
| alpha_alpha_H | alpha_H | beta_A to beta_H (all 8) | 2305 | 470 | 187be161c298 | 2305 | 470 |
| alpha_alpha_I | alpha_I | beta_A to beta_H (all 8) | 2305 | 470 | 187be161c298 | 2305 | 470 |
| alpha_alpha_J | alpha_J | beta_A to beta_H (all 8) | 2305 | 470 | 187be161c298 | 2305 | 470 |
| beta_beta_A | beta_A | the other 7 | 1968 | 444 | 9e655ac2cbfb | 1968 | 444 |
| beta_beta_B | beta_B | the other 7 | 2074 | 345 | 687e3db085b8 | 2074 | 345 |
| beta_beta_C | beta_C | the other 7 | 1966 | 470 | d17f6b345119 | 1966 | 470 |
| beta_beta_D | beta_D | the other 7 | 2117 | 425 | 0ade4440f4a6 | 2117 | 425 |
| beta_beta_E | beta_E | the other 7 | 1999 | 449 | d445d20dcf28 | 1999 | 449 |
| beta_beta_F | beta_F | the other 7 | 2019 | 306 | 992c826c1833 | 2019 | 306 |
| beta_beta_G | beta_G | the other 7 | 2022 | 385 | 97d01fa1d9b2 | 2022 | 385 |
| beta_beta_H | beta_H | the other 7 | 1970 | 466 | 2485a444b772 | 1970 | 466 |

## 6. Interpretations

The Beta-only M8 is a more conservative model on Alpha, not a less precise one. Its
Alpha false-corrected days fall from 74 (Phase 3) to 25 while its missed RPF days rise
from 712 to 1548; energy precision stays above 0.93 on both cohorts and the whole loss is
recall. A model that has seen one year of eight real stations, all with the Beta error
pattern, under-fires on Alpha's manufactured errors spread over three years. Whether
that is a fair test of M8 or an unfair one is a design question for the paper, not a
number this run settles: Phase 3 gave M8 the Alpha stations to learn from, this run does
not, and the two designs answer different questions.

On Beta the Beta-only M8 is essentially the Phase 3 M8. The pooled Energy IoU moves by
0.006 and the ordering of stations is the same: beta_B remains the failure (3 of 125 RPF
days corrected here, 8 in Phase 3), beta_F remains low (0.278, with 20 false-corrected
days against 1 in Phase 3), beta_A, beta_E and beta_G remain the strong stations. The ten
Alpha stations in Phase 3's training set therefore added little that transferred to Beta,
and the beta_B failure is a property of that station rather than of the training cohort.
Beta energy precision does drop below the 0.90 gate (0.888 from 0.932), driven by beta_F
and beta_H; M8 was not the gated default in Phase 3 either.

The macro columns are lower than the pooled columns for every method, and the reason is
mostly definitional rather than a property of the methods. Four of the eighteen stations
have no RPF day, and the frozen Energy IoU gives such a station 0 even when nothing was
corrected, so a quarter of the Alpha mean and an eighth of the Beta mean is a fixed zero.
Beyond that, one-vote-per-station exposes the small stations: M9's Beta macro Energy IoU
of 0.702 (pooled 0.895) is pulled down by beta_H (0.173 on 4 RPF days) as much as by the
zero of beta_C, and M8's Alpha macro of 0.442 (pooled 0.762) by alpha_B (0 on 27 RPF days)
and the three zero-RPF stations. If a macro number is to appear in the paper, the
treatment of zero-RPF stations needs a stated rule; the table reports the frozen
definition as it stands and takes no such decision.

M9 needs no interpretation here: it is unchanged on Beta and, on Alpha, it is the
intercept-transfer sandbox's untouched source pair, whose reading is already in that
sandbox's `RESULTS.md`. The M9 gate decision and default-method recommendation are the
Phase 3 ones.

Caveats. The M8 validation metrics are a temporal split within the training stations and
say nothing about held-out stations; they are reported because the bundle records them.
The M8 fit window covers ten months of one year, so seasonal coverage is thinner than in
Phase 3, which is part of what "Beta-only" means and was not compensated for. Nothing was
tuned; the M8 thresholds, seeds and hyperparameters are the Phase 3 values.

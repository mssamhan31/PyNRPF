# 2026-07-15 PRD: Reproducible m9_pbm Model Development and Forecast-Impact Notebooks

## Document Status

| Field | Value |
|---|---|
| Status | Approved design, not yet implemented |
| Created | 2026-07-15 |
| Scope | Journal-article Notebooks 02a-02g, Notebook 03, shared experiment code, tests, and publication outputs |
| Primary model | Interpretable, non-ML `m9_pbm` using F1 bridge improvement, F3 slope-continuity improvement, and F4 duration plausibility |
| Primary validation | Beta leave-one-substation-out, sure-only day-level F1 |
| Output root | `publication/2_journal_article/outputs/` |
| Historical source | Selected work in `notebooks/99_Misc/`; no paper-facing workflow may depend on that folder |

## 1. Executive Summary

The current `02_correction_validation.ipynb` is based on the older `m7_dtr` and `m8_xgb` workflow and no longer represents the journal article's intended method. It will be replaced by a readable sequence of seven notebooks, `02a` through `02g`, that reconstruct only the key `m9_pbm` experiments developed in `99_Misc`.

The new workflow will:

1. explain the physical model and create a journal-ready two-panel example for Alpha substation F on 2024-02-17;
2. build a reproducible candidate-window and feature cache from final Alpha and Beta data;
3. compare Beta-only, Beta-plus-Alpha, and Alpha-only training regimes using the compact equal-weight physical model;
4. evaluate all 511 nonempty equal-weight subsets of the nine physical features;
5. optimise the three selected feature weights and decision threshold using grid search and random search under strict nested Beta leave-one-substation-out validation;
6. compare DNN, random forest, and XGBoost decision models using only the same three physical features;
7. report final day, interval, candidate-window IoU, correction-energy IoU, and confidence-based manual-review reduction results; and
8. execute Notebook 03 for real using `m9_pbm`-corrected Gamma data and direct seven-day-ahead point forecasts.

The final paper method is fixed as a deterministic, non-ML model using F1, F3, and F4. The feature ablation and ML experiments provide evidence, but they do not change this final model family. The selected weights and thresholds will be learned without using labels from the held-out Beta substation.

All key CSV tables, figures, model manifests, and compact result artifacts will be written under the article-level `outputs/` folder. Large candidate caches, fold prediction audits, and temporary model files will remain reproducible local intermediates and will not be committed to Git.

## 2. Confirmed Decisions

The following decisions are locked for implementation:

- Use `02a`, `02b`, and so on rather than decimal notebook filenames.
- Replace `02_correction_validation.ipynb`; do not retain its m7/m8 experiment narrative.
- Do not include m7 or m8 as methods or benchmark rows in the new Notebook 02 or Notebook 03 results.
- Use the term **substation** consistently. Avoid generic `station` and `site` in paper-facing text and outputs.
- Treat Beta sure-only as the primary evaluation subset.
- Treat Beta all as a secondary subset that retains all final manual labels, including labels on unsure days.
- Compare three training regimes: Beta only, Beta plus Alpha, and Alpha only.
- Run the 511-model equal-weight feature ablation under the Beta-plus-Alpha regime.
- Use only F1 bridge improvement, F3 slope-continuity improvement, and F4 duration plausibility for weight optimisation and ML comparison.
- Select the final deterministic F1/F3/F4 model using optimised weights and an optimised threshold. ML cannot replace the final model.
- Optimise the final model using Beta only. Alpha must not influence the final deployed weights or thresholds.
- Use a nonnegative weight simplex with each feature weight at least `0.05` and all three weights summing to `1.0`.
- Use a `0.05` grid and 1,000 seeded random weight samples, with seed `9`.
- Select models using macro-substation F1, with precision as the first tie-breaker.
- Include interval metrics, candidate-window IoU, and correction-energy IoU.
- Include confidence coverage and manual-review burden analysis.
- Use Beta substation B as Gamma because it has the largest correction/data-error impact, not because it has the largest number of RPF days.
- Apply the Beta-B outer-fold `m9_pbm` model to Gamma. That model may use the other seven Beta substations but no Beta-B labels and no Alpha data.
- Keep forecast-model training fixed before September 2024.
- Produce direct point predictions for each 15-minute target exactly seven days after the latest observation available to that prediction.
- Compare raw, `m9_pbm`-corrected, and manually corrected data conditions in Notebook 03.

## 3. Background and Existing Evidence

### 3.1 Final datasets

The workflow will use only the final article datasets:

| Dataset | Substations | Period | Rows | RPF substation-days | Role |
|---|---:|---|---:|---:|---|
| Alpha | 10 | 2021-11-01 to 2024-09-30 | 1,011,264 | 3,423 | Simulated sign-error data with known reference |
| Beta | 8 | 2023-10-01 to 2024-09-30 | 280,800 | 630 | Manually reviewed actual sign-error data |
| Gamma | 1 (`beta_B`) | 2023-10-01 to 2024-09-30 | 35,136 | 152 | Forecast-impact case study |

Current Beta confidence composition:

| Confidence | Substation-days | RPF substation-days |
|---|---:|---:|
| Sure | 2,310 | 471 |
| Unsure | 618 | 159 |

Input files:

- `dataset/final/dataset_alpha.parquet`
- `dataset/final/dataset_beta.parquet`
- `dataset/final/dataset_gamma.parquet`
- `dataset/final/dataset_final_summary.csv`
- `dataset/final/gamma_selection_summary.csv`
- `dataset/final/sha256.txt`

### 3.2 Latest compact-model evidence

The final-review equal-weight subset search found the following Beta-sure results under Beta-plus-Alpha leave-one-substation-out evaluation:

| Feature count | Best subset | Precision | Recall | F1 |
|---:|---|---:|---:|---:|
| 2 | F2 roughness + F4 duration | 0.8362 | 0.9427 | 0.8862 |
| 3 | F1 bridge + F3 slope + F4 duration | 0.8561 | 0.9724 | 0.9105 |
| 4 | F1 bridge + F3 slope + F4 duration + F6 solar strength | 0.8686 | 0.9682 | 0.9157 |
| 5 | F1 bridge + F3 slope + F4 duration + F6 solar strength + F7 peak alignment | 0.8978 | 0.9321 | 0.9146 |

This supports the compact three-feature model: its recall is approximately `0.97`, its F1 is approximately `0.91`, and its F1 is close to the best larger subsets.

The feature interpretation must remain nuanced:

- F1 bridge is the strongest and most consistently useful feature.
- F4 duration is the second most consistently useful feature.
- F5 N-height is broadly helpful across many arbitrary subsets.
- F3 slope is interaction-dependent, but it is highly effective specifically with F1 bridge and F4 duration.
- Replacing F3 slope with F5 N-height in the compact model reduces Beta-sure F1 from `0.9105` to `0.8702`.

The paper must therefore say that F1/F3/F4 is the **best compact combination**, not that these are universally the three strongest standalone features.

### 3.3 Legacy-cache caveat

The latest `99_Misc` ablation reused a daily cache whose candidate window had originally been selected by an earlier bridge/roughness/slope core score. Consequently, the cached result is an important regression anchor, but it does not by itself prove that every reported subset is an end-to-end model using only its listed features.

The reconstructed workflow must resolve this transparently:

- cache candidate-level features before reducing to one row per day;
- let each ablation subset or weighted physical model score candidate windows using its own active features;
- select that model's highest-scoring candidate window for the day;
- retain one explicit regression check against the legacy fixed-window F1/F3/F4 result; and
- use the self-consistent candidate selection for all primary paper-facing metrics.

This may cause small differences from the cached exploratory results. Such differences must be reported, not hidden or corrected by hard-coding legacy values.

## 4. Product Goals

### 4.1 Primary goals

- Replace the outdated correction-validation notebook with a complete, readable, repeatable `m9_pbm` experiment sequence.
- Make the physical method understandable without requiring the reader to inspect helper source code.
- Reproduce the key scientific comparisons from `99_Misc` using final manual-review labels.
- Enforce leakage-safe substation splits and nested model selection.
- Produce compact, paper-ready CSV tables and figures from a clean execution.
- Create a stable final non-ML model artifact for downstream correction and forecasting.
- Execute the Gamma forecast-impact experiment with real outputs and no placeholder rows.
- Support work across laptops through Git-tracked notebooks, configuration, compact results, and manifests.

### 4.2 Secondary goals

- Preserve detailed audit outputs locally for troubleshooting.
- Make every long-running stage resumable from a validated cache.
- Make stale data, stale labels, or mismatched caches fail loudly.
- Record sufficient metadata to reproduce every headline number.

## 5. Non-Goals

- Recreating every exploratory experiment from `99_Misc`.
- Reintroducing m7, m8, logistic regression, calendar features, border-minima experiments, or error galleries into the main workflow.
- Using XGBoost from the conference paper as a correction benchmark.
- Selecting the final model from DNN, random forest, or XGBoost.
- Treating the 511-subset search as an untouched external test.
- Automatically changing final manual labels.
- Regenerating the oracle review application or review galleries.
- Committing large candidate-level caches or all fold-level predictions to Git.
- Changing Notebook 00 or Notebook 01 beyond compatibility fixes required by the new output contracts.
- Rewriting the manuscript as part of this implementation.

## 6. Terminology and Reporting Rules

### 6.1 Required terminology

- `substation`: an anonymised electrical substation.
- `substation-day`: one calendar day for one substation.
- `candidate window`, `W`: one contiguous interval considered for sign correction.
- `RPF sign-error day`: a substation-day containing at least one incorrectly positive RPF interval.
- `Beta sure`: Beta substation-days where reviewer confidence is `sure`.
- `Beta all`: all Beta substation-days using their final labels, including unsure days.
- `LOSO`: leave-one-substation-out validation.
- `auto accept`: accept either a confident positive decision for automatic correction or a confident negative decision for automatic retention.

### 6.2 Prohibited or discouraged terminology

- Do not use `station split`; use `substation split` or `leave-one-substation-out`.
- Do not use `site` in paper-facing labels when `substation` is intended.
- Do not use exploratory identifiers such as chunk names or letter-number codes in notebook prose, table labels, model names, or filenames.
- Do not call the compact features the three universally most important features.
- Do not describe `beta_B` as the substation with the most RPF days.

### 6.3 Development-result disclosure

The paper-facing narrative must state that:

- Beta labels are used for model development and cross-validation;
- the 511-subset ablation is exploratory model-development evidence;
- there is no completely untouched external Beta dataset;
- outer held-out-substation predictions are used for leakage-controlled performance estimation; and
- an additional independent dataset would still be required for final external validation.

## 7. Physical Method Specification

### 7.1 Core notation

Every notebook formula must be followed by a complete, human-readable notation block.

| Symbol | Meaning |
|---|---|
| `t` | A 15-minute timestamp within a substation-day |
| `d` | A substation-day |
| `y(t)` | Observed net load in MW; a sign-error RPF interval appears as a positive bump |
| `S(t)` | Estimated solar generation in MW |
| `W` | A contiguous candidate correction window |
| `Omega(W)` | `W` plus short left and right context shoulders |
| `U_no(t)` | Reconstructed underlying demand when no sign correction is applied |
| `U_corr,W(t)` | Reconstructed underlying demand when sign correction is applied inside `W` |
| `F_i(W)` | Value of physical feature `i` for candidate `W` |
| `w_i` | Nonnegative weight assigned to feature `i` |
| `Score(W)` | Weighted physical score for candidate `W` |
| `W_d*` | Highest-scoring candidate window on day `d` |
| `tau` | Day-level decision threshold selected from training data only |
| `epsilon` | Small positive constant preventing division by zero |

### 7.2 Demand reconstructions

The no-correction reconstruction is:

```text
U_no(t) = S(t) + y(t)
```

The candidate-corrected reconstruction is:

```text
U_corr,W(t) = S(t) - y(t),  when t is in W
              S(t) + y(t),  otherwise
```

The corrected net-load output is:

```text
y_corr,W(t) = -y(t),  when t is in W and the day is predicted positive
              y(t),   otherwise
```

### 7.3 Candidate-window generation

Candidate generation is deterministic and label-free:

- resolution: 15 minutes;
- scan interval: approximately 06:00 through 18:00;
- minimum duration: 2 slots, or 30 minutes;
- maximum duration: 32 slots, or 8 hours;
- candidate midpoint: within 14 slots, or 3.5 hours, of the daily solar peak;
- one candidate must be contiguous;
- all candidate constraints and boundary conventions must live in configuration;
- labels, reviewer confidence, and held-out-substation identity must not influence candidate generation.

The cache must record candidate counts, excluded candidates, missing-input counts, and the reason for any day with no valid candidate.

### 7.4 Nine physical features

#### F1: Bridge improvement

Let `L_W(t)` be the straight line joining demand anchors immediately before and after `W`. Define:

```text
E_bridge(U, W) = median over t in W of |U(t) - L_W(t)|

F1(W) = [E_bridge(U_no, W) - E_bridge(U_corr,W, W)]
        / [E_bridge(U_no, W) + E_bridge(U_corr,W, W) + epsilon]
```

Positive values mean correction produces a more plausible bridge between the surrounding demand anchors.

#### F2: Roughness improvement

Define total variation over the candidate window and its shoulders:

```text
TV(U, Omega(W)) = sum over adjacent t in Omega(W) of |U(t + Delta_t) - U(t)|

F2(W) = [TV(U_no, Omega(W)) - TV(U_corr,W, Omega(W))]
        / [TV(U_no, Omega(W)) + TV(U_corr,W, Omega(W)) + epsilon]
```

Positive values mean correction reduces local roughness.

#### F3: Slope-continuity improvement

Estimate robust slopes immediately outside and inside the left and right boundaries. Let `J(U,W)` be the sum of the left and right boundary slope mismatches:

```text
J(U, W) = |m_out,left(U) - m_in,left(U)|
          + |m_in,right(U) - m_out,right(U)|

F3(W) = [J(U_no, W) - J(U_corr,W, W)]
        / [J(U_no, W) + J(U_corr,W, W) + epsilon]
```

Positive values mean correction improves slope continuity at the candidate boundaries.

#### F4: Duration plausibility

```text
F4(W) = clip(duration_hours(W) / 1.5, 0, 1)
```

This is a soft duration feature. It does not replace the hard 30-minute to 8-hour candidate constraint.

#### F5: N-height ratio

```text
N_height(W) = max observed net load inside W
              - max observed net load at the two window boundaries

F5(W) = clip(N_height(W) / robust_day_substation_power_scale, 0, 1)
```

Larger values indicate a stronger N-shaped positive bounce.

#### F6: Solar-strength ratio

```text
F6(W) = clip(P95 solar inside W / median historical daytime-solar P95 for the substation, 0, 1)
```

Historical scaling is computed from unlabelled measurements only.

#### F7: Solar-peak alignment

```text
F7(W) = clip(1 - |midpoint(W) - daily solar-peak time| / 3.5 hours, 0, 1)
```

Larger values mean the candidate is closer to the daily solar peak.

#### F8: Substation-centred core score

```text
core(W) = F1(W) + F2(W) + F3(W)

F8(W) = robust_bound(core(W) - median historical daily best-core score for the substation)
```

The substation median must use unlabelled scores only.

#### F9: Substation-rank core score

```text
F9(d) = 2 * percentile_rank of the daily best-core score within the substation - 1
```

The rank distribution must use unlabelled scores only.

### 7.5 Candidate score and day decision

For an active feature set `A`:

```text
Score(W) = sum over i in A of w_i * F_i(W)

W_d* = argmax over candidate windows W on day d of Score(W)

predicted_RPF_day(d) = Score(W_d*) >= tau
```

If the day is predicted positive, `W_d*` becomes the predicted correction interval. If the day is predicted negative, no interval is corrected.

### 7.6 Missing values and finite outputs

- Candidate generation must interpolate only short, internal gaps using a documented deterministic rule.
- Remaining undefined feature values become zero only after their cause is counted in an audit table.
- Solar values are clipped to nonnegative values where required by the physical definition.
- All final scores must be finite.
- A run must fail if required day rows disappear during cache-to-label joins.

## 8. Validation Design and Leakage Controls

### 8.1 Evaluation regimes

| Regime | Training and tuning data | Held-out evaluation | Purpose |
|---|---|---|---|
| Beta only | Sure days from seven Beta substations | Eighth Beta substation | Primary real-data generalisation |
| Beta plus Alpha | Sure days from seven Beta substations plus Alpha | Eighth Beta substation | Test whether simulated Alpha data helps |
| Alpha only | Alpha data only | All Beta substations | Pure simulated-to-actual transfer |

For the first two regimes, repeat until every Beta substation has been held out once.

### 8.2 Threshold selection

- Thresholds are selected from training data only.
- The selection objective is macro-substation day-level F1.
- Tie-break order is higher macro precision, higher macro recall, and then the more conservative higher threshold.
- Beta-plus-Alpha tuning must give Alpha and Beta balanced total influence so Alpha's larger row count cannot dominate.
- Beta confidence may filter the training subset to sure days, but held-out confidence may only be used after prediction to create the sure-only report.

### 8.3 Nested weight and ML selection

For each outer held-out Beta substation:

1. Remove that substation and all its labels from model and threshold selection.
2. Use the remaining seven Beta substations for inner LOSO model selection.
3. For each inner fold, tune the threshold on six Beta substations and evaluate on the seventh.
4. Select weights or ML hyperparameters using aggregated inner macro-substation F1 and the defined tie-breaks.
5. Refit or retune the selected model on all seven outer-training substations.
6. Select the final outer-fold threshold using those seven substations only.
7. Predict the outer held-out substation once.

No Alpha data is permitted in final F1/F3/F4 weight optimisation or in the Beta-B model used by Gamma.

### 8.4 Model-development caveat

The feature set F1/F3/F4 was chosen using previous Beta development results. Nested LOSO prevents direct fold leakage during weight and threshold tuning, but it does not turn Beta into a previously untouched external dataset. This limitation must appear in notebook conclusions and the manuscript discussion.

## 9. Required Metrics

### 9.1 Day-level metrics

Report precision, recall, F1, support, positive support, TP, FP, FN, and TN for:

- Beta sure and Beta all;
- pooled held-out predictions;
- macro-substation averages; and
- each held-out Beta substation.

The paper headline is pooled held-out Beta-sure P/R/F1. Macro-substation results are required beside it.

### 9.2 Interval metrics

Expand each predicted positive candidate window to its 15-minute interval flags. Report interval-level precision, recall, F1, TP, FP, FN, and TN for Beta sure and Beta all.

Full-day interval metrics are primary. Daytime-only metrics are a diagnostic audit.

### 9.3 Candidate-window IoU

For true interval set `T_d` and predicted interval set `P_d`:

```text
window_IoU(d) = |T_d intersection P_d| / |T_d union P_d|
```

Report two scopes:

- `tp_days_only`: days where both true and predicted day labels are positive;
- `event_days_truth_or_prediction`: all days where either label is positive, assigning IoU zero to FP and FN days.

Report mean IoU, median IoU, proportion at least `0.50`, proportion at least `0.70`, and median absolute start/end boundary errors.

### 9.4 Correction-energy IoU

At each 15-minute interval, define correction energy:

```text
e(t) = 2 * max(y(t), 0) * 0.25 hours
```

Then:

```text
energy_IoU = sum e(t) over true-and-predicted intervals
             / sum e(t) over true-or-predicted intervals
```

Also report energy precision, energy recall, energy F1, manual correction MWh, predicted correction MWh, overlap MWh, and union MWh.

Pooled full-day energy metrics are primary. Daytime and per-substation values are required secondary outputs.

### 9.5 Confidence coverage and manual-review burden

For each held-out prediction:

```text
confidence_margin = |Score(W_d*) - tau|
```

Because final physical weights sum to one, score scale is controlled across folds. Sort predictions from highest to lowest confidence margin and evaluate auto-accept coverage at:

```text
50%, 60%, 70%, 80%, 90%, 95%, and 100%
```

An auto-accepted positive day is corrected automatically. An auto-accepted negative day is retained automatically. All remaining days are sent to manual review.

For each coverage level report:

- auto-accepted days;
- days remaining for manual review;
- auto TP, FP, FN, and TN;
- total auto errors;
- auto precision, recall, and F1;
- manual-review percentage; and
- true RPF days remaining for review.

The recommended operating point is the maximum held-out Beta-sure coverage satisfying both:

```text
precision >= 0.99
F1 >= 0.95
```

This is a model-development operating point and must be labelled accordingly.

## 10. Notebook Architecture

## 10.1 `02a_m9_pbm_method_and_example.ipynb`

### Purpose

Explain the model in plain language, define all notation, and create the main method visual.

### Required narrative

- Why RPF sign errors produce an N-shaped positive net-load curve.
- Why solar and observed net load can reconstruct underlying demand under two interpretations.
- Candidate-window constraints.
- The nine feature concepts, with full notation under every formula.
- The distinction between candidate-window selection, day classification, and interval correction.

### Required example

Use `alpha_F`, `2024-02-17`:

- 96 readings;
- true RPF interval from 10:00 through 13:45;
- 16 positive intervals;
- current regression anchor candidate slots 40 through 55.

Create one journal-ready two-panel figure:

- panel (a): observed net load `y(t)` and solar generation `S(t)`;
- panel (b): `U_no(t)` and `U_corr,W(t)`, with candidate shading, the two anchors, the linear bridge, and the true interval reference.

The figure must be generated from data and equations, not hand-drawn.

### Key outputs

- `outputs/tables/02a_m9_pbm_method_example/table01_alpha_F_2024-02-17_plot_data.csv`
- `outputs/figures/02a_m9_pbm_method_example/fig01_m9_pbm_alpha_F_2024-02-17.png`
- `outputs/manifests/02a_m9_pbm_method_example.json`

## 10.2 `02b_m9_pbm_candidate_features.ipynb`

### Purpose

Build and validate the reusable candidate-level feature cache once.

### Required processing

- Load final Alpha and Beta datasets.
- Validate schemas, hashes, substation counts, confidence counts, date coverage, and duplicate keys.
- Generate all valid candidate windows for every substation-day.
- Compute F1-F9 without using labels.
- Join labels only after candidate features are complete.
- Write a candidate-level Parquet cache and compact daily/cache summaries.
- Record missing-input handling and finite-feature checks.
- Verify the Alpha-F example numerically.

### Key outputs

- local-only `outputs/intermediate/02b_m9_pbm_candidate_features/candidate_feature_cache.parquet`
- local-only `outputs/intermediate/02b_m9_pbm_candidate_features/day_input_cache.parquet`
- `outputs/tables/02b_m9_pbm_candidate_features/table01_candidate_counts.csv`
- `outputs/tables/02b_m9_pbm_candidate_features/table02_feature_quality_summary.csv`
- `outputs/tables/02b_m9_pbm_candidate_features/table03_dataset_and_label_join_audit.csv`
- `outputs/manifests/02b_m9_pbm_candidate_features.json`

## 10.3 `02c_m9_pbm_training_regimes.ipynb`

### Purpose

Compare the three training regimes while holding the compact model form fixed.

### Model

Use equal weights for F1, F3, and F4. Select each day's candidate using that same equal-weight score. Tune only the threshold within each regime.

### Required comparisons

- Beta only;
- Beta plus Alpha; and
- Alpha only.

### Key outputs

- `outputs/metrics/02c_m9_pbm_training_regimes/01_day_metrics.csv`
- `outputs/metrics/02c_m9_pbm_training_regimes/02_thresholds_by_fold.csv`
- `outputs/tables/02c_m9_pbm_training_regimes/table01_regime_headline_metrics.csv`
- `outputs/tables/02c_m9_pbm_training_regimes/table02_regime_metrics_by_substation.csv`
- `outputs/figures/02c_m9_pbm_training_regimes/fig01_regime_precision_recall_f1.png`
- `outputs/figures/02c_m9_pbm_training_regimes/fig02_thresholds_by_heldout_substation.png`
- `outputs/manifests/02c_m9_pbm_training_regimes.json`

The notebook conclusion must state whether Alpha materially improves Beta-only training and must not overstate small differences.

## 10.4 `02d_m9_pbm_feature_ablation.ipynb`

### Purpose

Measure how model performance changes across every nonempty equal-weight subset of F1-F9.

### Design

- Search family: equal-weight physical features only.
- Number of feature subsets: `2^9 - 1 = 511`.
- Empty subset: excluded because it has no physical score.
- Regime: Beta plus Alpha.
- Evaluation: eight held-out Beta-substation folds.
- Threshold: selected independently within each fold from permitted training data.
- Candidate window: selected using the active subset for that model.

### Required analyses

- full ranking of all 511 models;
- best model for each feature count from 1 through 9;
- top-10, top-25, top-50, and top-100 feature frequency;
- paired marginal F1 effect from adding each feature to the same base subset;
- direct compact comparison of F1/F3/F4 against F1/F4/F5;
- a regression check against the legacy cached F1/F3/F4 result.

### Key outputs

- `outputs/metrics/02d_m9_pbm_feature_ablation/01_all_511_subset_metrics.csv`
- `outputs/metrics/02d_m9_pbm_feature_ablation/02_thresholds_by_subset_and_fold.csv`
- `outputs/tables/02d_m9_pbm_feature_ablation/table01_best_by_feature_count.csv`
- `outputs/tables/02d_m9_pbm_feature_ablation/table02_feature_frequency.csv`
- `outputs/tables/02d_m9_pbm_feature_ablation/table03_paired_marginal_effects.csv`
- `outputs/tables/02d_m9_pbm_feature_ablation/table04_compact_model_comparison.csv`
- `outputs/figures/02d_m9_pbm_feature_ablation/fig01_f1_by_feature_count.png`
- `outputs/figures/02d_m9_pbm_feature_ablation/fig02_top_subset_performance.png`
- `outputs/figures/02d_m9_pbm_feature_ablation/fig03_feature_frequency_and_marginal_effect.png`
- `outputs/manifests/02d_m9_pbm_feature_ablation.json`

The notebook must preserve the conclusion that F1/F3/F4 is the chosen compact feature set even if another larger subset has slightly higher F1.

## 10.5 `02e_m9_pbm_weight_optimisation.ipynb`

### Purpose

Optimise F1/F3/F4 weights and thresholds under strict Beta-only nested LOSO, and produce the final physical model artifacts.

### Weight constraints

```text
w1 + w3 + w4 = 1
w1 >= 0.05
w3 >= 0.05
w4 >= 0.05
```

### Grid search

- step size: `0.05`;
- positive simplex points satisfying the minimum weight: 171 candidates;
- equal weights included as a baseline through an explicit candidate if not represented exactly by the grid.

### Random search

- samples: 1,000;
- seed: `9`;
- sample `z` from a three-dimensional Dirichlet distribution;
- transform with `w_i = 0.05 + 0.85 * z_i` so weights sum to one and retain the minimum.

### Selection

- use the nested procedure in Section 8.3;
- select by inner macro-substation F1;
- tie-break using precision, recall, and conservative threshold;
- compare the best grid and random candidates under the same outer-fold protocol;
- select the better leakage-safe family;
- report weight stability across outer folds.

### Final artifacts

Produce three kinds of artifact:

1. outer-fold artifacts for leakage-controlled Beta performance;
2. a Beta-B outer-fold artifact trained using the other seven Beta substations, for Notebook 03; and
3. a deployment artifact fitted using all sure Beta substations, clearly marked as unsuitable for evaluating the existing Beta dataset.

### Key outputs

- `outputs/metrics/02e_m9_pbm_weight_optimisation/01_grid_search_results.csv`
- `outputs/metrics/02e_m9_pbm_weight_optimisation/02_random_search_results.csv`
- `outputs/metrics/02e_m9_pbm_weight_optimisation/03_nested_outer_fold_metrics.csv`
- `outputs/metrics/02e_m9_pbm_weight_optimisation/04_selected_weights_and_thresholds.csv`
- `outputs/tables/02e_m9_pbm_weight_optimisation/table01_equal_vs_grid_vs_random.csv`
- `outputs/tables/02e_m9_pbm_weight_optimisation/table02_weight_stability.csv`
- `outputs/figures/02e_m9_pbm_weight_optimisation/fig01_weight_simplex_performance.png`
- `outputs/figures/02e_m9_pbm_weight_optimisation/fig02_selected_weights_by_fold.png`
- `outputs/manifests/02e_m9_pbm_final_model.json`
- `outputs/manifests/02e_m9_pbm_beta_B_outer_fold_model.json`
- `outputs/manifests/02e_m9_pbm_weight_optimisation.json`

## 10.6 `02f_m9_pbm_ml_comparison.ipynb`

### Purpose

Test whether DNN, random forest, or XGBoost can improve the final day decision when given the same three physical features.

### Fair-comparison rule

- Select one candidate window per day using equal-weight F1/F3/F4.
- Provide only that candidate's F1, F3, and F4 values to the ML model.
- Do not include calendar, substation identity, raw load, solar, or any of F2/F5/F6/F7/F8/F9.
- ML replaces the final day decision only; it does not create a different candidate generator.

### Models

- DNN: reproducible two-hidden-layer `scikit-learn` MLP with standardised inputs.
- RF: `RandomForestClassifier` with class- and substation-balanced training weights.
- XGB: `XGBClassifier` using deterministic seeds and histogram tree construction.

Use small, declared hyperparameter grids and nested training-only selection. Do not launch broad exploratory searches.

### Regimes and reporting

Evaluate the three ML models under Beta only, Beta plus Alpha, and Alpha only. Report Beta sure and Beta all using the same pooled, macro-substation, and per-substation structure as Notebook 02c.

The final `m9_pbm` remains the deterministic optimised physical model regardless of ML ranking.

### Key outputs

- `outputs/metrics/02f_m9_pbm_ml_comparison/01_ml_nested_metrics.csv`
- `outputs/metrics/02f_m9_pbm_ml_comparison/02_selected_hyperparameters.csv`
- `outputs/tables/02f_m9_pbm_ml_comparison/table01_physical_vs_ml.csv`
- `outputs/tables/02f_m9_pbm_ml_comparison/table02_ml_by_substation.csv`
- `outputs/figures/02f_m9_pbm_ml_comparison/fig01_physical_vs_ml_precision_recall_f1.png`
- `outputs/manifests/02f_m9_pbm_ml_comparison.json`

## 10.7 `02g_m9_pbm_final_evaluation.ipynb`

### Purpose

Consolidate the final optimised physical model's held-out predictions and quantify classification, localisation, correction energy, and manual-review reduction.

### Required sections

1. Final model definition and selected fold weights.
2. Day-level Beta sure and Beta all results.
3. Per-substation performance and confusion counts.
4. Interval-level performance.
5. Candidate-window IoU.
6. Correction-energy metrics and energy IoU.
7. Confidence coverage.
8. Recommended development operating point.
9. Practical interpretation and limitations.

### Required auto-accept figure

Create a dual-axis plot with:

- horizontal axis: auto-accepted percentage;
- left vertical axis: number of substation-days remaining for manual review;
- right vertical axis: auto-accepted errors;
- stacked right-axis bars separating FP and FN errors; and
- markers identifying the maximum coverage satisfying precision at least `0.99` and F1 at least `0.95`.

Create a separate compact curve showing precision, recall, and F1 versus auto-accept coverage.

### Key outputs

- `outputs/metrics/02g_m9_pbm_final_evaluation/01_day_metrics.csv`
- `outputs/metrics/02g_m9_pbm_final_evaluation/02_interval_metrics.csv`
- `outputs/metrics/02g_m9_pbm_final_evaluation/03_window_iou_metrics.csv`
- `outputs/metrics/02g_m9_pbm_final_evaluation/04_energy_metrics.csv`
- `outputs/metrics/02g_m9_pbm_final_evaluation/05_confidence_coverage_metrics.csv`
- local-only `outputs/intermediate/02g_m9_pbm_final_evaluation/heldout_prediction_audit.parquet`
- `outputs/tables/02g_m9_pbm_final_evaluation/table01_final_headline_metrics.csv`
- `outputs/tables/02g_m9_pbm_final_evaluation/table02_final_metrics_by_substation.csv`
- `outputs/tables/02g_m9_pbm_final_evaluation/table03_localisation_and_energy.csv`
- `outputs/tables/02g_m9_pbm_final_evaluation/table04_recommended_auto_accept_operating_point.csv`
- `outputs/figures/02g_m9_pbm_final_evaluation/fig01_final_confusion_matrices.png`
- `outputs/figures/02g_m9_pbm_final_evaluation/fig02_window_iou_distribution.png`
- `outputs/figures/02g_m9_pbm_final_evaluation/fig03_energy_metric_summary.png`
- `outputs/figures/02g_m9_pbm_final_evaluation/fig04_auto_accept_manual_review_and_errors.png`
- `outputs/figures/02g_m9_pbm_final_evaluation/fig05_auto_accept_precision_recall_f1.png`
- `outputs/manifests/02g_m9_pbm_final_evaluation.json`

## 10.8 `03_gamma_forecast_impact.ipynb`

### Purpose

Quantify how RPF sign errors and `m9_pbm` correction affect direct seven-day-ahead net-load point forecasts for Gamma.

### Gamma definition

- Gamma remains `beta_B`.
- Selection rationale: highest raw-versus-reference data-error RMSE and largest practical correction impact.
- Current data-error RMSE: approximately `2.266 MW`.
- Do not claim that Beta B has the highest number of RPF days.

### Correction model

- Load `02e_m9_pbm_beta_B_outer_fold_model.json`.
- This artifact is trained and tuned using the other seven Beta substations only.
- Do not use Alpha data.
- Do not use Beta-B labels for candidate scoring, weight selection, or threshold selection.
- Apply the model to every Gamma day to produce `m9_pbm`-corrected net load.

### Data conditions

- raw uncorrected data;
- `m9_pbm`-corrected data; and
- manually corrected data as the ideal reference benchmark.

### Forecast definition

For every 15-minute target timestamp `t` in September 2024:

- forecast exactly one net-load value at `t`;
- set forecast origin to `t - 7 days`;
- use no observations after that origin;
- use a 14-day lookback window ending at the origin;
- include target-calendar features known at forecast time;
- train forecast models once using pre-September target examples; and
- do not refit forecast models during September.

This is a collection of direct seven-day-ahead point forecasts, not a forecast of a complete seven-day trajectory from one origin.

### Forecast models

- seasonal naive: value at exactly `t - 7 days`;
- linear regression; and
- XGBoost regression.

Train each model separately under each data condition. Evaluate every prediction against manually corrected net load.

### Required metrics

- RMSE in MW;
- MAE in MW;
- number of valid target timestamps;
- data-error-only RMSE and MAE for raw and `m9_pbm`-corrected series;
- absolute and percentage error reduction from correction; and
- per-model difference between raw and corrected training conditions.

### Key outputs

- `outputs/intermediate/03_gamma_forecast_impact/01_gamma_series.parquet`
- local-only forecast design matrices and prediction audits under the same intermediate folder;
- `outputs/metrics/03_gamma_forecast_impact/01_gamma_data_error_metrics.csv`
- `outputs/metrics/03_gamma_forecast_impact/02_gamma_forecast_metrics.csv`
- `outputs/tables/03_gamma_forecast_impact/table01_gamma_data_error_summary.csv`
- `outputs/tables/03_gamma_forecast_impact/table02_gamma_forecast_impact.csv`
- `outputs/figures/03_gamma_forecast_impact/fig01_gamma_raw_m9_manual_example_week.png`
- `outputs/figures/03_gamma_forecast_impact/fig02_gamma_data_error_rmse.png`
- `outputs/figures/03_gamma_forecast_impact/fig03_gamma_forecast_rmse.png`
- `outputs/figures/03_gamma_forecast_impact/fig04_gamma_forecast_residuals.png`
- `outputs/manifests/03_gamma_forecast_impact.json`

Final execution must contain no placeholder, smoke-only, or synthetic metric rows.

## 10.9 `04_publication_tables_figures.ipynb`

Notebook 04 must be updated after the new workflow exists so it:

- removes dependencies on obsolete Notebook 02 m7/m8 outputs;
- imports only compact paper-facing outputs from 02a-02g and 03;
- creates a clear inventory of missing upstream outputs;
- exports final CSV, Markdown, and LaTeX tables where already supported; and
- never reads directly from `99_Misc`.

## 11. Shared Code Architecture

Readable notebooks remain the primary experiment narrative. Shared code prevents duplicated implementation and keeps formulas consistent.

Recommended modules:

```text
notebooks/_m9_pbm_data.py
notebooks/_m9_pbm_features.py
notebooks/_m9_pbm_validation.py
notebooks/_m9_pbm_plotting.py
notebooks/_gamma_forecast.py
```

Responsibilities:

- `_m9_pbm_data.py`: schemas, paths, dataset loading, cache hashes, and joins;
- `_m9_pbm_features.py`: reconstructions, candidates, F1-F9, scoring, correction, and IoU;
- `_m9_pbm_validation.py`: LOSO folds, threshold sweeps, nested selection, metrics, weights, and ML wrappers;
- `_m9_pbm_plotting.py`: journal style and reusable publication figures;
- `_gamma_forecast.py`: direct seven-day-ahead examples, forecast models, and forecast metrics.

The existing `_experiment_helpers.py` may be reduced or adapted only where required for Notebooks 00, 01, and 04. New m9 logic should not be buried inside the old m7/m8 correction functions.

## 12. Configuration Contract

Extend `config/experiment_config.yaml` with explicit sections for:

- dataset paths and expected hashes;
- candidate scan start/end, minimum/maximum duration, shoulder length, anchor rules, and solar-peak radius;
- feature constants and `epsilon`;
- regime definitions;
- threshold objective and tie-breaks;
- ablation feature list;
- weight grid, random count, minimum weight, and seed;
- ML hyperparameter grids and seeds;
- confidence coverage levels and operating-point constraints;
- Gamma substation and forecast dates;
- forecast horizon and lookback;
- output overwrite/resume controls.

Notebooks must display the relevant configuration before execution. No headline parameter may exist only as an unexplained constant in a helper module.

## 13. Notebook Readability Standard

Every notebook must be understandable as a standalone research record.

Required structure:

1. title and research question;
2. concise summary of inputs, method, outputs, and expected runtime;
3. imports and resolved paths;
4. input validation;
5. thorough method explanation in Markdown;
6. visible formula and notation sections;
7. short, commented code cells that call named helpers;
8. visible key tables and figures;
9. plain-language findings;
10. limitations and leakage statement; and
11. output inventory with clickable relative paths where supported.

Markdown requirements:

- Explain why each experiment is being run, not only what function is called.
- Define every mathematical symbol directly below its formula.
- Explain each split using substation names and training/test roles.
- Explain Beta sure versus Beta all before displaying results.
- Avoid unexplained abbreviations and internal experiment identifiers.
- Do not paste large implementation functions into notebook cells.

Code requirements:

- Keep cells short enough to scan.
- Add comments for non-obvious transformations and leakage controls.
- Display assertions and audit summaries before long computations.
- Use deterministic seeds.
- Do not suppress exceptions that indicate stale caches or missing outputs.

## 14. Output Contract

All outputs live under:

```text
publication/2_journal_article/outputs/
```

Category-first layout remains consistent with the article workflow:

```text
outputs/intermediate/<notebook_slug>/
outputs/metrics/<notebook_slug>/
outputs/tables/<notebook_slug>/
outputs/figures/<notebook_slug>/
outputs/manifests/<notebook_slug>.json
```

Output rules:

- CSV is the required format for compact tables and metrics.
- Parquet is preferred for large candidate and prediction caches.
- Paper figures are 300-dpi PNG with stable dimensions and readable labels.
- Figure source data must be written as CSV when practical.
- Every notebook writes one JSON manifest.
- CSV schemas and column meanings must be documented in the PRD implementation notes or module docstrings.
- Re-running a notebook with unchanged inputs and seed must reproduce the same key metrics.
- No new output may be written under `99_Misc`.

## 15. Git and Cross-Laptop Artifact Policy

Commit:

- notebooks;
- shared Python modules;
- configuration;
- tests;
- compact headline metrics and tables;
- the full 511-row ablation ranking;
- final figures;
- compact model JSON artifacts;
- manifests and environment/version summaries.

Do not commit:

- candidate-level feature caches;
- full fold-by-candidate threshold sweeps;
- large prediction audits;
- serialized ML estimators;
- temporary notebook checkpoints;
- duplicate HTML galleries;
- generated copies of raw/final datasets.

Each manifest must record:

- input paths and SHA256 hashes;
- configuration hash;
- code/Git commit when available;
- Python and package versions;
- random seed;
- start/end timestamps and elapsed time;
- row counts;
- output paths and hashes;
- whether output is development-only or publication-ready.

## 16. Runtime and Caching Budget

The candidate scan is the expensive data stage and must be performed once. Later subset, threshold, weight, and ML experiments must reuse the validated candidate cache.

Known benchmark:

- the prior 766-model, 6,128-fold cached subset search completed in approximately 348 seconds;
- the new equal-weight ablation contains 511 models;
- weight optimisation contains 171 grid candidates plus 1,000 random candidates.

Runtime targets after the candidate cache exists:

| Notebook | Target |
|---|---:|
| 02c regime comparison | under 10 minutes |
| 02d 511-subset ablation | under 20 minutes |
| 02e weight optimisation | under 60 minutes |
| 02f ML comparison | under 60 minutes |
| 02g final metrics and figures | under 10 minutes |

Implementation requirements for meeting these targets:

- batch candidate score calculations with NumPy where practical;
- avoid rereading Parquet files inside model/fold loops;
- precompute fold indices and label arrays;
- vectorise threshold sweeps on sorted scores;
- process random weights in batches;
- write resumable stage checkpoints;
- benchmark a small batch before launching the full weight search; and
- fail with a clear estimate if projected weight-search runtime exceeds one hour rather than silently starting an uncontrolled run.

## 17. Tests and Verification

### 17.1 Unit tests

Add focused tests for:

- no-correction and corrected demand equations;
- candidate durations and daytime boundaries;
- solar-peak radius;
- bridge anchors and bridge error;
- total variation and slope-jump features;
- F4 duration values at 0.5, 1.5, and 8 hours;
- F5-F9 finite output and clipping;
- candidate argmax and deterministic tie-breaking;
- day thresholding and interval correction;
- window IoU including TP, FP, FN, and empty cases;
- correction-energy IoU;
- confidence coverage counts;
- seven-day forecast origin and no-lookahead rules.

### 17.2 Leakage tests

Tests must prove that:

- held-out Beta labels are absent from threshold selection;
- held-out Beta labels are absent from inner weight/ML selection;
- Beta-B labels are absent from the Gamma correction artifact;
- Alpha is absent from final weight optimisation and Gamma correction;
- reviewer confidence is used only for allowed filtering/reporting;
- feature scaling uses no labels; and
- each forecast target uses observations no later than `target - 7 days`.

### 17.3 Notebook execution tests

- Execute every notebook from a clean kernel in order.
- Confirm no notebook imports code or data from absolute user-specific paths.
- Confirm all expected output files exist and are nonempty.
- Confirm no final manifest contains placeholder status.
- Confirm figures are nonblank and labels do not overlap.
- Confirm the Alpha-F example contains 96 rows and 16 true intervals.
- Confirm the ablation contains exactly 511 unique nonempty subsets.
- Confirm weight vectors satisfy minimums and sum to one.
- Confirm Beta outer predictions contain exactly one prediction per Beta substation-day.
- Confirm Gamma remains exactly one substation and uses Beta B.

## 18. Acceptance Criteria

The implementation is complete only when:

- the outdated `02_correction_validation.ipynb` has been replaced by 02a-02g;
- all notebooks meet the readability standard;
- a clean 02a-02g execution produces all declared key outputs;
- the final model is deterministic F1/F3/F4 with optimised nonzero weights;
- final Beta metrics come only from outer held-out-substation predictions;
- Beta sure and Beta all are both reported;
- pooled, macro-substation, and per-substation metrics are available;
- interval, window IoU, and energy IoU outputs are complete;
- the dual-axis auto-accept/manual-review figure is generated;
- the maximum coverage satisfying precision `>= 0.99` and F1 `>= 0.95` is identified or explicitly reported as unavailable;
- Notebook 03 uses the Beta-B outer-fold `m9_pbm` artifact and no Alpha data;
- every September target prediction obeys the exact seven-day horizon;
- Notebook 03 contains real results with no placeholders;
- Notebook 04 consumes the new outputs;
- key compact artifacts are suitable for Git; and
- no paper-facing workflow depends on `99_Misc`.

## 19. Implementation Sequence

1. Archive the outdated experiment plan and preserve this PRD as the new source of truth.
2. Define output schemas, configuration, and shared module boundaries.
3. Implement and test reconstruction, candidate, and feature primitives.
4. Build Notebook 02a and verify the Alpha-F example.
5. Build Notebook 02b and create the validated candidate cache.
6. Build Notebook 02c and confirm the three regime definitions.
7. Build Notebook 02d and run the 511-subset ablation.
8. Build Notebook 02e and run nested grid/random optimisation.
9. Freeze the final physical model artifacts.
10. Build Notebook 02f and run the three-feature ML comparison.
11. Build Notebook 02g and generate final localisation, energy, and triage results.
12. Revise Notebook 03 to use `m9_pbm` and execute the real forecast experiment.
13. Update Notebook 04 output consolidation.
14. Execute all notebooks from clean kernels and validate manifests.
15. Review compact outputs for Git inclusion and update `.gitignore` narrowly.

## 20. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Self-consistent candidate selection changes legacy scores | Preserve a labelled regression check, use new end-to-end results as primary, and document differences |
| Feature search is mistaken for independent validation | Label it model development and keep final outer-fold predictions separate |
| Alpha dominates mixed training | Use dataset-balanced, macro-substation threshold selection |
| Substations with no sure positives distort macro F1 | Report pooled, all-substation macro, positive-substation macro, and per-substation confusion counts |
| Weight search exceeds one hour | Use 171 grid plus 1,000 random candidates, batching, runtime projection, and resumable checkpoints |
| ML comparison becomes an uncontrolled tuning exercise | Use only three inputs, small declared grids, nested selection, and fixed seeds |
| Confidence margins differ across folds | Normalise physical weights to sum to one and report fold thresholds/score ranges |
| Auto-accept operating point is overinterpreted | Label it development-only and report the full coverage curve |
| Beta-B Gamma evaluation leaks Beta-B labels | Use the Beta-B outer-fold artifact trained only on the other seven Beta substations |
| Gamma rationale conflicts with refreshed labels | Describe Beta B as highest data-error impact; retain the ranking table |
| Forecast experiment accidentally becomes a seven-day trajectory forecast | Assert one target, one origin, and `origin = target - 7 days` for every row |
| Large outputs make Git difficult across laptops | Commit compact tables/figures/manifests only; ignore reproducible caches and audits |
| Notebook prose becomes thin wrappers around helpers | Require thorough Markdown, visible formulas, visible audits, and plain-language conclusions |

## 21. Migration Map from `99_Misc`

The implementation should reuse validated logic selectively from:

- `2026-07-07_m9_physical_score_method_spec_v1.md` for method terminology and equations;
- `13_260707_physical_score_loso_experiments.py` for candidate features, LOSO metrics, IoU, energy, and confidence coverage;
- `19_overnight_weight_ml_timesplit_experiments.py` for nested weight and ML experiment patterns;
- `2026-07-10_manual_review_physical_feature_ablation_journal.md` for final-review compact-model findings;
- `outputs/20260710_physical_feature_subset_search/` for regression expectations;
- `tracked_best_model_cache/physical_score_c14_best/daily_feature_cache.csv` only as a temporary regression source, never as the final workflow dependency; and
- the existing `03_gamma_forecast_impact.ipynb` and forecast helpers for the direct seven-day-ahead point-forecast structure.

Do not copy exploratory naming, stale confidence counts, user-specific paths, or m7/m8 assumptions into the new workflow.

## 22. Definition of Done

The project is done when a collaborator can clone the repository on another laptop, obtain the final datasets through the documented data process, execute Notebooks 02a through 03 in order, and reproduce the paper's method figure, regime comparison, 511-subset ablation, compact weight search, ML comparison, final P/R/F1 and IoU tables, manual-review reduction plots, and Gamma forecast-impact results without opening or depending on `99_Misc`.


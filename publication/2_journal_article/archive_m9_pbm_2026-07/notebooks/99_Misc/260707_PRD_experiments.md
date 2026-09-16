# 260707 PRD: Fast LOSO Physical-Score Model Experiments

## Executive Summary

This PRD defines the next structured model-development experiment for the RPF sign-error detector. The goal is to move from ad hoc bridge-score experiments toward a single interpretable model family that can be tuned and evaluated consistently.

The experiment focuses on day-level detection first. Each model must output:

- a day-level RPF flag;
- one selected RPF window when the day is positive;
- a confidence score for review triage.

The primary model-development metric is:

> Beta leave-one-site-out sure-only day-level F1.

Secondary reporting must include Beta all-days, Alpha, per-site metrics, and confidence-coverage curves.

This is a model-development experiment, not final publication validation. The purpose is to identify a defensible model family and understand whether Beta labels and Alpha data should be used for tuning.

## Motivation

Previous experiments showed several important patterns:

- The original m7 and m8 baselines do not transfer cleanly to final Beta labels.
- The E4 v0.3 three-feature physical score is interpretable and keeps strong Alpha performance, but over-detects Beta RPF.
- Duration guards, especially E11/E12, substantially improve Beta sure-only F1 by reducing false positives.
- The m9.2 bridge development result showed that site-relative score normalisation can improve Beta performance, especially using site medians or site ranks.
- Fully complex ML models such as broad XGB feature sets are not the preferred next step because the aim is a journal-defensible, interpretable method.

The next experiment therefore keeps a single physical-score model family, expands it to at most nine interpretable components, and compares simple ablations against a lightweight logistic regression weight learner.

## Primary Question

Can a scale-free physical score, tuned under Beta leave-one-site-out validation, achieve strong day-level Beta performance while remaining simple enough for a journal paper and future Ausgrid triage?

## Training And Evaluation Regimes

The experiment will compare three training/tuning regimes.

| Regime | Training / tuning labels | Test labels | Purpose |
|---|---|---|---|
| R1 `beta_loso` | Other 7 Beta sites, sure days only | Held-out Beta site | Real-data site generalisation |
| R2 `beta_loso_plus_alpha` | Other 7 Beta sites sure days + all Alpha | Held-out Beta site | Test whether Alpha helps or hurts |
| R3 `alpha_only_to_beta` | Alpha only | All Beta sites and each Beta site | Pure Alpha-to-Beta transfer |

For R1 and R2, repeat over all 8 Beta held-out sites. For R3, fit/tune once on Alpha and evaluate on all Beta sites.

Beta `unsure` days are excluded from fitting and threshold selection. They are still evaluated in the Beta all-days report.

## Candidate Window Generation

Each site-day is scanned using dense v0.3-style candidate windows.

| Rule | Value |
|---|---|
| Daytime search window | 06:00-18:00 |
| Candidate duration | 0.5h to 8h |
| Solar-centred restriction | midpoint within 3.5h of daily solar peak |
| Output | zero or one RPF window per site-day |
| Metrics for this round | day-level only |

The implementation should reuse cached candidate/day score logic where possible so repeated iteration is fast.

## Model Components And Formulas

Each candidate window receives nine bounded, mostly scale-free components. Let:

```text
y[t]       = observed net load MW
S[t]       = solar generation MW
U_no[t]    = S[t] + y[t]
U_corr[t]  = S[t] - y[t] inside candidate window W, else S[t] + y[t]
eps        = 1e-9
clip01(x)  = min(1, max(0, x))
robust_bound(x) = clip x to [-3, 3], then divide by 3
```

For each site-day, define unlabelled robust scales:

```text
site_solar_scale = median over site-days of daytime p95(S)
site_net_scale   = median over site-days of daytime p95(abs(y))
day_net_scale    = max(p95(abs(y_daytime)), p95(S_daytime), eps)
```

These scales must be computed without labels. For R1/R2 Beta LOSO, any site-level scale for the held-out site may use that held-out site's unlabelled measurements, but never its labels.

| ID | Component | Explicit formula | Motivation |
|---|---|---|---|
| F1 | `bridge_improvement` | `(bridge_error_no(W) - bridge_error_corr(W)) / (bridge_error_no(W) + bridge_error_corr(W) + eps)` | Core physical signal: corrected demand should bridge better between surrounding anchors |
| F2 | `roughness_improvement` | `(roughness_no(W) - roughness_corr(W)) / (roughness_no(W) + roughness_corr(W) + eps)` | Corrected demand should be smoother |
| F3 | `slope_continuity_improvement` | `(slope_jump_no(W) - slope_jump_corr(W)) / (slope_jump_no(W) + slope_jump_corr(W) + eps)` | Reduces jumpy false windows |
| F4 | `duration_plausibility` | `clip01(duration_h / 1.5)` | Captures E11/E12 duration learning without a hard duration cutoff |
| F5 | `n_height_ratio` | `clip01((max(y inside W) - max(y at left anchor, y at right anchor)) / day_net_scale)` | Tests observed FP pattern where false windows have weak net-load bounce |
| F6 | `solar_strength_ratio` | `clip01(p95(S inside W) / max(site_solar_scale, eps))` | Reduces weak-solar false positives |
| F7 | `solar_peak_alignment` | `clip01(1 - abs(midpoint(W) - solar_peak_time) / 3.5h)` | Keeps event solar-centred |
| F8 | `site_centered_core_score` | `robust_bound(core_score(W) - median_site_daily_core_score)` | Incorporates m9.2 bridge site-median learning |
| F9 | `site_rank_core_score` | `2 * within_site_percentile_rank(core_score_day) - 1` | Incorporates m9.2 bridge rank learning |

Where:

```text
core_score(W) = F1 + F2 + F3
core_score_day = max over candidate windows of core_score(W)
median_site_daily_core_score = median core_score_day for that site using unlabelled site-days only
```

Bridge error is computed against a straight-line bridge between the pre-window and post-window anchor values of `U_no`. Roughness is mean absolute first difference over the candidate window plus shoulders. Slope continuity compares median slopes in 1-hour shoulders against median slopes in the first and last 1-hour inside the candidate.

All feature values should be finite. Missing or undefined feature values should be set to `0` and counted in the output manifest.

Site adaptation must use unlabelled site statistics only. No site-specific labelled thresholds are allowed.

## Score Definitions

For manual weighted-score variants:

```text
score = sum(w_i * F_i for i in 1..9)
```

For logistic regression:

```text
score = P(RPF day | F1..F9)
```

The candidate window with the highest score becomes the selected window for that site-day. The day-level flag is positive if the selected score is above the selected threshold.

## Model Variants

All regimes use the same model variants.

| Variant | Weights / learner | Purpose |
|---|---|---|
| M0 `all_equal` | all 9 weights = 1 | Main simple physical score |
| M1 `drop_bridge` | F1 = 0, all others = 1 | Ablate bridge |
| M2 `drop_roughness` | F2 = 0, all others = 1 | Ablate roughness |
| M3 `drop_slope` | F3 = 0, all others = 1 | Ablate slope continuity |
| M4 `drop_duration` | F4 = 0, all others = 1 | Ablate duration plausibility |
| M5 `drop_n_height` | F5 = 0, all others = 1 | Ablate N-height |
| M6 `drop_solar_strength` | F6 = 0, all others = 1 | Ablate solar-strength guard |
| M7 `drop_peak_alignment` | F7 = 0, all others = 1 | Ablate solar alignment |
| M8 `drop_site_centered` | F8 = 0, all others = 1 | Ablate site median normalisation |
| M9 `drop_site_rank` | F9 = 0, all others = 1 | Ablate site rank normalisation |
| M10 `logistic_all9` | L2 logistic regression on F1-F9 | Learn interpretable weights |

No XGBoost is included in this round.

## Experiment Permutations

### Regime-Level Run Count

| Regime | Manual variants | Logistic variants | Beta LOSO folds | Total evaluated fold-runs |
|---|---:|---:|---:|---:|
| R1 `beta_loso` | 10 | 1 | 8 | 88 |
| R2 `beta_loso_plus_alpha` | 10 | 1 | 8 | 88 |
| R3 `alpha_only_to_beta` | 10 | 1 | 1 transfer run | 11 |

Total planned evaluated runs: 187.

### Full Variant Matrix

| Regime | Fold definition | Variants |
|---|---|---|
| R1 `beta_loso` | hold out each Beta site once | M0-M10 |
| R2 `beta_loso_plus_alpha` | hold out each Beta site once | M0-M10 |
| R3 `alpha_only_to_beta` | train/tune on Alpha, evaluate all Beta | M0-M10 |

### Main Comparison Groups

| Comparison | Question |
|---|---|
| R1 vs R2 | Does adding Alpha help real-data site generalisation? |
| R2 vs R3 | Does real Beta tuning materially improve over pure Alpha transfer? |
| M0 vs M1-M9 | Which physical components actually matter? |
| M0 vs M10 | Does learned logistic weighting improve over equal weights? |
| All predictions vs confidence-filtered predictions | Can the model support review triage? |

## Fast-Iteration Implementation Strategy

The implementation must prioritise fast turnaround. Do not run the full experiment matrix as the first step. Build and execute in chunks, showing results after each chunk before moving to the next.

| Chunk | Run scope | Purpose | Expected output before continuing |
|---|---|---|---|
| C0 `smoke_feature_cache` | 1 Alpha site + 1 Beta site, M0 only | Validate feature formulas, selected windows, labels, confidence columns | Candidate/feature summary and a tiny metrics table |
| C1 `cached_daily_features` | Full Alpha + full Beta feature cache, no modelling | Pay the expensive candidate-scan cost once | Feature cache timing, candidate counts, missing-feature counts |
| C2 `manual_r1_quick` | R1 Beta LOSO, M0 only | First real LOSO result with equal weights | Beta sure/all pooled and per-site P/R/F1 |
| C3 `manual_ablation_r1` | R1 Beta LOSO, M0-M9 | Learn which components matter before adding Alpha/logistic | Ablation ranking table |
| C4 `regime_comparison_manual` | R1/R2/R3, M0 plus best 2 ablations | Check whether Alpha helps without running everything | Regime comparison table |
| C5 `logistic_check` | R1 and R2, M10 only | Test whether learned weights improve over manual scores | Logistic coefficients and metrics |
| C6 `full_matrix_if_needed` | R1/R2/R3, M0-M10 | Final complete model-development matrix only if earlier chunks justify it | Complete CSV outputs |

Fast-iteration requirements:

- Cache candidate-window features once and reuse them for all variants.
- Avoid recomputing candidate scans when only thresholds, ablations, or logistic weights change.
- Print or save a concise result summary after every chunk.
- Prefer CSV outputs first; figures and HTML examples are out of scope for this experiment.
- Each chunk should be independently runnable from notebook/script controls.
- The default run mode should be `smoke` or `quick`, not the full matrix.
- If C2 or C3 shows a clearly poor result, pause before running C4-C6.
- Long-running full matrix execution should be opt-in with an explicit variable such as `RUN_FULL_MATRIX = True`.

## Threshold Selection

| Variant type | Threshold rule |
|---|---|
| Manual weighted score | Select one threshold on the training/tuning split that maximises macro-site day F1 |
| Logistic regression | Fit coefficients on training/tuning split, then select probability threshold that maximises macro-site day F1 |
| Mixed Alpha + Beta tuning | Use site-balanced weights so Alpha cannot dominate the 7 Beta training sites |

Macro-site F1 is used for threshold selection to avoid high-RPF sites dominating the tuning objective.

## Model Selection Rule

After all regimes and variants finish, choose the best model using this deterministic order:

1. Keep only candidate models evaluated under R1 or R2 Beta LOSO. R3 is a transfer baseline, not the primary model-selection regime.
2. Rank by pooled held-out Beta LOSO `beta_sure_only` day-level F1.
3. Break ties within 0.01 F1 using higher site-macro `beta_sure_only` F1.
4. Break remaining ties using higher `beta_all` pooled F1.
5. Break remaining ties using better 70% auto-coverage `beta_sure_only` F1.
6. Break remaining ties by simplicity: manual equal/ablation score before logistic regression; fewer active components before more components.

The selected model must be reported with:

- selected regime;
- selected variant;
- active components or logistic coefficients;
- per-fold thresholds;
- pooled and per-site held-out Beta metrics;
- Beta all-days and Beta sure-only results;
- confidence-coverage table.

The selection result is still a model-development result because Beta labels are used in the LOSO development protocol. A later publication-ready run should freeze the selected method and rerun it cleanly.

## Leakage Prevention

The implementation must enforce fold boundaries explicitly.

| Item | Allowed | Not allowed |
|---|---|---|
| Held-out Beta labels | evaluation only | fitting logistic regression, threshold tuning, model selection inside the fold |
| Held-out Beta confidence | evaluation subset filtering only after prediction | threshold tuning or coefficient fitting |
| Held-out Beta unlabelled measurements | feature computation and site-scale normalisation | label-informed calibration |
| Other Beta sites | fitting/tuning in R1/R2 | leaking held-out site labels |
| Alpha labels | fitting/tuning in R2/R3 | changing held-out Beta labels |
| Site statistics | unlabelled medians, p95, ranks | site-specific labelled thresholds |

Specific safeguards:

- Split Beta by `substation_id` before fitting thresholds or logistic models.
- For each held-out Beta site, build the training/tuning label frame from other Beta sites only, plus Alpha only for R2.
- Select thresholds separately inside each fold using only that fold's training/tuning rows.
- Fit logistic regression separately inside each fold using only that fold's training/tuning rows.
- Compute `site_solar_scale`, `site_net_scale`, score medians, and score ranks from unlabelled measurements/scores only.
- Do not use final held-out fold metrics to revise thresholds, feature formulas, logistic settings, or active-component choices during the same run.
- Store every fold's train/test site lists and threshold source in `01_experiment_manifest.csv` and `03_threshold_selection.csv`.

## Confidence Reporting

Confidence is used for review triage only. It must not change the main prediction threshold in this first experiment.

| Model type | Confidence score |
|---|---|
| Manual score | absolute distance from selected score to selected threshold |
| Logistic | absolute distance from predicted probability to selected probability threshold |

Report performance at fixed auto-coverage levels:

| Auto-coverage | Interpretation |
|---:|---|
| 50% | Only the most confident half of site-days are auto-decided |
| 60% | Moderate automation |
| 70% | Target practical triage operating point |
| 80% | Aggressive automation |
| 90% | Very aggressive automation |
| 100% | Standard all-prediction result |

For each coverage level, report precision, recall, F1, support, positive support, TP, FP, FN, and TN.

## Required Outputs

The future implementation should write CSV outputs only under a new ignored misc output folder.

Recommended folder:

```text
publication/2_journal_article/notebooks/99_Misc/outputs/260707_physical_score_loso_experiments/
```

| Output | Contents |
|---|---|
| `01_experiment_manifest.csv` | run settings, regimes, variants, feature list |
| `02_feature_component_summary.csv` | component distributions by dataset/site/truth group |
| `03_threshold_selection.csv` | selected threshold per regime/fold/variant |
| `04_day_level_metrics.csv` | pooled, macro-site, and per-site P/R/F1 |
| `05_confidence_coverage_metrics.csv` | P/R/F1 at 50/60/70/80/90/100% auto-coverage |
| `06_selected_windows_audit.csv` | site/date, selected window, score, threshold, truth, prediction |
| `07_logistic_coefficients.csv` | coefficients, intercept, signs, fold/regime metadata |
| `08_ablation_rankings.csv` | effect of dropping each component on Beta LOSO sure-only F1 |

## Main Metrics Table Shape

`04_day_level_metrics.csv` should include at least:

```text
regime, fold_id, heldout_site, variant, dataset, subset,
summary_scope, substation_id,
support, positive_support, tp, fp, fn, tn,
precision, recall, f1,
threshold, threshold_source, notes
```

Where:

- `summary_scope` is `pooled`, `macro_site_average`, or `site`;
- `subset` is `beta_sure_only`, `beta_all`, or `alpha_all`;
- `substation_id` is blank for pooled and macro-site rows.

## Success Criteria

| Criterion | Target |
|---|---|
| Primary metric | highest Beta LOSO sure-only day F1 |
| Interpretability | <=9 physical/site-normalised components |
| Site robustness | no catastrophic hidden site failure without being flagged |
| Alpha usefulness | R2 should show whether Alpha improves or hurts R1 |
| Triage usefulness | confidence curves should show whether high-confidence subsets exceed all-prediction F1 |

The final decision should prioritise:

1. Beta LOSO sure-only F1.
2. Beta LOSO all-days F1.
3. Site-macro F1 and worst-site behaviour.
4. Confidence-coverage performance at 70%.
5. Simplicity and interpretability.

## Explicit Non-Goals

- No XGBoost in this experiment round.
- No interval-level optimisation.
- No site-specific labelled thresholds.
- No month/time split as the primary Beta validation.
- No final publication claim until the selected model is rerun cleanly and documented.

## Assumptions

- Beta LOSO station split is the primary practical validation because Beta has only one labelled year.
- Site normalisation may use unlabelled site statistics, but not site-specific labels.
- Logistic regression is acceptable because it is fast and interpretable.
- Alpha can be used either as auxiliary labelled data or as a pure transfer training source.
- Confidence is a reporting and triage layer in v1, not a separate optimised model.

# 2026-06-25 Counterfactual m9_rpf Implementation Plan

## Summary

Create a new exploratory misc notebook implementing the main CGPT Pro recommendation: a **physics-guided counterfactual structured window ranker** for RPF detection.

The notebook will remain fully inside `publication/2_journal_article/notebooks/99_Misc/` and write outputs only under the ignored misc output folder. It will not modify the main journal notebooks, config, package code, or current m7/m8/m9 helpers.

The implementation goal is to test whether dense bounded windows + null candidate + counterfactual reconstruction features + site-day ranking can improve Beta performance compared with current `m8_xgb`, `m9_hybrid`, and plateau-aware candidate-generator diagnostics.

## Proposed Notebook

Create:

```text
publication/2_journal_article/notebooks/99_Misc/07_m9_counterfactual_window_ranker.ipynb
```

Write outputs to:

```text
publication/2_journal_article/notebooks/99_Misc/outputs/07_m9_counterfactual_window_ranker/
```

Use notebook-local code only.

Default controls:

```python
RUN_MODE = "smoke"  # "smoke", "focus", "full"
TRAIN_MODE = "alpha_loso"  # only used when RUN_MODE = "full"
FOCUS_SITES = ["beta_B", "beta_D", "beta_E"]
FOCUS_DATES = ["2023-10-13"]  # plus auto-selected current failure examples
WINDOW_START_HOUR = 6
WINDOW_END_HOUR = 18
MIN_DURATION_MINUTES = 30
MAX_DURATION_MINUTES = 8 * 60
REQUIRE_SOLAR_PEAK = True
ALLOW_TOP3_SOLAR_PEAKS = True
WRITE_HTML = True
RANDOM_SEED = 9
```

Mode meanings:

- `smoke`: tiny deterministic subset to validate feature construction, ranking labels, decoding, and outputs.
- `focus`: Beta diagnostic-only mode for visual and oracle analysis; no training claims.
- `full`: Alpha-only training/validation plus Beta transfer evaluation.

## Data Flow

1. Load final Alpha and Beta parquet datasets.
2. Build cached site-day objects containing:
   - timestamps;
   - raw net load `y`;
   - solar `S`;
   - pseudo-load `S - y`;
   - forward reconstruction `S + y`;
   - daytime masks;
   - missingness masks;
   - true labelled window for evaluation/training labels.
3. Generate dense bounded candidate windows:
   - one null candidate per site-day;
   - all daytime windows from 0.5 to 8 hours;
   - default require window to contain daily solar peak;
   - if `ALLOW_TOP3_SOLAR_PEAKS`, also allow windows containing one of the top-three solar local peaks;
   - include diagnostic variant allowing windows within +/-60 minutes of main solar peak.
4. Compute counterfactual reconstruction features for each candidate.
5. Assign ranking relevance labels on Alpha only.
6. Train ranker or fallback classifier using Alpha groups.
7. Decode each day by comparing best non-null candidate with null candidate.
8. Select ranking margin/threshold using Alpha out-of-fold predictions only.
9. Evaluate Beta transfer.
10. Write metrics, diagnostics, examples, and manifest.

## Candidate Window Generation

For each site-day, define eligible timestamps from 06:00 to 18:00 at 15-minute resolution.

Generate:

- null candidate:
  - `candidate_type = "null"`;
  - no start/end;
  - means no RPF correction.
- non-null candidates:
  - every contiguous window with duration 30 minutes to 8 hours;
  - start/end align to existing timestamps;
  - initially require candidate to contain the daily solar peak timestamp;
  - include windows containing top-three solar local peaks if the solar curve is multi-peaked;
  - store candidate metadata:
    - `start_time`;
    - `end_time`;
    - `duration_minutes`;
    - `contains_daily_solar_peak`;
    - `contains_top3_solar_peak`;
    - `distance_midpoint_to_solar_peak_minutes`;
    - `candidate_id`;
    - `candidate_type`.

Implementation guardrails:

- Do not fabricate output labels at missing timestamps.
- For feature construction only, allow short-gap interpolation on net load/solar if needed.
- Keep raw arrays and interpolated arrays separate.
- Mask missing timestamps in residual/roughness calculations.

Expected candidate volume:

- 06:00-18:00 gives roughly 49 timestamps.
- Duration bounds keep windows to a few hundred per day before solar-peak filtering.
- This is acceptable for exploratory ranking.

## Counterfactual Reconstructions

For raw net load `y_t` and solar `S_t`:

```text
Forward/no-correction underlying load:
U_empty(t) = alpha_site * S_t + y_t

Candidate correction:
U_W(t) = alpha_site * S_t - y_t, if t in W
       = alpha_site * S_t + y_t, otherwise
```

For v1, set:

```text
alpha_site = 1.0
```

Add an optional later ablation:

- estimate `alpha_site` from low-solar/forward-flow days;
- clip to a conservative range such as `[0.7, 1.3]`;
- report with and without solar scale.

## Feature Set

Keep features lean but physically meaningful.

### Window Metadata

- start hour;
- end hour;
- midpoint hour;
- duration hours;
- month;
- weekday;
- weekend flag;
- contains daily solar peak;
- contains top-three solar peak;
- distance from candidate midpoint to daily solar peak.

### Reconstruction Improvement Features

Compute for `U_empty` and `U_W`, then include both absolute values and differences:

```text
delta_feature = feature(U_empty) - feature(U_W)
```

Include:

- full-day roughness;
- inside-window roughness;
- full-day curvature;
- inside-window curvature;
- robust residual to historical profile;
- robust residual to same-day bridge profile;
- fraction of negative reconstructed demand;
- p05/p50/p95 reconstructed demand;
- reconstructed demand range.

Positive delta means the candidate correction improves plausibility.

### Boundary Continuity Features

At start and end boundary:

- jump size in `U_W`;
- jump size in `U_empty`;
- delta boundary jump;
- raw net load at boundary;
- solar at boundary;
- pseudo-load at boundary;
- absolute slope before/after boundary.

### Shape And Co-Movement Features

Inside candidate window:

- solar-net-load correlation;
- derivative same-sign fraction;
- mean derivative product;
- solar bell-shape score;
- net-load N-shape score;
- pseudo-load standard deviation;
- pseudo-load roughness;
- pseudo-load range;
- pseudo-load roughness relative to daytime pseudo-load roughness.

### Missingness Features

- missing net-load count inside window;
- missing solar count inside window;
- missing count near boundaries;
- maximum missing run inside window;
- fraction of candidate window observed.

## Historical / Baseline Profile

Implement two simple profile baselines.

### Baseline A: Robust Site-Time Profile

For each site, estimate typical underlying load by:

- using days with low solar or no obvious RPF labels in Alpha training folds;
- grouping by `(month, weekday/weekend, time_of_day)`;
- using robust median and IQR.

For Beta transfer:

- build site profile from the same Beta site without using labels, using clearly forward-flow / low-solar periods only;
- do not use Beta labels.

Feature:

```text
profile_residual = median_abs((U - profile_median) / profile_iqr)
delta_profile_residual = residual(U_empty) - residual(U_W)
```

### Baseline B: Same-Day Bridge

For each candidate window:

- use pre-window and post-window observed reconstructed load anchors;
- fit a linear or monotonic bridge across the candidate window;
- compute residual of `U_W` and `U_empty` against the bridge.

This helps when site historical profile is weak or seasonal.

## Ranking Labels

For Alpha training:

Compute IoU between each non-null candidate and the true labelled window.

Assign relevance:

```text
4: IoU >= 0.85
3: 0.70 <= IoU < 0.85
2: 0.50 <= IoU < 0.70
1: 0.25 <= IoU < 0.50
0: IoU < 0.25
```

For non-RPF days:

- null candidate relevance = 4;
- all non-null candidates relevance = 0.

For RPF days:

- null candidate relevance = 0;
- non-null candidates get IoU-based relevance.

Store diagnostic columns:

- `iou_with_true`;
- `start_error_minutes`;
- `end_error_minutes`;
- `relevance`;
- `is_null_candidate`;
- `label_day`.

## Model

Primary model:

- `xgboost.XGBRanker`
- grouped by site-day;
- objective: ranking objective such as `rank:pairwise` or `rank:ndcg`;
- train on Alpha only.

Fallback if XGBRanker integration is awkward:

- `XGBClassifier` with relevance-derived sample weights;
- still grouped for threshold/margin selection and decoding.

Sample weighting fallback:

```text
relevance 4 -> weight 8
relevance 3 -> weight 4
relevance 2 -> weight 2
relevance 1 -> weight 1
relevance 0 -> weight 1
```

The null candidate must be included in every training and inference group.

## Decoder

For each site-day:

1. Score all candidates including null.
2. Identify:
   - best null score;
   - best non-null score.
3. Predict RPF if:

```text
best_non_null_score - null_score >= margin
```

4. Otherwise predict no RPF.
5. If positive, output exactly the selected non-null candidate window.

Select `margin` using Alpha out-of-fold predictions:

```text
maximise day F1 subject to precision >= P_min
```

Default:

```text
P_min = 0.70
```

Also report unconstrained max-F1 margin for diagnostics.

## Validation Protocol

### Alpha Validation

Use leave-one-substation-out Alpha validation.

For each Alpha site:

- train ranker on other Alpha sites;
- score held-out site;
- decode with candidate scores;
- save out-of-fold predictions.

Use pooled Alpha out-of-fold predictions to choose the final margin.

### Beta Transfer

Train final model on all Alpha.

Apply to Beta without using Beta labels for model fitting or threshold selection.

Evaluate:

- day precision/recall/F1;
- interval precision/recall/F1;
- macro-average site F1;
- event IoU;
- boundary start MAE;
- boundary end MAE;
- false-positive duration distribution;
- fragmentation check, which should always be one or zero windows by construction.

## Outputs

Write local outputs only:

```text
outputs/07_m9_counterfactual_window_ranker/
```

CSV outputs:

- `01_candidate_window_summary.csv`
- `02_alpha_oof_candidate_scores.csv`
- `03_alpha_oof_decoded_days.csv`
- `04_alpha_margin_sweep.csv`
- `05_beta_candidate_scores.csv`
- `06_beta_decoded_days.csv`
- `07_metric_summary.csv`
- `08_site_metric_summary.csv`
- `09_error_examples.csv`
- `10_feature_importance.csv`

Figures:

- candidate-count distribution;
- Alpha margin sweep;
- Beta site day-F1 bar chart;
- confusion matrices for day and interval levels;
- event IoU distribution;
- boundary error distribution;
- counterfactual plausibility feature distributions for TP/FP/FN/TN.

HTML examples:

- TP/FP/FN/TN examples by site;
- focused examples for `beta_B`, `beta_D`, `beta_E`;
- plots show:
  - raw net load;
  - solar;
  - `U_empty = S + y`;
  - selected `U_W`;
  - manual window;
  - selected candidate window;
  - null-vs-window score margin.

Manifest:

- run mode;
- candidate settings;
- model settings;
- selected margin;
- whether Beta labels were used for training or threshold selection;
- elapsed runtime;
- warning that misc results are exploratory.

## Implementation Phases

### Phase 1: Smoke Counterfactual Candidate Lab

Implement:

- dense bounded window generator;
- null candidate;
- counterfactual reconstruction;
- core features;
- IoU relevance labels;
- no model yet.

Acceptance:

- `beta_B 2023-10-13` has many valid dense windows;
- at least one window has high IoU with the manual label;
- null candidate exists for every site-day;
- feature table is finite and manageable.

### Phase 2: Ranking Model Smoke Run

Implement:

- small Alpha subset train;
- XGBRanker or weighted XGBClassifier fallback;
- group-wise decoding;
- margin sweep on Alpha smoke predictions.

Acceptance:

- model trains;
- decoder emits exactly one or zero windows per day;
- output files are written;
- no Beta labels used in training.

### Phase 3: Full Alpha LOSO + Beta Transfer

Implement:

- Alpha LOSO out-of-fold scoring;
- Alpha margin selection;
- final all-Alpha model;
- Beta transfer scoring;
- full metrics and figures.

Acceptance:

- run completes;
- results compared against Notebook 2 `m8_xgb`, current `m9_hybrid`, and plateau candidate oracle;
- outputs clearly marked exploratory.

### Phase 4: Ablations

Run controlled variants:

- dense windows with daily solar peak only;
- dense windows with top-three solar peaks;
- with/without historical profile features;
- with/without same-day bridge features;
- with/without `alpha_site` solar scaling;
- XGBRanker versus weighted classifier fallback.

## Test And Acceptance Criteria

Core checks:

- every site-day has exactly one null candidate;
- non-null candidate durations are between 0.5 and 8 hours;
- no candidate starts before 06:00 or ends after 18:00;
- candidate feature rows contain no unhandled infinities;
- ranking groups match site-day boundaries;
- Beta labels are not used for training or margin selection;
- decoder emits no more than one non-null window per day.

Metric checks:

- report pooled and macro site-level day F1;
- report interval F1;
- report event IoU and boundary errors;
- include current baselines for comparison.

Performance checks:

- smoke mode should run in under a few minutes;
- full mode can be slower but should write intermediate candidate files so later debugging does not repeat all feature work unnecessarily.

## Open Risks

- Dense windows may create many candidates; candidate filtering and caching are essential.
- Historical profiles may be noisy or biased if built from unreviewed Beta data.
- `alpha_site` solar scaling may improve transfer but add another calibration degree of freedom.
- XGBRanker may require careful group handling; fallback weighted classifier should be available.
- Current Beta labels are still provisional, so final conclusions must wait for reviewed oracle data.

## Default Decisions

- Use dense bounded windows with null candidate as the main design.
- Use one-window decoder.
- Use Alpha-only margin selection.
- Use `P_min = 0.70` as the default precision guardrail, while also reporting unconstrained F1.
- Use misc-only implementation until the method clearly outperforms current baselines.


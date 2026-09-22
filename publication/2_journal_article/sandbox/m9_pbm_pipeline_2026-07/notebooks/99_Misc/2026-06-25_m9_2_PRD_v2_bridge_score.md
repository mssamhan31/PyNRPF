# 2026-06-25 m9.2 PRD v2: Bridge-Score Counterfactual Model

## Summary

Implement a new exploratory `m9.2` variant based on the CGPT Pro 0.778 Beta day-F1 experiment.

The key change from the current local `m9.2_physics` implementation is to move away from an XGBRanker-first candidate-window scorer and toward a compact, transferable, label-free physical day score:

```text
bridge_ratio_p99
  + label-free site adjustment
  + Alpha-learned seasonal prior
  + conservative Alpha threshold
  + deterministic one-window decoder
```

This should be implemented only in the misc folder first. Do not modify Notebook 2, production helpers, package code, or the main journal config.

## Proposed Model Name

Use:

```text
m9.2_bridge_score
```

This distinguishes it from the current `m9.2_physics` XGBRanker prototype.

## Motivation

The current local `m9.2_physics` full run achieved:

| Dataset | Level | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Alpha LOSO | day | 0.957 | 0.935 | 0.946 |
| Beta transfer | day | 0.454 | 0.659 | 0.537 |
| Alpha LOSO | interval | 0.808 | 0.909 | 0.856 |
| Beta transfer | interval | 0.461 | 0.566 | 0.508 |

This means it fits/transfers across synthetic Alpha sites well, but it does not transfer strongly to Beta.

CGPT Pro's bridge-score prototype reported:

| Dataset | Level | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Beta | day | 0.772 | 0.785 | 0.778 |
| Beta | interval | 0.766 | 0.798 | 0.782 |

The main learning is that a compact physical bridge score can transfer better than a broad XGB ranker trained on many Alpha candidate features.

## Scope

Create a new misc notebook:

```text
publication/2_journal_article/notebooks/99_Misc/08_m9_2_bridge_score_development.ipynb
```

Write outputs only under:

```text
publication/2_journal_article/notebooks/99_Misc/outputs/08_m9_2_bridge_score_development/
```

Optional helper scripts may be created in the same misc folder if needed for speed and reproducibility.

## Non-Goals

Do not:

- modify the main journal notebooks;
- modify `_experiment_helpers.py`;
- modify `experiment_config.yaml`;
- modify production `src/` code;
- promote the model into Notebook 2 yet;
- run complete 10-site Alpha LOSO in this version;
- add Viterbi smoothing in v2;
- add broad supervised models such as LightGBM/logistic regression in v2;
- use Beta labels to fit the deployable score or threshold.

## Validation Strategy For This Version

To keep iteration fast, do not run complete Alpha LOSO.

Use:

1. **Beta full evaluation** for exploratory diagnosis.
2. **One Alpha held-out-site evaluation** for sanity checking Alpha-side behaviour.

Default Alpha held-out site should be configurable:

```python
ALPHA_HELDOUT_SITE = "alpha_F"
```

Rationale:

- `alpha_F` has high RPF support and is useful for detecting obvious failures.
- Full 10-site LOSO took too long for fast research iteration.
- We can run complete Alpha LOSO later only if the v2 method looks promising.

Important caveat:

Beta evaluation in this misc notebook is exploratory. The method family has already been inspired by Beta performance, so this is not a clean external validation estimate.

## Data Inputs

Use the final journal datasets:

```text
publication/2_journal_article/dataset/final/dataset_alpha.parquet
publication/2_journal_article/dataset/final/dataset_beta.parquet
```

Required columns:

```text
substation_id
date
timestamp
net_load_MW
solar_MW
label_interval
label_day
```

Treat timestamps as dataset wall-clock time, consistent with the existing journal workflow.

## Daily Matrix Representation

Convert each site-day into a fixed 96-row daily representation:

```text
net_load[0:96]
solar[0:96]
label_interval[0:96]
```

For feature calculation:

- use 15-minute slots;
- maintain original raw values;
- interpolate short missing gaps for feature arrays where reasonable;
- keep missingness indicators;
- do not change raw labels;
- do not write corrected measurements back to the dataset.

Recommended missing-data handling:

1. If a day has a few missing points, interpolate for feature calculation.
2. If a day has substantial missingness, still compute features but record missingness flags.
3. Do not automatically reject a day because of missingness.

This differs from earlier candidate-generation logic where missing values could effectively veto candidate generation.

## Candidate Window Scan

For each site-day, generate dense daytime candidate windows.

Default bounds:

```python
WINDOW_START_SLOT = 24  # 06:00
WINDOW_END_SLOT = 72    # 18:00
MIN_DURATION_SLOTS = 2  # 0.5 h
MAX_DURATION_SLOTS = 32 # 8 h
```

For a candidate window `[L, R]`:

- `L` and `R` are inclusive 15-minute slot indices;
- duration is `R - L + 1`;
- duration must be between 2 and 32 slots;
- midpoint should be reasonably close to the daily solar peak.

The CGPT Pro prototype used:

```python
abs((L + R) / 2 - solar_peak_slot) <= 14
```

This allows candidate midpoints within about 3.5 hours of the solar peak.

Recommended v2 default:

```python
MIDPOINT_PEAK_TOLERANCE_SLOTS = 14
```

This is broader than our current "must contain peak" rule. It may help recover events whose true window begins/ends asymmetrically around the solar peak.

## Core Counterfactual Definitions

For raw net load `y` and solar `S`:

```text
No-correction underlying load:
U_plus = S + y

Reverse-flow interpretation:
U_minus = S - y
```

For each candidate window `[L, R]`, compare whether `U_minus[L:R]` forms a better bridge between the uncorrected surrounding periods than `U_plus[L:R]`.

## Bridge MSE

For a candidate `[L, R]`, define the bridge line using the uncorrected reconstruction just outside the candidate:

```text
left boundary point  = U_plus[L - 1] if available, otherwise U_plus[L]
right boundary point = U_plus[R + 1] if available, otherwise U_plus[R]
```

Fit a straight line between those two points.

Then compute:

```text
bridge_mse_plus  = MSE(U_plus[L:R], bridge_line)
bridge_mse_minus = MSE(U_minus[L:R], bridge_line)
```

The improvement is:

```text
bridge_improve = bridge_mse_plus - bridge_mse_minus
```

The ratio score is:

```text
bridge_ratio = bridge_improve / (bridge_mse_plus + bridge_mse_minus + eps)
```

Interpretation:

- high positive `bridge_ratio`: reverse-flow interpretation makes the candidate segment bridge better;
- near zero: no clear gain;
- negative: correction makes the bridge worse.

## Daily Score

For each day, compute `bridge_ratio` for every candidate window.

Then compute:

```text
bridge_ratio_p99 = 99th percentile of bridge_ratio over candidate windows
```

This should be the primary raw day score.

Why p99:

- more robust than a single maximum;
- captures whether the day has a cluster of plausible RPF windows;
- avoids overreacting to one cherry-picked candidate;
- worked best in the CGPT Pro artifacts.

Also compute these diagnostics, but do not necessarily include them in the default score:

```text
bridge_ratio_max
bridge_ratio_p95
bridge_improve_p99
full_tv_ratio_p99
combined_p99
candidate_count
missing_net_count
missing_solar_count
```

## Label-Free Site Adjustment

The raw score distribution differs by site. Apply label-free site adjustment.

Let:

```text
raw_day_score = bridge_ratio_p99
site_median = median(raw_day_score for all days at same site)
site_rank = within-site percentile rank(raw_day_score) - 0.5
```

Default v2 score candidate:

```text
z0 = raw_day_score - 0.425 * site_median
z1 = z0 + 0.075 * site_rank
```

Conservative strict-calibration candidate:

```text
z0 = raw_day_score - 0.425 * site_median
z1 = z0
```

Use both as named variants:

```text
v2_dev_best
v2_alpha_strict
```

## Seasonal Prior

Use an Alpha-learned day-of-year seasonal prior.

Procedure:

1. Use Alpha labels only.
2. Estimate labelled RPF probability by day-of-year.
3. Smooth with a 31-day window.
4. Convert to centered log-odds.

Then add:

```text
z2 = z1 + seasonal_weight * alpha_season_prior[day_of_year]
```

Default variants:

```text
v2_dev_best: seasonal_weight = 0.050
v2_alpha_strict: seasonal_weight = 0.025
```

Do not estimate the seasonal prior from Beta labels.

## Rolling Score Component

Use a small centered rolling mean within each site:

```text
rolling_z = centered_rolling_mean(z2, window_days)
score = (z2 + rolling_weight * rolling_z) / (1 + rolling_weight)
```

Default variants:

```text
v2_dev_best:
  rolling_window = 5
  rolling_weight = 0.1

v2_alpha_strict:
  rolling_window = 7
  rolling_weight = 0.25
```

This is label-free and only uses nearby score values from the same site.

Deployment caveat:

- centered rolling uses future dates in a retrospective annual dataset;
- this is acceptable for offline oracle/data-cleaning experiments;
- for online deployment, use trailing rolling windows instead.

## Threshold Calibration

Select threshold from Alpha only.

Do not tune threshold using Beta labels.

Recommended threshold selection:

```text
choose threshold that maximises Alpha F1 subject to Alpha precision >= P_MIN
```

Default:

```python
P_MIN = 0.9995
```

Calibration modes:

1. `alpha_pre_oct_strict`
   - Fit seasonal prior and threshold using Alpha dates before `2023-10-01`.
   - Evaluate Alpha dates from `2023-10-01` onward as an Alpha time-split sanity check.

2. `one_site_loso`
   - Exclude `ALPHA_HELDOUT_SITE` from threshold/seasonal calibration.
   - Evaluate on `ALPHA_HELDOUT_SITE`.

For fast iteration, run both modes only for one held-out Alpha site, not all 10 sites.

## Day Classification

For each site-day:

```text
pred_day = score >= threshold
```

Save:

```text
raw_bridge_ratio_p99
site_median
site_rank
season_prior
rolling_score
final_score
threshold
pred_day
true_label_day
confusion_group
```

## Interval Decoder

When `pred_day = False`:

```text
predict no RPF interval
```

When `pred_day = True`:

select exactly one candidate window.

Default decoder:

```text
choose candidate with greatest full-day total variation improvement
```

For diagnostics, also compute candidate choices under:

```text
bridge_ratio
bridge_improve
full_tv_ratio
full_tv_improve
combined_score
```

But default v2 should use the decoder that matches the CGPT Pro note:

```text
full-day total variation improvement
```

## Metrics

Compute:

Beta:

- pooled day precision/recall/F1;
- pooled interval precision/recall/F1;
- site-level day precision/recall/F1;
- site-level interval precision/recall/F1;
- macro-average site F1;
- day confusion counts;
- interval confusion counts.

Alpha one-site sanity:

- held-out site day precision/recall/F1;
- held-out site interval precision/recall/F1;
- threshold calibration table;
- optional time-split Alpha metrics.

Do not run complete Alpha LOSO in v2 by default.

## Output Files

Write under:

```text
publication/2_journal_article/notebooks/99_Misc/outputs/08_m9_2_bridge_score_development/
```

Suggested outputs:

```text
csv/01_daily_scores_alpha.csv
csv/02_daily_scores_beta.csv
csv/03_threshold_calibration.csv
csv/04_alpha_one_site_metrics.csv
csv/05_beta_overall_metrics.csv
csv/06_beta_site_metrics.csv
csv/07_beta_error_examples.csv
csv/08_candidate_recall_summary.csv
csv/09_interval_decoder_comparison.csv
csv/10_bootstrap_summary.csv

figures/fig01_beta_score_distribution.png
figures/fig02_beta_site_day_f1.png
figures/fig03_beta_confusion_by_month.png
figures/fig04_candidate_recall.png
figures/fig05_threshold_sweep.png

html_examples/
manifests/run_manifest.json
```

## Fast Run Controls

Notebook controls:

```python
RUN_MODE = "smoke"  # smoke, beta_fast, full_beta
MODEL_NAME = "m9.2_bridge_score"
ALPHA_HELDOUT_SITE = "alpha_F"
P_MIN = 0.9995
WRITE_HTML = True
BOOTSTRAP_N = 0
```

Mode meanings:

- `smoke`: tiny subset, validate candidate scan and score construction.
- `beta_fast`: all Beta days, one Alpha held-out site, no bootstrap, limited HTML.
- `full_beta`: all Beta days, one Alpha held-out site, optional bootstrap and diagnostics.

Do not include a `full_alpha_loso` mode as the default path. If it is added later, it should be opt-in and clearly marked slow.

## Bootstrap

Optional only.

If run:

- use 14-day blocks;
- report day and interval F1 intervals;
- default `BOOTSTRAP_N = 1000` for local development;
- allow `BOOTSTRAP_N = 10000` only for final reporting.

State clearly:

```text
Bootstrap intervals capture sampling variability, not model-selection optimism.
```

## Candidate Recall Diagnostics

For every labelled RPF day, compute whether any dense candidate satisfies:

```text
IoU >= 0.50
IoU >= 0.70
both boundaries within +/-30 minutes
```

Report candidate recall overall and by site.

Target from CGPT Pro artifacts:

```text
Beta IoU >= 0.50 candidate recall: about 0.975
Beta IoU >= 0.70 candidate recall: about 0.926
Beta +/-30 min boundary recall: about 0.890
```

This will confirm that our implementation matches the high-recall candidate scan.

## Error Analysis

For Beta false positives and false negatives, save examples sorted by:

False positives:

- highest final score;
- highest raw bridge score;
- highest site-rank score.

False negatives:

- score closest below threshold;
- high candidate IoU but low final score;
- high raw bridge score but suppressed by site adjustment.

Generate limited HTML examples:

```text
2 FP + 2 FN per Beta site
```

Plots should show:

- raw net load;
- solar;
- `U_plus = S + y`;
- selected `U_minus = S - y` inside candidate;
- manual label window;
- predicted window;
- score components.

## Implementation Notes

### Efficient Daily Arrays

Use daily 96-slot arrays and vectorised candidate scans.

The CGPT Pro script precomputed candidate index pairs by solar peak slot:

```python
CACHE[p] = candidate windows around solar peak slot p
```

Use the same idea.

### Prefix Sums

Use prefix sums for:

- segment mean;
- segment variance;
- total variation;
- bridge residuals if possible.

Avoid per-candidate loops where a vectorised array expression is reasonable.

### Smoothing

The prototype used 3-point smoothing for net load and solar before feature calculation.

Recommended v2:

- use 3-point smoothing for bridge-score feature calculation;
- keep raw values for plots and final dataset outputs;
- expose `SMOOTHING_WINDOW = 3` as a control;
- test `SMOOTHING_WINDOW = 1` only as an ablation, not the default.

### Missing Data

The CGPT Pro script uses `nan_to_num` inside the scan. For v2, prefer:

- interpolate short gaps;
- fill remaining missing values conservatively;
- include missingness features.

This is cleaner and easier to explain.

## Planned Notebook Sections

1. Title and method summary.
2. Imports, controls, and output paths.
3. Load final Alpha/Beta datasets.
4. Convert to daily arrays.
5. Generate dense candidate cache.
6. Compute bridge-score candidate summaries.
7. Compute label-free site adjustment.
8. Compute Alpha seasonal prior.
9. Calibrate threshold on Alpha only.
10. Evaluate one Alpha held-out site.
11. Evaluate full Beta.
12. Decode intervals.
13. Candidate recall diagnostics.
14. Error examples and HTML plots.
15. Figures.
16. Manifest.
17. Interpretation notes and caveats.

Each major helper call should be preceded by a markdown explanation of what it does and what to inspect if results look wrong.

## Acceptance Checks

The notebook should pass these checks:

1. Candidate scan writes daily scores for all Beta site-days.
2. Candidate recall roughly matches CGPT Pro's reported high-recall values.
3. `bridge_ratio_p99` exists and is finite for almost all site-days.
4. Beta pooled day F1 is reported.
5. Beta site-level metrics are reported.
6. One Alpha held-out-site metric is reported.
7. Threshold is selected using Alpha only.
8. Beta labels are used only for evaluation/diagnostics.
9. Output files stay inside the misc output folder.
10. No full Alpha LOSO is run by default.

## Success Criteria

Exploratory target:

```text
Beta day F1 >= 0.75
Beta interval F1 >= 0.75
```

Minimum useful target:

```text
Beat current m9.2_physics Beta day F1 = 0.537
Beat previous m9_hybrid Beta day F1 = 0.511
```

Additional reporting target:

```text
Macro-average Beta site F1 should be reported alongside pooled F1.
```

## Risks And Caveats

1. Model-selection optimism.

The formulation is inspired by Beta-guided CGPT Pro exploration. Beta results are therefore development-set results, not clean external validation.

2. Provisional Beta labels.

Current Beta labels are not yet the final manually reviewed oracle.

3. Site adaptation may hide distribution shift.

Label-free site centering is deployable, but it uses the target site's score distribution. This is appropriate for offline annual data cleaning, but online deployment would need trailing estimates.

4. Low-support sites remain unstable.

`beta_C` and `beta_H` have few positives, so site-level F1 can move substantially.

5. Centered rolling component uses future days.

This is fine for retrospective oracle creation, but should be replaced with trailing rolling for real-time deployment.

6. High Alpha precision calibration may reduce Alpha F1.

This is acceptable if the target is high-confidence operational correction, but it should be documented.

## Recommendation

Implement `m9.2_bridge_score` next as a separate misc notebook.

Prioritise reproducing:

1. `bridge_ratio_p99`;
2. site median/rank adjustment;
3. Alpha seasonal prior;
4. conservative Alpha threshold;
5. deterministic one-window decoder.

Do not spend time tuning broad supervised models until this compact score is reproduced locally.

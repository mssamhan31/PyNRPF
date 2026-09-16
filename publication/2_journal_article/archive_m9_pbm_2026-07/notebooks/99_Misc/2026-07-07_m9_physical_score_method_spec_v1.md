# m9 Physical-Score Method Spec v1

Date: 2026-07-07

Scope: current best model-development candidate from the fast physical-score LOSO experiment ladder.

This specification freezes the current leading day/window detector so it can be reviewed, visualised, and later promoted into a cleaner journal workflow if desired.

## 1. Model Name

Working name:

`m9_physical_score_v1`

Current best evaluated configuration:

`R1_beta_loso / M9_drop_site_rank`

Interpretation:

- Tune one global threshold per held-out Beta site using the other seven Beta sites.
- Use only Beta sure-confidence days for threshold selection.
- Use the physical score with the F9 site-rank component removed.
- Predict zero or one RPF window per site-day.

## 2. Input Data

Required columns:

`substation_id,date,timestamp,net_load_MW,solar_MW,label_interval,label_day`

For Dataset Beta development evaluation, `confidence` is also used to create the sure-only subset.

The current final Beta labels are from the manually reviewed oracle final dataset:

- 2,928 site-days
- 634 RPF site-days
- 2,330 `sure` site-days
- 598 `unsure` site-days

## 3. Candidate Windows

For each site-day, enumerate dense candidate windows:

| Rule | Value |
|---|---:|
| Search period | 06:00-18:00 |
| Candidate duration | 0.5 h to 8 h |
| Solar-centred restriction | window midpoint within 3.5 h of daily solar peak |
| Output per day | zero or one selected RPF window |

Each candidate is scored, and the highest-scoring candidate becomes the selected candidate for the day.

## 4. Counterfactual Demand Reconstruction

Let:

```text
y[t]      = observed net load
S[t]      = solar generation
U_no[t]   = S[t] + y[t]
U_corr[t] = S[t] - y[t] inside candidate window W, else S[t] + y[t]
eps       = 1e-9
clip01(x) = min(1, max(0, x))
```

Physical motivation:

If positive net-load readings are wrong-sign RPF intervals, flipping the sign inside the correct window should produce a more plausible underlying demand curve.

## 5. Score Components

The current model uses eight components. The original ninth component, F9 site-rank core score, is removed because the ablation showed better Beta LOSO performance without it.

| ID | Component | Definition summary | Included in current best? |
|---|---|---|---|
| F1 | `bridge_improvement` | improvement in straight-line bridge residual after counterfactual correction | Yes |
| F2 | `roughness_improvement` | reduction in roughness / total variation after correction | Yes |
| F3 | `slope_continuity_improvement` | reduction in boundary slope discontinuity after correction | Yes |
| F4 | `duration_plausibility` | `clip01(duration_h / 1.5)` | Yes |
| F5 | `n_height_ratio` | net-load bounce height divided by robust day scale | Yes |
| F6 | `solar_strength_ratio` | candidate-window solar p95 divided by unlabelled site solar scale | Yes |
| F7 | `solar_peak_alignment` | closeness between window midpoint and daily solar peak | Yes |
| F8 | `site_centered_core_score` | core score minus unlabelled site median core score, robustly bounded | Yes |
| F9 | `site_rank_core_score` | within-site percentile rank of core score | No |

Core score:

```text
core_score = F1 + F2 + F3
```

Current best score:

```text
score = F1 + F2 + F3 + F4 + F5 + F6 + F7 + F8
```

## 6. Decision Rule

For each site-day:

1. Generate candidate windows.
2. Compute F1-F8 for each candidate.
3. Select the candidate with maximum score.
4. Predict RPF day if:

```text
selected_score >= threshold
```

5. If positive, output the selected candidate start/end as the predicted RPF window.
6. If negative, output no RPF window.

## 7. Threshold Selection

Current best development regime:

`R1_beta_loso`

For each held-out Beta site:

1. Remove that site from threshold tuning.
2. Use the remaining seven Beta sites.
3. Use only `confidence == "sure"` days.
4. Choose the threshold that maximises macro-site day-level F1 on the tuning sites.
5. Evaluate on the held-out site.

This is a model-development validation scheme, not a final locked external validation.

## 8. Confidence Score

The current confidence score is:

```text
confidence_score = abs(selected_score - threshold)
```

Interpretation:

- Large distance above threshold: confident RPF prediction.
- Large distance below threshold: confident no-RPF prediction.
- Near threshold: uncertain, should be prioritised for manual review.

This confidence score is used only for triage analysis. It does not change the decision threshold.

## 9. Current Key Results

Best full-coverage day-level Beta sure-only result:

| Metric | Value |
|---|---:|
| Precision | 0.8875 |
| Recall | 0.8838 |
| F1 | 0.8857 |

Best full-coverage interval-level Beta sure-only result:

| Metric | Value |
|---|---:|
| Precision | 0.8733 |
| Recall | 0.8717 |
| F1 | 0.8725 |

Window quality on Beta sure-only true-positive days:

| Metric | Value |
|---|---:|
| Median IoU | 0.9167 |
| IoU >= 0.5 rate | 0.9695 |
| IoU >= 0.7 rate | 0.8944 |
| Median absolute start error | 15 min |
| Median absolute end error | 15 min |

Confidence triage at 80% auto-coverage on Beta sure-only days:

| Metric | Value |
|---|---:|
| Auto-accepted rows | 1,864 |
| Manual rows left | 466 |
| Auto precision | 0.9918 |
| Auto recall | 0.9878 |
| Auto F1 | 0.9898 |
| Auto errors | 5 |

## 10. Main Findings

1. The bridge-improvement component is essential.
2. Removing the site-rank component improves pooled Beta LOSO performance.
3. Logistic regression did not beat the simple physical score.
4. Alpha-only transfer is recall-heavy and loses precision on Beta.
5. Confidence triage is strong and may be more practically useful than fully automated review.
6. Interval/window localisation is strong when the day is correctly detected.

## 11. Known Caveats

- Beta labels were used in the R1/R2 model-development scheme, so final paper validation should clearly separate model development from locked evaluation.
- Site-level F1 is awkward for sites with zero or very few positive sure RPF days.
- The current method is day-first; interval/window quality is evaluated after day selection rather than independently optimised.
- Remaining errors should be visually reviewed before adding new components.

## 12. Recommended Next Step

Do targeted visual error review for:

- high-confidence false positives and false negatives;
- the five auto-accepted errors at 80% confidence coverage;
- low-confidence residual errors;
- weak sites such as Beta A, Beta D, and Beta H.

Only add another rule or feature if the visual review identifies a consistent, physically defensible failure pattern.

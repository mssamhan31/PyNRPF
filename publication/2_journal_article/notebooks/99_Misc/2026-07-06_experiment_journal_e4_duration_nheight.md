# 2026-07-06 Experiment Journal: Bridge Ladder, E4 Diagnostics, Duration, and N-Height

## Executive Summary

Today we focused on fast, interpretable day-level RPF detection experiments using the final reviewer-B Beta labels. The main practical goal was to improve Beta F1 without sacrificing the strong Alpha result too much.

The strongest baseline remains `E4_v03_three_feature`, a three-feature structured counterfactual window method. It performs very well on Alpha but creates too many false positives on Beta:

| Model | Alpha F1 | Beta all F1 | Beta sure-only F1 |
|---|---:|---:|---:|
| E4 v0.3 three-feature | 0.9486 | 0.6498 | 0.7384 |

We explored a full ladder from E1 to E13. The most important lesson is that Beta false positives are often short, low-prominence local windows that look smoother under sign flip, but do not look like real sustained RPF events. Duration guards and N-height/prominence guards reduce Beta false positives substantially, but hard guards damage Alpha recall because Alpha contains many true short labelled RPF events.

Current headline:

- Best Alpha-preserving simple method: still `E4`.
- Best Beta sure-only F1 among today's hard duration variants: `E12_duration_ge_1.5h`, F1 0.8608.
- Best next direction: combine duration and N-height as **soft penalties**, not hard gates.

## Background

The core problem is detecting wrong-sign reverse power flow readings in actual sites. During these intervals, the measured positive net load should probably have been negative. We want a model that is:

- accurate on Beta, especially day-level F1;
- validated using Alpha training/selection where possible;
- interpretable and defensible;
- low-parameter;
- fast to iterate.

Today's experiments were all in the misc workflow and did not train XGBoost or other ML models. The main workflow used cached daily scores and final datasets.

## Ground Truth And Scope

- Alpha truth: `dataset/final/dataset_alpha.parquet`.
- Beta truth: `dataset/final/dataset_beta.parquet`, rebuilt from reviewer-B-final oracle labels.
- Beta sure-only filter: reviewer-B confidence equal to `sure`.
- Main metric: day-level precision, recall, and F1.
- Main comparison sets:
  - Alpha all days.
  - Beta all days.
  - Beta sure-only days.

## Method Ladder

Today's method ladder now contains E1-E13:

| ID | Method | Intent |
|---|---|---|
| E1 | `bridge_only` | One-feature bridge p99 score. |
| E2 | `bridge_plus_roughness` | Existing bridge score plus roughness proxy. |
| E3 | `bridge_plus_site_median` | Tests simple site-median normalisation. |
| E4 | `v03_three_feature` | Candidate-window max of bridge + roughness + slope continuity. |
| E5 | `v03_without_slope` | E4 ablation without slope continuity. |
| E6 | `v03_three_feature_site_median` | E4 with fixed site-median correction. |
| E7 | `v03_three_feature_site_centered` | E4 minus each site’s median E4 score. |
| E8 | `v03_three_feature_site_rank` | E4 converted to within-site percentile rank. |
| E9 | `v03_site_solar_scaled` | E4 after scaling net and solar by site solar scale. |
| E10 | `v03_site_combined_scaled` | E4 after scaling net and solar by combined site scale. |
| E11 | `duration_ge_1h` | E4 prediction allowed only if selected duration >= 1h. |
| E12 | `duration_ge_1.5h` | E4 prediction allowed only if selected duration >= 1.5h. |
| E13 | `soft_duration_penalty` | E4 with fixed penalty for selected windows shorter than 2.5h. |

All thresholds were selected on Alpha pooled day-level F1 unless stated otherwise.

## Key Results

| Experiment | Alpha P | Alpha R | Alpha F1 | Beta all P | Beta all R | Beta all F1 | Beta sure P | Beta sure R | Beta sure F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m7 DTR time split | 0.9120 | 0.9630 | 0.9370 | n/a | n/a | n/a | n/a | n/a | n/a |
| m8 XGB time split | 0.9540 | 0.9580 | 0.9560 | n/a | n/a | n/a | n/a | n/a | n/a |
| m7 DTR spatiotemporal split | 0.5298 | 0.9895 | 0.6901 | 0.2218 | 0.7837 | 0.3457 | 0.2120 | 0.8033 | 0.3355 |
| m8 XGB spatiotemporal split | 0.9740 | 0.8185 | 0.8895 | 0.5933 | 0.3670 | 0.4535 | 0.7427 | 0.3706 | 0.4945 |
| E1 bridge only | 0.9358 | 0.9240 | 0.9299 | 0.4578 | 0.9920 | 0.6265 | 0.5552 | 1.0000 | 0.7140 |
| E2 bridge + roughness | 0.9362 | 0.9340 | 0.9351 | 0.4667 | 0.9872 | 0.6337 | 0.5730 | 1.0000 | 0.7285 |
| E3 bridge + site median | 0.9425 | 0.8531 | 0.8956 | 0.4804 | 0.9808 | 0.6449 | 0.5605 | 0.9876 | 0.7151 |
| E4 v0.3 three-feature | 0.9518 | 0.9454 | 0.9486 | 0.4894 | 0.9663 | 0.6498 | 0.5896 | 0.9876 | 0.7384 |
| E5 without slope | 0.9439 | 0.9690 | 0.9563 | 0.4545 | 0.9760 | 0.6202 | 0.5486 | 0.9938 | 0.7069 |
| E6 E4 + site median | 0.9464 | 0.9074 | 0.9265 | 0.4878 | 0.9631 | 0.6476 | 0.5701 | 0.9855 | 0.7223 |
| E7 site centered | 0.8650 | 0.6854 | 0.7648 | 0.5574 | 0.8173 | 0.6628 | 0.5871 | 0.8447 | 0.6927 |
| E8 site rank | 0.4795 | 0.8834 | 0.6216 | 0.3427 | 0.9535 | 0.5042 | 0.3509 | 0.9627 | 0.5144 |
| E9 solar scaled | 0.9518 | 0.9454 | 0.9486 | 0.4894 | 0.9663 | 0.6498 | 0.5896 | 0.9876 | 0.7384 |
| E10 combined scaled | 0.9518 | 0.9454 | 0.9486 | 0.4894 | 0.9663 | 0.6498 | 0.5896 | 0.9876 | 0.7384 |
| E11 duration >= 1h | 0.9630 | 0.8069 | 0.8781 | 0.5943 | 0.9343 | 0.7265 | 0.7093 | 0.9752 | 0.8213 |
| E12 duration >= 1.5h | 0.9791 | 0.7382 | 0.8418 | 0.6643 | 0.9135 | 0.7692 | 0.7757 | 0.9669 | 0.8608 |
| E13 soft duration penalty | 0.9472 | 0.8744 | 0.9093 | 0.5394 | 0.9647 | 0.6920 | 0.6421 | 0.9917 | 0.7795 |
| m9.2 bridge v2 dev best | 0.9983 | 0.5297 | 0.6921 | 0.7948 | 0.7324 | 0.7623 | 0.8705 | 0.7930 | 0.8299 |

Note: the first m7/m8 baseline rows come from stored final-label re-evaluation prediction artifacts in `outputs/10_final_label_day_model_reevaluation/01_model_day_metrics.csv`. Their Alpha rows cover 3,657 stored Alpha prediction site-days, whereas E1-E13 are scored over the full 10,643 Alpha site-days in the bridge ladder output. The conference time-split rows come from `publication/1_conference_paper/outputs/publication_tables/table2_model_performance_summary.csv`, using the conference-paper day-level test split only; that experiment did not evaluate the current Beta oracle dataset. The `m9.2 bridge v2 dev best` row also comes from `outputs/10_final_label_day_model_reevaluation/01_model_day_metrics.csv`, using stored bridge-score predictions re-scored against the final labels.

## Key Finding 1: E4 Is The Best Alpha-Preserving Baseline

`E4_v03_three_feature` remains the cleanest Alpha-preserving simple method. It achieves:

- Alpha F1: 0.9486.
- Beta all F1: 0.6498.
- Beta sure-only F1: 0.7384.

It is interpretable: it evaluates whether a candidate sign-flipped window makes the counterfactual demand shape smoother and more physically plausible using bridge, roughness, and slope-continuity features.

The weakness is not recall. On Beta sure-only, E4 has only 6 FNs but 332 FPs.

## Key Finding 2: E1-E3 Are Useful But Not Enough

E1-E3 confirmed the bridge-family idea is meaningful, but not sufficient:

- E1 bridge only gives very high Beta recall but poor precision.
- E2 roughness adds a small improvement.
- E3 site-median correction does not solve the core issue and reduces Alpha recall.

These methods helped establish the baseline behaviour: simple bridge scores tend to over-detect Beta RPF.

## Key Finding 3: The Full v0.3 Three-Feature E4 Is Better Than Earlier Bridge Scores

E4 improves over E1-E3 on Alpha and Beta sure-only:

- E2 Beta sure-only F1: 0.7285.
- E4 Beta sure-only F1: 0.7384.
- E4 Alpha F1: 0.9486.

The slope-continuity feature helps Alpha substantially, although the E5 ablation shows that removing slope improves Alpha F1 but hurts Beta.

## Key Finding 4: E5 Without Slope Is Not The Beta Direction

E5 achieves the highest Alpha F1 among E1-E10:

- Alpha F1: 0.9563.

But it worsens Beta:

- Beta sure-only F1: 0.7069.

This suggests slope-continuity is useful for transfer to Beta, even if it slightly reduces Alpha F1 relative to E5.

## Key Finding 5: Site-Relative Score Normalisation Did Not Solve The Problem

E6-E8 tested site-relative transforms:

- E6 fixed site-median E4 correction.
- E7 site-centered E4.
- E8 site rank.

E7 improves Beta all-days F1 modestly, but it damages Alpha badly:

- E7 Alpha F1: 0.7648.
- E7 Beta all F1: 0.6628.
- E7 Beta sure-only F1: 0.6927.

Conclusion: site-relative score transforms alone are not robust enough. They can help Beta in some views but break Alpha validation.

## Key Finding 6: Input-Level Site Scaling Had No Effect

E9 and E10 scaled net load and solar before computing E4:

- E9 site solar scale.
- E10 combined site scale.

Both produced effectively identical results to E4. This is expected because the E4 features are ratio-style improvements, so uniformly scaling both net load and solar by the same site factor preserves the score.

Conclusion: simple site-scale normalisation is not a useful lever for this E4 formulation.

## Key Finding 7: Beta False Positives Are Mostly Short Local Windows

The HTML diagnostics showed that many Beta false positives are short local wiggle windows rather than sustained RPF-like intervals.

For Beta sure-only E4 predicted-positive days:

| Group | Median selected duration |
|---|---:|
| TP | 5.5 h |
| FP | 1.0 h |

This led to E11 and E12:

- E11 duration >= 1h improves Beta sure-only F1 from 0.7384 to 0.8213.
- E12 duration >= 1.5h improves Beta sure-only F1 to 0.8608.

But the cost is high Alpha recall loss:

- E11 Alpha F1: 0.8781.
- E12 Alpha F1: 0.8418.

Conclusion: duration is a real Beta FP signal, but hard duration gates are too blunt.

## Key Finding 8: Alpha Contains Many True Short RPF Events

Alpha duration analysis showed that true labelled RPF events are often short:

- Alpha median true RPF duration: about 3.5 hours.
- About 35.6% of true Alpha RPF days are shorter than 2.5 hours.
- About 42.3% are shorter than 3.0 hours.
- Sites such as `alpha_B` and `alpha_J` contain many short positive events.

This explains why Beta-inspired hard duration rules harm Alpha. The rule is physically plausible for Beta but not universally safe across Alpha labels.

## Key Finding 9: N-Height Is A Strong Beta FP Signal But Also Too Harsh As A Hard Gate

We tested:

```text
n_height_ratio =
  (window_net_peak - mean(left_edge_net_load, right_edge_net_load))
  / daytime_abs_net_load_p95
```

This captures whether the selected window has a meaningful N-shaped net-load bump.

For Beta sure-only E4 predicted-positive days:

| Feature | TP median | FP median |
|---|---:|---:|
| Window net max | 2.338 MW | 1.018 MW |
| Window peak / day p95 | 0.572 | 0.181 |
| N-height ratio | 0.413 | 0.054 |
| Window/day range ratio | 0.448 | 0.100 |

For beta_A-C, the separation was even stronger:

| Feature | TP median | FP median |
|---|---:|---:|
| Window net max | 4.822 MW | 1.054 MW |
| Window peak / day p95 | 0.506 | 0.059 |
| N-height ratio | 0.349 | 0.021 |
| Window/day range ratio | 0.362 | 0.042 |

Hard guard results:

| Variant | Alpha F1 | Beta all F1 | Beta sure-only F1 |
|---|---:|---:|---:|
| E4 baseline | 0.9486 | 0.6498 | 0.7384 |
| N-height >= 0.02 | 0.9172 | 0.7081 | 0.7993 |
| N-height >= 0.05 | 0.8725 | 0.7249 | 0.8170 |
| N-height >= 0.10 | 0.8009 | 0.7228 | 0.8123 |

Conclusion: N-height is a real, interpretable FP signal, but as a hard gate it hurts Alpha recall too much. It should be tested next as a soft penalty.

## Other Finding: Edge-Minima Logic Is Directional But Weaker

We checked whether selected window edges are minima-like.

For Beta sure-only E4 predicted-positive days:

| Edge feature | TP rate | FP rate |
|---|---:|---:|
| Both edges inside-min | 0.870 | 0.623 |
| Both local-min | 0.723 | 0.560 |

For beta_A-C:

| Edge feature | TP rate | FP rate |
|---|---:|---:|
| Both edges inside-min | 0.893 | 0.705 |
| Both local-min | 0.767 | 0.676 |

This confirms the intuition but the separation is not as strong as duration or N-height. Edge-minima is better treated as a secondary weak penalty, not the next main rule.

## Visual Artifacts

Generated one HTML per Beta site and confusion group for sure-only E4 errors:

| Site | Group | Days | Date range | File |
|---|---:|---:|---|---|
| beta_A | FP | 77 | 2023-10-02 to 2024-09-30 | `E4_beta_A_FP_77days.html` |
| beta_B | FN | 4 | 2024-06-25 to 2024-07-14 | `E4_beta_B_FN_4days.html` |
| beta_B | FP | 23 | 2023-10-22 to 2024-09-25 | `E4_beta_B_FP_23days.html` |
| beta_C | FP | 5 | 2023-11-17 to 2024-08-28 | `E4_beta_C_FP_5days.html` |
| beta_D | FP | 75 | 2023-11-06 to 2024-09-30 | `E4_beta_D_FP_75days.html` |
| beta_E | FP | 31 | 2023-10-02 to 2024-09-30 | `E4_beta_E_FP_31days.html` |
| beta_F | FP | 60 | 2023-10-04 to 2024-09-28 | `E4_beta_F_FP_60days.html` |
| beta_G | FP | 48 | 2023-10-18 to 2024-09-19 | `E4_beta_G_FP_48days.html` |
| beta_H | FN | 2 | 2023-10-25 to 2024-09-21 | `E4_beta_H_FN_2days.html` |
| beta_H | FP | 13 | 2023-10-06 to 2024-09-22 | `E4_beta_H_FP_13days.html` |

The HTML panels now:

- show only 06:00-18:00;
- show raw net load and solar;
- shade E4 selected/predicted candidate windows;
- shade actual/manual RPF windows;
- show bridge, roughness, slope, total score, and threshold.

## Files Updated Today

- `11_minimal_bridge_method_ladder.py`
- `11_minimal_bridge_method_ladder.ipynb`
- `12_e4_beta_fp_fn_visual_examples.py`
- `12_e4_beta_fp_fn_visual_examples.ipynb`
- `2026-07-06_experiment_journal_e4_duration_nheight.md`

Main output folders:

- `outputs/11_minimal_bridge_method_ladder/`
- `outputs/12_e4_beta_fp_fn_visual_examples_by_site_group_sure_only/`

## Recommended Next Step

Do not add another hard gate immediately. The evidence points to a small soft-penalty experiment:

```text
E14_soft_physical_plausibility =
  E4_score
  - a * short_duration_penalty
  - b * low_n_height_penalty
```

Keep it fast:

- tiny Alpha-only grid for `a` and `b`;
- one threshold selected on Alpha;
- report Alpha, Beta all, Beta sure-only;
- no visualisation unless the metrics look promising.

The key design target should be preserving Alpha F1 near 0.93-0.95 while lifting Beta sure-only F1 above E4.

## Open Questions

- Are short Alpha labelled RPF events physically equivalent to the Beta events we care most about?
- Should model selection prioritise Alpha validation, Beta all-days, or Beta sure-only?
- Should suspicious Beta visual cases be corrected in `final_review` before more tuning?
- Can duration and N-height be combined softly enough to reduce Beta FPs without killing Alpha recall?
- Is edge-minima useful only as a tie-breaker or weak penalty?

# 2026-07-10 Research Journal: Final Review Refresh And Physical-Feature Ablation

## Summary

Today we completed a final manual-review refresh of the Beta oracle labels, regenerated the active final Beta/Gamma datasets, and recomputed the non-XGB physical-score model results using the refreshed labels. The main finding is encouraging: the current best physical model remains strong, but a much smaller 4-feature model nearly matches it, and a 3-feature model still reaches about 97% recall and 91% F1 on the Beta sure-only subset.

## Activities

1. Served the oracle review web app locally for final manual inspection.
2. Exported the updated final-review oracle annotations into the active reviewed oracle dataset.
3. Refreshed the final journal Beta/Gamma datasets from the final-review CSV.
4. Recomputed the journal E4 three-feature model and current C14 best physical model against the refreshed final Beta labels.
5. Ran a physical-feature subset/ablation search excluding XGB and excluding any new gallery generation.

## Refreshed Dataset State

The final refresh used:

- Final-review annotations: `manual_oracle_annotations_final_review.csv`
- Active reviewed oracle output: `dataset/oracle_data_creation/actual_pynrpf_dataset_reflagged.parquet`
- Final Beta output: `dataset/final/dataset_beta.parquet`
- Final Gamma output: `dataset/final/dataset_gamma.parquet`

Final Beta after refresh:

| Quantity | Value |
|---|---:|
| Rows | 280,800 |
| Substations | 8 |
| Site-days | 2,928 |
| RPF day site-days | 630 |
| RPF intervals | 12,181 |
| Sure site-days | 2,310 |
| Unsure site-days | 618 |

Final Gamma after refresh:

| Quantity | Value |
|---|---:|
| Selected substation | `beta_B` |
| Rows | 35,136 |
| RPF day site-days | 152 |
| RPF intervals | 3,222 |

## Current Best Physical Model

The refreshed current best interpretable physical model is still:

`C14_best_G_b1p0_r1p5_sc0p5`

Regime:

- `R2_beta_loso_plus_alpha`
- Leave one Beta substation out.
- Tune threshold on all Alpha plus the other seven sure-confidence Beta substations.
- Evaluate on the held-out Beta substation.

Weights:

| Feature | Meaning | Weight |
|---|---|---:|
| F1 | Bridge improvement | 1.0 |
| F2 | Roughness improvement | 1.5 |
| F3 | Slope-continuity improvement | 1.0 |
| F4 | Duration plausibility | 1.0 |
| F5 | N-height ratio | 1.0 |
| F6 | Solar-strength ratio | 1.0 |
| F7 | Solar peak alignment | 1.0 |
| F8 | Site-centered core score | 0.5 |
| F9 | Site-rank core score | 0.0 |

Refreshed pooled day-level performance:

| Subset | Precision | Recall | F1 |
|---|---:|---:|---:|
| Beta all | 0.7714 | 0.8571 | 0.8120 |
| Beta sure-only | 0.8916 | 0.9427 | 0.9164 |

This is an improvement over the previous tracked-cache sure-only F1 of about 0.8994.

## E4 Three-Feature Baseline

The original journal E4 model uses:

- F1 bridge improvement
- F2 roughness improvement
- F3 slope-continuity improvement

with one Alpha-selected threshold. After the final label refresh:

| Subset | Precision | Recall | F1 |
|---|---:|---:|---:|
| Beta all | 0.4951 | 0.9683 | 0.6552 |
| Beta sure-only | 0.6165 | 1.0000 | 0.7628 |

This confirms the older E4 model is very sensitive but too permissive: it catches almost all sure-positive days, but creates too many false positives.

## Feature Ablation Search

We then ran a lightweight physical-feature subset search from the tracked daily feature cache, with refreshed final labels rejoined from `dataset/final`.

Scope:

- XGB excluded.
- No candidate-window rebuild.
- No FP/FN gallery regeneration.
- 766 physical-feature subset models evaluated.
- 6,128 held-out-substation threshold folds evaluated.

Search families:

1. `c14_weighted_ablation`: all non-empty subsets of the 8 nonzero C14 features, retaining C14 weights.
2. `equal_weight_subset`: all non-empty subsets of all 9 physical features with equal weights.

Main outputs:

- `outputs/20260710_physical_feature_subset_search/05_subset_ranking_all.csv`
- `outputs/20260710_physical_feature_subset_search/06_best_by_feature_count.csv`
- `outputs/20260710_physical_feature_subset_search/10_feature_frequency_top_models.csv`

## Best Subsets By Feature Count

Best C14-style ablations by feature count:

| Feature count | Features | Beta sure P | Beta sure R | Beta sure F1 | Beta all F1 |
|---:|---|---:|---:|---:|---:|
| 1 | F1 | 0.6349 | 0.9490 | 0.7609 | 0.6628 |
| 2 | F2, F4 | 0.8281 | 0.9512 | 0.8854 | 0.7712 |
| 3 | F1, F3, F4 | 0.8561 | 0.9724 | 0.9105 | 0.7913 |
| 4 | F1, F3, F4, F6 | 0.8686 | 0.9682 | 0.9157 | 0.7972 |
| 5 | F1, F3, F4, F6, F7 | 0.8978 | 0.9321 | 0.9146 | 0.8076 |
| 6 | F1, F3, F4, F5, F6, F7 | 0.8770 | 0.9533 | 0.9135 | 0.8059 |
| 7 | F1, F3, F4, F5, F6, F7, F8 | 0.8902 | 0.9299 | 0.9097 | 0.8112 |
| 8 | F1, F2, F3, F4, F5, F6, F7, F8 | 0.8916 | 0.9427 | 0.9164 | 0.8120 |

Key finding:

The 4-feature model `F1 + F3 + F4 + F6` almost matches the full best model:

- Full 8-feature C14 best: Beta sure-only F1 = 0.9164
- Compact 4-feature model: Beta sure-only F1 = 0.9157

This is only a 0.0007 absolute F1 gap while using half the effective features.

## Important Three-Feature Finding

The 3-feature subset to highlight is:

`F1 + F3 + F4`

That is:

- F1 bridge improvement
- F3 slope-continuity improvement
- F4 duration plausibility

Performance:

| Subset | Precision | Recall | F1 |
|---|---:|---:|---:|
| Beta sure-only | 0.8561 | 0.9724 | 0.9105 |
| Beta all | 0.7100 | 0.8937 | 0.7913 |

This is a strong result for the paper story: with only three interpretable physical features, the method reaches about 97% recall and 91% F1 on sure-confidence Beta days. It is a compact high-recall model that may be useful if the objective is to minimise missed RPF cases while still keeping the model interpretable.

Note: in discussion this was written as "F1, F3, and F3"; the result table indicates the intended 3-feature subset is `F1 + F3 + F4`.

## Feature Importance Pattern

Among the top C14-weighted ablation models:

| Feature | Frequency in top 25 |
|---|---:|
| F4 duration plausibility | 100% |
| F1 bridge improvement | 96% |
| F3 slope-continuity improvement | 96% |
| F6 solar-strength ratio | 56% |
| F7 solar peak alignment | 56% |
| F5 N-height ratio | 48% |
| F2 roughness improvement | 40% |
| F8 site-centered core score | 40% |
| F9 site-rank core score | 0% |

Interpretation:

- F4 duration plausibility is the most consistently useful feature.
- F1 bridge and F3 slope continuity are the two core physical-shape features.
- F6 and F7 help some compact models, especially when trying to lift precision.
- F9 is not useful under the current best setup and already has weight zero.

## Paper-Facing Takeaways

1. The refreshed final labels improved the current best sure-only F1 to 0.9164.
2. The current best model is still the 8-effective-feature C14 weighted model with F9 weight zero.
3. A compact 4-feature model, `F1 + F3 + F4 + F6`, almost matches the full best model at F1 0.9157.
4. A very compact 3-feature model, `F1 + F3 + F4`, reaches recall 0.9724 and F1 0.9105 on Beta sure-only days.
5. If the story is "minimise manual checking", the 5-feature model `F1 + F3 + F4 + F6 + F7` is interesting because it has higher sure-only precision than the full 8-feature model, with F1 still 0.9146.
6. The ablation supports a simpler and more explainable model story: bridge shape, slope continuity, duration plausibility, and optionally solar strength/peak alignment carry most of the performance.

## Recommended Next Step

For the manuscript and slide deck, compare three candidates:

| Candidate | Role |
|---|---|
| `F1 + F3 + F4` | Minimal high-recall interpretable model |
| `F1 + F3 + F4 + F6` | Best compact F1 model |
| Full C14 best | Best headline F1 model |

This would let the paper report both the best achieved result and the simpler model that explains nearly all of the gain.

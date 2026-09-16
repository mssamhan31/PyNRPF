# 2026-06-25 Learning From CGPT Pro 0.778 F1 Experiment

## Purpose

This note summarises what we learned from the CGPT Pro prototype artifacts stored in:

```text
publication/2_journal_article/notebooks/99_Misc/outputs/99_cgpt_pro_experiments/
```

The purpose is not to treat the result as publication-ready. The purpose is to identify which modelling ideas appear to explain the large jump in Dataset Beta performance, and which ideas should be carried into the next `m9.2` implementation.

## Headline Result

CGPT Pro reported a counterfactual structured-window prototype with:

| Evaluation | Precision | Recall | F1 |
|---|---:|---:|---:|
| Beta day level | 0.772 | 0.785 | 0.778 |
| Beta interval level | 0.766 | 0.798 | 0.782 |

The day-level confusion counts were:

| TP | FP | FN | TN |
|---:|---:|---:|---:|
| 437 | 129 | 120 | 2242 |

The stricter calibration check, where the threshold and seasonal prior were learned only from Alpha observations before October 2023, gave a very similar Beta day result:

| Precision | Recall | F1 |
|---:|---:|---:|
| 0.793 | 0.761 | 0.777 |

This is important because it suggests the result is not only caused by a fragile final threshold choice. However, the overall formulation was still selected after inspecting Beta results, so the number remains an exploratory development-set result.

## Comparison With Our Current Methods

The current local `m9.2_physics` full run gave:

| Method | Dataset | Level | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|
| current `m9.2_physics` | Alpha LOSO | day | 0.957 | 0.935 | 0.946 |
| current `m9.2_physics` | Alpha LOSO | interval | 0.808 | 0.909 | 0.856 |
| current `m9.2_physics` | Beta transfer | day | 0.454 | 0.659 | 0.537 |
| current `m9.2_physics` | Beta transfer | interval | 0.461 | 0.566 | 0.508 |

Previous `m9_hybrid` gave:

| Method | Dataset | Level | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|
| previous `m9_hybrid` | Beta transfer | day | 0.411 | 0.677 | 0.511 |
| previous `m9_hybrid` | Beta transfer | interval | 0.517 | 0.642 | 0.573 |

The CGPT Pro result is therefore materially better on Beta:

- day F1: `0.537 -> 0.778` compared with our current `m9.2_physics`;
- interval F1: `0.508 -> 0.782` compared with our current `m9.2_physics`.

## Main Difference From Our Current m9.2

Our current `m9.2_physics` implementation is primarily:

```text
dense candidate windows
  + per-candidate counterfactual features
  + XGBRanker
  + null-vs-best-candidate margin
```

The CGPT Pro best result is different. It is closer to:

```text
dense candidate windows
  + compact physical candidate scan
  + per-day upper-tail bridge plausibility score
  + label-free site adjustment
  + Alpha-calibrated threshold
  + one-window decoder
```

This distinction matters. The 0.778 result did not mainly come from training a stronger general supervised model. It came from finding a physically meaningful scalar score that transferred better from Alpha to Beta.

## What The Best Model Actually Uses

The supplied `final8_eval.py` uses the following score:

```python
raw = bridge_ratio_p99
z = raw - 0.425 * site_median(raw)
z = z + 0.075 * within_site_rank(raw)
z = z + 0.05 * alpha_season_prior(day_of_year)
score = (z + 0.1 * rolling_mean_5(z)) / 1.1
```

The best row in `targeted_small8.csv` was:

| feature | rolling window | site median weight | site rank weight | season weight | rolling weight | threshold | Beta P | Beta R | Beta F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bridge_ratio_p99` | 5 | 0.425 | 0.075 | 0.050 | 0.1 | 0.55859 | 0.772 | 0.785 | 0.778 |

The stricter pre-October Alpha-calibrated row in `targeted_trainonly_fast.csv` was:

| feature | rolling window | site median weight | site rank weight | season weight | rolling weight | pmin | threshold | Beta P | Beta R | Beta F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bridge_ratio_p99` | 7 | 0.425 | 0.000 | 0.025 | 0.25 | 0.9995 | 0.51892 | 0.793 | 0.761 | 0.777 |

This tells us the most robust signal is not a large feature set. It is specifically the high percentile of a counterfactual bridge-ratio score, adjusted in a label-free way for site and season.

## Candidate Generation Insight

Earlier candidate-generation diagnostics showed current m7-style candidate recall was a hard ceiling:

| Candidate generator | Beta strict candidate recall |
|---|---:|
| current m7 candidates | about 0.659 |
| current plus plateau | about 0.820 |
| combined shape plus plateau | about 0.867 |

CGPT Pro dense scanning went further. The attached note reports Beta candidate recall of:

| Criterion | Beta candidate recall |
|---|---:|
| at least one candidate with IoU >= 0.50 | 0.975 |
| at least one candidate with IoU >= 0.70 | 0.926 |
| both boundaries within +/-30 minutes | 0.890 |

This changes the problem. Candidate generation is no longer the main ceiling. The remaining bottleneck becomes:

- day-score calibration;
- separating true RPF days from site-specific lookalikes;
- selecting the correct window among many candidates.

## Counterfactual Bridge Ratio

For raw net load `y` and solar `S`, the model considers two interpretations:

```text
Forward/no-correction interpretation:
U_plus = S + y

Reverse/RPF interpretation inside a candidate window:
U_minus = S - y
```

For each candidate window, the code computes whether `U_minus` inside the window forms a more plausible bridge between the uncorrected periods before and after the window.

In `rpf_stage2_scan8.py`, this appears as:

```python
bup = bridge_mse(up, L, R, up)
bum = bridge_mse(um, L, R, up)
babs = bup - bum
brat = babs / (bup + bum + 1e-6)
```

Where:

- `up = S + y`;
- `um = S - y`;
- `bridge_mse` compares the candidate segment against a line between the uncorrected values just outside the candidate boundaries;
- `bridge_ratio` is high when the reverse-flow interpretation bridges better than the forward-flow interpretation.

The strongest day-level variable is:

```text
bridge_ratio_p99
```

That is, the 99th percentile of bridge-ratio plausibility across all dense candidate windows in the day. This is important because it does not depend on a single brittle candidate.

## Why p99 Seems Better Than Best Candidate Only

Using an upper-tail percentile rather than only the single maximum candidate is probably useful because:

- it is less sensitive to one strange cherry-picked window;
- it captures whether the day has a cluster of plausible RPF-like windows;
- it still responds strongly when there is a physically coherent RPF period;
- it keeps the model day-level and compact.

This is a major learning for `m9.2` v2.

## Label-Free Site Adaptation

The raw bridge score differs by substation. CGPT Pro improved transfer using label-free site adaptation:

- subtract a fraction of each site's annual median score;
- optionally add a within-site rank score;
- add an Alpha-learned seasonal prior;
- add a small centered rolling-score component.

No Beta labels are required for these calculations. They use only the distribution of the score at the target site.

This likely explains much of the improvement over our current XGBRanker version. The current ranker sees many features but still transfers poorly because the score scale and lookalike patterns differ by site.

## Conservative Alpha Calibration

The threshold was selected under a very high Alpha precision requirement, approximately:

```text
Alpha precision >= 0.9995
```

The intuition is sensible: correcting sign errors is operationally sensitive, so false positives should be strongly discouraged.

However, there is a tradeoff. In the strict calibration table, the Alpha training F1 can look modest because the threshold is extremely conservative. The surprising part is that this conservative threshold transfers to Beta with much better F1 than our ranker.

## Interval Decoder

The day score classifies a day as RPF or not. If a day is positive, the interval decoder chooses exactly one candidate window.

The attached note says the best decoder selects the candidate that produces the greatest improvement in full-day total variation after counterfactual sign correction.

This is another useful simplification:

- use `bridge_ratio_p99` for day classification;
- use full-day total-variation improvement for window selection.

This avoids asking a supervised ranker to solve both the day-classification and boundary-selection tasks at once.

## Site-Level Result

From `m10_site_metrics.csv`:

| Site | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| beta_A | 0.518 | 0.879 | 0.652 | 29 | 27 | 4 | 306 |
| beta_B | 0.800 | 0.738 | 0.768 | 96 | 24 | 34 | 212 |
| beta_C | 0.500 | 0.333 | 0.400 | 1 | 1 | 2 | 362 |
| beta_D | 0.805 | 0.688 | 0.742 | 66 | 16 | 30 | 254 |
| beta_E | 0.889 | 0.800 | 0.842 | 48 | 6 | 12 | 300 |
| beta_F | 0.759 | 0.881 | 0.816 | 104 | 33 | 14 | 215 |
| beta_G | 0.815 | 0.830 | 0.822 | 88 | 20 | 18 | 240 |
| beta_H | 0.714 | 0.455 | 0.556 | 5 | 2 | 6 | 353 |

The pooled F1 is much higher than the macro-average site F1:

- pooled day F1: about 0.778;
- macro site F1: about 0.700.

Both should be reported if this method is later used in the journal workflow.

## Bootstrap Uncertainty

From `m10_block_bootstrap.csv`, the 14-day block bootstrap intervals were approximately:

| Metric | 2.5% | Median | 97.5% |
|---|---:|---:|---:|
| day F1 | 0.729 | 0.778 | 0.817 |
| interval F1 | 0.731 | 0.782 | 0.821 |

This captures sampling variability but not model-selection optimism.

## Alternative Approaches Tested

The attached note lists these alternatives:

| Approach | Best Beta day F1 | Main learning |
|---|---:|---|
| Solar-scale-optimised bridge model | 0.755 | Strong ranking, but weaker accepted-day FP discrimination than main model. |
| Unsupervised 3-component Gaussian mixture | 0.753 | Useful separation signal, but unstable on low-support sites. |
| Temporally structured combined-physics score | 0.742 | Good recall, more low-load false positives. |
| Logistic regression day features | about 0.589 | Generic supervised model transferred poorly. |
| LightGBM day features | about 0.616 | Generic supervised model transferred poorly. |
| Viterbi-style smoother | about 0.779 | Negligible gain over simpler model, likely not worth complexity. |

The clearest message is that compact, physically invariant scoring worked better than a broad supervised feature learner.

## What Worked Well

1. Dense daytime candidate scan.

This removed the hard candidate-recall ceiling from m7-style local-peak candidates.

2. Bridge-ratio counterfactual plausibility.

The strongest signal was whether the reverse-flow interpretation creates a more plausible bridge between surrounding no-correction periods.

3. Upper-tail day summary.

`bridge_ratio_p99` worked better than depending on a single candidate.

4. Label-free site adjustment.

Site median and site rank adjustments improved transfer without using Beta labels.

5. Alpha-learned seasonal prior.

The seasonal prior helped encode the known August-November/April temporal concentration without using Beta labels.

6. Conservative threshold calibration.

Very high Alpha precision produced a threshold that transferred surprisingly well to Beta.

7. Separate day classification and interval selection.

The day decision uses a robust daily score; interval selection can use a deterministic physical decoder.

## What Did Not Work As Well

1. Generic supervised day-feature models.

Logistic regression and LightGBM did not approach the compact bridge-score result.

2. Adding temporal smoothing/Viterbi.

The best smoother improved only about 0.001 in F1 and was obtained after Beta-guided screening.

3. Broad combined physics scores.

The combined score was useful but retained more low-load false positives.

4. Relying on learned candidate rankers alone.

Our current XGBRanker-based `m9.2_physics` has strong Alpha LOSO F1 but weak Beta transfer F1. This suggests the ranker is learning Alpha-specific scoring patterns rather than a compact transferable physics rule.

## Remaining Problems

The best CGPT Pro model still has 120 Beta false negatives.

The attached note says:

- 110 of 120 false-negative days still had a candidate with IoU >= 0.50;
- 91 had a candidate with IoU >= 0.70;
- 87 had both boundaries within +/-30 minutes.

So after dense generation, most remaining false negatives are not candidate-availability failures. They are calibration/scoring failures.

The weaker sites remain:

- `beta_A`: many false positives;
- `beta_C`: only 3 positives, unstable estimate;
- `beta_H`: low support and lower recall;
- `beta_D`: still meaningful false negatives despite decent precision.

## Validation Caveat

The 0.778 result is not yet a final external-validation estimate.

Reasons:

- CGPT Pro inspected Beta results while choosing feature families and formulations.
- The result is therefore Beta-guided exploratory model development.
- Current Beta labels are still provisional until the manual oracle review is complete.
- Site-level performance is uneven.
- The bootstrap interval does not account for model-selection optimism.

The right framing is:

> The CGPT Pro artifacts identify a much stronger modelling direction. They do not yet prove publication-ready generalisation.

## Main Takeaway For m9.2 v2

The next implementation should not just tune the current XGBRanker.

The next implementation should rebuild `m9.2` around:

```text
dense daytime candidate scan
  + bridge_ratio_p99 daily score
  + label-free site median/rank adjustment
  + Alpha-learned seasonal prior
  + conservative Alpha threshold calibration
  + deterministic one-window decoder
```

The goal is to preserve the compact transferable physics score while making the implementation reproducible inside the PyNRPF misc workflow.

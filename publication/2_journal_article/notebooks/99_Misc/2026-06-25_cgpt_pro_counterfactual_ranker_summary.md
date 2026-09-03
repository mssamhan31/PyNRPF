# 2026-06-25 CGPT Pro Counterfactual Ranker Recommendation Summary

## Purpose

This document records the external CGPT Pro recommendation received on 2026-06-25 for improving the real-site RPF detection method. It is written as a research handoff note for future modelling work in the misc notebook folder.

The central message is that the bottleneck is not generic XGBoost hyperparameter tuning. The stronger direction is to redesign the problem as a **physics-guided structured sign-inference problem**.

## Main Diagnosis

The main failure is a mismatch between:

- the candidate-generation mechanism;
- the training labels and the actual day-level F1 objective;
- the structured one-event nature of RPF days;
- Alpha-to-Beta domain shift.

The original conference-style model performs strongly on easier Alpha settings, but the journal experiment correctly stresses unseen substations and manually labelled real data. The collapse from strong Alpha performance to weaker Beta performance is therefore informative: the model has learned Alpha-specific morphology more than a transferable physical definition of wrong-sign RPF.

## Error Taxonomy From The Response

| Failure mechanism | Evidence noted by CGPT Pro | Consequence |
|---|---|---|
| m8 day-gate failure | Most m8 false-negative days never reach the interval model. | Interval-threshold tuning cannot recover most recall loss. |
| Cascade/objective mismatch | m8 day gate alone has similar or better day F1 than the final cascade, while the interval stage trades recall for precision. | The cascade is not aligned with day-level F1. |
| Site-dependent domain shift | Some Beta sites have very low recall while others overproduce false positives. | A single global threshold cannot fix all sites. |
| Candidate-generation ceiling | Current m9-style generator misses many Beta RPF days, especially under strict boundary criteria. | A perfect scorer cannot recover days with no useful candidate. |
| Missing-data veto | Many missed candidates are caused by rejecting whole days with missing net-load values. | Missingness is being treated as evidence of no RPF. |
| Training-label mismatch | Near-correct candidates are labelled negative if they fail the strict +/-30 minute boundary rule. | Physically useful candidates are taught as negatives. |
| Independent interval predictions | Most Beta RPF days have one contiguous event, but interval models create fragmented predictions. | Predictions violate the observed event structure. |
| FP/FN morphology separation | FN windows often have strong solar-net-load co-movement while FP windows often have weak or opposite structure. | There are physical signals not being used effectively. |

## Key Recommendation

Redefine the future `m9_rpf` as a **physics-guided counterfactual window-ranking model**, not another independent interval classifier.

For a positive raw meter reading `y_t` and estimated solar generation `S_t`, there are two physical interpretations of underlying demand:

```text
Normal forward-flow interpretation:
U_t^F = S_t + y_t

Wrong-sign RPF interpretation:
U_t^R = S_t - y_t
```

For every candidate window `W = [s, e]`, reconstruct a full-day underlying-load curve:

```text
U_t(W) = S_t - y_t, if t in W
       = S_t + y_t, if t not in W
```

Then score whether `U(W)` is more physically plausible than the no-correction reconstruction:

```text
U(empty) = S + y
```

The core question becomes:

> Does applying the sign correction inside this candidate window make the full-day reconstructed underlying demand curve more physically plausible?

## Dense Bounded Candidate Windows

CGPT Pro recommends replacing brittle local-peak candidate generation with dense, bounded candidate windows:

- include one null candidate representing no RPF;
- generate every daytime window from 0.5 to 8 hours;
- initially require the window to contain the daily solar peak;
- for cloudy or unusual days, allow top-three solar peaks or windows within +/-60 minutes of the main peak.

This is computationally feasible because 06:00-18:00 has only about 49 fifteen-minute timestamps, so the number of duration-constrained windows is small enough for per-day ranking.

The recommendation is strongly aligned with current diagnostics:

- current local-peak candidate generation fails plateau days;
- plateau-aware candidates greatly improve recall;
- most Beta RPF days are one contiguous event;
- most labelled RPF events contain the daily solar peak.

## Counterfactual Plausibility Features

Pseudo-load `S - y` is useful, but CGPT Pro argues that the stronger feature idea is the improvement in the whole reconstructed load curve.

Candidate features should include:

- improvement in fit to a site-specific historical underlying-load profile;
- reduction in full-day roughness and curvature;
- continuity at sign-transition boundaries;
- raw net-load level at the boundaries;
- fraction of reconstructed demand below zero;
- solar-net-load correlation and derivative agreement inside the window;
- pseudo-load stability relative to comparable non-RPF periods;
- duration, start time, end time, and distance from solar peak;
- missingness and telemetry-quality indicators.

A useful scalar feature is:

```text
Delta(W) = L(U(empty)) - L(U(W))
```

where `L` combines robust profile residual, boundary discontinuity, roughness, and physical-constraint penalties. Positive `Delta(W)` means the candidate correction improves plausibility.

The historical profile can be estimated without Beta labels using:

- robust site-hour medians from low-solar or clearly forward-flow days;
- weekday/month conditioning;
- same-day interpolation or bridge between pre-window and post-window load;
- optional site solar scale `alpha_site`, using `U = alpha_site * S +/- y`, to account for solar-estimation bias.

## Ranking Instead Of Binary Candidate Classification

The response recommends `XGBRanker` grouped by site-day.

On an RPF day, candidate relevance should be based on overlap with the labelled event:

- relevance 4: IoU >= 0.85;
- relevance 3: 0.70 <= IoU < 0.85;
- relevance 2: 0.50 <= IoU < 0.70;
- relevance 1: 0.25 <= IoU < 0.50;
- relevance 0: IoU < 0.25.

On a non-RPF day:

- the null candidate receives the highest relevance;
- non-null candidates receive low relevance.

This solves two current problems:

- near-correct boundary candidates are no longer treated as fully negative;
- positive days without an exact +/-30 minute candidate no longer become all-negative training groups.

The day is predicted positive when the best non-null candidate exceeds the null candidate by an Alpha-calibrated margin.

## Structured Decoder

The decoder should return exactly one of:

- null/no RPF;
- one continuous candidate window.

This directly encodes the observed structure:

- most Beta RPF days contain exactly one contiguous event;
- fragmented interval predictions are physically unlikely;
- day-level F1 is the primary goal.

For rare multi-segment days, use the enclosing window for now. A later sensitivity experiment can allow two segments with a substantial transition penalty.

## Why This Is Stronger Than More XGBoost Tuning

The counterfactual ranker addresses several bottlenecks at once:

- no local-peak candidate ceiling;
- no unrecoverable day gate;
- no whole-day missingness veto;
- no fragmented independent interval predictions;
- less reliance on raw MW morphology;
- no binary punishment of near-correct boundaries;
- stronger physical interpretability.

It also strengthens the journal contribution: the method becomes a structured physical sign-inference framework, not simply an optimised classifier.

## Alternatives Noted

### Alternative 1: Repair m9 Without Full Redesign

Lowest effort. Keep the current candidate-classification architecture but:

- interpolate short gaps for candidate generation;
- add fallback solar-centred windows;
- use nearby/multiple minima instead of global minima;
- use IoU-soft labels rather than strict binary boundaries;
- avoid selecting high-IoU positive-day candidates as hard negatives;
- normalise magnitude features by robust site scales;
- add m7/m8 outputs as meta-features;
- keep the one-window decoder.

This is likely useful as an ablation, but remains morphology-dependent.

### Alternative 2: Constrained HSMM / Viterbi Model

Use a three-state left-to-right model:

```text
forward-before -> RPF -> forward-after
```

Each interval receives an emission score, and Viterbi/HSMM decoding enforces:

- at most one RPF segment;
- plausible duration;
- daytime occurrence;
- transition penalties;
- optional solar-peak inclusion.

This removes explicit candidate generation, but risks becoming a more complex interval classifier if counterfactual features and normalisation are weak.

### Alternative 3: Realistic Alpha Domain Randomisation

Generate Alpha variants with real telemetry distortions:

- site-level net/solar scaling;
- solar-estimation bias and lag;
- missing blocks;
- flatlines, plateaus, spikes;
- meter deadband around zero;
- boundary jitter;
- additive offsets and noise;
- cloudy multi-peak solar;
- imperfect absolute-value behaviour.

This is complementary but does not remove structural candidate/decoder issues by itself.

## Validation Recommendations

CGPT Pro emphasised that current misc searches are exploratory and Beta-guided. For final validation:

- finalise Beta labels with confirmed RPF, confirmed non-RPF, and uncertain categories;
- adjudicate model disagreements and a representative true-negative sample;
- use nested Alpha leave-one-substation-out validation for feature/model/threshold selection;
- report pooled F1 and macro-average site F1;
- select threshold or ranking margin by maximising F1 subject to minimum precision;
- report boundary MAE and event IoU separately from day F1;
- lock a new confirmatory actual-data set before final claims if possible.

## Recommended Development Order

1. Repair missing-data handling and produce a formal FN/FP taxonomy.
2. Implement dense bounded windows and a null candidate.
3. Replace binary boundary labels with IoU ranking.
4. Add fully normalised counterfactual reconstruction features.
5. Add historical underlying-load plausibility score.
6. Freeze the model and evaluate on a locked actual-data set.

## Current Interpretation For Our Work

This recommendation matches our recent findings:

- plateau-aware generation helps because current m7 is brittle;
- broad candidate recall is now less limiting, but choosing/scoring windows is still hard;
- the next serious m9 should probably be a counterfactual structured ranker;
- the current candidate-generator notebook is useful as a fast lab, but the next method needs a new notebook for dense windows, null candidates, counterfactual features, and ranking.


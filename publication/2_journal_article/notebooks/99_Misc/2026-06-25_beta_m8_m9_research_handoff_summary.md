# 2026-06-25 Beta m8/m9 Research Handoff Summary

## Purpose Of This Document

This note summarises the current state of the Dataset Beta RPF-detection problem for external brainstorming.

The intended reader is someone who has not followed the whole development process but can help think deeply about the modelling problem. The goal is to make the situation clear enough for a smart collaborator, ChatGPT Pro, or another ML researcher to suggest better methods.

This is a research handoff, not a publication-ready result summary.

## High-Level Goal

We want to detect wrong-sign reverse power flow (RPF) readings in real substation net-load time series.

The physical problem is:

- `net_load_MW` should sometimes become negative during strong solar export.
- In the problematic real data, some intervals appear as positive net load even though they are believed to be wrong-sign readings.
- The correction is not to change the raw measurement in the source dataset, but to label which intervals are wrong-sign RPF intervals.
- Once an interval is labelled as RPF, the corrected/reference interpretation is effectively the negative of the raw positive net-load value.

The modelling goal is:

- Primary metric: maximise **day-level F1** on Dataset Beta.
- Secondary metric: interval-level F1.
- False positives are worse than false negatives in practice, but we still want to optimise F1 rather than precision alone.
- The final deployable method should train/tune using Dataset Alpha only and transfer to Dataset Beta without Beta-specific calibration.

## Datasets And Current Caveats

Dataset Alpha:

- Synthetic/paper-facing dataset.
- Site IDs are `alpha_A` to `alpha_J`.
- Used as the training and validation source for publication-facing methods.
- Contains many labelled RPF examples with known synthetic labels.

Dataset Beta:

- Real/paper-facing dataset.
- Site IDs are `beta_A` to `beta_H`.
- Current Beta is based on `actual_pynrpf_dataset.csv`, filtered to `2023-10-01` through `2024-09-30`.
- Current Beta labels are the existing/manual labels from the original actual dataset.
- Final manual oracle review is still in progress, so current Beta labels may contain noise or ambiguous cases.

Important Beta support:

- Total Beta site-days: `2,928`.
- Labelled Beta RPF days: `557`.
- Beta daytime interval support in Notebook 2 metrics: `152,100` intervals.

## Main Framing

The central question is not just "how do we tune XGBoost?"

The better framing seems to be:

1. Can we generate at least one plausible candidate RPF window on most true RPF days?
2. Can a model/rule score those candidates well enough to separate true RPF from non-RPF lookalikes?
3. Can the final decoder enforce the event structure: at most one contiguous RPF window per site-day?
4. Can we tune thresholds using Alpha only and still transfer well to real Beta sites?

The current evidence suggests that candidate generation is already a major bottleneck.

## Publication Workflow Baseline: Notebook 2

Notebook:

- `publication/2_journal_article/notebooks/02_correction_validation.ipynb`

Key output:

- `publication/2_journal_article/outputs/metrics/02_correction_validation/01_correction_metrics.csv`

Notebook 2 compares:

- `m8_xgb`: main trainable correction method.
- `m7_dtr`: deterministic benchmark.

Current Beta transfer result from Notebook 2:

| Method | Level | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `m8_xgb` | day | `0.606` | `0.420` | `0.496` |
| `m8_xgb` | interval | `0.659` | `0.246` | `0.358` |
| `m7_dtr` | day | `0.192` | `0.761` | `0.307` |
| `m7_dtr` | interval | `0.370` | `0.646` | `0.470` |

Interpretation:

- `m8_xgb` is much more precise than `m7_dtr`, but recall is too low, especially at interval level.
- `m7_dtr` catches many more RPF days/intervals but produces too many false positives.
- The Beta result is much weaker than the best Alpha holdout results, showing a real transfer/domain-shift issue.

## Diagnostics Performed So Far

The misc folder contains diagnostic notebooks. These are exploratory and should not be treated as publication-ready validation.

### 01: Beta XGB1 Diagnostics

Notebook:

- `publication/2_journal_article/notebooks/99_Misc/01_beta_xgb1_diagnostics.ipynb`

Goal:

- Diagnose why the `m8_xgb` candidate-day logic underperforms on Beta.
- Separate day-level candidate behaviour from final interval filtering.
- Inspect TP/TN/FP/FN examples interactively.

Key findings:

- Beta transfer failures are site-specific.
- Some sites have obvious RPF-like days that the model misses.
- Some sites have low-load or unusual non-RPF days that look RPF-like and create false positives.
- The problem does not look like a single global threshold issue.

### 02: Beta m8 Threshold And Variant Search

Notebook:

- `publication/2_journal_article/notebooks/99_Misc/02_beta_m8_xgb_threshold_and_variant_search.ipynb`

Outputs:

- `publication/2_journal_article/notebooks/99_Misc/outputs/02_beta_m8_xgb_threshold_and_variant_search/`

This notebook explored threshold calibration, site-specific calibration, site normalisation, and small hyperparameter variants.

Important caveat:

- This is Beta-guided exploratory work.
- Beta labels were used to compare/tune variants.
- These results are useful for diagnosis, not for final model validation.

Headline findings:

| Variant Type | Best Variant | Calibration | Mean Held-Out Beta Day F1 |
|---|---|---|---:|
| Raw threshold calibration | `baseline_raw_features` | global split-validated thresholds | `0.520` |
| Raw site-specific thresholds | `baseline_raw_features` | site split-validated thresholds | `0.597` |
| Site-normalised | `site_joint_p95` | site split-validated thresholds | `0.640` |
| Site-normalised | `site_daytime_joint_p95` | global split-validated thresholds | `0.637` |

Optimistic all-Beta upper-bound examples:

| Variant | Level | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `baseline_raw_features` | day | `0.525` | `0.634` | `0.574` |
| `baseline_raw_features` | interval | `0.577` | `0.436` | `0.497` |
| `site_joint_p95` | day | `0.532` | `0.795` | `0.637` |
| `site_joint_p95` | interval | `0.682` | `0.638` | approximately `0.659` |

Interpretation:

- Normalisation and calibration help substantially.
- However, site-specific calibration is not ideal because the intended use case is random/new sites.
- Even with Beta-guided normalisation/calibration, the result is not close to near-human performance.
- This suggests that the limitation is not just hyperparameters. It is likely representation, candidate generation, and event-structure modelling.

## m9 Direction So Far

We started moving beyond `m8_xgb` toward a new exploratory method family called `m9_hybrid`.

The central design idea:

- Do not predict each 15-minute interval independently.
- Treat each site-day as having either:
  - no RPF; or
  - one contiguous RPF window.
- Generate candidate windows using deterministic/domain-inspired logic.
- Score candidate windows with XGBoost.
- Decode each site-day to either no RPF or the single best candidate window.

This better matches the suspected physical structure of the problem.

## m9 Candidate-Window Design

Current m9 candidate generator is based on m7-style logic:

- Find local positive net-load peaks that could represent wrong-sign RPF bumps.
- For each peak, choose left and right minima around the peak as candidate boundaries.
- Convert boundaries into a predicted interval.
- Score each candidate with deterministic shape and co-movement features.

Training label:

- Collapse each true RPF day into one enclosing true window.
- Candidate is positive if predicted start and end are both within plus/minus 30 minutes of the true start/end.

Inference:

- Score all candidates.
- Select the highest scoring candidate if it passes threshold.
- Otherwise predict no RPF.
- Output exactly one window or no window per site-day.

## Current Candidate-Generation Ceiling

This is probably the most important current finding.

With the current m7-style candidate generator:

Alpha candidate recall:

- Alpha RPF days: `3,423`.
- Alpha RPF days with at least one positive candidate: `2,418`.
- Alpha candidate recall: `0.706`.
- Average candidates per candidate-day: `3.54`.

Beta candidate recall:

- Beta RPF days: `557`.
- Beta RPF days with at least one positive candidate: `367`.
- Beta candidate recall: `0.659`.
- Average candidates per candidate-day: `3.56`.

Beta site-level candidate recall:

| Site | RPF Days | Hits | Candidate Recall |
|---|---:|---:|---:|
| `beta_A` | `33` | `29` | `0.879` |
| `beta_B` | `130` | `64` | `0.492` |
| `beta_C` | `3` | `3` | `1.000` |
| `beta_D` | `96` | `15` | `0.156` |
| `beta_E` | `60` | `26` | `0.433` |
| `beta_F` | `118` | `115` | `0.975` |
| `beta_G` | `106` | `104` | `0.981` |
| `beta_H` | `11` | `11` | `1.000` |

Interpretation:

- Current candidate generation is a hard ceiling.
- No classifier can recover the roughly 34 percent of Beta RPF days where no good candidate is generated.
- The ceiling is especially bad for `beta_D`, `beta_B`, and `beta_E`.
- `beta_F` and `beta_G` are not mainly candidate-recall problems; they are more likely scoring/precision/decoding problems.

This strongly suggests that the next method should first diagnose and improve candidate generation, not just train a stronger classifier.

## m9 Hybrid Results

Notebook:

- `publication/2_journal_article/notebooks/99_Misc/03_m9_hybrid_development.ipynb`

Outputs:

- `publication/2_journal_article/notebooks/99_Misc/outputs/03_m9_hybrid_development/`

Current m9 Beta transfer result:

| Method | Level | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| current `m9_hybrid` | day | `0.411` | `0.677` | `0.511` |
| current `m9_hybrid` | interval | `0.517` | `0.642` | `0.573` |

Interpretation:

- `m9_hybrid` improved recall and interval F1 relative to Notebook 2 `m8_xgb`.
- It is still not good enough at day level because precision is too low.
- The model produces too many false positives.

## m9 v1b Precision Gate

Notebook:

- `publication/2_journal_article/notebooks/99_Misc/04_m9_v1b_precision_gate_search.ipynb`

This tried simple precision-focused gates on top of m9.

Best currently noted gate:

- Candidate threshold around `0.2`.
- Minimum `solar_p95_inside` around `2.0`.
- Duration less than or equal to `8` hours.

Result:

| Method | Level | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `v1b_existing_gate` | day | `0.436` | `0.657` | `0.524` |
| `v1b_existing_gate` | interval | `0.533` | `0.630` | `0.578` |

Interpretation:

- Simple gates reduce false positives a little.
- The improvement is real but modest.
- It does not solve the fundamental problem.

## Pseudo-Load Experiment

Notebook:

- `publication/2_journal_article/notebooks/99_Misc/05_m9_pseudoload_iterative_search.ipynb`

Outputs:

- `publication/2_journal_article/notebooks/99_Misc/outputs/05_m9_pseudoload_iterative_search/`

Motivation:

During wrong-sign RPF, define:

```text
pseudo_load = solar_MW - net_load_MW
```

If the raw positive net load is actually a sign error, then `solar_MW - net_load_MW` may approximate underlying demand and may be relatively stable during the wrong-sign window.

The search tried 11 pseudo-load variants:

- pseudo-load features on original m7 candidates;
- pseudo-load stability gates;
- combined v1b and pseudo-load gates;
- pseudo-load score reranking;
- pseudo-load hard negative training;
- boundary-expanded candidates;
- pseudo-load constant-segment candidates;
- m7 plus pseudo-load segment candidates.

This was explicitly Beta-guided exploratory work, not publication-ready validation.

Best pseudo-load result:

| Variant | Level | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `v03_pseudo_xgb_m7_precision90` | day | `0.466` | `0.677` | `0.552` |
| `v03_pseudo_xgb_m7_precision90` | interval | `0.550` | `0.663` | `0.601` |

Comparison:

| Method | Day Precision | Day Recall | Day F1 | Interval F1 |
|---|---:|---:|---:|---:|
| current `m9_hybrid` | `0.411` | `0.677` | `0.511` | `0.573` |
| `v1b_existing_gate` | `0.436` | `0.657` | `0.524` | `0.578` |
| best pseudo-load variant | `0.466` | `0.677` | `0.552` | `0.601` |

Important interpretation:

- Pseudo-load helped mostly by reducing false positives.
- It did not improve recall because it still depended on the same m7-style candidates.
- Boundary-expanded candidates and pseudo-load segment candidates performed worse because they created many extra false positives.
- Pseudo-load currently looks useful as an additional feature/ranking signal, not as a standalone candidate generator.

## Current Best Result Is Still Not Good Enough

The current best exploratory Beta day F1 is around `0.552` for the pseudo-load variant.

That is not good enough for the intended final method.

The likely reasons are:

1. Candidate generation misses too many true Beta RPF days.
2. Some non-RPF real days look RPF-like under current features.
3. Current Alpha-to-Beta transfer is affected by site morphology and scale differences.
4. Existing Beta labels may include ambiguous or noisy manual decisions.
5. The current methods may not fully encode the physics/shape constraints that a human reviewer uses visually.

## Observed Error Patterns And Hypotheses

From diagnostics and interactive examples:

- `beta_B` and `beta_G` have many visually plausible RPF cases, but their raw morphology differs from Alpha.
- `beta_D` is a major candidate-generation problem. Current m7-style candidates hit only about `15.6%` of its labelled RPF days.
- `beta_F` and `beta_G` have high candidate recall, so their remaining errors are more likely scoring/decoding problems.
- Some false positives are low-load days or days where net load and solar shape coincidentally look RPF-like.
- Some false negatives are clear-sky high-solar days where the net-load shape visually appears wrong-sign, but the current candidate generator or scorer fails.
- The first-derivative co-movement idea is probably important:
  - On wrong-sign RPF intervals, solar and raw positive net load may move in the same direction.
  - On normal days, solar increasing should usually make net load decrease, and solar decreasing should usually make net load increase.
- One-contiguous-window structure is likely important:
  - The user suspects most true RPF days have one continuous RPF interval.
  - Notebook 1 characterisation work was updated to analyse number of contiguous RPF events per RPF day.

## Open Obstacles

### 1. Candidate Generation Ceiling

The current m7-style generator only achieves about:

- `0.706` Alpha candidate recall;
- `0.659` Beta candidate recall.

This is too low. Even perfect scoring cannot recover days where no candidate exists.

Open question:

- What deterministic or ML-assisted candidate generator can raise candidate recall while not exploding false positives?

### 2. False Positive Control

When candidate generation is expanded, false positives increase rapidly.

Open question:

- How can we generate more candidates for missed true RPF days without drowning the scorer in non-RPF lookalikes?

### 3. Alpha-To-Beta Transfer

Site normalisation helped in Beta-guided diagnostics, but site-specific thresholding is not desirable for new/random sites.

Open question:

- What normalisation can be learned or applied using Alpha only and still generalise to Beta?

### 4. Event Structure

The target phenomenon is not an arbitrary set of intervals. It is usually one continuous wrong-sign segment in the daytime.

Open question:

- Should the model be formulated as structured prediction or candidate-window ranking rather than interval classification?

### 5. Label Ambiguity

Some days may be ambiguous even for humans. Current Beta labels are also not yet the final reviewed oracle.

Open question:

- Should the method explicitly handle uncertain/borderline labels or margin-based evaluation?

### 6. Objective Function

Day-level F1 is primary, interval F1 secondary, and false positives are worse than false negatives.

Open question:

- Should threshold selection optimise F1 directly, or optimise F1 subject to a minimum precision constraint?

## Current Working Hypothesis

The best next method is likely not:

- more generic hyperparameter tuning;
- pure interval classification;
- pure site-specific calibration;
- pseudo-load-only segmentation.

The best next method is probably:

1. A **higher-recall candidate generator** that uses solar shape, net-load morphology, derivative co-movement, and perhaps pseudo-load stability.
2. A **candidate-window ML scorer**, likely XGBoost first, trained on Alpha candidate windows.
3. A **structured one-window decoder** that selects zero or one window per site-day.
4. Alpha-only threshold selection, possibly with a precision guardrail.
5. Beta evaluation only after the method is fixed.

## Suggested Next Analyses

### A. Candidate Oracle Analysis

For every Beta FN, split it into:

- no candidate generated;
- candidate generated but scored too low;
- candidate generated and scored high but rejected by gate/threshold.

For every Beta FP, split it into:

- plausible RPF-like day;
- low-solar or low-load false positive;
- co-movement mismatch;
- pseudo-load instability;
- possible label ambiguity.

This will tell us whether to work on candidate generation, scoring, or decoding.

### B. Design A Better High-Recall Candidate Generator

Possible ideas:

- generate candidate windows around high-solar clear-sky periods;
- generate candidates where raw net load and solar have high same-sign derivative agreement;
- generate candidates where pseudo-load becomes unusually stable;
- generate candidates where raw net load forms an N-shaped positive bump during a bell-shaped solar day;
- generate candidates using bounded dynamic programming over start/end times, but with pruning so we do not enumerate every pair blindly.

Important guardrail:

- The candidate generator should be high recall but not unbounded.
- Expanding candidates naively has already produced many false positives.

### C. Candidate Scorer Features

Useful feature families seen so far:

- timing: start, end, midpoint, duration, month, weekday/weekend;
- solar: peak/p95, bell-shape score;
- net load: p05/p95/range, N-shape score, positive bump shape;
- co-movement: derivative same-sign fraction, solar-net correlation, ramp-up/ramp-down agreement;
- pseudo-load: inside-window stability, roughness, range, mean absolute slope, stability relative to full daytime;
- context: whether candidate contains daily solar peak, distance to solar peak, missingness.

The feature set should remain lean. Too many features may overfit Alpha and obscure the real issue.

### D. Consider Structured Or Ranking Formulations

Possible formulations:

- binary candidate classification;
- candidate ranking per day;
- learning-to-rank where the best candidate on a labelled RPF day should outrank all other candidates;
- structured loss that rewards overlap/IoU with the true window;
- two-stage model:
  - first detect candidate RPF day;
  - then choose window.

Current simple candidate classification may be too blunt because each RPF day has at most one true positive candidate but many negatives.

### E. Validate Against Human Visual Logic

The user believes many FP/FN cases are obvious visually:

- FN: clear-sky solar plus N-shaped raw net load should probably be detected.
- FP: solar and net load moving in opposite directions should usually be rejected.

Any proposed model should be checked against this visual logic.

## What Would Be Most Helpful From A Reviewer

Please suggest:

1. A stronger candidate-generation strategy that raises candidate recall on sites like `beta_B`, `beta_D`, and `beta_E`.
2. A way to preserve precision when candidate generation expands.
3. Whether this should be formulated as classification, ranking, structured prediction, or another ML setup.
4. How to incorporate physics/domain constraints without making a brittle hand-built rule system.
5. A validation protocol that is fair, since many of the current misc results are Beta-guided exploratory experiments.
6. Whether pseudo-load stability is a useful signal and how to use it better.
7. How to handle site/domain shift without per-site Beta calibration.

## Key Artifacts To Share

Primary notebooks:

- `publication/2_journal_article/notebooks/02_correction_validation.ipynb`
- `publication/2_journal_article/notebooks/99_Misc/01_beta_xgb1_diagnostics.ipynb`
- `publication/2_journal_article/notebooks/99_Misc/02_beta_m8_xgb_threshold_and_variant_search.ipynb`
- `publication/2_journal_article/notebooks/99_Misc/03_m9_hybrid_development.ipynb`
- `publication/2_journal_article/notebooks/99_Misc/04_m9_v1b_precision_gate_search.ipynb`
- `publication/2_journal_article/notebooks/99_Misc/05_m9_pseudoload_iterative_search.ipynb`

Important output folders:

- `publication/2_journal_article/outputs/metrics/02_correction_validation/`
- `publication/2_journal_article/notebooks/99_Misc/outputs/01_beta_xgb1_diagnostics/`
- `publication/2_journal_article/notebooks/99_Misc/outputs/02_beta_m8_xgb_threshold_and_variant_search/`
- `publication/2_journal_article/notebooks/99_Misc/outputs/03_m9_hybrid_development/`
- `publication/2_journal_article/notebooks/99_Misc/outputs/04_m9_v1b_precision_gate_search/`
- `publication/2_journal_article/notebooks/99_Misc/outputs/05_m9_pseudoload_iterative_search/`

Especially useful files:

- Notebook 2 Beta transfer metrics:
  - `publication/2_journal_article/outputs/metrics/02_correction_validation/01_correction_metrics.csv`
- m8 threshold/normalisation leaderboard:
  - `publication/2_journal_article/notebooks/99_Misc/outputs/02_beta_m8_xgb_threshold_and_variant_search/csv/08_ranked_leaderboard.csv`
- m9 candidate-day summary:
  - `publication/2_journal_article/notebooks/99_Misc/outputs/03_m9_hybrid_development/intermediate/03_alpha_candidate_day_summary.csv`
- m9 Beta decoded days:
  - `publication/2_journal_article/notebooks/99_Misc/outputs/03_m9_hybrid_development/intermediate/07_beta_decoded_days.csv`
- pseudo-load variant leaderboard:
  - `publication/2_journal_article/notebooks/99_Misc/outputs/05_m9_pseudoload_iterative_search/metrics/02_variant_leaderboard.csv`
- pseudo-load interactive examples:
  - `publication/2_journal_article/notebooks/99_Misc/outputs/05_m9_pseudoload_iterative_search/html/`

## Bottom Line

Current approaches have improved from the original Beta transfer result, but not enough.

The most important insight is that current candidate generation is probably limiting the achievable result. The best pseudo-load variant improved precision and F1, but it did not improve recall because it still relied on the same m7-style candidates.

Therefore, the next breakthrough likely requires a better high-recall candidate-generation method plus a structured candidate-ranking/decoding approach, not just more threshold tuning.


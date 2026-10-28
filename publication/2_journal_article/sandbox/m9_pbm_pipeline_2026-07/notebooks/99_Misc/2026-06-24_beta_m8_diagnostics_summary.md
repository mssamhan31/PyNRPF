# 2026-06-24 Beta m8_xgb Diagnostics Summary

## Purpose

Today we paused the main Notebook 2 workflow and used misc diagnostic notebooks to understand why `m8_xgb` transfers less well from Dataset Alpha to Dataset Beta.

The goal was diagnostic and exploratory only. We did not change the main Notebook 2 algorithm, config, or publication workflow.

## What We Investigated

- Compared Alpha and Beta day-level `m8_xgb` behaviour, separating:
  - XGB1 candidate-day selection;
  - final day result after XGB2 interval filtering.
- Inspected site-level Beta errors, especially `beta_B`, `beta_G`, `beta_F`, and `beta_D`.
- Created an interactive diagnostic notebook with Plotly examples for TP, TN, FP, and FN days.
- Ran a bounded threshold and variant search notebook:
  - current raw-feature thresholds;
  - global threshold calibration;
  - site-specific threshold calibration;
  - site-scale normalisation variants;
  - a small light hyperparameter search.

## Key Findings

- The original current-config Beta `m8_xgb` day-level F1 is about `0.49`.
- Raw-feature threshold calibration improves the result only modestly:
  - global threshold calibration reaches about `0.52` mean held-out Beta day F1;
  - site-specific thresholds on raw features reach about `0.60`.
- Site-scale normalisation helps more:
  - best split-validated result was `site_joint_p95` plus site-specific thresholds;
  - mean held-out Beta day F1 was about `0.64`.
- The all-Beta optimistic upper-bound for the best normalised variants is similar, around `0.64`, so the split result is not just a single lucky fold.

## Interpretation

Normalisation and calibration clearly help, but they do not fully solve the Beta transfer problem.

The failure is site-specific:

- `beta_B` and `beta_G` contain many labelled RPF days that stay relatively positive in raw net load, so the original Alpha-trained XGB1 often does not identify them as candidate days.
- `beta_D` has many low-load non-RPF days, creating false positives because normal low-load behaviour can look RPF-like.
- `beta_F` is a different failure mode: XGB1 often finds the day, but XGB2 interval filtering can still be too strict.

This suggests the current `m8_xgb` design is limited by representation and site morphology, not just by hyperparameters.

## Current Conclusion

We should not treat the current `m8_xgb` as the final real-site method.

The most promising next direction is a new `m9` model or method family that keeps the useful parts of XGB but changes the modelling idea:

- site-normalised features;
- site-calibrated thresholds or calibration layers;
- shape-aware features around the solar and net-load profile;
- constraints or post-processing that reflect expected RPF event structure, such as the suspicion that most RPF days contain one contiguous RPF interval.

The diagnostic result supports the paper story that real-site transfer exposes a scale/domain shift, and that a future `m9` should explicitly address that shift.

## Important Caveat

The misc diagnostic outputs are exploratory and are not publication-ready validation results. Any future `m9` result should be promoted into the main journal workflow only after we agree on the method design and validation protocol.

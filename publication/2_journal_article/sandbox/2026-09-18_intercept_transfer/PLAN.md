# Intercept transfer: can M9 be ported to a new population with one small reviewed sample?

Date: 18 September 2026. Sandbox only; reads committed Phase 3 scores, labels and energies; nothing frozen is touched.

## Question

With the scorer frozen and the calibration slope frozen from a *different* population, how many reviewed days of the new population are needed to estimate the intercept so that the operating point at c = 0.7 is recovered? Is one intercept per population enough, or is a per-station intercept needed? Is the untouched foreign calibration safe (no over-correction) before any review?

## Design

- Directions: source Alpha → target Beta, and source Beta → target Alpha. The slope is fitted on all stations of the source cohort (headline, input_ok days); the source cohort shares no station with the target, so nothing leaks.
- Target held out one station at a time (the Phase 3 folds of the target cohort).
- Adaptation levels, with the slope frozen and only the intercept fitted by maximum likelihood:
  - population: k labelled days drawn from the *other* stations of the target cohort (leakage-safe stand-in for "calibrate once per deployment"), applied to the held-out station's days;
  - station: k labelled days drawn from the held-out station itself, evaluated on that station's remaining days.
- Sampling designs: random; and the review design (a third of the sample from the largest provisional corrections by p × candidate MWh, a third nearest the provisional threshold p = c, a third random), where the provisional p uses the source calibration untouched.
- k ∈ {9, 18, 27, 54, 108}; 20 seeded draws per cell; labels are headline days only.
- Benchmarks: k = 0 (the source pair untouched, the safe-default check); all labels (intercept from every other station of the target cohort); R0 (frozen within-cohort pair, slope and intercept).
- A sample with only one class cannot fit an intercept; the default intercept is kept and the event counted.
- Metrics pooled over held-out target stations and reported as the median and 5–95 percentile range over draws: Energy IoU, energy precision (gate 0.90 on Beta), sure-day recall, day precision, false corrections; per-station spread at k = 27; the fitted intercept versus the all-labels intercept.

## Fixed in advance

Success reading: the population-level intercept from 27 review-design days recovers Beta energy precision ≥ 0.90 and Energy IoU ≥ 0.875 in the median draw, and Alpha Energy IoU ≥ 0.849 in the reverse direction; the k = 0 source pair over-corrects on neither cohort. Station-level adaptation is reported to show whether it adds anything beyond the population level.

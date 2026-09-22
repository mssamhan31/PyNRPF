# How the paper judges a method

## Two questions, two levels

Fixing a wrong sign is two questions: is this day wrong, and which slots flip. The paper
scores both and never merges them.

At the day level the outcomes are the usual four (true and false positive, true and false
negative), summarised as site-day precision, recall and F1. A false positive plants a new
error in clean data, so precision is the safety number and recall is what review effort buys.

At the interval level the score is energy. Flipping a fifteen-minute reading of `y` MW
changes the recorded energy by `2·y·0.25` MWh. With `P` the slots a method flips and `R`
the slots the reference flips:

    Energy IoU        = Σ_{P∩R} E_t / Σ_{P∪R} E_t         the main metric
    energy precision  = Σ_{P∩R} E_t / Σ_{P} E_t           the safety gate, at least 0.90 on Beta sure days

Energies are pooled over all site-days before dividing, so a station with more energy at
stake weighs more; per-station tables and a macro (unweighted) mean sit beside the pooled
numbers.

## Station held out, fitted on real errors only

Each of the 18 stations is held out once. The two previous methods and M9 score the held-out
station on the same complete site-days with the same metrics. Anything fitted is fitted on
the other Beta stations only: M8 (the XGBoost baseline) is retrained per fold, and M9's two
calibration numbers are refitted per fold; an Alpha station is scored with the fit on all
eight Beta stations. Alpha, whose truth is exact but manufactured, is never fitted on and
serves as the check that corrected hours are the right hours. Beta sure days are the
headline population; unsure days appear only in a sensitivity table.

Uncertainty on the headline numbers comes from a station bootstrap (1,000 draws of stations
with replacement, 2.5 and 97.5 percentiles).

## The gate and the operating point

A method is trusted when its energy precision on Beta sure days reaches 0.90; the gate is
checked on the pooled numbers and the outcome is recorded in `results/04_metrics/gate.json`.
The control `c` then trades review effort against silent errors: the operating-point study
turns a precision target (0.90 to 0.99) into a `c` through the calibrated probabilities on
the other stations, and reports the review days and recall each target costs.

## The forecasting case study

Gamma asks what the correction is worth downstream: seven-day-ahead net-load forecasts
trained on raw, M9-corrected and manually corrected history for one station, compared on
the September 2024 test month.

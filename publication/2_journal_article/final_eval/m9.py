"""M9 revision 2 through the frozen m9_dev scorer, calibrated within cohort per fold.

Inputs:  the complete-day interval frame of a cohort, the folds, the settings.
Outputs: a cached score table (one row per site-day, label-free), the M9 site-day
         prediction table (probability, outcome, window) and the calibration fits per
         fold, plus the M9 interval prediction table in the common schema.
Key steps: the sigma floor is the smallest non-zero overnight step in the cohort
         (label-free); every site-day is scored once with the frozen settings; for each
         fold the two calibration coefficients ``cal_intercept`` and ``cal_slope`` are
         fitted by logistic regression on the signed-log evidence of the other stations
         of the same cohort (headline-confidence days only) and applied to the held-out
         station with the public control c.

The scorer is imported from ``m9_dev``; this module adds no scoring logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import AUTO_CORRECT, INTERVAL_COLUMNS, UNCERTAIN, finish_interval_table
from .config import Settings, import_m9_dev
from .data import siteday_arrays
from .folds import Fold

SCORE_COLUMNS = ["input_ok", "n_admissible", "sigma", "best_start", "best_end", "best_score",
                 "runner_start", "runner_end", "runner_score", "margin_window", "r_best"]


def sigma_floor(intervals: pd.DataFrame, settings: Settings) -> float:
    """Smallest non-zero absolute overnight step of the reconstructed load across the cohort.

    Identical to ``m9_dev.m9_eval.sigma_floor``: label-free, computed on complete days.
    """
    ms, _ = import_m9_dev(settings)
    steps = []
    for _, _, _, y, s, _ in siteday_arrays(intervals):
        steps.append(np.abs(np.diff((y + s)[ms.OVERNIGHT])))
    steps = np.concatenate(steps)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    return float(steps.min()) if steps.size else 1e-3


def score_cohort(intervals: pd.DataFrame, index: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Score every complete site-day of a cohort with the frozen M9 settings (no labels used)."""
    ms, _ = import_m9_dev(settings)
    m9 = settings["m9"]
    floor = sigma_floor(intervals, settings)
    rows = []
    for cohort, station, date, y, s, _ in siteday_arrays(intervals):
        r = ms.score_siteday(y, s, floor, variant=m9["variant"], p_exp=float(m9["p_exp"]), sigma=None,
                             scale=m9["sigma_mode"], stat=m9["stat"], missing=m9["missing"], edges=m9["edges"])
        rows.append(dict(cohort=cohort, station=station, date=date, sigma_floor=floor, **{k: r[k] for k in SCORE_COLUMNS}))
    scores = pd.DataFrame(rows)
    labels = index.loc[index["complete"], ["cohort", "station", "date", "confidence", "headline", "rpf", "true_start", "true_end", "true_slots"]]
    return scores.merge(labels, on=["cohort", "station", "date"], how="left", validate="one_to_one")


def calibrate_fold(scores: pd.DataFrame, fold: Fold, settings: Settings) -> tuple[pd.DataFrame, dict]:
    """Held-out probabilities and outcomes for one fold; the fit uses other stations only."""
    ms, _ = import_m9_dev(settings)
    c = float(settings["m9"]["c"])
    eligible = scores["input_ok"] & scores["headline"]
    train = scores[scores["station"].isin(fold.m9_calibration) & eligible]
    if fold.held_out in set(train["station"]):
        raise ValueError(f"Fold {fold.fold_id}: held-out station inside the calibration set.")
    test = scores[scores["station"] == fold.held_out].copy()
    cal = ms.Calibrator().fit(train["r_best"].to_numpy(float), train["rpf"].to_numpy(int))
    p = np.where(test["input_ok"], cal.probability(test["r_best"].fillna(0.0).to_numpy(float)), np.nan)
    test["p"] = p
    test["outcome"] = [ms.decide(v, c) if np.isfinite(v) else UNCERTAIN for v in p]
    # A day cannot be corrected without a window; the frozen harness keeps this guard too.
    no_window = test["best_start"] < 0
    test.loc[no_window, "outcome"] = test.loc[no_window, "outcome"].replace(AUTO_CORRECT, UNCERTAIN)
    test["fold_id"] = fold.fold_id
    test["method"] = "m9"
    fit = dict(fold_id=fold.fold_id, cohort=fold.cohort, held_out=fold.held_out, n_train=len(train),
               n_train_rpf=int(train["rpf"].sum()), cal_intercept=cal.alpha, cal_slope=cal.beta,
               raw_threshold_correct=cal.raw_threshold(c), raw_threshold_keep=cal.raw_threshold(1 - c))
    return test, fit


def predict_cohort(scores: pd.DataFrame, folds: list[Fold], settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Site-day predictions for every held-out station of a cohort, and the fits table."""
    cohort = scores["cohort"].iloc[0]
    parts, fits = [], []
    for fold in folds:
        if fold.cohort != cohort:
            continue
        test, fit = calibrate_fold(scores, fold, settings)
        parts.append(test)
        fits.append(fit)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(fits)


def interval_table(site_days: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Expand M9 site-day decisions to the common interval schema.

    pred_interval marks the best window (the candidate) whatever the outcome, so the
    coverage analysis can re-decide days at other values of c; the applied correction
    is derived from the outcome downstream.
    """
    keys = site_days[["cohort", "station", "date", "fold_id", "best_start", "best_end", "p", "outcome"]]
    table = intervals[["cohort", "station", "date", "slot", "ts"]].merge(keys, on=["cohort", "station", "date"], how="inner")
    in_window = (table["best_start"] >= 0) & (table["slot"] >= table["best_start"]) & (table["slot"] <= table["best_end"])
    table["pred_interval"] = in_window.to_numpy()
    table["pred_day"] = (table["outcome"] == AUTO_CORRECT).to_numpy()
    table["prob_day"] = table["p"].to_numpy(float)
    table["prob_interval"] = np.nan
    out = []
    for fold_id, g in table.groupby("fold_id", sort=True):
        out.append(finish_interval_table(g, "m9", fold_id))
    return pd.concat(out, ignore_index=True)[INTERVAL_COLUMNS]

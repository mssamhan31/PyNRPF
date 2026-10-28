"""M9 through the ``pynrpf`` package, calibrated per fold on the fold's calibration stations.

Inputs:  the complete-day interval frame of a cohort, the folds, the settings.
Outputs: a score table (one row per site-day, label-free), the M9 site-day prediction
         table (probability, outcome, window) and the calibration fit per fold, plus
         the M9 interval prediction table in the common schema.
Key steps: the evidence floor ``phi`` is the smallest non-zero overnight step of demand
         in the cohort (label-free, ``sigma_floor``); every site-day is scored once
         with ``pynrpf.m9.score_siteday``; for each fold the two calibration numbers
         are fitted with ``pynrpf.fit_calibration`` on the signed-log evidence of the
         fold's calibration stations (headline-confidence days only) and applied to
         the held-out station with the control ``c``. The calibration stations come
         from the fold manifest: Beta stations under ``beta_only``, so the fit draws
         from a pool of every cohort's scores and the held-out cohort's table supplies
         only the test rows.

This module adds no scoring logic: the method is the package's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pynrpf import Score, fit_calibration
from pynrpf.m9 import AUTO_CORRECT, UNCERTAIN, decide, score_siteday

from .config import Settings
from .data import INTERVAL_COLUMNS, finish_interval_table, siteday_arrays
from .folds import Fold

OVERNIGHT = slice(0, 24)   # 00:00 to 06:00, the slots the floor is measured on (no solar, no wrong sign)

SCORE_COLUMNS = ["input_ok", "n_admissible", "best_start", "best_end", "best_score",
                 "runner_start", "runner_end", "runner_score", "r_best"]


def sigma_floor(intervals: pd.DataFrame) -> float:
    """Smallest non-zero absolute overnight step of the reconstructed demand across the cohort, MW.

    This is ``phi`` of the package (``pynrpf.m9.RELEASE_PHI`` is the Beta value of the
    reference run): label-free, computed on complete days only.
    """
    steps = []
    for _, _, _, y, s, _ in siteday_arrays(intervals):
        steps.append(np.abs(np.diff((y + s)[OVERNIGHT])))
    steps = np.concatenate(steps)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    return float(steps.min()) if steps.size else 1e-3


def score_row(score: Score) -> dict:
    """The ``SCORE_COLUMNS`` of one site-day from the package's ``Score``."""
    ok = score.input_ok
    return dict(
        input_ok=ok, n_admissible=score.n_admissible,
        best_start=score.winner.start, best_end=score.winner.end,
        best_score=score.winner.evidence if ok else np.nan,
        runner_start=score.runner_up.start, runner_end=score.runner_up.end,
        runner_score=score.runner_up.evidence if ok else np.nan,
        r_best=score.evidence,
    )


def score_cohort(intervals: pd.DataFrame, index: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Score every complete site-day of a cohort with the released method (no labels used).

    Args:
        intervals: complete-day interval frame of the cohort (``data.load_population``).
        index: the cohort's site-day index (labels are attached for the evaluation only).
        settings: the evaluation settings (unused by the scorer; kept for symmetry with the baselines).

    Returns:
        One row per site-day: ``sigma_floor`` (phi, MW), the ``SCORE_COLUMNS`` and the labels.
    """
    del settings
    floor = sigma_floor(intervals)
    rows = []
    for cohort, station, date, y, s, _ in siteday_arrays(intervals):
        score = score_siteday(y, s, floor)
        rows.append(dict(cohort=cohort, station=station, date=date, sigma_floor=floor, **score_row(score)))
    scores = pd.DataFrame(rows)
    labels = index.loc[index["complete"], ["cohort", "station", "date", "confidence", "headline", "rpf",
                                          "true_start", "true_end", "true_slots"]]
    return scores.merge(labels, on=["cohort", "station", "date"], how="left", validate="one_to_one")


def calibrate_fold(scores: pd.DataFrame, fold: Fold, settings: Settings,
                   calibration_pool: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Held-out probabilities and outcomes for one fold; the fit uses the calibration stations only.

    Args:
        scores: score table of the held-out station's cohort (``score_cohort``).
        fold: the fold; ``m9_calibration`` names the stations the fit may read labels from.
        settings: the evaluation settings (``m9.c`` is the control).
        calibration_pool: score rows of every cohort, from which the calibration stations
            are drawn; defaults to ``scores``. Under ``beta_only`` an Alpha fold calibrates
            on Beta stations, so the caller passes the pool of both cohorts.

    Returns:
        (test, fit): the held-out station's site-days with probability ``p`` and
        ``outcome`` added, and the fit record (the two coefficients and the evidence
        thresholds that ``c`` implies).
    """
    c = float(settings["m9"]["c"])
    pool = scores if calibration_pool is None else calibration_pool
    eligible = pool["input_ok"] & pool["headline"]
    train = pool[pool["station"].isin(fold.m9_calibration) & eligible]
    if fold.held_out in set(train["station"]):
        raise ValueError(f"Fold {fold.fold_id}: held-out station inside the calibration set.")
    if train.empty:
        raise ValueError(f"Fold {fold.fold_id}: none of its calibration stations {fold.m9_calibration} is present.")
    test = scores[scores["station"] == fold.held_out].copy()
    cal = fit_calibration(train["r_best"].to_numpy(float), train["rpf"].to_numpy(int), slope=None)
    p = np.where(test["input_ok"], cal.probability(test["r_best"].fillna(0.0).to_numpy(float)), np.nan)
    test["p"] = p
    test["outcome"] = [decide(v, c) for v in p]
    # A day cannot be corrected without a window; the package's run keeps this guard too.
    no_window = test["best_start"] < 0
    test.loc[no_window, "outcome"] = test.loc[no_window, "outcome"].replace(AUTO_CORRECT, UNCERTAIN)
    test["fold_id"] = fold.fold_id
    test["method"] = "m9"
    fit = dict(fold_id=fold.fold_id, cohort=fold.cohort, held_out=fold.held_out, n_train=len(train),
               n_train_rpf=int(train["rpf"].sum()), cal_intercept=cal.intercept, cal_slope=cal.slope,
               raw_threshold_correct=cal.evidence_at(c), raw_threshold_keep=cal.evidence_at(1 - c))
    return test, fit


def predict_cohort(scores: pd.DataFrame, folds: list[Fold], settings: Settings,
                   calibration_pool: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Site-day predictions for every held-out station of a cohort, and the fits table.

    Args:
        scores: score table of one cohort.
        folds: all folds; those of the cohort are predicted.
        settings: the evaluation settings.
        calibration_pool: score rows of every cohort (see ``calibrate_fold``).

    Returns:
        (site_days, fits) concatenated over the cohort's folds.
    """
    cohort = scores["cohort"].iloc[0]
    parts, fits = [], []
    for fold in folds:
        if fold.cohort != cohort:
            continue
        test, fit = calibrate_fold(scores, fold, settings, calibration_pool)
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
    table = intervals[["cohort", "station", "date", "slot", "ts"]].merge(keys, on=["cohort", "station", "date"],
                                                                         how="inner")
    has_window = table["best_start"] >= 0
    in_window = has_window & (table["slot"] >= table["best_start"]) & (table["slot"] <= table["best_end"])
    table["pred_interval"] = in_window.to_numpy()
    table["pred_day"] = (table["outcome"] == AUTO_CORRECT).to_numpy()
    table["prob_day"] = table["p"].to_numpy(float)
    table["prob_interval"] = np.nan
    out = []
    for fold_id, g in table.groupby("fold_id", sort=True):
        out.append(finish_interval_table(g, "m9", fold_id))
    return pd.concat(out, ignore_index=True)[INTERVAL_COLUMNS]

"""The site-day decision-and-impact table: energy terms, window agreement, minimum demand.

Inputs:  an interval table with outcomes and applied corrections (from outcomes.py),
         joined with the recorded net load ``y`` and the labelled slots ``truth``.
Outputs: one row per (method, cohort, station, date) with the locked energy terms,
         slot-level confusion counts, window IoU, boundary errors, and the site-day
         minimum-demand impact of the method and of the reference correction.
Key steps: the correction energy of a slot is 2·y·0.25 MWh (flipping the sign changes
         the reading by 2y); proposed = applied slots, required = labelled slots,
         correct = their intersection; missing readings carry no energy. The candidate
         energy (the method's proposed slots whatever the outcome) is kept so the
         coverage analysis can re-decide days at other confidence levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import KEY

HOURS_PER_SLOT = 0.25
SITE_DAY_COLUMNS = ["method", "cohort", "station", "fold_id", "date", "confidence", "headline", "rpf",
                    "outcome", "prob_day", "n_pred_slots", "pred_start", "pred_end",
                    "true_start", "true_end", "true_slots",
                    "candidate_mwh", "candidate_correct_mwh", "proposed_mwh", "required_mwh", "correct_mwh", "false_mwh",
                    "slot_tp", "slot_fp", "slot_fn", "window_iou", "start_error", "end_error",
                    "raw_min_mw", "applied_min_mw", "min_change_mw", "reference_min_mw", "reference_min_change_mw"]


def slot_energy(y: np.ndarray) -> np.ndarray:
    """Correction energy per slot if that slot were flipped, MWh. Missing readings carry none."""
    return 2.0 * np.nan_to_num(np.asarray(y, dtype=float), nan=0.0) * HOURS_PER_SLOT


def energy_terms(y: np.ndarray, truth: np.ndarray, proposed: np.ndarray) -> tuple[float, float, float]:
    """(proposed, required, correctly corrected) MWh for arbitrary slot masks.

    Generalises ``m9_dev.m9_metrics.energy_terms`` from one contiguous window to any
    set of slots, so M7 and M8 flags that are not contiguous are scored without change.
    """
    e = slot_energy(y)
    proposed = np.asarray(proposed, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    return float(e[proposed].sum()), float(e[truth].sum()), float(e[proposed & truth].sum())


def minimum_demand(y: np.ndarray, flipped: np.ndarray) -> tuple[float, float, float]:
    """(raw minimum, minimum after flipping the marked slots, signed change) in MW.

    NaN readings are ignored; a day with no finite reading returns NaN throughout.
    """
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).any():
        return np.nan, np.nan, np.nan
    raw = float(np.nanmin(y))
    applied = float(np.nanmin(np.where(np.asarray(flipped, dtype=bool), -y, y)))
    return raw, applied, applied - raw


def _span(mask: np.ndarray) -> tuple[int, int, int]:
    idx = np.flatnonzero(mask)
    return (int(idx[0]), int(idx[-1]), int(idx.size)) if idx.size else (-1, -1, 0)


def siteday_row(y: np.ndarray, truth: np.ndarray, candidate: np.ndarray, applied: np.ndarray) -> dict:
    """All impact quantities for one site-day from its slot masks."""
    cand_mwh, required, cand_correct = energy_terms(y, truth, candidate)
    proposed, _, correct = energy_terms(y, truth, applied)
    pred_start, pred_end, n_pred = _span(candidate)
    true_start, true_end, n_true = _span(truth)
    tp = int((applied & truth).sum())
    fp = int((applied & ~truth).sum())
    fn = int((~applied & truth).sum())
    union = int((applied | truth).sum())
    raw_min, applied_min, change = minimum_demand(y, applied)
    _, ref_min, ref_change = minimum_demand(y, truth)
    return dict(
        n_pred_slots=n_pred, pred_start=pred_start, pred_end=pred_end,
        candidate_mwh=cand_mwh, candidate_correct_mwh=cand_correct,
        proposed_mwh=proposed, required_mwh=required, correct_mwh=correct, false_mwh=proposed - correct,
        slot_tp=tp, slot_fp=fp, slot_fn=fn,
        window_iou=(tp / union) if union else np.nan,
        start_error=(pred_start - true_start) if (n_pred and n_true and applied.any()) else np.nan,
        end_error=(pred_end - true_end) if (n_pred and n_true and applied.any()) else np.nan,
        raw_min_mw=raw_min, applied_min_mw=applied_min, min_change_mw=change,
        reference_min_mw=ref_min, reference_min_change_mw=ref_change,
    )


def siteday_table(intervals: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """The site-day decision-and-impact rows for one method's interval table.

    Args:
        intervals: output of ``outcomes.attach_outcomes`` joined with ``y`` and ``truth``.
        index: the cohort site-day index (confidence, headline flag, labelled window).
    """
    rows = []
    for (cohort, station, date), g in intervals.groupby(KEY, sort=True):
        y = g["y"].to_numpy(float)
        truth = g["truth"].to_numpy(bool)
        row = dict(method=g["method"].iloc[0], cohort=cohort, station=station, fold_id=g["fold_id"].iloc[0], date=date,
                   outcome=g["outcome"].iloc[0], prob_day=float(np.nanmax(g["prob_day"].to_numpy(float))) if g["prob_day"].notna().any() else np.nan)
        row.update(siteday_row(y, truth, g["pred_interval"].to_numpy(bool), g["applied"].to_numpy(bool)))
        rows.append(row)
    table = pd.DataFrame(rows)
    labels = index[KEY + ["confidence", "headline", "rpf", "true_start", "true_end", "true_slots"]]
    table = table.merge(labels, on=KEY, how="left", validate="one_to_one")
    return table[SITE_DAY_COLUMNS]

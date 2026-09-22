"""The reference-side terms: outcomes, applied corrections, energy, window agreement, minimum demand.

Inputs:  an interval prediction table in the common schema (any method) joined with
         the recorded net load ``y`` (MW) and the labelled slots ``truth``; for M9, its
         site-day outcomes; the cohort site-day index.
Outputs: the same interval table with ``outcome``, ``applied`` (the slots actually
         flipped) and ``applied_y`` (the net load after the method's correction, MW);
         then one row per (method, cohort, station, date) with the energy terms,
         slot-level confusion counts, window IoU, boundary errors and the site-day
         minimum-demand impact of the method and of the reference correction.
Key steps: M7 and M8 are binary at their native thresholds: a day is AUTO_CORRECT
         when the method flags at least one slot, AUTO_KEEP otherwise. M9 is three-way
         from its calibrated probability and c. Only AUTO_CORRECT days carry applied
         energy; the recorded series is never altered on AUTO_KEEP or UNCERTAIN days.
         The correction energy of a slot is 2·y·0.25 MWh (``pynrpf.impact.slot_energy``):
         proposed = applied slots, required = labelled slots, correct = their
         intersection; missing readings carry no energy. The candidate energy (the
         method's proposed slots whatever the outcome) is kept so the coverage analysis
         can re-decide days at other confidence levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pynrpf.impact import slot_energy
from pynrpf.m9 import AUTO_CORRECT, AUTO_KEEP

from .data import KEY

SITE_DAY_COLUMNS = ["method", "cohort", "station", "fold_id", "date", "confidence", "headline", "rpf",
                    "outcome", "prob_day", "n_pred_slots", "pred_start", "pred_end",
                    "true_start", "true_end", "true_slots",
                    "candidate_mwh", "candidate_correct_mwh", "proposed_mwh", "required_mwh", "correct_mwh",
                    "false_mwh",
                    "slot_tp", "slot_fp", "slot_fn", "window_iou", "start_error", "end_error",
                    "raw_min_mw", "applied_min_mw", "min_change_mw", "reference_min_mw", "reference_min_change_mw"]


# ----------------------------------------------------------------------------- outcomes

def binary_outcomes(intervals: pd.DataFrame) -> pd.Series:
    """Site-day outcome for a binary method: AUTO_CORRECT if any slot is flagged."""
    any_flag = intervals.groupby(KEY, sort=False)["pred_interval"].transform("any")
    return pd.Series(np.where(any_flag, AUTO_CORRECT, AUTO_KEEP), index=intervals.index)


def attach_outcomes(intervals: pd.DataFrame, m9_site_days: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add outcome, applied and applied_y to an interval table.

    Args:
        intervals: common-schema interval predictions joined with the recorded net load
            in column ``y`` (MW).
        m9_site_days: M9 site-day table with ``outcome``; required when the method is m9.
    """
    out = intervals.copy()
    method = out["method"].iloc[0]
    if method == "m9":
        if m9_site_days is None:
            raise ValueError("M9 outcomes need the M9 site-day table.")
        outcome = out[KEY].merge(m9_site_days[KEY + ["outcome"]], on=KEY, how="left", validate="many_to_one")["outcome"]
        out["outcome"] = outcome.to_numpy()
    else:
        out["outcome"] = binary_outcomes(out).to_numpy()
    out["applied"] = out["pred_interval"] & (out["outcome"] == AUTO_CORRECT)
    out["applied_y"] = np.where(out["applied"], -out["y"], out["y"])
    return out


# ----------------------------------------------------------------------------- one site-day

def energy_terms(y: np.ndarray, truth: np.ndarray, proposed: np.ndarray) -> tuple[float, float, float]:
    """(proposed, required, correctly corrected) MWh for arbitrary slot masks.

    Any set of slots is accepted, so M7 and M8 flags that are not contiguous are scored
    without change.
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
    """All reference-side quantities for one site-day from its slot masks."""
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
        intervals: output of ``attach_outcomes`` joined with ``y`` and ``truth``.
        index: the cohort site-day index (confidence, headline flag, labelled window).
    """
    rows = []
    for (cohort, station, date), g in intervals.groupby(KEY, sort=True):
        y = g["y"].to_numpy(float)
        truth = g["truth"].to_numpy(bool)
        prob = g["prob_day"].to_numpy(float)
        row = dict(method=g["method"].iloc[0], cohort=cohort, station=station, fold_id=g["fold_id"].iloc[0], date=date,
                   outcome=g["outcome"].iloc[0],
                   prob_day=float(np.nanmax(prob)) if g["prob_day"].notna().any() else np.nan)
        row.update(siteday_row(y, truth, g["pred_interval"].to_numpy(bool), g["applied"].to_numpy(bool)))
        rows.append(row)
    table = pd.DataFrame(rows)
    labels = index[KEY + ["confidence", "headline", "rpf", "true_start", "true_end", "true_slots"]]
    table = table.merge(labels, on=KEY, how="left", validate="one_to_one")
    return table[SITE_DAY_COLUMNS]

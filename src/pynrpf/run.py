"""Run M9 on a frame of readings: validate, score every site-day, calibrate, decide, report.

    readings -> site-days -> steps 1 to 7 (score) -> step 8 (probability) -> step 9 (outcome)
             -> proposed energy and minimum-demand change -> two tables

Inputs:  a pandas frame of fifteen-minute readings (see ``validate``), the control ``c``,
         the calibration and the evidence floor.
Outputs: ``Result.site_days`` and ``Result.intervals`` (see ``schemas``). Raw readings are
         never altered: the corrected series is a separate column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import impact, schemas, validate
from .m9 import RELEASE_CALIBRATION, RELEASE_PHI, Calibration, Score, score_siteday
from .m9.decision import AUTO_CORRECT, DEFAULT_C, UNCERTAIN, decide
from .m9.winner import NO_CORRECTION


@dataclass
class Result:
    """The two output tables of a run."""

    site_days: pd.DataFrame
    intervals: pd.DataFrame

    def summary(self) -> dict:
        """Counts and energies a caller can log: days per outcome and MWh flipped."""
        d = self.site_days
        corrected = d["outcome"] == AUTO_CORRECT
        return {
            "sites": int(d["site"].nunique()),
            "site_days": int(len(d)),
            "scored": int(d["input_ok"].sum()),
            **{f"days_{o.lower()}": int((d["outcome"] == o).sum()) for o in ("AUTO_CORRECT", "AUTO_KEEP", "UNCERTAIN")},
            "corrected_mwh": float(d.loc[corrected, "proposed_mwh"].sum()),
        }


def run(frame: pd.DataFrame, *, columns: dict | None = None, c: float = DEFAULT_C,
        calibration: Calibration = RELEASE_CALIBRATION, phi: float = RELEASE_PHI) -> Result:
    """Detect and correct a wrong reverse-power-flow sign in every site-day of ``frame``.

    Args:
        frame: fifteen-minute readings; columns per ``columns`` (defaults in ``validate``).
        columns: mapping of ``site``, ``timestamp``, ``net_load``, ``solar`` to column names.
        c: the control; correct at ``p >= c``, keep at ``p <= 1 - c``, review otherwise.
        calibration: the two numbers of step 8; the release pair unless refitted.
        phi: the evidence floor of step 6, MW.

    Returns:
        A ``Result`` with the site-day table and the interval table.
    """
    prepared = validate.prepare(frame, columns)
    day_rows, interval_parts = [], []
    for site, date, rows in validate.site_days(prepared):
        y = rows["y"].to_numpy(float)
        if validate.is_complete(rows):
            score = score_siteday(y, rows["s"].to_numpy(float), phi)
        else:
            score = Score(False, 0, NO_CORRECTION, NO_CORRECTION, float("nan"))
        p = float(calibration.probability(score.evidence)) if score.input_ok else float("nan")
        outcome = decide(p, c)
        if outcome == AUTO_CORRECT and score.winner.is_null:
            outcome = UNCERTAIN                         # a day cannot be corrected without a window
        day_rows.append(_site_day_row(site, date, len(rows), score, p, outcome, y))
        interval_parts.append(_interval_rows(rows, score, outcome))
    if not day_rows:
        return Result(schemas.empty(schemas.SITE_DAYS), schemas.empty(schemas.INTERVALS))
    site_days = schemas.cast(pd.DataFrame(day_rows), schemas.SITE_DAYS)
    intervals = schemas.cast(pd.concat(interval_parts, ignore_index=True), schemas.INTERVALS)
    return Result(site_days, intervals)


def _site_day_row(site: str, date: str, n_slots: int, score: Score, p: float, outcome: str, y: np.ndarray) -> dict:
    window = score.winner
    recorded_min, corrected_min, change = impact.minimum_demand(y, window)
    return dict(
        site=site, date=date, n_slots=n_slots, input_ok=score.input_ok, n_admissible=score.n_admissible,
        evidence=score.evidence, p=p, outcome=outcome,
        window_start=window.start, window_end=window.end,
        runner_start=score.runner_up.start, runner_end=score.runner_up.end, runner_evidence=score.runner_up.evidence,
        proposed_mwh=impact.proposed_energy_mwh(y, window),
        recorded_min_mw=recorded_min, corrected_min_mw=corrected_min, min_change_mw=change,
    )


def _interval_rows(rows: pd.DataFrame, score: Score, outcome: str) -> pd.DataFrame:
    slot = rows["slot"].to_numpy()
    window = score.winner
    in_window = (not window.is_null) & (slot >= window.start) & (slot <= window.end)
    corrected = in_window & (outcome == AUTO_CORRECT)
    y = rows["y"].to_numpy(float)
    return pd.DataFrame({
        "site": rows["site"].to_numpy(),
        "timestamp": rows["timestamp"].to_numpy(),
        "net_load_mw": y,
        "in_window": in_window,
        "corrected": corrected,
        "net_load_corrected_mw": np.where(corrected, -y, y),
    })

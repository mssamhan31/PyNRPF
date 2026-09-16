"""Decision outcomes and applied corrections, one policy for all three methods.

Inputs:  an interval prediction table in the common schema (any method) and, for M9,
         its site-day outcomes.
Outputs: the same interval table with ``outcome`` (per site-day), ``applied`` (the
         slots actually flipped) and ``applied_y`` (the net load after the method's
         correction, MW).
Key steps: M7 and M8 are binary at their native thresholds: a day is AUTO_CORRECT
         when the method flags at least one slot, AUTO_KEEP otherwise. M9 is three-way
         from its calibrated probability and c. Only AUTO_CORRECT days carry applied
         energy; the recorded series is never altered on AUTO_KEEP or UNCERTAIN days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import AUTO_CORRECT, AUTO_KEEP
from .data import KEY


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

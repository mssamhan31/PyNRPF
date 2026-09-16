"""Shared vocabulary: outcome names, the common interval prediction schema, small helpers.

Inputs:  none.
Outputs: constants and two helpers used by every method module.
Key steps: one schema for the interval prediction table so M7, M8 and M9 are joined
         and scored by the same code; the three outcome names of the decision policy.
"""

from __future__ import annotations

import pandas as pd

AUTO_CORRECT = "AUTO_CORRECT"
AUTO_KEEP = "AUTO_KEEP"
UNCERTAIN = "UNCERTAIN"
OUTCOMES = (AUTO_CORRECT, AUTO_KEEP, UNCERTAIN)
METHODS = ("m7", "m8", "m9")
METHOD_LABELS = {"m7": "M7", "m8": "M8", "m9": "M9"}

# One row per quarter-hour of every complete held-out site-day, for every method.
# pred_interval marks the slots the method proposes to flip; pred_day is the method's
# own day decision; prob_day and prob_interval are graded scores where the method has
# them (NaN otherwise). The applied correction is derived in outcomes.py.
INTERVAL_COLUMNS = ["cohort", "station", "fold_id", "method", "date", "slot", "ts",
                    "pred_interval", "pred_day", "prob_day", "prob_interval"]


def finish_interval_table(table: pd.DataFrame, method: str, fold_id: str) -> pd.DataFrame:
    """Stamp method and fold identifiers and order the columns of an interval table."""
    table = table.copy()
    table["method"] = method
    table["fold_id"] = fold_id
    return table[INTERVAL_COLUMNS]


def read_parquet_or_none(path) -> pd.DataFrame | None:
    """Read a parquet file if it exists; None otherwise (for resumable stages)."""
    return pd.read_parquet(path) if path.exists() else None

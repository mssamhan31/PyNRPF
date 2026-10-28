"""The two previous methods the paper compares M9 against.

M7 is a deterministic threshold rule (``m7``); M8 is a two-stage XGBoost classifier
(``m8``, features in ``m8_features``) fitted once per station-held-out fold. Both were
carried over from the conference codebase with their numerics unchanged: the same
features, the same classifier parameters, seeds and thresholds, the same fit and
validation split handling, and the same library calls in the same order. What was
removed is the plugin, registry, configuration-system and artefact-store machinery of
the old package; a bundle is now a local pickle under ``results/02_baselines/bundles/``.

Inputs:  the interval frame in the configuration's column names (``data.model_input``).
Outputs: interval prediction tables in the common schema (``data.INTERVAL_COLUMNS``).
"""

from __future__ import annotations

import pandas as pd

INTERVAL_MINUTES = 15


def check_frame(frame: pd.DataFrame, columns: dict[str, str]) -> None:
    """The input checks both baselines rely on: columns present, fifteen-minute grid, unique keys.

    Args:
        frame: interval rows with the four columns named in ``columns``.
        columns: the configuration's ``columns`` block (site, timestamp, net_load, solar).

    Raises:
        KeyError: a required column is missing.
        ValueError: a timestamp is off the grid or a (site, timestamp) key repeats.
    """
    required = [columns[k] for k in ("site", "timestamp", "net_load", "solar")]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    ts = pd.to_datetime(frame[columns["timestamp"]])
    if ts.isna().any():
        raise ValueError(f"{int(ts.isna().sum())} timestamps are null")
    off_grid = (ts.dt.second != 0) | (ts.dt.microsecond != 0) | (ts.dt.minute % INTERVAL_MINUTES != 0)
    if off_grid.any():
        raise ValueError(f"{int(off_grid.sum())} timestamps are not on the {INTERVAL_MINUTES}-minute grid")
    duplicates = int(frame.duplicated(subset=[columns["site"], columns["timestamp"]], keep=False).sum())
    if duplicates:
        raise ValueError(f"Duplicate (site, timestamp) keys: {duplicates} rows")

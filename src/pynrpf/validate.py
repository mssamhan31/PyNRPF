"""Input checks, and the split of a frame into site-days.

A run takes one frame of fifteen-minute readings with a site identifier, a timestamp, the
recorded net load and a solar estimate. The checks are the ones the method relies on: the
columns exist, the timestamps parse and sit on fifteen-minute boundaries, and no site has
two readings at the same time. A calendar day with all 96 readings is scored; a day with
fewer is reported but not scored. Missing values inside a complete day are allowed and are
handled by the method (they disqualify the windows that touch them).

Timestamps with a time zone are converted to UTC before the calendar day is taken; naive
timestamps are taken as they are.

Inputs:  a pandas frame and the mapping from the four logical names to its column names.
Outputs: a standardised frame (site, ts, date, slot, y, s, timestamp as given) and an
         iterator over site-days.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

DEFAULT_COLUMNS = {
    "site": "substation_id",
    "timestamp": "timestamp",
    "net_load": "net_load_MW",
    "solar": "solar_MW",
}
INTERVAL_MINUTES = 15
SLOTS_PER_DAY = 96


def resolve_columns(columns: dict | None) -> dict:
    """The four column names, defaults overridden by ``columns``."""
    resolved = dict(DEFAULT_COLUMNS)
    resolved.update(columns or {})
    unknown = set(resolved) - set(DEFAULT_COLUMNS)
    if unknown:
        raise KeyError(f"Unknown column roles {sorted(unknown)}; expected {sorted(DEFAULT_COLUMNS)}")
    return resolved


def prepare(frame: pd.DataFrame, columns: dict | None = None) -> pd.DataFrame:
    """Check the frame and return it standardised, sorted by site and time.

    Returns a frame with ``site`` (string), ``ts`` (datetime), ``date`` (YYYY-MM-DD),
    ``slot`` (0 to 95), ``y`` and ``s`` (float, MW) and ``timestamp`` (the value as given).

    Raises:
        KeyError: a required column is missing.
        ValueError: timestamps do not parse, are off the fifteen-minute grid, or repeat within a site.
    """
    cols = resolve_columns(columns)
    missing = [name for name in cols.values() if name not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}")
    ts = pd.to_datetime(frame[cols["timestamp"]], errors="coerce")
    if ts.isna().any():
        raise ValueError(f"{int(ts.isna().sum())} timestamps could not be parsed")
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC")
    off_grid = (ts.dt.minute % INTERVAL_MINUTES != 0) | (ts.dt.second != 0)
    if off_grid.any():
        raise ValueError(f"{int(off_grid.sum())} timestamps are not on the {INTERVAL_MINUTES}-minute grid")
    out = pd.DataFrame({
        "site": frame[cols["site"]].astype(str).to_numpy(),
        "ts": ts.to_numpy(),
        "timestamp": frame[cols["timestamp"]].astype(str).to_numpy(),   # kept as given, for the output
        "y": pd.to_numeric(frame[cols["net_load"]], errors="coerce").astype(float).to_numpy(),
        "s": pd.to_numeric(frame[cols["solar"]], errors="coerce").astype(float).to_numpy(),
    })
    out["ts"] = pd.to_datetime(out["ts"])
    if out.duplicated(["site", "ts"]).any():
        raise ValueError("A site has two readings at the same timestamp")
    out["date"] = out["ts"].dt.strftime("%Y-%m-%d")
    out["slot"] = (out["ts"].dt.hour * 4 + out["ts"].dt.minute // INTERVAL_MINUTES).astype(int)
    return out.sort_values(["site", "ts"]).reset_index(drop=True)


def site_days(prepared: pd.DataFrame) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """Yield ``(site, date, rows)`` for every site-day of a prepared frame, in order."""
    for (site, date), rows in prepared.groupby(["site", "date"], sort=True):
        yield site, date, rows


def is_complete(rows: pd.DataFrame) -> bool:
    """True when the day has all 96 readings, one per slot."""
    return len(rows) == SLOTS_PER_DAY and bool(np.array_equal(rows["slot"].to_numpy(), np.arange(SLOTS_PER_DAY)))

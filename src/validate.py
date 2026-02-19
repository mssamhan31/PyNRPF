from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    """Raise KeyError if any required columns are missing from *df*."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def make_timestamp_local_naive(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Parse *ts_col* to datetime and strip any timezone info (no conversion).

    - String columns are parsed via ``pd.to_datetime``.
    - Timezone-aware columns have their tz removed with ``tz_localize(None)``.
    """
    if ts_col not in df.columns:
        raise KeyError(f"Timestamp column not found: {ts_col}")

    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=False)

    if df[ts_col].dt.tz is not None:
        df[ts_col] = df[ts_col].dt.tz_localize(None)

    return df


def coerce_timestamp(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Simple ``pd.to_datetime`` cast on *ts_col*."""
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    return df


def assert_no_null_timestamps(df: pd.DataFrame, ts_col: str) -> None:
    """Raise if *ts_col* contains any null / NaT values."""
    n = int(df[ts_col].isna().sum())
    if n:
        raise ValueError(f"Unparsable/null timestamps in '{ts_col}': {n} rows")


def assert_interval_alignment(
    df: pd.DataFrame, ts_col: str, interval_minutes: int
) -> None:
    """Raise if any timestamp is not aligned to *interval_minutes* boundaries."""
    if interval_minutes <= 0 or 60 % interval_minutes != 0:
        raise ValueError(f"Invalid interval_minutes: {interval_minutes}")

    ts = df[ts_col]
    bad = int(((ts.dt.second != 0) | (ts.dt.microsecond != 0) | (ts.dt.minute % interval_minutes != 0)).sum())

    if bad:
        raise ValueError(
            f"Timestamps not aligned to {interval_minutes}-minute boundary: {bad} rows"
        )


def duplicate_key_count(df: pd.DataFrame, key_cols: List[str]) -> int:
    """Return the number of duplicate rows based on *key_cols*."""
    return int(df.duplicated(subset=key_cols, keep=False).sum())


def assert_unique_keys(df: pd.DataFrame, key_cols: List[str]) -> None:
    """Raise if any duplicate key combinations exist."""
    dup = duplicate_key_count(df, key_cols)
    if dup:
        raise ValueError(f"Duplicate keys on {key_cols}: {dup} rows")


def missingness_summary(df: pd.DataFrame, cols: List[str]) -> Dict[str, int]:
    """Return ``{null_<col>: count}`` for each column in *cols*."""
    return {f"null_{c}": int(df[c].isna().sum()) for c in cols}


def basic_validate(
    df: pd.DataFrame,
    cols_required: List[str],
    site_col: str,
    ts_col: str,
    key_cols: List[str],
    interval_minutes: int,
    strip_timezone: bool = True,
    enforce_interval_alignment: bool = True,
    enforce_unique_keys: bool = True,
) -> Dict[str, Any]:
    """Run all validation checks and return ``{"df": <cleaned>, "summary": {...}}``."""
    require_columns(df, cols_required)

    df2 = (
        make_timestamp_local_naive(df, ts_col)
        if strip_timezone
        else coerce_timestamp(df, ts_col)
    )
    assert_no_null_timestamps(df2, ts_col)

    if enforce_interval_alignment:
        assert_interval_alignment(df2, ts_col, interval_minutes)

    dup_n = duplicate_key_count(df2, key_cols)
    if enforce_unique_keys and dup_n:
        raise ValueError(f"Duplicate keys on {key_cols}: {dup_n} rows")

    summary: Dict[str, Any] = {
        "n_rows": len(df2),
        "n_sites": int(df2[site_col].nunique()),
        "n_duplicate_keys": dup_n,
        "min_ts": df2[ts_col].min(),
        "max_ts": df2[ts_col].max(),
        **missingness_summary(df2, cols_required),
    }
    return {"df": df2, "summary": summary}

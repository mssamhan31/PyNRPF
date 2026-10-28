"""The two tables every run returns, column by column, with units.

``site_days``: one row per site and calendar day. ``intervals``: one row per input reading,
in the input order. The same definitions drive the pandas run, the Spark adapter (through
the DDL strings) and the command line.
"""

from __future__ import annotations

import pandas as pd

# (name, pandas dtype, Spark type, meaning)
SITE_DAYS = [
    ("site", "string", "string", "site identifier as given"),
    ("date", "string", "string", "calendar day, YYYY-MM-DD"),
    ("n_slots", "int64", "long", "readings found for the day; 96 is complete"),
    ("input_ok", "bool", "boolean", "the day was complete and had at least one admissible window"),
    ("n_admissible", "int64", "long", "candidate windows that could be scored"),
    ("evidence", "float64", "double", "r*: evidence of the best window; NaN when not scored"),
    ("p", "float64", "double", "calibrated probability that the day carries a wrong sign"),
    ("outcome", "string", "string", "AUTO_CORRECT, AUTO_KEEP or UNCERTAIN"),
    ("window_start", "int64", "long", "first slot of the best window (0 = 00:00), -1 when none"),
    ("window_end", "int64", "long", "last slot of the best window, inclusive, -1 when none"),
    ("runner_start", "int64", "long", "first slot of the runner-up window, -1 when none"),
    ("runner_end", "int64", "long", "last slot of the runner-up window, -1 when none"),
    ("runner_evidence", "float64", "double", "evidence of the runner-up window"),
    ("proposed_mwh", "float64", "double", "energy the best window would flip, MWh"),
    ("recorded_min_mw", "float64", "double", "minimum recorded net load of the day, MW"),
    ("corrected_min_mw", "float64", "double", "minimum after flipping the best window, MW"),
    ("min_change_mw", "float64", "double", "corrected minus recorded minimum, MW"),
]

INTERVALS = [
    ("site", "string", "string", "site identifier as given"),
    ("timestamp", "string", "string", "the reading's timestamp as given, ISO 8601"),
    ("net_load_mw", "float64", "double", "recorded net load, MW"),
    ("in_window", "bool", "boolean", "the slot lies inside the best window"),
    ("corrected", "bool", "boolean", "the sign was flipped: in the window and the day is AUTO_CORRECT"),
    ("net_load_corrected_mw", "float64", "double", "recorded net load with the sign flipped where corrected, MW"),
]


def names(table: list[tuple]) -> list[str]:
    return [c[0] for c in table]


def empty(table: list[tuple]) -> pd.DataFrame:
    """An empty frame with the table's columns and dtypes."""
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype, _, _ in table})


def cast(frame: pd.DataFrame, table: list[tuple]) -> pd.DataFrame:
    """Order and type a frame to the table."""
    out = frame[names(table)].copy()
    for name, dtype, _, _ in table:
        out[name] = out[name].astype(dtype)
    return out


def spark_ddl(table: list[tuple]) -> str:
    """The table as a Spark DDL schema string, for ``applyInPandas``."""
    return ", ".join(f"{name} {spark}" for name, _, spark, _ in table)


def markdown(table: list[tuple]) -> str:
    """The table's columns as a Markdown table, for the documentation."""
    lines = ["| column | type | meaning |", "|---|---|---|"]
    lines += [f"| `{name}` | {dtype} | {meaning} |" for name, dtype, _, meaning in table]
    return "\n".join(lines)

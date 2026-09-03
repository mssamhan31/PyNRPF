"""Convert caller frames to and from pandas, and validate their schema.

Inputs:  a pandas or Spark DataFrame of interval net load readings, plus the
         resolved inference configuration.
Outputs: a cleaned pandas frame and a data-quality summary; on the way out, a
         frame restored to the caller's original type.
Key steps: detect the input type, delegate schema and interval checks to the
         legacy validator, and rebuild a Spark frame when the caller supplied one.
"""

from __future__ import annotations

from typing import Any, Literal, Tuple

import pandas as pd

from ._legacy.validate import basic_validate

InputKind = Literal["pandas", "spark"]


def to_pandas_input(data: Any) -> Tuple[InputKind, pd.DataFrame, Any]:
    """Coerce a caller frame to pandas, remembering what it was.

    Args:
        data: A pandas DataFrame, which is copied, or a Spark DataFrame, which is
            collected to pandas.

    Returns:
        Tuple of the input kind (``"pandas"`` or ``"spark"``), the pandas frame,
        and the originating Spark session or None.

    Raises:
        TypeError: If the input is neither type, or a Spark frame carries no
            reachable session.
    """
    if isinstance(data, pd.DataFrame):
        return "pandas", data.copy(), None

    to_pandas = getattr(data, "toPandas", None)
    if callable(to_pandas):
        spark_session = getattr(data, "sparkSession", None)
        if spark_session is None:
            sql_ctx = getattr(data, "sql_ctx", None)
            spark_session = getattr(sql_ctx, "sparkSession", None)
        if spark_session is None:
            raise TypeError("Spark DataFrame detected but no sparkSession found.")
        return "spark", to_pandas(), spark_session

    raise TypeError(
        "Unsupported data input type. Expected pandas DataFrame or Spark DataFrame."
    )


def from_pandas_output(df: pd.DataFrame, kind: InputKind, spark_session: Any) -> Any:
    """Restore a scored pandas frame to the caller's original frame type.

    Args:
        df: The scored pandas frame.
        kind: The input kind recorded by :func:`to_pandas_input`.
        spark_session: Session to rebuild with, when kind is ``"spark"``.

    Returns:
        The frame as pandas, or as a Spark DataFrame.
    """
    if kind == "pandas":
        return df
    return spark_session.createDataFrame(df)


def validate_dataframe(
    df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check an interval frame against the configured schema and clean it.

    Args:
        df: Raw interval-level frame.
        cfg: Resolved inference configuration. The ``columns`` block names the
            required site, timestamp, net load and solar columns; ``runtime``
            gives the interval length in minutes and whether strict validation
            applies.

    Returns:
        Tuple of the cleaned frame, with timezones stripped from timestamps, and
        a data-quality summary of what was checked and dropped.

    Raises:
        KeyError: If a required column is missing.
        ValueError: Under strict validation, if intervals are misaligned to the
            configured length or site-timestamp keys are not unique.
    """
    cols = cfg["columns"]
    runtime = cfg["runtime"]
    required = [cols["site"], cols["timestamp"], cols["net_load"], cols["solar"]]

    strict = bool(runtime.get("strict_validation", True))
    result = basic_validate(
        df=df,
        cols_required=required,
        site_col=cols["site"],
        ts_col=cols["timestamp"],
        key_cols=[cols["site"], cols["timestamp"]],
        interval_minutes=int(runtime.get("interval_minutes", 15)),
        strip_timezone=True,
        enforce_interval_alignment=strict,
        enforce_unique_keys=strict,
    )
    return result["df"], result["summary"]

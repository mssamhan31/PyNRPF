from __future__ import annotations

from typing import Any, Literal, Tuple

import pandas as pd

from ._legacy.validate import basic_validate

InputKind = Literal["pandas", "spark"]


def to_pandas_input(data: Any) -> Tuple[InputKind, pd.DataFrame, Any]:
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
    if kind == "pandas":
        return df
    return spark_session.createDataFrame(df)


def validate_dataframe(
    df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
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

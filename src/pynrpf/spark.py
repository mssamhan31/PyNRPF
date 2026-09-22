"""The same run on a Spark DataFrame, one site at a time.

M9 works one site-day at a time and needs nothing across sites, so Spark's contribution is
distribution, not arithmetic. ``run_per_site`` groups the frame by site and runs
``pynrpf.run`` on each group as a pandas frame (``applyInPandas``), so the numerics are the
ones tested in pandas. Each group holds one site's readings; for very long histories pass
``group_by=("site", "month")`` to bound the group size, which does not change the result
because days are scored independently.

Inputs:  a Spark DataFrame with the four columns (see ``validate``), and the run options.
Outputs: two Spark DataFrames with the schemas of ``schemas.SITE_DAYS`` and
         ``schemas.INTERVALS``. Each is produced by its own pass over the data.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from . import schemas, validate
from .m9 import RELEASE_CALIBRATION, RELEASE_PHI, Calibration
from .m9.decision import DEFAULT_C
from .run import run


def run_per_site(sdf, *, columns: dict | None = None, c: float = DEFAULT_C,
                 calibration: Calibration = RELEASE_CALIBRATION, phi: float = RELEASE_PHI,
                 group_by: tuple[str, ...] = ("site",)):
    """Score every site-day of a Spark frame; returns ``(site_days, intervals)`` Spark frames.

    Args:
        sdf: Spark DataFrame of fifteen-minute readings.
        columns, c, calibration, phi: as for ``pynrpf.run``.
        group_by: ``("site",)`` or ``("site", "month")``; the grouping columns are derived
            from the site and timestamp columns and never leave the adapter.
    """
    from pyspark.sql import functions as F

    cols = validate.resolve_columns(columns)
    keyed = sdf.withColumn("_site", F.col(cols["site"]).cast("string"))
    keys = ["_site"]
    if "month" in group_by:
        keyed = keyed.withColumn("_month", F.date_format(F.col(cols["timestamp"]).cast("timestamp"), "yyyy-MM"))
        keys.append("_month")

    def per_group(table_name: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
        def apply(pdf: pd.DataFrame) -> pd.DataFrame:
            result = run(pdf.drop(columns=keys), columns=cols, c=c, calibration=calibration, phi=phi)
            return getattr(result, table_name)
        return apply

    grouped = keyed.groupBy(*keys)
    site_days = grouped.applyInPandas(per_group("site_days"), schema=schemas.spark_ddl(schemas.SITE_DAYS))
    intervals = grouped.applyInPandas(per_group("intervals"), schema=schemas.spark_ddl(schemas.INTERVALS))
    return site_days, intervals

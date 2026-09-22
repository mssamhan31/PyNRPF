"""The common site-day population, the interval frames every method receives, and the shared vocabulary.

Inputs:  the Alpha and Beta parquet files (interval rows with labels).
Outputs: a site-day index per cohort (one row per station-date with completeness,
         confidence and the labelled window) and the interval frame of complete days.
Key steps: parse timestamps as UTC; count slots per site-day; keep only the days with
         all 96 quarter-hours (one population definition shared by M7, M8 and M9);
         give Alpha the confidence 'controlled'.

Column names follow the configuration's ``columns`` block; the evaluation-side names
are fixed: cohort, station, date, slot, y (net load, MW), s (solar, MW), truth. The
interval prediction table every method produces has one schema (``INTERVAL_COLUMNS``)
so the three methods are joined and scored by the same code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings

KEY = ["cohort", "station", "date"]
METHODS = ("m7", "m8", "m9")
METHOD_LABELS = {"m7": "M7", "m8": "M8", "m9": "M9"}

# One row per quarter-hour of every complete held-out site-day, for every method.
# pred_interval marks the slots the method proposes to flip; pred_day is the method's
# own day decision; prob_day and prob_interval are graded scores where the method has
# them (NaN otherwise). The applied correction is derived in reference.py.
INTERVAL_COLUMNS = ["cohort", "station", "fold_id", "method", "date", "slot", "ts",
                    "pred_interval", "pred_day", "prob_day", "prob_interval"]


def finish_interval_table(table: pd.DataFrame, method: str, fold_id: str) -> pd.DataFrame:
    """Stamp method and fold identifiers and order the columns of an interval table."""
    table = table.copy()
    table["method"] = method
    table["fold_id"] = fold_id
    return table[INTERVAL_COLUMNS]


def load_cohort(settings: Settings, cohort: str) -> pd.DataFrame:
    """Read one cohort's interval rows, parse the timestamp to UTC and fill the confidence.

    Returns the frame sorted by station and timestamp with the evaluation-side columns
    cohort, station, date, ts, y, s, truth and confidence added beside the originals.
    """
    cols = settings.columns
    df = pd.read_parquet(settings.dataset(cohort))
    df["cohort"] = cohort
    df["station"] = df[cols["site"]].astype(str)
    df["ts"] = pd.to_datetime(df[cols["timestamp"]], utc=True)
    df["date"] = df["ts"].dt.strftime("%Y-%m-%d")
    df["y"] = df[cols["net_load"]].astype(float)
    df["s"] = df[cols["solar"]].astype(float)
    df["truth"] = df[cols["label_interval"]].fillna(False).astype(bool)
    if cols["confidence"] in df.columns:
        df["confidence"] = df[cols["confidence"]].astype(str)
    else:
        df["confidence"] = settings["population"]["alpha_confidence"]
    df = df.sort_values(["station", "ts"]).reset_index(drop=True)
    df["slot"] = df.groupby(["station", "date"]).cumcount()
    return df


def siteday_index(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """One row per station-date: completeness, confidence, RPF label and labelled window.

    A day is complete when it has exactly ``population.slots_per_day`` rows. Only
    complete days enter the evaluation; the count of dropped days is reported in the
    fold manifest so the population is auditable.
    """
    slots = int(settings["population"]["slots_per_day"])
    headline = set(settings["population"]["headline_confidence"])

    def window(g: pd.Series) -> tuple[int, int, int]:
        idx = np.flatnonzero(g.to_numpy())
        return (int(idx[0]), int(idx[-1]), int(idx.size)) if idx.size else (-1, -1, 0)

    grouped = df.groupby(KEY, sort=True)
    index = grouped.agg(
        n_slots=("slot", "size"),
        n_finite=("y", lambda v: int(np.isfinite(v).sum())),
        confidence=("confidence", "first"),
        rpf=("truth", "any"),
    ).reset_index()
    windows = grouped["truth"].apply(window)
    index[["true_start", "true_end", "true_slots"]] = pd.DataFrame(windows.tolist(), index=windows.index).to_numpy()
    index["complete"] = index["n_slots"] == slots
    index["headline"] = index["confidence"].isin(headline)
    index["rpf"] = index["rpf"].astype(int)
    return index


def complete_intervals(df: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Interval rows of complete site-days only, in station-date-slot order."""
    keep = index.loc[index["complete"], KEY]
    out = df.merge(keep, on=KEY, how="inner")
    return out.sort_values(["station", "date", "slot"]).reset_index(drop=True)


def model_input(intervals: pd.DataFrame, settings: Settings, with_labels: bool = False) -> pd.DataFrame:
    """The four columns the baselines read (plus labels for training), tz-naive UTC.

    The baselines take naive timestamps; passing naive UTC keeps the (site, timestamp)
    key identical on the way in and the way out.
    """
    cols = settings.columns
    out = pd.DataFrame({
        cols["site"]: intervals["station"].to_numpy(),
        cols["timestamp"]: intervals["ts"].dt.tz_localize(None).to_numpy(),
        cols["net_load"]: intervals["y"].to_numpy(),
        cols["solar"]: intervals["s"].to_numpy(),
    })
    if with_labels:
        out[cols["label_day"]] = intervals.groupby(["station", "date"])["truth"].transform("any").to_numpy()
        out[cols["label_interval"]] = intervals["truth"].to_numpy()
    return out


def siteday_arrays(intervals: pd.DataFrame):
    """Yield (cohort, station, date, y, s, truth) per complete site-day, arrays of 96."""
    for (cohort, station, date), g in intervals.groupby(KEY, sort=True):
        yield cohort, station, date, g["y"].to_numpy(float), g["s"].to_numpy(float), g["truth"].to_numpy(bool)


def load_population(settings: Settings) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Site-day index and complete-day interval frame for every cohort in the configuration."""
    indexes: dict[str, pd.DataFrame] = {}
    intervals: dict[str, pd.DataFrame] = {}
    for cohort in settings["population"]["cohorts"]:
        df = load_cohort(settings, cohort)
        indexes[cohort] = siteday_index(df, settings)
        intervals[cohort] = complete_intervals(df, indexes[cohort])
    return indexes, intervals

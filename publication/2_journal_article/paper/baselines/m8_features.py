"""The M8 feature matrices: XGB1 (one row per site-day) and XGB2 (one row per daytime interval).

A faithful pandas reproduction of the Databricks PySpark feature pipeline of the
conference codebase; column names and their order match it.

Inputs:  interval rows (tz-naive timestamps) with net load and solar in MW and a
         ground-truth column encoded as the conference pipeline expects: negative for a
         wrong-sign row, positive otherwise (a constant positive column at prediction time).
Outputs: the wide feature table, the ordered feature column list and the label column name.
Key steps: XGB1 pivots net load and solar into 96 slot columns per day, adds calendar,
         holiday, missingness and station one-hot columns; XGB2 restricts to the
         candidate days and the daytime hours, joins the day features and adds the
         underlying load (net load + solar), its rates of change and the slot one-hot.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

XGB1_LABEL = "is_having_reverse_power_flow_issue"
XGB2_LABEL = "sign_is_wrong"

# 96 quarter-hour slot labels covering the full day, and the 52 daytime ones (06:00 .. 18:45).
_SLOTS_96 = [(datetime(2000, 1, 1) + timedelta(minutes=15 * i)).strftime("%H%M") for i in range(96)]
_SLOTS_DAYTIME = [f"{h:02d}{m:02d}" for h in range(6, 19) for m in (0, 15, 30, 45)]
_MONTH_ABBREVS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
_CALENDAR = ["is_holiday", "is_weekend"] + [f"is_{m}" for m in _MONTH_ABBREVS]


def _sanitise(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


def holiday_dates(start_date: str, end_date: str, country: str, subdivision: str) -> set:
    """Public holidays (national and state) between two ISO dates inclusive."""
    import holidays as pyhol

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    years = list(range(start.year, end.year + 1))
    national = pyhol.country_holidays(country, years=years)
    state = pyhol.country_holidays(country, subdiv=subdivision, years=years)
    return {d for d in national if start <= d <= end} | {d for d in state if start <= d <= end}


def _pivot_column(df: pd.DataFrame, col_site: str, value_col: str, prefix: str) -> pd.DataFrame:
    """Pivot ``value_col`` by slot label for each (site, date): columns ``prefix_0000`` .. ``prefix_2345``."""
    sub = df[[col_site, "date", "hhmm", value_col]].copy()
    sub = sub.drop_duplicates(subset=[col_site, "date", "hhmm"], keep="first")
    piv = sub.pivot_table(index=[col_site, "date"], columns="hhmm", values=value_col, aggfunc="first")
    for s in _SLOTS_96:
        if s not in piv.columns:
            piv[s] = np.nan
    piv = piv[_SLOTS_96]
    piv.columns = [f"{prefix}_{s}" for s in _SLOTS_96]
    return piv.reset_index()


def build_xgb1_features(df: pd.DataFrame, m8_cfg: dict[str, Any], columns: dict[str, str], col_gt: str,
                        holiday_range: tuple[str, str]) -> tuple[pd.DataFrame, list[str], str]:
    """The XGB1 day-level feature matrix.

    Args:
        df: interval rows, tz-naive timestamps.
        m8_cfg: the ``m8.m8_xgb`` block (noon hours, holiday country and subdivision).
        columns: the configuration's ``columns`` block.
        col_gt: ground-truth column; a day is positive when any value is negative.
        holiday_range: (start, end) ISO dates the holiday calendar is built for.

    Returns:
        ``(features, feature_columns, label_column)``: one row per (site, date).
    """
    col_site, col_ts = columns["site"], columns["timestamp"]
    col_net, col_solar = columns["net_load"], columns["solar"]
    noon_start = int(m8_cfg.get("noon_hour_start", 6))
    noon_end = int(m8_cfg.get("noon_hour_end", 18))

    work = df[[col_site, col_ts, col_net, col_solar, col_gt]].copy()
    work["date"] = work[col_ts].dt.date
    work["hhmm"] = work[col_ts].dt.strftime("%H%M")
    work["hour"] = work[col_ts].dt.hour
    work = work.sort_values(col_ts)
    work = work.drop_duplicates(subset=[col_site, "date", "hhmm"], keep="first")
    base = work[[col_site, "date"]].drop_duplicates()

    label_daily = (work.groupby([col_site, "date"])[col_gt].apply(lambda s: int((s < 0).any()))
                   .reset_index(name=XGB1_LABEL))

    holiday_set = holiday_dates(holiday_range[0], holiday_range[1],
                                m8_cfg.get("holiday_country", "AU"), m8_cfg.get("holiday_subdivision", "NSW"))
    cal = base.copy()
    cal_dates = pd.to_datetime(cal["date"])
    cal["is_weekend"] = cal_dates.dt.dayofweek.isin([5, 6]).astype(int).values
    for i, m in enumerate(_MONTH_ABBREVS, start=1):
        cal[f"is_{m}"] = (cal_dates.dt.month == i).astype(int).values
    cal["is_holiday"] = cal["date"].apply(lambda d: int(d in holiday_set))

    mw_piv = _pivot_column(work, col_site, col_net, "MW")
    solar_piv = _pivot_column(work, col_site, col_solar, "Solar_MW")
    mw_agg = work.groupby([col_site, "date"]).agg(MW_max=(col_net, "max")).reset_index()
    noon = work.loc[(work["hour"] >= noon_start) & (work["hour"] <= noon_end)]
    mw_noon = noon.groupby([col_site, "date"]).agg(MW_noon_min=(col_net, "min")).reset_index()
    solar_agg = work.groupby([col_site, "date"]).agg(Solar_MW_max=(col_solar, "max"),
                                                     Solar_MW_sum=(col_solar, "sum")).reset_index()

    wide = base.copy()
    for rhs in [cal, label_daily, mw_piv, solar_piv, mw_agg, mw_noon, solar_agg]:
        wide = wide.merge(rhs, on=[col_site, "date"], how="left")

    mw_cols = [f"MW_{s}" for s in _SLOTS_96]
    sol_cols = [f"Solar_MW_{s}" for s in _SLOTS_96]
    wide["mw_missing_count"] = wide[mw_cols].isna().sum(axis=1).astype(int)
    wide["solar_missing_count"] = wide[sol_cols].isna().sum(axis=1).astype(int)

    # Station one-hot columns; a station unseen in training gets zeros at prediction time.
    ohe_cols = []
    for name in sorted(wide[col_site].unique()):
        cname = f"is_{_sanitise(name)}"
        wide[cname] = (wide[col_site] == name).astype(int)
        ohe_cols.append(cname)

    primary_cols = [col_site, "date"]
    final_cols = (primary_cols + [XGB1_LABEL] + ohe_cols + _CALENDAR + mw_cols + ["MW_max", "MW_noon_min"]
                  + sol_cols + ["Solar_MW_max", "Solar_MW_sum"] + ["mw_missing_count", "solar_missing_count"])
    final_cols = [c for c in final_cols if c in wide.columns]
    wide = wide[final_cols]
    feat_cols = [c for c in wide.columns if c not in primary_cols + [XGB1_LABEL]]
    return wide, feat_cols, XGB1_LABEL


def build_xgb2_features(df: pd.DataFrame, m8_cfg: dict[str, Any], columns: dict[str, str], col_gt: str,
                        day_features: pd.DataFrame,
                        candidate_keys: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    """The XGB2 interval-level feature matrix for the candidate site-days.

    Args:
        df: interval rows, tz-naive timestamps.
        m8_cfg: the ``m8.m8_xgb`` block.
        columns: the configuration's ``columns`` block.
        col_gt: ground-truth column; a row is positive when its value is negative.
        day_features: output of ``build_xgb1_features``.
        candidate_keys: (site, date) pairs of the site-days to score.

    Returns:
        ``(features, feature_columns, label_column)``: one row per daytime interval of a candidate day.
    """
    col_site, col_ts = columns["site"], columns["timestamp"]
    col_net, col_solar = columns["net_load"], columns["solar"]
    noon_start = int(m8_cfg.get("noon_hour_start", 6))
    noon_end = int(m8_cfg.get("noon_hour_end", 18))
    pkeys2 = [col_site, "date", col_ts]

    work = df[[col_site, col_ts, col_net, col_solar, col_gt]].copy()
    work["date"] = work[col_ts].dt.date
    work["hhmm"] = work[col_ts].dt.strftime("%H%M")
    work["hour"] = work[col_ts].dt.hour
    work["MW_underlying"] = work[col_net] + work[col_solar]
    work = work.sort_values(col_ts)
    work_dedup = work.drop_duplicates(subset=[col_site, "date", "hhmm"], keep="first")

    # Rate of change: the next reading minus this one, within the site-day.
    work_dedup = work_dedup.sort_values([col_site, "date", col_ts])
    for src, dst in [(col_net, "roc_MW"), (col_solar, "roc_Solar_MW"), ("MW_underlying", "roc_MW_underlying")]:
        shifted = work_dedup.groupby([col_site, "date"])[src].shift(-1)
        work_dedup[dst] = shifted - work_dedup[src]

    mw_und_piv = _pivot_column(work_dedup, col_site, "MW_underlying", "MW_underlying")
    mw_und_agg = work_dedup.groupby([col_site, "date"]).agg(MW_underlying_max=("MW_underlying", "max")).reset_index()
    noon = work_dedup.loc[(work_dedup["hour"] >= noon_start) & (work_dedup["hour"] <= noon_end)]
    mw_und_noon = noon.groupby([col_site, "date"]).agg(MW_underlying_noon_min=("MW_underlying", "min")).reset_index()
    roc_mw_piv = _pivot_column(work_dedup, col_site, "roc_MW", "roc_MW")
    roc_sol_piv = _pivot_column(work_dedup, col_site, "roc_Solar_MW", "roc_Solar_MW")
    roc_und_piv = _pivot_column(work_dedup, col_site, "roc_MW_underlying", "roc_MW_underlying")
    roc_aggs = work_dedup.groupby([col_site, "date"]).agg(
        roc_MW_max=("roc_MW", "max"), roc_MW_min=("roc_MW", "min"),
        roc_Solar_MW_max=("roc_Solar_MW", "max"), roc_Solar_MW_min=("roc_Solar_MW", "min"),
        roc_MW_underlying_max=("roc_MW_underlying", "max"), roc_MW_underlying_min=("roc_MW_underlying", "min"),
    ).reset_index()

    daytime = work.loc[(work["hour"] >= noon_start) & (work["hour"] <= noon_end)].copy()
    daytime = daytime.dropna(subset=[col_gt])
    daytime[XGB2_LABEL] = (daytime[col_gt] < 0).astype(int)
    daytime = daytime.merge(candidate_keys[[col_site, "date"]], on=[col_site, "date"], how="inner")

    day_feat_cols = [c for c in day_features.columns if c not in {col_site, "date", XGB1_LABEL}]
    joined = daytime.merge(day_features[[col_site, "date"] + day_feat_cols], on=[col_site, "date"], how="inner")
    for rhs in [mw_und_piv, mw_und_agg, mw_und_noon, roc_mw_piv, roc_sol_piv, roc_und_piv, roc_aggs]:
        joined = joined.merge(rhs, on=[col_site, "date"], how="left")
    for v in _SLOTS_DAYTIME:
        joined[f"hhmm_{v}"] = (joined["hhmm"] == v).astype(int)

    calendar_set = set(_CALENDAR)
    substation_cols = sorted(c for c in day_feat_cols if c.startswith("is_") and c not in calendar_set)
    feat_cols2 = (
        substation_cols
        + [f"hhmm_{v}" for v in _SLOTS_DAYTIME]
        + [c for c in _CALENDAR if c in joined.columns]
        + [f"MW_{s}" for s in _SLOTS_96] + ["MW_max", "MW_noon_min"]
        + [f"Solar_MW_{s}" for s in _SLOTS_96] + ["Solar_MW_max", "Solar_MW_sum"]
        + [f"MW_underlying_{s}" for s in _SLOTS_96] + ["MW_underlying_max", "MW_underlying_noon_min"]
        + [f"roc_MW_{s}" for s in _SLOTS_96] + [f"roc_Solar_MW_{s}" for s in _SLOTS_96]
        + [f"roc_MW_underlying_{s}" for s in _SLOTS_96]
        + ["roc_MW_max", "roc_MW_min", "roc_Solar_MW_max", "roc_Solar_MW_min",
           "roc_MW_underlying_max", "roc_MW_underlying_min"]
        + ["mw_missing_count", "solar_missing_count"]
    )
    feat_cols2 = [c for c in feat_cols2 if c in joined.columns]
    result = joined[pkeys2 + [XGB2_LABEL] + feat_cols2].copy()
    return result, feat_cols2, XGB2_LABEL

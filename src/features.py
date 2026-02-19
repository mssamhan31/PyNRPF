"""Feature engineering for XGB1 (day-level) and XGB2 (interval-level).

Faithfully reproduces the Databricks PySpark feature pipeline in pure pandas.
Column names and ordering match the production implementation.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.io import req

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 96 quarter-hour slot labels covering the full 24 h
_SLOTS_96: List[str] = [
    (datetime(2000, 1, 1) + timedelta(minutes=15 * i)).strftime("%H%M")
    for i in range(96)
]

# 52 daytime quarter-hour slot labels (06:00 .. 18:45)
_SLOTS_DAYTIME: List[str] = [
    f"{h:02d}{m:02d}" for h in range(6, 19) for m in (0, 15, 30, 45)
]

_MONTH_ABBREVS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov"]

_SANITIZE = lambda s: re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")


# ---------------------------------------------------------------------------
# Holiday helper
# ---------------------------------------------------------------------------

def _build_holiday_set(
    start_date: str,
    end_date: str,
    country: str = "AU",
    subdivision: str = "NSW",
) -> set:
    """Return the set of holiday dates (AU national ∪ state)."""
    import holidays as pyhol

    sd = datetime.strptime(start_date, "%Y-%m-%d").date()
    ed = datetime.strptime(end_date, "%Y-%m-%d").date()
    years = list(range(sd.year, ed.year + 1))

    try:
        au_nat = pyhol.country_holidays(country, years=years)
        au_sub = pyhol.country_holidays(country, subdiv=subdivision, years=years)
    except Exception:
        au_nat = pyhol.Australia(years=years)
        au_sub = pyhol.Australia(prov=subdivision, years=years)

    return {d for d in au_nat if sd <= d <= ed} | {d for d in au_sub if sd <= d <= ed}


# ---------------------------------------------------------------------------
# Pivot helper
# ---------------------------------------------------------------------------

def _pivot_column(
    df: pd.DataFrame,
    col_site: str,
    value_col: str,
    prefix: str,
    slots: List[str] | None = None,
) -> pd.DataFrame:
    """Pivot *value_col* by hhmm for each (site, date).

    De-duplicates by taking the first occurrence per (site, date, hhmm).
    Returns a DataFrame with columns ``prefix_0000`` .. ``prefix_2345``.
    """
    if slots is None:
        slots = _SLOTS_96

    sub = df[[col_site, "date", "hhmm", value_col]].copy()
    # De-dup: keep first row per (site, date, hhmm)
    sub = sub.drop_duplicates(subset=[col_site, "date", "hhmm"], keep="first")

    piv = sub.pivot_table(
        index=[col_site, "date"],
        columns="hhmm",
        values=value_col,
        aggfunc="first",
    )
    # Ensure all slots present
    for s in slots:
        if s not in piv.columns:
            piv[s] = np.nan
    piv = piv[slots]
    piv.columns = [f"{prefix}_{s}" for s in slots]
    return piv.reset_index()


# ---------------------------------------------------------------------------
# Public: XGB1 features (day-level)
# ---------------------------------------------------------------------------

def build_xgb1_features(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    col_site: str,
    col_ts: str,
    col_net: str,
    col_solar: str,
    col_gt: str,
) -> Tuple[pd.DataFrame, List[str], str]:
    """Build the XGB1 day-level feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Interval-level data (must already be tz-naive).
    cfg : dict
        Full run config.
    col_site, col_ts, col_net, col_solar, col_gt : str
        Column names.

    Returns
    -------
    (features_df, feat_cols, label_col)
        *features_df* has one row per (site, date).
        *feat_cols* is the ordered list of feature column names.
        *label_col* is the name of the label column.
    """
    m8 = req(cfg, "m8_xgb")
    noon_start = int(m8.get("noon_hour_start", 6))
    noon_end = int(m8.get("noon_hour_end", 18))
    split = req(cfg, "split")
    hol_country = m8.get("holiday_country", "AU")
    hol_subdiv = m8.get("holiday_subdivision", "NSW")

    label_col = "is_having_reverse_power_flow_issue"

    # --- Enrich ---
    work = df[[col_site, col_ts, col_net, col_solar, col_gt]].copy()
    work["date"] = work[col_ts].dt.date
    work["hhmm"] = work[col_ts].dt.strftime("%H%M")
    work["hour"] = work[col_ts].dt.hour

    # --- De-dup per (site, date, hhmm): keep first by TS ---
    work = work.sort_values(col_ts)
    work = work.drop_duplicates(subset=[col_site, "date", "hhmm"], keep="first")

    # --- Base keys ---
    base = work[[col_site, "date"]].drop_duplicates()

    # --- Label: 1 if any ground_truth < 0 on that day ---
    label_daily = (
        work.groupby([col_site, "date"])[col_gt]
        .apply(lambda s: int((s < 0).any()))
        .reset_index(name=label_col)
    )

    # --- Holidays ---
    holiday_set = _build_holiday_set(
        split["train_start"], split["test_end"], hol_country, hol_subdiv
    )

    # --- Calendar ---
    cal = base.copy()
    cal_dates = pd.to_datetime(cal["date"])
    cal["is_weekend"] = cal_dates.dt.dayofweek.isin([5, 6]).astype(int).values
    for i, m in enumerate(_MONTH_ABBREVS, start=1):
        cal[f"is_{m}"] = (cal_dates.dt.month == i).astype(int).values
    cal["is_holiday"] = cal["date"].apply(lambda d: int(d in holiday_set))

    # --- Pivot MW ---
    mw_piv = _pivot_column(work, col_site, col_net, "MW")
    solar_piv = _pivot_column(work, col_site, col_solar, "Solar_MW")

    # --- MW aggregates ---
    mw_agg = work.groupby([col_site, "date"]).agg(
        MW_max=(col_net, "max"),
    ).reset_index()

    mw_noon = work.loc[
        (work["hour"] >= noon_start) & (work["hour"] <= noon_end)
    ].groupby([col_site, "date"]).agg(
        MW_noon_min=(col_net, "min"),
    ).reset_index()

    # --- Solar aggregates ---
    solar_agg = work.groupby([col_site, "date"]).agg(
        Solar_MW_max=(col_solar, "max"),
        Solar_MW_sum=(col_solar, "sum"),
    ).reset_index()

    # --- Join everything ---
    wide = base.copy()
    for rhs in [cal, label_daily, mw_piv, solar_piv, mw_agg, mw_noon, solar_agg]:
        wide = wide.merge(rhs, on=[col_site, "date"], how="left")

    # --- Missingness counts (before null->NaN, but they ARE NaN from pivot) ---
    mw_cols = [f"MW_{s}" for s in _SLOTS_96]
    sol_cols = [f"Solar_MW_{s}" for s in _SLOTS_96]
    wide["mw_missing_count"] = wide[mw_cols].isna().sum(axis=1).astype(int)
    wide["solar_missing_count"] = wide[sol_cols].isna().sum(axis=1).astype(int)

    # --- Substation OHE ---
    subs = sorted(wide[col_site].unique())
    ohe_cols = []
    for name in subs:
        cname = f"is_{_SANITIZE(name)}"
        wide[cname] = (wide[col_site] == name).astype(int)
        ohe_cols.append(cname)

    # --- Column ordering (matches Databricks) ---
    primary_cols = [col_site, "date"]
    label_cols = [label_col]
    calendar_cols = ["is_holiday", "is_weekend"] + [f"is_{m}" for m in _MONTH_ABBREVS]
    mw_time_cols = mw_cols
    mw_agg_cols = ["MW_max", "MW_noon_min"]
    sol_time_cols = sol_cols
    sol_agg_cols = ["Solar_MW_max", "Solar_MW_sum"]
    extra_cols = ["mw_missing_count", "solar_missing_count"]

    final_cols = (
        primary_cols + label_cols + ohe_cols + calendar_cols
        + mw_time_cols + mw_agg_cols + sol_time_cols + sol_agg_cols + extra_cols
    )
    # Defensive: only keep existing columns
    final_cols = [c for c in final_cols if c in wide.columns]
    wide = wide[final_cols]

    # --- Feature column list ---
    feat_cols = [c for c in wide.columns if c not in primary_cols + label_cols]

    print(f"XGB1 features: {len(feat_cols)} columns, {len(wide):,} rows "
          f"({wide[label_col].sum():,} positive)")

    return wide, feat_cols, label_col


# ---------------------------------------------------------------------------
# Public: XGB2 features (interval-level)
# ---------------------------------------------------------------------------

def build_xgb2_features(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    day_features_df: pd.DataFrame,
    xgb1_positive_keys: pd.DataFrame,
    col_site: str,
    col_ts: str,
    col_net: str,
    col_solar: str,
    col_gt: str,
) -> Tuple[pd.DataFrame, List[str], str]:
    """Build the XGB2 interval-level feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Full interval-level data.
    cfg : dict
        Full run config.
    day_features_df : pd.DataFrame
        Output of ``build_xgb1_features`` (day-level wide table).
    xgb1_positive_keys : pd.DataFrame
        DataFrame with columns (col_site, date) for XGB1-predicted-positive
        site-days.
    col_site, col_ts, col_net, col_solar, col_gt : str
        Column names.

    Returns
    -------
    (features_df, feat_cols, label_col)
    """
    m8 = req(cfg, "m8_xgb")
    noon_start = int(m8.get("noon_hour_start", 6))
    noon_end = int(m8.get("noon_hour_end", 18))

    label_col2 = "sign_is_wrong"
    pkeys2 = [col_site, "date", col_ts]

    # --- Enrich df ---
    work = df[[col_site, col_ts, col_net, col_solar, col_gt]].copy()
    work["date"] = work[col_ts].dt.date
    work["hhmm"] = work[col_ts].dt.strftime("%H%M")
    work["hour"] = work[col_ts].dt.hour

    # --- MW_underlying = MW + Solar_MW ---
    work["MW_underlying"] = work[col_net] + work[col_solar]

    # --- De-dup for pivots ---
    work = work.sort_values(col_ts)
    work_dedup = work.drop_duplicates(subset=[col_site, "date", "hhmm"], keep="first")

    # --- ROC (forward difference within site-day) on de-duped data ---
    work_dedup = work_dedup.sort_values([col_site, "date", col_ts])
    for src, dst in [
        (col_net, "roc_MW"),
        (col_solar, "roc_Solar_MW"),
        ("MW_underlying", "roc_MW_underlying"),
    ]:
        work_dedup[dst] = work_dedup.groupby([col_site, "date"])[src].diff(periods=-1).mul(-1)
        # diff(-1) gives next - current but we need lead-current, so diff(periods=1 shift)
        # Actually: use shift(-1) - current
    # Redo properly: lead - current
    for src, dst in [
        (col_net, "roc_MW"),
        (col_solar, "roc_Solar_MW"),
        ("MW_underlying", "roc_MW_underlying"),
    ]:
        shifted = work_dedup.groupby([col_site, "date"])[src].shift(-1)
        work_dedup[dst] = shifted - work_dedup[src]

    # --- Pivot MW_underlying (96 cols) ---
    mw_und_piv = _pivot_column(work_dedup, col_site, "MW_underlying", "MW_underlying")

    # MW_underlying aggregates
    mw_und_agg = work_dedup.groupby([col_site, "date"]).agg(
        MW_underlying_max=("MW_underlying", "max"),
    ).reset_index()

    mw_und_noon = work_dedup.loc[
        (work_dedup["hour"] >= noon_start) & (work_dedup["hour"] <= noon_end)
    ].groupby([col_site, "date"]).agg(
        MW_underlying_noon_min=("MW_underlying", "min"),
    ).reset_index()

    # --- ROC pivots (96 cols each) ---
    roc_mw_piv = _pivot_column(work_dedup, col_site, "roc_MW", "roc_MW")
    roc_sol_piv = _pivot_column(work_dedup, col_site, "roc_Solar_MW", "roc_Solar_MW")
    roc_und_piv = _pivot_column(work_dedup, col_site, "roc_MW_underlying", "roc_MW_underlying")

    # ROC aggregates
    roc_aggs = work_dedup.groupby([col_site, "date"]).agg(
        roc_MW_max=("roc_MW", "max"),
        roc_MW_min=("roc_MW", "min"),
        roc_Solar_MW_max=("roc_Solar_MW", "max"),
        roc_Solar_MW_min=("roc_Solar_MW", "min"),
        roc_MW_underlying_max=("roc_MW_underlying", "max"),
        roc_MW_underlying_min=("roc_MW_underlying", "min"),
    ).reset_index()

    # --- Filter to daytime hours 6..18 inclusive ---
    daytime = work.loc[
        (work["hour"] >= noon_start) & (work["hour"] <= noon_end)
    ].copy()

    # --- Drop rows with null ground truth ---
    daytime = daytime.dropna(subset=[col_gt])

    # --- Label ---
    daytime[label_col2] = (daytime[col_gt] < 0).astype(int)

    # --- Restrict to XGB1-positive days ---
    daytime = daytime.merge(
        xgb1_positive_keys[[col_site, "date"]],
        on=[col_site, "date"],
        how="inner",
    )

    # --- Join XGB1 day-level features (exclude label + primary keys) ---
    xgb1_label = "is_having_reverse_power_flow_issue"
    day_feat_cols = [c for c in day_features_df.columns
                     if c not in {col_site, "date", xgb1_label}]
    day_feats = day_features_df[[col_site, "date"] + day_feat_cols].copy()

    joined = daytime.merge(day_feats, on=[col_site, "date"], how="inner")

    # --- Join MW_underlying day features ---
    joined = joined.merge(mw_und_piv, on=[col_site, "date"], how="left")
    joined = joined.merge(mw_und_agg, on=[col_site, "date"], how="left")
    joined = joined.merge(mw_und_noon, on=[col_site, "date"], how="left")

    # --- Join ROC day features ---
    joined = joined.merge(roc_mw_piv, on=[col_site, "date"], how="left")
    joined = joined.merge(roc_sol_piv, on=[col_site, "date"], how="left")
    joined = joined.merge(roc_und_piv, on=[col_site, "date"], how="left")
    joined = joined.merge(roc_aggs, on=[col_site, "date"], how="left")

    # --- Interval-time OHE (52 cols: 06:00..18:45) ---
    for v in _SLOTS_DAYTIME:
        joined[f"hhmm_{v}"] = (joined["hhmm"] == v).astype(int)

    # --- Build deterministic column ordering ---

    # Substation OHE (from day_feat_cols, starts with is_ but not is_holiday/is_weekend/is_Jan.. etc)
    calendar_set = {"is_holiday", "is_weekend"} | {f"is_{m}" for m in _MONTH_ABBREVS}
    substation_cols = sorted([c for c in day_feat_cols if c.startswith("is_") and c not in calendar_set])

    interval_ohe_cols = [f"hhmm_{v}" for v in _SLOTS_DAYTIME]

    calendar_cols = (["is_holiday", "is_weekend"]
                     + [f"is_{m}" for m in _MONTH_ABBREVS])
    calendar_cols = [c for c in calendar_cols if c in joined.columns]

    mw_interval_cols = [f"MW_{s}" for s in _SLOTS_96]
    mw_agg_cols_list = ["MW_max", "MW_noon_min"]

    solar_interval_cols = [f"Solar_MW_{s}" for s in _SLOTS_96]
    solar_agg_cols_list = ["Solar_MW_max", "Solar_MW_sum"]

    mw_und_interval_cols = [f"MW_underlying_{s}" for s in _SLOTS_96]
    mw_und_agg_cols_list = ["MW_underlying_max", "MW_underlying_noon_min"]

    roc_mw_cols = [f"roc_MW_{s}" for s in _SLOTS_96]
    roc_sol_cols = [f"roc_Solar_MW_{s}" for s in _SLOTS_96]
    roc_und_cols = [f"roc_MW_underlying_{s}" for s in _SLOTS_96]
    roc_agg_list = [
        "roc_MW_max", "roc_MW_min",
        "roc_Solar_MW_max", "roc_Solar_MW_min",
        "roc_MW_underlying_max", "roc_MW_underlying_min",
    ]

    missingness_cols = ["mw_missing_count", "solar_missing_count"]

    feat_cols2 = (
        substation_cols
        + interval_ohe_cols
        + calendar_cols
        + mw_interval_cols + mw_agg_cols_list
        + solar_interval_cols + solar_agg_cols_list
        + mw_und_interval_cols + mw_und_agg_cols_list
        + roc_mw_cols + roc_sol_cols + roc_und_cols + roc_agg_list
        + missingness_cols
    )

    # Defensive: only keep existing columns
    feat_cols2 = [c for c in feat_cols2 if c in joined.columns]

    # Final column selection
    final_cols = pkeys2 + [label_col2] + feat_cols2
    result = joined[final_cols].copy()

    print(f"XGB2 features: {len(feat_cols2)} columns, {len(result):,} rows "
          f"({result[label_col2].sum():,} positive)")

    return result, feat_cols2, label_col2

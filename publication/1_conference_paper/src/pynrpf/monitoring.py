from __future__ import annotations

from typing import Any

import pandas as pd


def _confidence_summary(values: pd.Series) -> dict[str, Any]:
    clean = values.dropna()
    if clean.empty:
        return {"status": "skipped", "reason": "no_confidence_values"}
    return {
        "status": "ok",
        "count": int(clean.shape[0]),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "p05": float(clean.quantile(0.05)),
        "p50": float(clean.quantile(0.50)),
        "p95": float(clean.quantile(0.95)),
    }


def _drift_summary(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    monitoring_cfg = cfg.get("monitoring", {})
    reference_stats = monitoring_cfg.get("reference_stats", {})
    if not isinstance(reference_stats, dict) or not reference_stats:
        return {"status": "skipped", "reason": "no_reference_stats"}

    cols = cfg["columns"]
    out: dict[str, Any] = {"status": "ok", "features": {}}
    for logical_name in ("net_load", "solar"):
        feature_ref = reference_stats.get(logical_name)
        if not isinstance(feature_ref, dict):
            continue

        col = cols[logical_name]
        cur_mean = float(df[col].mean(skipna=True))
        cur_std = float(df[col].std(skipna=True))

        ref_mean = feature_ref.get("mean")
        ref_std = feature_ref.get("std")
        z_score = None
        if ref_mean is not None and ref_std not in (None, 0):
            z_score = (cur_mean - float(ref_mean)) / float(ref_std)

        out["features"][logical_name] = {
            "current_mean": cur_mean,
            "current_std": cur_std,
            "reference_mean": ref_mean,
            "reference_std": ref_std,
            "mean_z_score": z_score,
        }

    if not out["features"]:
        return {"status": "skipped", "reason": "reference_stats_missing_expected_features"}
    return out


def build_operational_summary(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    model_name: str,
    data_quality_summary: dict[str, Any],
) -> dict[str, Any]:
    cols = cfg["columns"]
    site_col = cols["site"]
    ts_col = cols["timestamp"]

    work = df.copy()
    if "date" not in work.columns:
        work["date"] = pd.to_datetime(work[ts_col]).dt.date

    interval_flag_col = "pynrpf_interval_flag"
    day_flag_col = "pynrpf_day_flag"
    confidence_col = "pynrpf_confidence"

    rows_processed = int(len(work))
    rows_corrected = int(work.get(interval_flag_col, pd.Series(False)).fillna(False).sum())

    if day_flag_col in work.columns:
        day_flags = (
            work.groupby([site_col, "date"])[day_flag_col]
            .first()
            .fillna(False)
            .astype(bool)
        )
        predicted_positive_days = int(day_flags.sum())
    else:
        predicted_positive_days = 0

    predicted_positive_intervals = rows_corrected

    monitoring_cfg = cfg.get("monitoring", {})
    confidence_summary = (
        _confidence_summary(work[confidence_col])
        if monitoring_cfg.get("enable_confidence_summary", True) and confidence_col in work.columns
        else {"status": "skipped", "reason": "disabled_or_missing"}
    )
    drift_summary = (
        _drift_summary(work, cfg)
        if monitoring_cfg.get("enable_drift_summary", True)
        else {"status": "skipped", "reason": "disabled"}
    )

    return {
        "model": model_name,
        "rows_processed": rows_processed,
        "rows_corrected": rows_corrected,
        "predicted_positive_days": predicted_positive_days,
        "predicted_positive_intervals": predicted_positive_intervals,
        "data_quality": data_quality_summary,
        "confidence_summary": confidence_summary,
        "drift_summary": drift_summary,
    }

"""Leakage-safe direct seven-day-ahead forecasting for the Gamma case study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from _m9_pbm_features import select_best_candidates
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

CONDITION_COLUMNS = {
    "raw_uncorrected": "raw_uncorrected_MW",
    "m9_pbm_corrected": "m9_pbm_corrected_MW",
    "manually_corrected": "manually_corrected_MW",
}

CONDITION_LABELS = {
    "raw_uncorrected": "Raw uncorrected",
    "m9_pbm_corrected": "m9_pbm corrected",
    "manually_corrected": "Manually corrected",
}

MODEL_LABELS = {
    "seasonal_naive": "Seasonal naive",
    "linear_regression": "Linear regression",
    "xgboost": "XGBoost",
}


def load_beta_b_model(path: Path) -> dict[str, Any]:
    """Load and validate the label-isolated Beta-B m9_pbm artifact."""

    with path.open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    expected_features = {
        "F1_bridge_improvement",
        "F3_slope_continuity_improvement",
        "F4_duration_plausibility",
    }
    if model.get("heldout_substation") != "beta_B":
        raise ValueError("The Gamma correction artifact must hold out beta_B.")
    if model.get("heldout_labels_used") is not False:
        raise ValueError("The Gamma correction artifact must not use Beta-B labels.")
    if model.get("alpha_used") is not False:
        raise ValueError("The Gamma correction artifact must not use Alpha data.")
    if set(model.get("features", [])) != expected_features:
        raise ValueError("The Gamma correction artifact has unexpected features.")
    if set(model.get("weights", {})) != expected_features:
        raise ValueError("The Gamma correction artifact has unexpected weights.")
    return model


def apply_m9_pbm_correction(
    gamma: pd.DataFrame,
    candidates: pd.DataFrame,
    model: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create raw, m9-corrected, and manually corrected Gamma series."""

    substations = gamma["substation_id"].astype(str).unique().tolist()
    if substations != [model["heldout_substation"]]:
        raise ValueError(
            "Gamma substation does not match the correction artifact: "
            f"{substations!r}."
        )
    if candidates["substation_id"].astype(str).unique().tolist() != substations:
        raise ValueError("Candidate cache does not contain only the Gamma substation.")

    best = select_best_candidates(candidates, model["weights"])
    best["predicted_day"] = best["score"].ge(float(model["threshold"]))
    day_predictions = best[
        [
            "substation_id",
            "date",
            "candidate_id",
            "left_slot",
            "right_slot",
            "score",
            "predicted_day",
        ]
    ].copy()
    day_predictions = day_predictions.rename(columns={"score": "m9_score"})

    series = gamma.sort_values(["substation_id", "timestamp"]).copy()
    series["slot"] = series.groupby(["substation_id", "date"]).cumcount()
    if not series.groupby(["substation_id", "date"]).size().eq(96).all():
        raise ValueError("Gamma must contain exactly 96 quarter-hour slots per day.")
    series = series.merge(
        day_predictions,
        on=["substation_id", "date"],
        how="left",
        validate="many_to_one",
    )
    if series["m9_score"].isna().any():
        raise ValueError("At least one Gamma day is missing a candidate prediction.")

    series["predicted_interval"] = (
        series["predicted_day"]
        & series["slot"].ge(series["left_slot"])
        & series["slot"].le(series["right_slot"])
    )
    raw = pd.to_numeric(series["net_load_MW"], errors="coerce")
    series["raw_uncorrected_MW"] = raw
    series["m9_pbm_corrected_MW"] = raw.where(~series["predicted_interval"], -raw)
    series["manually_corrected_MW"] = raw.where(~series["label_interval"], -raw)
    return series, day_predictions


def _regression_errors(reference: pd.Series, estimate: pd.Series) -> dict[str, float]:
    truth = pd.to_numeric(reference, errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(estimate, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(truth) & np.isfinite(pred)
    if not valid.any():
        return {"n_valid": 0, "rmse_MW": np.nan, "mae_MW": np.nan}
    residual = pred[valid] - truth[valid]
    return {
        "n_valid": int(valid.sum()),
        "rmse_MW": float(np.sqrt(np.mean(np.square(residual)))),
        "mae_MW": float(np.mean(np.abs(residual))),
    }


def gamma_data_error_metrics(
    series: pd.DataFrame,
    *,
    test_start: str,
    test_end: str,
) -> pd.DataFrame:
    """Measure raw and m9 data errors against the manual correction."""

    timestamp = pd.to_datetime(series["timestamp"], utc=True)
    start = pd.Timestamp(test_start, tz="UTC")
    end = pd.Timestamp(f"{test_end} 23:45:00", tz="UTC")
    scopes = {
        "full_gamma": pd.Series(True, index=series.index),
        "forecast_test_month": timestamp.between(start, end, inclusive="both"),
    }
    rows: list[dict[str, Any]] = []
    for scope, mask in scopes.items():
        scope_rows: list[dict[str, Any]] = []
        for condition, column in list(CONDITION_COLUMNS.items())[:2]:
            metrics = _regression_errors(
                series.loc[mask, "manually_corrected_MW"],
                series.loc[mask, column],
            )
            scope_rows.append(
                {
                    "scope": scope,
                    "data_condition": condition,
                    "data_condition_label": CONDITION_LABELS[condition],
                    **metrics,
                }
            )
        raw = next(row for row in scope_rows if row["data_condition"] == "raw_uncorrected")
        for row in scope_rows:
            row["rmse_reduction_vs_raw_MW"] = raw["rmse_MW"] - row["rmse_MW"]
            row["mae_reduction_vs_raw_MW"] = raw["mae_MW"] - row["mae_MW"]
            row["rmse_reduction_vs_raw_pct"] = (
                100 * row["rmse_reduction_vs_raw_MW"] / raw["rmse_MW"]
                if raw["rmse_MW"]
                else np.nan
            )
            row["mae_reduction_vs_raw_pct"] = (
                100 * row["mae_reduction_vs_raw_MW"] / raw["mae_MW"]
                if raw["mae_MW"]
                else np.nan
            )
        rows.extend(scope_rows)
    return pd.DataFrame(rows)


def _utc_timestamp(value: str, *, end_of_day: bool = False) -> pd.Timestamp:
    suffix = " 23:45:00" if end_of_day else " 00:00:00"
    return pd.Timestamp(f"{value}{suffix}", tz="UTC")


def forecast_feature_columns(lookback_days: int = 14) -> list[str]:
    """Return the declared direct-forecast feature columns."""

    lag_columns = ["origin_value_MW"] + [
        f"origin_minus_{day}d_MW" for day in range(1, lookback_days)
    ]
    summary_columns = [
        "lookback_mean_MW",
        "lookback_std_MW",
        "lookback_min_MW",
        "lookback_max_MW",
        "lookback_p05_MW",
        "lookback_p95_MW",
        "last_day_mean_MW",
        "last_day_min_MW",
        "last_day_max_MW",
    ]
    calendar_columns = [
        "target_time_sin",
        "target_time_cos",
        "target_dow_sin",
        "target_dow_cos",
        "target_month_sin",
        "target_month_cos",
        "target_is_weekend",
    ]
    return lag_columns + summary_columns + calendar_columns


def build_forecast_examples(
    series: pd.DataFrame,
    series_column: str,
    *,
    target_start: str,
    target_end: str,
    horizon_days: int = 7,
    lookback_days: int = 14,
) -> pd.DataFrame:
    """Build direct point-forecast examples without observations after the origin."""

    work = series.sort_values("timestamp").copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    if work["timestamp"].duplicated().any():
        raise ValueError("Forecast input timestamps must be unique.")
    work = work.set_index("timestamp")
    expected_frequency = work.index.to_series().diff().dropna()
    if not expected_frequency.eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("Forecast input must be a complete 15-minute time grid.")

    horizon_slots = horizon_days * 96
    lookback_slots = lookback_days * 96
    values = pd.to_numeric(work[series_column], errors="coerce")
    reference = pd.to_numeric(work["manually_corrected_MW"], errors="coerce")
    history_at_origin = values.shift(horizon_slots)

    examples = pd.DataFrame(index=work.index)
    examples["target_timestamp"] = work.index
    examples["origin_timestamp"] = work.index - pd.Timedelta(days=horizon_days)
    examples["origin_value_MW"] = history_at_origin
    for day in range(1, lookback_days):
        examples[f"origin_minus_{day}d_MW"] = values.shift(horizon_slots + day * 96)

    rolling = history_at_origin.rolling(lookback_slots, min_periods=1)
    examples["lookback_mean_MW"] = rolling.mean()
    examples["lookback_std_MW"] = rolling.std(ddof=0)
    examples["lookback_min_MW"] = rolling.min()
    examples["lookback_max_MW"] = rolling.max()
    examples["lookback_p05_MW"] = rolling.quantile(0.05)
    examples["lookback_p95_MW"] = rolling.quantile(0.95)
    examples["history_observations"] = rolling.count()

    last_day = history_at_origin.rolling(96, min_periods=1)
    examples["last_day_mean_MW"] = last_day.mean()
    examples["last_day_min_MW"] = last_day.min()
    examples["last_day_max_MW"] = last_day.max()

    minute = work.index.hour * 60 + work.index.minute
    day_of_week = work.index.dayofweek
    month = work.index.month
    examples["target_time_sin"] = np.sin(2 * np.pi * minute / 1440)
    examples["target_time_cos"] = np.cos(2 * np.pi * minute / 1440)
    examples["target_dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    examples["target_dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    examples["target_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    examples["target_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    examples["target_is_weekend"] = (day_of_week >= 5).astype(int)
    examples["y_condition"] = values
    examples["y_reference"] = reference

    first_complete_target = work.index.min() + pd.Timedelta(
        days=horizon_days + lookback_days
    )
    start = max(_utc_timestamp(target_start), first_complete_target)
    end = _utc_timestamp(target_end, end_of_day=True)
    selected = examples.loc[examples.index.to_series().between(start, end)]
    return selected.reset_index(drop=True)


def fit_direct_forecasts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    data_condition: str,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit LR and XGBoost once, then forecast all September targets."""

    lookback_days = int(config["forecast"]["lookback_days"])
    feature_columns = forecast_feature_columns(lookback_days)
    fit_rows = train["y_condition"].notna()
    if not fit_rows.any():
        raise ValueError(f"No finite training targets for {data_condition}.")
    x_train = train.loc[fit_rows, feature_columns]
    y_train = train.loc[fit_rows, "y_condition"]
    x_test = test[feature_columns]

    linear = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    xgb_config = config["forecast"]["xgboost"]
    xgboost = XGBRegressor(
        objective=xgb_config["objective"],
        tree_method=xgb_config["tree_method"],
        learning_rate=float(xgb_config["eta"]),
        n_estimators=int(xgb_config["n_estimators"]),
        max_depth=int(xgb_config["max_depth"]),
        subsample=float(xgb_config["subsample"]),
        colsample_bytree=float(xgb_config["colsample_bytree"]),
        random_state=int(xgb_config["seed"]),
        n_jobs=4,
        verbosity=0,
    )

    common = test[
        [
            "target_timestamp",
            "origin_timestamp",
            "y_condition",
            "y_reference",
            "history_observations",
        ]
    ].copy()
    frames: list[pd.DataFrame] = []
    for model_name, estimator in [
        ("linear_regression", linear),
        ("xgboost", xgboost),
    ]:
        estimator.fit(x_train, y_train)
        predicted = common.copy()
        predicted["model"] = model_name
        predicted["y_pred"] = estimator.predict(x_test)
        frames.append(predicted)

    seasonal = common.copy()
    seasonal["model"] = "seasonal_naive"
    seasonal["y_pred"] = test["origin_value_MW"].to_numpy(dtype=float)
    frames.insert(0, seasonal)
    predictions = pd.concat(frames, ignore_index=True)
    predictions["data_condition"] = data_condition
    predictions["status"] = "complete"
    predictions["is_placeholder"] = False

    audit = pd.DataFrame(
        [
            {
                "data_condition": data_condition,
                "training_examples": int(fit_rows.sum()),
                "training_target_start": train.loc[fit_rows, "target_timestamp"].min(),
                "training_target_end": train.loc[fit_rows, "target_timestamp"].max(),
                "test_examples": len(test),
                "test_target_start": test["target_timestamp"].min(),
                "test_target_end": test["target_timestamp"].max(),
                "fit_count_per_learned_model": 1,
                "maximum_test_observation_timestamp": test["origin_timestamp"].max(),
            }
        ]
    )
    return predictions, audit


def forecast_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute RMSE and MAE against manually corrected Gamma targets."""

    rows: list[dict[str, Any]] = []
    for (condition, model), group in predictions.groupby(
        ["data_condition", "model"], sort=True
    ):
        metrics = _regression_errors(group["y_reference"], group["y_pred"])
        rows.append(
            {
                "data_condition": condition,
                "data_condition_label": CONDITION_LABELS[condition],
                "model": model,
                "model_label": MODEL_LABELS[model],
                "n_targets_total": len(group),
                **metrics,
                "status": "complete",
                "is_placeholder": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "data_condition"]).reset_index(drop=True)


def forecast_impact_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare raw, m9-corrected, and ideal manual training conditions by model."""

    wide = metrics.pivot(index="model", columns="data_condition", values="rmse_MW")
    rows = []
    for model, values in wide.iterrows():
        raw = float(values["raw_uncorrected"])
        corrected = float(values["m9_pbm_corrected"])
        manual = float(values["manually_corrected"])
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "raw_rmse_MW": raw,
                "m9_pbm_corrected_rmse_MW": corrected,
                "manually_corrected_rmse_MW": manual,
                "m9_rmse_reduction_vs_raw_MW": raw - corrected,
                "m9_rmse_reduction_vs_raw_pct": 100 * (raw - corrected) / raw,
                "remaining_gap_to_manual_rmse_MW": corrected - manual,
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)

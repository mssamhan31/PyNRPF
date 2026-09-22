"""Gamma case study: seven-day-ahead net-load forecasts on raw, M9-corrected and manually corrected data.

Inputs:  the Gamma dataset (Beta station B, one year, every day), the M9 site-day
         decisions for beta_B from its held-out fold, the settings.
Outputs: the three Gamma series, the data-error metrics, the forecast predictions and
         metrics, the impact table, and four figures: example week, data-error RMSE,
         grouped forecast RMSE, residuals.
Key steps: apply the held-out M9 decision to each Gamma day (raw kept unless the day is
         AUTO_CORRECT, then the sign flips inside the window); build direct point-forecast
         examples with no observation after the origin; fit linear regression and
         XGBoost once per data condition on targets before the test month; forecast the
         test month; score against the manually corrected reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from pynrpf.m9 import AUTO_CORRECT, UNCERTAIN

from .config import Settings
from .style import COLORS, apply_journal_style, save_figure, style_axis

CONDITION_COLUMNS = {"raw_uncorrected": "raw_uncorrected_MW", "m9_corrected": "m9_corrected_MW",
                     "manually_corrected": "manually_corrected_MW"}
CONDITION_LABELS = {"raw_uncorrected": "Raw uncorrected", "m9_corrected": "M9 corrected",
                    "manually_corrected": "Manually corrected"}
MODEL_LABELS = {"seasonal_naive": "Seasonal naive", "linear_regression": "Linear regression", "xgboost": "XGBoost"}
MODELS = list(MODEL_LABELS)
CONDITIONS = list(CONDITION_COLUMNS)


# ----------------------------------------------------------------------------- series

def load_gamma(settings: Settings) -> pd.DataFrame:
    """The Gamma interval rows with UTC timestamps, date strings and a slot index."""
    cols = settings.columns
    df = pd.read_parquet(settings.dataset("gamma"))
    df["station"] = df[cols["site"]].astype(str)
    df["timestamp"] = pd.to_datetime(df[cols["timestamp"]], utc=True)
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)
    df["slot"] = df.groupby(["station", "date"]).cumcount()
    if not df.groupby(["station", "date"]).size().eq(96).all():
        raise ValueError("Gamma must contain exactly 96 quarter-hour slots per day.")
    return df


def apply_m9_correction(gamma: pd.DataFrame, m9_site_days: pd.DataFrame,
                        settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw, M9-corrected and manually corrected Gamma series from the held-out M9 decisions.

    A Gamma day without an M9 decision (a day that was incomplete in the Beta
    population and therefore never scored) keeps its raw values; the audit table
    counts such days.
    """
    station = settings["gamma"]["station"]
    if gamma["station"].unique().tolist() != [station]:
        raise ValueError(f"Gamma must contain only {station}.")
    decision_cols = ["date", "outcome", "prob_day", "pred_start", "pred_end", "fold_id"]
    decisions = m9_site_days[(m9_site_days["station"] == station)][decision_cols].copy()
    if decisions["fold_id"].nunique() > 1:
        raise ValueError("Gamma decisions must come from one held-out fold.")
    series = gamma.merge(decisions, on="date", how="left", validate="many_to_one")
    scored = series["outcome"].notna()
    series["m9_applied"] = (scored & (series["outcome"] == AUTO_CORRECT) & (series["pred_start"] >= 0)
                            & (series["slot"] >= series["pred_start"]) & (series["slot"] <= series["pred_end"]))
    raw = pd.to_numeric(series[settings.columns["net_load"]], errors="coerce")
    series["raw_uncorrected_MW"] = raw
    series["m9_corrected_MW"] = raw.where(~series["m9_applied"], -raw)
    series["manually_corrected_MW"] = raw.where(~series[settings.columns["label_interval"]].astype(bool), -raw)
    audit = pd.DataFrame([{
        "station": station, "gamma_days": series["date"].nunique(),
        "days_with_m9_decision": int(decisions["date"].nunique()),
        "days_without_decision_kept_raw": int(series.loc[~scored, "date"].nunique()),
        "days_auto_corrected": int(decisions["outcome"].eq(AUTO_CORRECT).sum()),
        "days_uncertain": int(decisions["outcome"].eq(UNCERTAIN).sum()),
        "held_out_fold": decisions["fold_id"].iloc[0] if len(decisions) else "",
    }])
    return series, audit


def _regression_errors(reference: pd.Series, estimate: pd.Series) -> dict[str, float]:
    truth = pd.to_numeric(reference, errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(estimate, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(truth) & np.isfinite(pred)
    if not valid.any():
        return {"n_valid": 0, "rmse_MW": np.nan, "mae_MW": np.nan}
    residual = pred[valid] - truth[valid]
    return {"n_valid": int(valid.sum()), "rmse_MW": float(np.sqrt(np.mean(residual ** 2))),
            "mae_MW": float(np.mean(np.abs(residual)))}


def data_error_metrics(series: pd.DataFrame, test_start: str, test_end: str) -> pd.DataFrame:
    """Raw and M9 data error against the manual correction, full year and test month."""
    start = pd.Timestamp(test_start, tz="UTC")
    end = pd.Timestamp(f"{test_end} 23:45:00", tz="UTC")
    scopes = {"full_gamma": pd.Series(True, index=series.index),
              "forecast_test_month": series["timestamp"].between(start, end, inclusive="both")}
    rows: list[dict[str, Any]] = []
    for scope, mask in scopes.items():
        scope_rows = []
        for condition in CONDITIONS[:2]:
            metrics = _regression_errors(series.loc[mask, "manually_corrected_MW"],
                                         series.loc[mask, CONDITION_COLUMNS[condition]])
            scope_rows.append({"scope": scope, "data_condition": condition,
                               "data_condition_label": CONDITION_LABELS[condition], **metrics})
        raw = scope_rows[0]
        for row in scope_rows:
            row["rmse_reduction_vs_raw_MW"] = raw["rmse_MW"] - row["rmse_MW"]
            row["mae_reduction_vs_raw_MW"] = raw["mae_MW"] - row["mae_MW"]
            for error in ("rmse", "mae"):
                reduction = row[f"{error}_reduction_vs_raw_MW"]
                baseline = raw[f"{error}_MW"]
                row[f"{error}_reduction_vs_raw_pct"] = 100 * reduction / baseline if baseline else np.nan
        rows.extend(scope_rows)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- forecasting

def _utc(value: str, end_of_day: bool = False) -> pd.Timestamp:
    return pd.Timestamp(f"{value}{' 23:45:00' if end_of_day else ' 00:00:00'}", tz="UTC")


def forecast_feature_columns(lookback_days: int = 14) -> list[str]:
    """The declared direct-forecast feature columns."""
    lags = ["origin_value_MW"] + [f"origin_minus_{d}d_MW" for d in range(1, lookback_days)]
    summary = ["lookback_mean_MW", "lookback_std_MW", "lookback_min_MW", "lookback_max_MW", "lookback_p05_MW",
               "lookback_p95_MW", "last_day_mean_MW", "last_day_min_MW", "last_day_max_MW"]
    calendar = ["target_time_sin", "target_time_cos", "target_dow_sin", "target_dow_cos", "target_month_sin",
                "target_month_cos", "target_is_weekend"]
    return lags + summary + calendar


def build_forecast_examples(series: pd.DataFrame, series_column: str, target_start: str, target_end: str,
                            horizon_days: int = 7, lookback_days: int = 14) -> pd.DataFrame:
    """Direct point-forecast examples; no feature observes a value after the origin."""
    work = series.sort_values("timestamp").set_index("timestamp")
    if work.index.duplicated().any():
        raise ValueError("Forecast input timestamps must be unique.")
    if not work.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("Forecast input must be a complete 15-minute grid.")
    horizon = horizon_days * 96
    values = pd.to_numeric(work[series_column], errors="coerce")
    reference = pd.to_numeric(work["manually_corrected_MW"], errors="coerce")
    at_origin = values.shift(horizon)
    ex = pd.DataFrame(index=work.index)
    ex["target_timestamp"] = work.index
    ex["origin_timestamp"] = work.index - pd.Timedelta(days=horizon_days)
    ex["origin_value_MW"] = at_origin
    for d in range(1, lookback_days):
        ex[f"origin_minus_{d}d_MW"] = values.shift(horizon + d * 96)
    rolling = at_origin.rolling(lookback_days * 96, min_periods=1)
    ex["lookback_mean_MW"], ex["lookback_std_MW"] = rolling.mean(), rolling.std(ddof=0)
    ex["lookback_min_MW"], ex["lookback_max_MW"] = rolling.min(), rolling.max()
    ex["lookback_p05_MW"], ex["lookback_p95_MW"] = rolling.quantile(0.05), rolling.quantile(0.95)
    ex["history_observations"] = rolling.count()
    last = at_origin.rolling(96, min_periods=1)
    ex["last_day_mean_MW"], ex["last_day_min_MW"], ex["last_day_max_MW"] = last.mean(), last.min(), last.max()
    minute = work.index.hour * 60 + work.index.minute
    dow, month = work.index.dayofweek, work.index.month
    ex["target_time_sin"], ex["target_time_cos"] = np.sin(2 * np.pi * minute / 1440), np.cos(2 * np.pi * minute / 1440)
    ex["target_dow_sin"], ex["target_dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    month_angle = 2 * np.pi * (month - 1) / 12
    ex["target_month_sin"], ex["target_month_cos"] = np.sin(month_angle), np.cos(month_angle)
    ex["target_is_weekend"] = (dow >= 5).astype(int)
    ex["y_condition"], ex["y_reference"] = values, reference
    first_complete = work.index.min() + pd.Timedelta(days=horizon_days + lookback_days)
    start = max(_utc(target_start), first_complete)
    selected = ex.loc[ex.index.to_series().between(start, _utc(target_end, end_of_day=True))]
    return selected.reset_index(drop=True)


def fit_direct_forecasts(train: pd.DataFrame, test: pd.DataFrame, data_condition: str,
                         settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit linear regression and XGBoost once, then forecast every test target."""
    g = settings["gamma"]
    features = forecast_feature_columns(int(g["lookback_days"]))
    fit_rows = train["y_condition"].notna()
    if not fit_rows.any():
        raise ValueError(f"No finite training targets for {data_condition}.")
    x_train, y_train, x_test = train.loc[fit_rows, features], train.loc[fit_rows, "y_condition"], test[features]
    linear = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                       ("regressor", LinearRegression())])
    xg = g["xgboost"]
    xgboost = XGBRegressor(objective=xg["objective"], tree_method=xg["tree_method"], learning_rate=float(xg["eta"]),
                           n_estimators=int(xg["n_estimators"]), max_depth=int(xg["max_depth"]),
                           subsample=float(xg["subsample"]), colsample_bytree=float(xg["colsample_bytree"]),
                           random_state=int(xg["seed"]), n_jobs=4, verbosity=0)
    common = test[["target_timestamp", "origin_timestamp", "y_condition", "y_reference", "history_observations"]].copy()
    frames = []
    seasonal = common.copy()
    seasonal["model"], seasonal["y_pred"] = "seasonal_naive", test["origin_value_MW"].to_numpy(float)
    frames.append(seasonal)
    for name, estimator in (("linear_regression", linear), ("xgboost", xgboost)):
        estimator.fit(x_train, y_train)
        predicted = common.copy()
        predicted["model"], predicted["y_pred"] = name, estimator.predict(x_test)
        frames.append(predicted)
    predictions = pd.concat(frames, ignore_index=True)
    predictions["data_condition"] = data_condition
    audit = pd.DataFrame([{
        "data_condition": data_condition, "training_examples": int(fit_rows.sum()),
        "training_target_start": train.loc[fit_rows, "target_timestamp"].min(),
        "training_target_end": train.loc[fit_rows, "target_timestamp"].max(),
        "test_examples": len(test), "test_target_start": test["target_timestamp"].min(),
        "test_target_end": test["target_timestamp"].max(),
        "fit_count_per_learned_model": 1, "maximum_test_observation_timestamp": test["origin_timestamp"].max(),
    }])
    return predictions, audit


def forecast_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """RMSE and MAE against the manually corrected targets per condition and model."""
    rows = []
    for (condition, model), group in predictions.groupby(["data_condition", "model"], sort=True):
        rows.append({"data_condition": condition, "data_condition_label": CONDITION_LABELS[condition], "model": model,
                     "model_label": MODEL_LABELS[model], "n_targets_total": len(group),
                     **_regression_errors(group["y_reference"], group["y_pred"])})
    return pd.DataFrame(rows).sort_values(["model", "data_condition"]).reset_index(drop=True)


def forecast_impact_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Raw versus M9-corrected versus manual training conditions, per model."""
    wide = metrics.pivot(index="model", columns="data_condition", values="rmse_MW")
    rows = []
    for model, v in wide.iterrows():
        raw, corrected, manual = float(v["raw_uncorrected"]), float(v["m9_corrected"]), float(v["manually_corrected"])
        rows.append({"model": model, "model_label": MODEL_LABELS[model], "raw_rmse_MW": raw,
                     "m9_corrected_rmse_MW": corrected, "manually_corrected_rmse_MW": manual,
                     "m9_rmse_reduction_vs_raw_MW": raw - corrected,
                     "m9_rmse_reduction_vs_raw_pct": 100 * (raw - corrected) / raw if raw else np.nan,
                     "remaining_gap_to_manual_rmse_MW": corrected - manual})
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


# ----------------------------------------------------------------------------- figures

def draw_example_week(axis: Any, series: pd.DataFrame, test_start: str, test_end: str) -> pd.Timestamp:
    """The test-month week with the largest manual correction: raw, M9 and manual overlaid. Returns its start."""
    in_month = series["timestamp"].between(_utc(test_start), _utc(test_end, end_of_day=True), inclusive="both")
    month = series[in_month].copy()
    month["week_start"] = month["timestamp"].dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time
    month["manual_change"] = (month["manually_corrected_MW"] - month["raw_uncorrected_MW"]).abs()
    week_start = month.groupby("week_start")["manual_change"].sum().idxmax()
    naive = month["timestamp"].dt.tz_localize(None)
    example = month[naive.between(week_start, week_start + pd.Timedelta(days=7), inclusive="left")]
    axis.plot(example["timestamp"], example["raw_uncorrected_MW"], color=COLORS["grey"], linewidth=1.2,
              label="Raw uncorrected")
    axis.plot(example["timestamp"], example["m9_corrected_MW"], color=COLORS["orange"], linewidth=1.6,
              label="M9 corrected")
    axis.plot(example["timestamp"], example["manually_corrected_MW"], color=COLORS["dark_blue"], linewidth=1.2,
              linestyle="--", label="Manually corrected reference")
    axis.axhline(0, color=COLORS["dark_blue"], linewidth=0.7)
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Net load (MW)")
    axis.xaxis.set_major_locator(mdates.DayLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    lo = float(example[list(CONDITION_COLUMNS.values())].min().min())
    hi = float(example[list(CONDITION_COLUMNS.values())].max().max())
    span = max(hi - lo, 1.0)
    axis.set_ylim(lo - 0.03 * span, hi + 0.28 * span)
    axis.legend(ncol=3, loc="upper center")
    style_axis(axis)
    return week_start


def plot_example_week(series: pd.DataFrame, test_start: str, test_end: str, path: Path) -> Path:
    apply_journal_style()
    figure, axis = plt.subplots(figsize=(10.2, 4.2))
    week_start = draw_example_week(axis, series, test_start, test_end)
    axis.set_title(f"Gamma substation B: highest-impact week of the test month ({week_start.date()})")
    return save_figure(figure, path)[0]


def plot_data_error(data_metrics: pd.DataFrame, path: Path) -> Path:
    """Raw versus M9 data-error RMSE over the full year and the test month."""
    apply_journal_style()
    order, labels = ["full_gamma", "forecast_test_month"], ["Full Gamma year", "September 2024"]
    raw = data_metrics[data_metrics["data_condition"] == "raw_uncorrected"].set_index("scope").loc[order, "rmse_MW"]
    corrected = data_metrics[data_metrics["data_condition"] == "m9_corrected"].set_index("scope").loc[order, "rmse_MW"]
    x, width = np.arange(len(order)), 0.34
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(x - width / 2, raw, width, color=COLORS["grey"], label="Raw uncorrected")
    axis.bar(x + width / 2, corrected, width, color=COLORS["orange"], label="M9 corrected")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Data-error RMSE (MW)")
    axis.set_title("Error against the manually corrected Gamma reference")
    axis.legend()
    style_axis(axis)
    return save_figure(figure, path)[0]


def draw_forecast_rmse(axis: Any, metrics: pd.DataFrame) -> None:
    """Grouped bars: data condition on the x axis, one bar per forecast model."""
    colors = [COLORS["dark_blue"], COLORS["orange"], COLORS["grey"]]
    condition_labels = ["Raw uncorrected", "M9 corrected", "Manual reference"]
    x, width = np.arange(len(CONDITIONS)), 0.24
    for k, (model, color) in enumerate(zip(MODELS, colors, strict=True)):
        values = metrics[metrics["model"] == model].set_index("data_condition").loc[CONDITIONS, "rmse_MW"]
        axis.bar(x + (k - 1) * width, values, width, color=color, label=MODEL_LABELS[model])
    axis.set_xticks(x, condition_labels)
    axis.set_ylabel("Seven-day-ahead RMSE (MW)")
    axis.set_ylim(0, float(metrics["rmse_MW"].max()) * 1.25)
    axis.legend(ncol=3, loc="upper center")
    style_axis(axis)


def plot_forecast_rmse(metrics: pd.DataFrame, path: Path) -> Path:
    apply_journal_style()
    figure, axis = plt.subplots(figsize=(8.4, 4.5))
    draw_forecast_rmse(axis, metrics)
    axis.set_title("Direct September 2024 point forecasts")
    return save_figure(figure, path)[0]


def plot_forecast_residuals(predictions: pd.DataFrame, path: Path) -> Path:
    """Residual distributions by model and data condition."""
    apply_journal_style()
    data = predictions.assign(residual_MW=predictions["y_pred"] - predictions["y_reference"])
    colors, labels = [COLORS["grey"], COLORS["orange"], COLORS["dark_blue"]], ["Raw", "M9", "Manual"]
    positions, values, box_colors, ticks = [], [], [], []
    for mi, model in enumerate(MODELS):
        centre = mi * 4 + 2
        ticks.append(centre)
        for ci, condition in enumerate(CONDITIONS):
            selected = (data["model"] == model) & (data["data_condition"] == condition)
            residual = data.loc[selected, "residual_MW"].dropna()
            positions.append(centre + ci - 1)
            values.append(residual.to_numpy())
            box_colors.append(colors[ci])
    figure, axis = plt.subplots(figsize=(9.4, 4.6))
    boxes = axis.boxplot(values, positions=positions, widths=0.72, patch_artist=True, showfliers=False,
                         medianprops={"color": COLORS["dark_blue"], "linewidth": 1.0})
    for box, color in zip(boxes["boxes"], box_colors, strict=True):
        box.set_facecolor(color)
        box.set_alpha(0.82)
    axis.axhline(0, color=COLORS["dark_blue"], linewidth=0.8)
    axis.set_xticks(ticks, [MODEL_LABELS[m] for m in MODELS])
    axis.set_ylabel("Forecast residual (MW)")
    axis.set_title("September forecast residuals against the manual reference")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, label=lb) for c, lb in zip(colors, labels, strict=True)]
    axis.legend(handles=handles, ncol=3, loc="upper center")
    style_axis(axis)
    return save_figure(figure, path)[0]


def run_study(gamma: pd.DataFrame, m9_site_days: pd.DataFrame, settings: Settings, folder: Path) -> dict[str, Any]:
    """The whole Gamma study: series, data error, forecasts, metrics, impact, four figures."""
    g = settings["gamma"]
    test_start, test_end = g["forecast_test_start"], g["forecast_test_end"]
    series, audit = apply_m9_correction(gamma, m9_site_days, settings)
    data_errors = data_error_metrics(series, test_start, test_end)
    train_start = series["timestamp"].min().strftime("%Y-%m-%d")
    train_end = (pd.Timestamp(test_start) - pd.Timedelta(minutes=15)).strftime("%Y-%m-%d")
    horizon, lookback = int(g["horizon_days"]), int(g["lookback_days"])
    predictions, fit_audits = [], []
    for condition, column in CONDITION_COLUMNS.items():
        train = build_forecast_examples(series, column, train_start, train_end, horizon, lookback)
        test = build_forecast_examples(series, column, test_start, test_end, horizon, lookback)
        if not train["target_timestamp"].max() < _utc(test_start):
            raise ValueError("A training target lies inside the test month.")
        pred, fit_audit = fit_direct_forecasts(train, test, condition, settings)
        predictions.append(pred)
        fit_audits.append(fit_audit)
    predictions = pd.concat(predictions, ignore_index=True)
    metrics = forecast_metric_rows(predictions)
    impact = forecast_impact_table(metrics)
    folder.mkdir(parents=True, exist_ok=True)
    outputs = {
        "series": folder / "gamma_series.parquet", "correction_audit": folder / "gamma_correction_audit.csv",
        "data_error_metrics": folder / "gamma_data_error_metrics.csv",
        "forecast_predictions": folder / "gamma_forecast_predictions.parquet",
        "fit_audit": folder / "gamma_forecast_fit_audit.csv", "forecast_metrics": folder / "gamma_forecast_metrics.csv",
        "forecast_impact": folder / "gamma_forecast_impact.csv",
    }
    series.to_parquet(outputs["series"], index=False)
    audit.to_csv(outputs["correction_audit"], index=False)
    data_errors.to_csv(outputs["data_error_metrics"], index=False)
    predictions.to_parquet(outputs["forecast_predictions"], index=False)
    pd.concat(fit_audits, ignore_index=True).to_csv(outputs["fit_audit"], index=False)
    metrics.to_csv(outputs["forecast_metrics"], index=False)
    impact.to_csv(outputs["forecast_impact"], index=False)
    figures = [
        plot_example_week(series, test_start, test_end, folder / "fig01_gamma_raw_m9_manual_example_week.png"),
        plot_data_error(data_errors, folder / "fig02_gamma_data_error_rmse.png"),
        plot_forecast_rmse(metrics, folder / "fig03_gamma_forecast_rmse.png"),
        plot_forecast_residuals(predictions, folder / "fig04_gamma_forecast_residuals.png"),
    ]
    return {"outputs": list(outputs.values()) + figures, "audit": audit, "data_errors": data_errors,
            "metrics": metrics, "impact": impact}

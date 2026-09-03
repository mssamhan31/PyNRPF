from __future__ import annotations

import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ARTICLE_ROOT = Path(__file__).resolve().parents[1] / "publication" / "2_journal_article"
NOTEBOOK_DIR = ARTICLE_ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOK_DIR))

import _experiment_helpers as helpers  # noqa: E402


@lru_cache(maxsize=None)
def cached_config() -> dict:
    return helpers.load_config(ARTICLE_ROOT)


@lru_cache(maxsize=None)
def cached_dataset(dataset_key: str) -> pd.DataFrame:
    return helpers.load_dataset(ARTICLE_ROOT, cached_config(), dataset_key)


def tiny_forecast_config() -> dict:
    cfg = deepcopy(cached_config())
    cfg["windows"]["gamma_forecast_test_start"] = "2024-09-01"
    cfg["windows"]["gamma_forecast_test_end"] = "2024-09-02"
    return cfg


def tiny_gamma_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-08-01 00:00", "2024-09-02 23:45", freq="15min")
    minutes = timestamps.hour * 60 + timestamps.minute
    net_load = 1.8 + 0.4 * pd.Series(np.sin(2 * np.pi * minutes / 1440)).to_numpy()
    label_interval = (timestamps >= "2024-09-01 10:00") & (timestamps <= "2024-09-01 13:45")
    frame = pd.DataFrame(
        {
            "substation_id": "beta_B",
            "date": timestamps.strftime("%Y-%m-%d"),
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "net_load_MW": net_load,
            "solar_MW": 0.0,
            "label_interval": label_interval,
            "label_day": label_interval,
            "confidence": "sure",
        }
    )
    return helpers.prepare_dataset(frame[helpers.CONFIDENCE_COLUMNS], "Tiny Gamma")


def test_journal_palette_constants_are_available() -> None:
    assert helpers.JOURNAL_COLORS["orange"] == "#eb932c"
    assert helpers.JOURNAL_COLORS["dark_blue"] == "#22303d"
    assert helpers.JOURNAL_COLORS["grey"] == "#2F4D67"
    assert helpers.JOURNAL_COLORS["light_grey"] == "#5C7D99"
    assert helpers.JOURNAL_COLORS["light_white"] == "#ebe3e3"
    assert len(helpers.JOURNAL_BAR_COLORS) >= 4


def test_config_paths_and_schema_resolve() -> None:
    cfg = cached_config()
    paths = helpers.article_paths(ARTICLE_ROOT, cfg)

    assert cfg["schema_version"] == "journal_v2"
    assert (ARTICLE_ROOT / cfg["paths"]["alpha_dataset_path"]).exists()
    assert (ARTICLE_ROOT / cfg["paths"]["beta_dataset_path"]).exists()
    assert (ARTICLE_ROOT / cfg["paths"]["gamma_dataset_path"]).exists()
    assert cfg["paths"]["alpha_dataset_path"].endswith(".parquet")
    assert cfg["paths"]["beta_dataset_path"].endswith(".parquet")
    assert cfg["paths"]["gamma_dataset_path"].endswith(".parquet")
    assert not any(key.endswith("_csv") for key in cfg["paths"])
    assert paths.final.name == "final"
    assert paths.intermediate.name == "intermediate"
    assert paths.metrics.name == "metrics"


def test_misc_diagnostics_outputs_are_ignored_but_summary_is_tracked() -> None:
    gitignore = (ARTICLE_ROOT.parents[1] / ".gitignore").read_text(encoding="utf-8")
    summary_path = (
        NOTEBOOK_DIR / "99_Misc" / "2026-06-24_beta_m8_diagnostics_summary.md"
    )
    summary = summary_path.read_text(encoding="utf-8")

    assert "publication/2_journal_article/notebooks/99_Misc/outputs/" in gitignore
    assert summary_path.exists()
    assert "site-specific" in summary
    assert "normalisation" in summary
    assert "0.64" in summary
    assert "m9" in summary


def test_real_data_rankings_match_current_labels() -> None:
    cfg = cached_config()
    alpha = cached_dataset("alpha")
    beta = cached_dataset("beta")
    gamma = cached_dataset("gamma")

    assert sorted(alpha["substation_id"].unique()) == [
        "alpha_A",
        "alpha_B",
        "alpha_C",
        "alpha_D",
        "alpha_E",
        "alpha_F",
        "alpha_G",
        "alpha_H",
        "alpha_I",
        "alpha_J",
    ]
    assert sorted(beta["substation_id"].unique()) == [
        "beta_A",
        "beta_B",
        "beta_C",
        "beta_D",
        "beta_E",
        "beta_F",
        "beta_G",
        "beta_H",
    ]
    assert helpers.alpha_loso_sites(alpha, cfg) == [
        "alpha_F",
        "alpha_E",
        "alpha_G",
        "alpha_C",
        "alpha_J",
        "alpha_I",
        "alpha_B",
        "alpha_A",
        "alpha_D",
        "alpha_H",
    ]
    assert helpers.select_gamma_site(beta, cfg) == "beta_B"
    assert beta["date"].min() == "2023-10-01"
    assert beta["date"].max() == "2024-09-30"
    assert len(beta) == 280_800
    assert beta[["substation_id", "date"]].drop_duplicates().shape[0] == 2_928
    beta_confidence = beta[["substation_id", "date", "confidence"]].drop_duplicates()
    assert beta_confidence["confidence"].value_counts().to_dict() == {
        "sure": 2310,
        "unsure": 618,
    }
    assert gamma["substation_id"].nunique() == 1
    assert gamma["substation_id"].iloc[0] == "beta_B"
    assert len(gamma) == 35_136
    gamma_confidence = gamma[["substation_id", "date", "confidence"]].drop_duplicates()
    assert gamma_confidence["confidence"].value_counts().to_dict() == {
        "sure": 231,
        "unsure": 135,
    }


def test_binary_metrics_and_daytime_interval_scope() -> None:
    cfg = cached_config()
    frame = pd.DataFrame(
        {
            "substation_id": ["A"] * 4,
            "date": ["2024-09-01"] * 4,
            "hour": [5, 6, 12, 19],
            "label_interval": [True, True, False, True],
            "pred_interval": [True, False, True, True],
        }
    )

    metrics = helpers.evaluate_prediction_frame(frame, cfg, "Beta", "unit", "m8_xgb")
    interval = metrics.loc[metrics["level"] == "interval"].iloc[0]

    assert int(interval["support"]) == 2
    assert int(interval["tp"]) == 0
    assert int(interval["fp"]) == 1
    assert int(interval["fn"]) == 1


def test_correction_smoke_metrics_are_finite_placeholders() -> None:
    cfg = cached_config()
    alpha = cached_dataset("alpha")
    beta = cached_dataset("beta")

    metrics = helpers.correction_smoke_metrics(alpha, beta, cfg)

    assert len(metrics) == 44
    assert metrics.groupby(["dataset", "fold_id", "method", "level"]).size().eq(1).all()
    assert set(metrics["level"]) == {"day", "interval"}
    assert metrics["is_placeholder"].eq(True).all()
    assert metrics["status"].eq("placeholder_smoke_only").all()
    assert metrics[["support", "positive_support", "tp", "fp", "fn", "tn"]].notna().all().all()
    assert metrics[["precision", "recall", "f1"]].notna().all().all()
    assert metrics[["precision", "recall", "f1"]].ge(0).all().all()
    assert metrics[["precision", "recall", "f1"]].le(1).all().all()


def test_correction_beta_site_metrics_cover_all_beta_sites_in_rpf_order() -> None:
    cfg = cached_config()
    beta = cached_dataset("beta")

    assert helpers.beta_top_rpf_sites(beta) == ["beta_F", "beta_B", "beta_G"]
    assert helpers.beta_rpf_site_order(beta) == [
        "beta_F",
        "beta_B",
        "beta_G",
        "beta_D",
        "beta_E",
        "beta_A",
        "beta_H",
        "beta_C",
    ]

    metrics = helpers.correction_smoke_beta_site_metrics(beta, cfg)

    assert len(metrics) == 32
    assert metrics["substation_id"].drop_duplicates().tolist() == [
        "beta_F",
        "beta_B",
        "beta_G",
        "beta_D",
        "beta_E",
        "beta_A",
        "beta_H",
        "beta_C",
    ]
    assert set(metrics["method"]) == {"m8_xgb", "m7_dtr"}
    assert set(metrics["level"]) == {"day", "interval"}
    assert metrics.groupby(["substation_id", "method", "level"]).size().eq(1).all()


def test_correction_metrics_table_merges_beta_site_rows() -> None:
    cfg = cached_config()
    alpha = cached_dataset("alpha")
    beta = cached_dataset("beta")
    metrics = helpers.correction_smoke_metrics(alpha, beta, cfg)
    site_metrics = helpers.correction_smoke_beta_site_metrics(beta, cfg)

    table = helpers.correction_metrics_table(metrics, site_metrics)

    assert "summary_scope" in table.columns
    assert "substation_id" in table.columns
    assert set(table["summary_scope"]) == {
        "alpha_loso_fold",
        "beta_overall",
        "beta_site",
    }
    assert len(table.loc[table["summary_scope"] == "alpha_loso_fold"]) == 40
    assert len(table.loc[table["summary_scope"] == "beta_site"]) == 32
    assert table.loc[
        table["summary_scope"] == "alpha_loso_fold", "substation_id"
    ].drop_duplicates().tolist() == [
        "alpha_F",
        "alpha_E",
        "alpha_G",
        "alpha_C",
        "alpha_J",
        "alpha_I",
        "alpha_B",
        "alpha_A",
        "alpha_D",
        "alpha_H",
    ]
    assert table.loc[
        table["summary_scope"] == "beta_site", "substation_id"
    ].drop_duplicates().tolist() == [
        "beta_F",
        "beta_B",
        "beta_G",
        "beta_D",
        "beta_E",
        "beta_A",
        "beta_H",
        "beta_C",
    ]


def test_beta_site_metrics_from_predictions_use_supplied_frames() -> None:
    cfg = cached_config()
    beta = cached_dataset("beta")
    pred = beta.copy()
    pred["pred_interval"] = pred["label_interval"]
    pred["corrected_net_load_MW"] = pred["reference_net_load_MW"]

    metrics = helpers.correction_beta_site_metrics_from_predictions(
        beta,
        cfg,
        {"m8_xgb": pred, "m7_dtr": pred},
    )

    assert len(metrics) == 32
    assert metrics["precision"].eq(1.0).all()
    assert metrics["recall"].eq(1.0).all()
    assert metrics["f1"].eq(1.0).all()


def test_reusable_prediction_validation_checks_columns_and_keys(tmp_path: Path) -> None:
    beta = cached_dataset("beta").head(4).copy()
    pred = beta[helpers.EXPECTED_COLUMNS].copy()
    pred["pred_interval"] = [True, False, False, True]
    pred["corrected_net_load_MW"] = beta["net_load_MW"].to_numpy()
    path = tmp_path / "prediction.csv"
    pred.to_csv(path, index=False)

    reused = helpers.load_reusable_correction_prediction(path, beta, require_existing=True)

    assert reused is not None
    assert reused["pred_interval"].tolist() == [True, False, False, True]
    assert "confidence" in reused.columns
    assert list(helpers._model_input_columns(beta).columns) == helpers.EXPECTED_COLUMNS

    missing = pred.drop(columns=["pred_interval"])
    missing_path = tmp_path / "missing.csv"
    missing.to_csv(missing_path, index=False)
    with pytest.raises(ValueError):
        helpers.load_reusable_correction_prediction(missing_path, beta, require_existing=True)

    mismatched = pred.copy()
    mismatched.loc[0, "timestamp"] = "1999-01-01 00:00:00+00:00"
    mismatch_path = tmp_path / "mismatch.csv"
    mismatched.to_csv(mismatch_path, index=False)
    with pytest.raises(ValueError):
        helpers.load_reusable_correction_prediction(mismatch_path, beta, require_existing=True)


def test_beta_confidence_split_metrics_filter_sure_site_days() -> None:
    cfg = cached_config()
    beta = cached_dataset("beta")
    pred = beta.copy()
    pred["pred_interval"] = pred["label_interval"]
    pred["corrected_net_load_MW"] = pred["reference_net_load_MW"]

    metrics = helpers.correction_beta_confidence_split_metrics_from_predictions(
        beta,
        cfg,
        {"m8_xgb": pred, "m7_dtr": pred},
    )

    assert len(metrics) == 72
    assert set(metrics["confidence_scope"]) == {"all", "sure"}
    assert set(metrics["method"]) == {"m8_xgb", "m7_dtr"}
    site_order = (
        metrics.loc[metrics["summary_scope"] == "beta_site", "substation_id"]
        .drop_duplicates()
        .tolist()
    )
    assert site_order == [
        "beta_F",
        "beta_B",
        "beta_G",
        "beta_D",
        "beta_E",
        "beta_A",
        "beta_H",
        "beta_C",
    ]
    sure_day = metrics.loc[
        (metrics["summary_scope"] == "beta_overall")
        & (metrics["confidence_scope"] == "sure")
        & (metrics["method"] == "m8_xgb")
        & (metrics["level"] == "day")
    ].iloc[0]
    assert int(sure_day["support"]) == 2310
    assert int(sure_day["positive_support"]) == 471


def test_correction_pooled_metrics_sum_alpha_loso_counts() -> None:
    cfg = cached_config()
    alpha = cached_dataset("alpha")
    beta = cached_dataset("beta")
    metrics = helpers.correction_smoke_metrics(alpha, beta, cfg)

    pooled = helpers.correction_pooled_metrics(metrics)
    source = metrics.loc[
        (metrics["dataset"] == "Alpha")
        & (metrics["method"] == "m8_xgb")
        & (metrics["level"] == "day")
    ]
    pooled_row = pooled.loc[
        (pooled["summary_group"] == "Alpha CV pooled")
        & (pooled["method"] == "m8_xgb")
        & (pooled["level"] == "day")
    ].iloc[0]

    assert int(pooled_row["support"]) == int(source["support"].sum())
    assert int(pooled_row["tp"]) == int(source["tp"].sum())
    assert int(pooled_row["fp"]) == int(source["fp"].sum())
    assert int(pooled_row["fn"]) == int(source["fn"].sum())
    assert int(pooled_row["tn"]) == int(source["tn"].sum())


def test_correction_placeholder_figures_are_written(tmp_path: Path) -> None:
    cfg = cached_config()
    alpha = cached_dataset("alpha")
    beta = cached_dataset("beta")
    metrics = helpers.correction_smoke_metrics(alpha, beta, cfg)
    alpha_site_metrics = helpers.correction_alpha_site_metrics_from_loso_metrics(metrics)
    beta_site_metrics = helpers.correction_smoke_beta_site_metrics(beta, cfg)

    figure_paths = helpers.write_correction_figures(
        metrics,
        tmp_path,
        alpha_site_metrics=alpha_site_metrics,
        beta_site_metrics=beta_site_metrics,
    )

    assert [path.name for path in figure_paths] == [
        "fig01a_confusion_matrices_day.png",
        "fig01b_confusion_matrices_interval.png",
        "fig02a_precision_recall_f1_day.png",
        "fig02b_precision_recall_f1_interval.png",
        "fig03_beta_site_precision_recall_f1_boxplot.png",
        "fig04_alpha_site_precision_recall_f1_boxplot.png",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in figure_paths)


def test_characterisation_new_temporal_event_summaries() -> None:
    alpha = cached_dataset("alpha")
    beta = cached_dataset("beta")
    events = pd.concat(
        [
            helpers.extract_rpf_events(alpha, "Alpha"),
            helpers.extract_rpf_events(beta, "Beta"),
        ],
        ignore_index=True,
    )

    daytype_summary = pd.concat(
        [
            helpers.rpf_daytype_summary(alpha, "Alpha"),
            helpers.rpf_daytype_summary(beta, "Beta"),
        ],
        ignore_index=True,
    )
    event_counts = helpers.rpf_event_count_by_day(events)

    observed_site_days = sum(
        len(frame[["substation_id", "date"]].drop_duplicates()) for frame in [alpha, beta]
    )
    assert set(daytype_summary["month"]) == set(range(1, 13))
    assert set(daytype_summary["daytype"]) == {"Weekday", "Weekend"}
    assert int(daytype_summary["total_site_days"].sum()) == observed_site_days

    expected_rpf_site_days = events[["dataset", "substation_id", "date"]].drop_duplicates()
    assert int(daytype_summary["rpf_site_days"].sum()) == len(expected_rpf_site_days)
    assert int(event_counts["n_rpf_site_days"].sum()) == len(expected_rpf_site_days)
    assert set(event_counts["plot_category"]).issubset({"1", "2", "3", "4", "5+"})


def test_publication_inventory_uses_new_notebook2_figure_names() -> None:
    cfg = cached_config()
    paths = helpers.article_paths(ARTICLE_ROOT, cfg)
    figures = helpers.publication_expected_figures(paths)

    assert "fig01a_confusion_matrices_day" in figures
    assert "fig01b_confusion_matrices_interval" in figures
    assert "fig02a_precision_recall_f1_day" in figures
    assert "fig02b_precision_recall_f1_interval" in figures
    assert "fig03_beta_site_precision_recall_f1_boxplot" in figures
    assert "fig04_alpha_site_precision_recall_f1_boxplot" in figures
    assert "fig04_month_daytype_rpf_heatmap_alpha_beta" in figures
    assert "fig05_rpf_events_per_day_doughnut_alpha_beta" in figures
    assert not any(
        path.name == "fig01_correction_confusion_matrices.png" for path in figures.values()
    )
    assert not any(
        path.name == "fig02_correction_precision_recall_f1.png" for path in figures.values()
    )
    assert "fig02_gamma_forecast_rmse" in figures
    assert not any(
        path.name == "fig02a_gamma_perfect_model_baseline_rmse.png"
        for path in figures.values()
    )
    assert not any(path.name == "fig02b_gamma_forecast_rmse.png" for path in figures.values())


def test_forecast_examples_are_exactly_seven_days_ahead_with_14_day_lookback() -> None:
    cfg = cached_config()
    timestamps = pd.date_range("2024-08-01 00:00", "2024-09-30 23:45", freq="15min")
    frame = pd.DataFrame(
        {
            "substation_id": "beta_B",
            "date": timestamps.strftime("%Y-%m-%d"),
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "net_load_MW": range(len(timestamps)),
            "solar_MW": 0.0,
            "label_interval": False,
            "label_day": False,
            "confidence": "sure",
        }
    )
    gamma = helpers.prepare_dataset(frame[helpers.CONFIDENCE_COLUMNS], "Gamma unit")

    examples = helpers.build_forecast_examples(
        gamma,
        "net_load_MW",
        cfg,
        "2024-09-01",
        "2024-09-01",
    )

    assert len(examples) == 96
    first = examples.iloc[0]
    target = pd.Timestamp(first["target_timestamp"])
    origin = pd.Timestamp(first["origin_timestamp"])
    assert target - origin == pd.Timedelta(days=7)
    assert first["origin_value"] == gamma.set_index("_timestamp_dt").loc[origin, "net_load_MW"]


def test_gamma_forecast_smoke_rows_cover_placeholder_models_and_conditions() -> None:
    cfg = tiny_forecast_config()
    gamma = tiny_gamma_frame()
    gamma["raw_uncorrected_MW"] = gamma["net_load_MW"]
    gamma["reference_corrected_MW"] = gamma["reference_net_load_MW"]
    gamma["m8_xgb_corrected_MW"] = helpers.placeholder_m8_corrected_series(gamma)

    baseline = helpers.perfect_model_baseline(gamma, cfg)
    placeholder = helpers.placeholder_forecast_rows(gamma, cfg)
    forecasts = pd.concat([baseline, placeholder], ignore_index=True)
    metrics = helpers.forecast_metric_rows(forecasts)

    assert set(metrics["model"]) == {
        "perfect_model_baseline",
        "seasonal_naive",
        "linear_regression",
        "xgboost",
    }
    assert set(metrics["data_condition"]) == {
        "raw_uncorrected",
        "m8_xgb_corrected",
        "reference_corrected",
    }
    baseline_metrics = metrics.loc[metrics["model"] == "perfect_model_baseline"]
    assert len(baseline_metrics) == 3
    assert baseline_metrics["is_placeholder"].eq(False).all()
    assert (
        metrics.loc[metrics["model"] != "perfect_model_baseline", "is_placeholder"]
        .eq(True)
        .all()
    )
    manual_baseline = baseline_metrics.loc[
        baseline_metrics["data_condition"] == "reference_corrected"
    ].iloc[0]
    assert np.isclose(manual_baseline["rmse_MW"], 0.0)
    assert np.isclose(manual_baseline["mae_MW"], 0.0)
    for condition in metrics["data_condition"].unique():
        condition_metrics = metrics.loc[metrics["data_condition"] == condition]
        baseline_rmse = float(
            condition_metrics.loc[
                condition_metrics["model"] == "perfect_model_baseline", "rmse_MW"
            ].iloc[0]
        )
        model_rmse = condition_metrics.loc[
            condition_metrics["model"] != "perfect_model_baseline", "rmse_MW"
        ]
        assert model_rmse.gt(baseline_rmse).all()
    assert set(metrics["data_condition_label"]) == {
        "Uncorrected data",
        "m8_xgb-corrected data",
        "Manually corrected data",
    }
    public_labels = set(metrics["data_condition_label"]).union(set(metrics["model_label"]))
    assert not {"Raw", "Reference", "Data error"}.intersection(public_labels)


def test_gamma_forecast_smoke_figures_are_written(tmp_path: Path) -> None:
    cfg = tiny_forecast_config()
    gamma = tiny_gamma_frame()
    gamma["raw_uncorrected_MW"] = gamma["net_load_MW"]
    gamma["reference_corrected_MW"] = gamma["reference_net_load_MW"]
    gamma["m8_xgb_corrected_MW"] = helpers.placeholder_m8_corrected_series(gamma)
    baseline = helpers.perfect_model_baseline(gamma, cfg)
    forecasts = pd.concat(
        [baseline, helpers.placeholder_forecast_rows(gamma, cfg)], ignore_index=True
    )
    metrics = helpers.forecast_metric_rows(forecasts)

    paths = [
        helpers.write_gamma_series_figure(gamma, tmp_path, "beta_B"),
        helpers.write_forecast_metric_figure(metrics, tmp_path),
        helpers.write_forecast_residual_figure(forecasts, tmp_path),
    ]

    assert [path.name for path in paths if path is not None] == [
        "fig01_gamma_series_raw_corrected_reference.png",
        "fig02_gamma_forecast_rmse.png",
        "fig03_gamma_forecast_residuals.png",
    ]
    assert all(path is not None and path.exists() and path.stat().st_size > 0 for path in paths)


def test_extract_rpf_events_reports_contiguous_duration_hours() -> None:
    timestamps = pd.date_range("2024-09-01 00:00", periods=6, freq="15min")
    frame = pd.DataFrame(
        {
            "substation_id": "beta_A",
            "date": timestamps.strftime("%Y-%m-%d"),
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "net_load_MW": [1.0, 1.1, 1.2, -0.5, 1.3, 1.4],
            "solar_MW": 0.0,
            "label_interval": [True, True, False, True, True, True],
            "label_day": True,
        }
    )
    prepared = helpers.prepare_dataset(frame[helpers.EXPECTED_COLUMNS], "Event unit")

    events = helpers.extract_rpf_events(prepared, "Beta")

    assert list(events["duration_minutes"]) == [30, 45]
    assert list(events["duration_hours"]) == [0.5, 0.75]

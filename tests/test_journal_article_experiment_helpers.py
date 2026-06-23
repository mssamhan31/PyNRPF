from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ARTICLE_ROOT = Path(__file__).resolve().parents[1] / "publication" / "2_journal_article"
NOTEBOOK_DIR = ARTICLE_ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOK_DIR))

import _experiment_helpers as helpers  # noqa: E402


def test_config_paths_and_schema_resolve() -> None:
    cfg = helpers.load_config(ARTICLE_ROOT)
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


def test_real_data_rankings_match_current_labels() -> None:
    cfg = helpers.load_config(ARTICLE_ROOT)
    alpha = helpers.load_dataset(ARTICLE_ROOT, cfg, "alpha")
    beta = helpers.load_dataset(ARTICLE_ROOT, cfg, "beta")
    gamma = helpers.load_dataset(ARTICLE_ROOT, cfg, "gamma")

    assert helpers.alpha_loso_sites(alpha, cfg) == ["syn_F", "syn_E", "syn_G"]
    assert helpers.select_gamma_site(beta, cfg) == "act_B"
    assert beta["date"].min() == "2023-10-01"
    assert beta["date"].max() == "2024-09-30"
    assert len(beta) == 280_800
    assert beta[["substation_id", "date"]].drop_duplicates().shape[0] == 2_928
    assert gamma["substation_id"].nunique() == 1
    assert gamma["substation_id"].iloc[0] == "act_B"
    assert len(gamma) == 35_136


def test_binary_metrics_and_daytime_interval_scope() -> None:
    cfg = helpers.load_config(ARTICLE_ROOT)
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
    interval = metrics.loc[metrics["level"] == "interval_daytime"].iloc[0]

    assert int(interval["support"]) == 2
    assert int(interval["tp"]) == 0
    assert int(interval["fp"]) == 1
    assert int(interval["fn"]) == 1


def test_forecast_examples_are_exactly_seven_days_ahead_with_14_day_lookback() -> None:
    cfg = helpers.load_config(ARTICLE_ROOT)
    timestamps = pd.date_range("2024-08-01 00:00", "2024-09-30 23:45", freq="15min")
    frame = pd.DataFrame(
        {
            "substation_id": "act_B",
            "date": timestamps.strftime("%Y-%m-%d"),
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "net_load_MW": range(len(timestamps)),
            "solar_MW": 0.0,
            "label_interval": False,
            "label_day": False,
        }
    )
    gamma = helpers.prepare_dataset(frame[helpers.EXPECTED_COLUMNS], "Gamma unit")

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

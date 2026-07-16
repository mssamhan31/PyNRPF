from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

ARTICLE_ROOT = Path(__file__).resolve().parents[1] / "publication" / "2_journal_article"
NOTEBOOK_DIR = ARTICLE_ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOK_DIR))

from _experiment_helpers import rpf_daytype_summary  # noqa: E402
from _figure_sources import (  # noqa: E402
    figure_source_path,
    load_figure_source,
    render_only_enabled,
    write_figure_source,
)
from _journal_figure_style import (  # noqa: E402
    COLORS,
    align_twin_y_axes,
    apply_journal_style,
    style_axis,
)
from _m9_pbm_plotting import (  # noqa: E402
    plot_ablation_feature_evidence,
    plot_gamma_forecast_rmse,
    plot_method_example,
    plot_regime_metrics,
)


def test_shared_style_uses_requested_palette_and_typography() -> None:
    assert COLORS == {
        "orange": "#eb932c",
        "dark_blue": "#22303d",
        "grey": "#2F4D67",
        "light_grey": "#5C7D99",
        "light_white": "#ebe3e3",
        "red": "#B64A4A",
    }

    apply_journal_style()

    assert plt.rcParams["xtick.labelsize"] == 13
    assert plt.rcParams["legend.fontsize"] == 13
    assert plt.rcParams["axes.labelsize"] == 14
    assert plt.rcParams["axes.titlesize"] == 15
    assert plt.rcParams["axes.axisbelow"] is True


def test_style_axis_keeps_complete_frame_and_limits_continuous_ticks() -> None:
    apply_journal_style()
    figure, axis = plt.subplots()
    axis.plot(np.arange(100), np.arange(100))

    style_axis(axis, x_continuous=True, y_continuous=True)
    figure.canvas.draw()

    assert all(spine.get_visible() for spine in axis.spines.values())
    assert len(axis.get_xticks()) <= 5
    assert len(axis.get_yticks()) <= 5
    assert axis.get_axisbelow() is True
    plt.close(figure)


def test_twin_axes_use_aligned_grid_positions() -> None:
    apply_journal_style()
    figure, left_axis = plt.subplots()
    right_axis = left_axis.twinx()
    left_axis.plot([0, 1], [0, 93])
    right_axis.bar([0, 1], [0, 17])

    align_twin_y_axes(left_axis, right_axis)

    left_normalized = left_axis.get_yticks() / left_axis.get_ylim()[1]
    right_normalized = right_axis.get_yticks() / right_axis.get_ylim()[1]
    assert len(left_normalized) == len(right_normalized) == 5
    assert np.allclose(left_normalized, right_normalized)
    assert all(spine.get_visible() for spine in left_axis.spines.values())
    assert all(spine.get_visible() for spine in right_axis.spines.values())
    plt.close(figure)


def test_figure_source_round_trip_and_integrity(tmp_path: Path) -> None:
    source = pd.DataFrame({"metric": ["precision", "recall"], "value": [0.8, 0.9]})
    path = figure_source_path(tmp_path, "02c", "regime_metrics")

    write_figure_source(source, path, required_columns=["metric", "value"])

    pd.testing.assert_frame_equal(
        load_figure_source(path, required_columns=["metric", "value"]),
        source,
    )
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_figure_source(path)


def test_render_only_environment_parser() -> None:
    assert render_only_enabled({}) is False
    assert render_only_enabled({"PYNRPF_RENDER_ONLY": "1"}) is True
    assert render_only_enabled({"PYNRPF_RENDER_ONLY": "false"}) is False
    with pytest.raises(ValueError, match="PYNRPF_RENDER_ONLY"):
        render_only_enabled({"PYNRPF_RENDER_ONLY": "sometimes"})


def test_daytype_summary_uses_observed_site_days_as_denominator() -> None:
    frame = pd.DataFrame(
        {
            "substation_id": ["A", "A", "A", "A", "B", "B"],
            "date": [
                "2024-01-05",
                "2024-01-05",
                "2024-01-06",
                "2024-01-06",
                "2024-01-05",
                "2024-01-06",
            ],
            "label_interval": [True, False, False, False, False, True],
        }
    )

    summary = rpf_daytype_summary(frame, "Test")
    january = summary.loc[summary["month"].eq(1)].set_index("daytype")

    assert january.loc["Weekday", "total_site_days"] == 2
    assert january.loc["Weekday", "rpf_site_days"] == 1
    assert january.loc["Weekday", "rpf_site_day_pct"] == 50
    assert january.loc["Weekend", "total_site_days"] == 2
    assert january.loc["Weekend", "rpf_site_days"] == 1
    assert january.loc["Weekend", "rpf_site_day_pct"] == 50


def test_requested_regrouped_figures_render(tmp_path: Path) -> None:
    regimes = pd.DataFrame(
        {
            "regime": ["beta_only", "beta_plus_alpha", "alpha_only"],
            "precision": [0.8, 0.81, 0.7],
            "recall": [0.7, 0.72, 0.6],
            "f1": [0.75, 0.76, 0.65],
        }
    )
    features = pd.DataFrame(
        {
            "feature_number": range(1, 10),
            "feature_short": [f"F{number}" for number in range(1, 10)],
            "top_25_frequency_pct": np.linspace(20, 100, 9),
            "mean_paired_delta_f1": np.linspace(-0.02, 0.04, 9),
        }
    )
    forecast = pd.DataFrame(
        [
            {"model": model, "data_condition": condition, "rmse_MW": value}
            for value, (model, condition) in enumerate(
                (
                    (model, condition)
                    for model in ["seasonal_naive", "linear_regression", "xgboost"]
                    for condition in [
                        "raw_uncorrected",
                        "m9_pbm_corrected",
                        "manually_corrected",
                    ]
                ),
                start=1,
            )
        ]
    )
    outputs = [
        plot_regime_metrics(regimes, tmp_path / "regimes.png"),
        plot_ablation_feature_evidence(features, tmp_path / "features.png"),
        plot_gamma_forecast_rmse(forecast, tmp_path / "forecast.png"),
    ]

    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)


def test_method_example_colors_and_legends_do_not_overlap_panels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = pd.date_range("2024-02-17", periods=8, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "observed_net_load_MW": [6, 5, 3, 1, 0, 2, 7, 6],
            "solar_generation_MW": [0, 0, 2, 5, 7, 5, 1, 0],
            "candidate_window": [False, False, True, True, True, True, False, False],
            "bridge_anchor": [False, True, False, False, False, False, True, False],
            "uncorrected_demand_MW": [6, 5, 5, 6, 7, 7, 8, 6],
            "corrected_demand_MW": [6, 5, 6, 7, 8, 8, 8, 6],
            "linear_bridge_MW": [np.nan, 5, 5.6, 6.2, 6.8, 7.4, 8, np.nan],
        }
    )

    with monkeypatch.context() as patch:
        patch.setattr(plt, "close", lambda _figure: None)
        output = plot_method_example(frame, tmp_path / "method.png")
        figure = plt.gcf()

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    header_axes = [axis for axis in figure.axes if axis.get_legend() is not None]
    data_axes = [axis for axis in figure.axes if axis.get_ylabel()]
    observed_lines = {
        line.get_label(): line.get_color() for line in data_axes[0].get_lines()
    }

    assert output.exists()
    assert observed_lines[r"Observed net load, $y(t)$"] == COLORS["dark_blue"]
    assert observed_lines[r"Solar generation, $S(t)$"] == COLORS["orange"]
    assert len(header_axes) == len(data_axes) == 2
    for header_axis, data_axis in zip(header_axes, data_axes, strict=True):
        legend_box = header_axis.get_legend().get_window_extent(renderer)
        title_box = header_axis.texts[0].get_window_extent(renderer)
        assert not legend_box.overlaps(title_box)
        assert not legend_box.overlaps(data_axis.get_window_extent(renderer))
        assert data_axis.get_legend() is None

    plt.close(figure)
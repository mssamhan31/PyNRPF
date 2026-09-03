"""Journal plotting style and reusable m9_pbm figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _journal_figure_style import (
    COLORS,
    align_twin_y_axes,
    apply_journal_style,
    journal_colormap,
    style_axes,
    style_axis,
    style_colorbar,
)


def plot_method_example(plot_data: pd.DataFrame, output_path: Path) -> Path:
    """Create the required two-panel observed/reconstructed-demand example."""

    apply_journal_style()
    data = plot_data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    candidate = data.loc[data["candidate_window"]]
    anchors = data.loc[data["bridge_anchor"]]
    if candidate.empty or len(anchors) != 2:
        raise ValueError("Method example requires one candidate window and exactly two anchors.")

    start = candidate["timestamp"].min()
    end = candidate["timestamp"].max()
    fig = plt.figure(figsize=(10.2, 8.8))
    grid = fig.add_gridspec(
        4,
        1,
        height_ratios=(0.30, 1.0, 0.50, 1.0),
        hspace=0.08,
    )
    header_axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[2])]
    axes = [fig.add_subplot(grid[1]), fig.add_subplot(grid[3])]
    axes[1].sharex(axes[0])
    for header_axis in header_axes:
        header_axis.set_axis_off()

    axes[0].plot(
        data["timestamp"],
        data["observed_net_load_MW"],
        color=COLORS["dark_blue"],
        linewidth=2.0,
        label=r"Observed net load, $y(t)$",
    )
    axes[0].plot(
        data["timestamp"],
        data["solar_generation_MW"],
        color=COLORS["orange"],
        linewidth=1.8,
        label=r"Solar generation, $S(t)$",
    )
    axes[0].axvspan(start, end, color=COLORS["light_grey"], alpha=0.42, label=r"Window, $W$")
    axes[0].axhline(0, color=COLORS["grey"], linewidth=0.8)
    axes[0].set_ylabel("Power (MW)")

    axes[1].plot(
        data["timestamp"],
        data["uncorrected_demand_MW"],
        color=COLORS["orange"],
        linewidth=2.0,
        label=r"Uncorrected demand, $U_{no}(t)$",
    )
    axes[1].plot(
        data["timestamp"],
        data["corrected_demand_MW"],
        color=COLORS["dark_blue"],
        linewidth=2.1,
        label=r"Corrected demand, $U_{corr,W}(t)$",
    )
    bridge = data["linear_bridge_MW"].notna()
    axes[1].plot(
        data.loc[bridge, "timestamp"],
        data.loc[bridge, "linear_bridge_MW"],
        color=COLORS["grey"],
        linewidth=1.6,
        linestyle="--",
        label=r"Linear bridge, $L_W(t)$",
    )
    axes[1].scatter(
        anchors["timestamp"],
        anchors["uncorrected_demand_MW"],
        marker="D",
        s=[42, 68],
        color=[COLORS["light_grey"], COLORS["orange"]],
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="Bridge anchors",
    )
    for number, row in enumerate(anchors.itertuples(index=False), start=1):
        axes[1].annotate(
            f"Anchor {number}",
            (row.timestamp, row.uncorrected_demand_MW),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=COLORS["dark_blue"],
        )
    axes[1].axvspan(start, end, color=COLORS["light_grey"], alpha=0.42)
    axes[1].set_ylabel("Underlying demand (MW)")
    axes[1].set_xlabel("Time on 17 February 2024")
    axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    for header_axis, axis, title, columns in zip(
        header_axes,
        axes,
        ["(a) Observed readings", "(b) Two interpretations of underlying demand"],
        [3, 2],
        strict=True,
    ):
        header_axis.text(
            0.5,
            0.98,
            title,
            ha="center",
            va="top",
            fontsize=15,
            color=COLORS["dark_blue"],
        )
        handles, labels = axis.get_legend_handles_labels()
        header_axis.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            frameon=False,
            ncol=columns,
        )

    for axis in axes:
        axis.margins(x=0)
    style_axes(axes, x_dates=True, y_continuous=True)
    axes[0].tick_params(labelbottom=False)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.98, bottom=0.09)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output_path


def plot_regime_metrics(metrics: pd.DataFrame, output_path: Path) -> Path:
    """Group precision/recall/F1 with training-regime variants inside each group."""

    apply_journal_style()
    data = metrics.copy()
    score_columns = ["precision", "recall", "f1"]
    regime_labels = {
        "beta_only": "Beta only",
        "beta_plus_alpha": "Beta + Alpha",
        "alpha_only": "Alpha only",
    }
    regime_order = ["beta_only", "beta_plus_alpha", "alpha_only"]
    data = data.set_index("regime").loc[regime_order]
    figure, axis = plt.subplots(figsize=(8.2, 4.5))
    centers = np.arange(len(score_columns))
    width = 0.23
    for offset, (regime, color) in enumerate(
        zip(regime_order, [COLORS["dark_blue"], COLORS["orange"], COLORS["grey"]], strict=True)
    ):
        positions = centers + (offset - 1) * width
        values = data.loc[regime, score_columns].to_numpy(dtype=float)
        bars = axis.bar(
            positions,
            values,
            width=width,
            color=color,
            label=regime_labels[regime],
        )
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    axis.set_xticks(centers, ["Precision", "Recall", "F1"])
    axis.set_ylim(0, 1.18)
    axis.set_ylabel("Held-out Beta sure score")
    axis.set_title("Training-regime comparison for equal-weight F1/F3/F4")
    axis.legend(frameon=False, ncol=3, loc="upper center")
    style_axis(axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return output_path


def plot_regime_thresholds(thresholds: pd.DataFrame, output_path: Path) -> Path:
    """Show fold-specific day thresholds for Beta LOSO regimes."""

    apply_journal_style()
    data = thresholds.loc[thresholds["heldout_substation"].ne("all_beta")].copy()
    regime_labels = {
        "beta_only": "Beta only",
        "beta_plus_alpha": "Beta + Alpha",
    }
    figure, axis = plt.subplots(figsize=(8.4, 4.4))
    for regime, color, marker in [
        ("beta_only", COLORS["dark_blue"], "o"),
        ("beta_plus_alpha", COLORS["orange"], "s"),
    ]:
        subset = data.loc[data["regime"].eq(regime)].sort_values("heldout_substation")
        axis.plot(
            subset["heldout_substation"],
            subset["threshold"],
            color=color,
            marker=marker,
            linewidth=1.8,
            label=regime_labels[regime],
        )
    axis.set_xlabel("Held-out Beta substation")
    axis.set_ylabel(r"Selected threshold, $\tau$")
    axis.set_title("Training-only threshold selected for each outer fold")
    axis.legend(frameon=False, ncol=2)
    style_axis(axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return output_path


def plot_ablation_by_feature_count(
    metrics: pd.DataFrame,
    best_by_count: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Scatter all subset F1 values and connect the best result at each size."""

    apply_journal_style()
    figure, axis = plt.subplots(figsize=(8.4, 4.7))
    axis.scatter(
        metrics["feature_count"],
        metrics["beta_sure_f1"],
        s=18,
        alpha=0.35,
        color=COLORS["light_grey"],
        edgecolor="none",
        label="All nonempty subsets",
    )
    axis.plot(
        best_by_count["feature_count"],
        best_by_count["beta_sure_f1"],
        color=COLORS["orange"],
        marker="o",
        linewidth=2.2,
        label="Best subset at each size",
    )
    for row in best_by_count.itertuples(index=False):
        axis.annotate(
            f"{row.beta_sure_f1:.3f}",
            (row.feature_count, row.beta_sure_f1),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    axis.set_xticks(range(1, 10))
    axis.set_xlabel("Number of physical features")
    axis.set_ylabel("Held-out Beta sure F1")
    axis.set_title("Self-consistent equal-weight feature ablation")
    score_min = float(metrics["beta_sure_f1"].min())
    score_max = float(best_by_count["beta_sure_f1"].max())
    score_span = max(score_max - score_min, 0.1)
    axis.set_ylim(score_min - 0.05 * score_span, score_max + 0.22 * score_span)
    axis.legend(frameon=False, ncol=2, loc="upper center")
    style_axis(axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return output_path


def plot_top_ablation_subsets(metrics: pd.DataFrame, output_path: Path, top_n: int = 20) -> Path:
    """Horizontal F1 ranking for the strongest feature subsets."""

    apply_journal_style()
    data = metrics.nlargest(top_n, "beta_sure_f1").sort_values("beta_sure_f1")
    figure, axis = plt.subplots(figsize=(8.5, 5.8))
    bars = axis.barh(
        data["feature_set_short"],
        data["beta_sure_f1"],
        color=[
            COLORS["orange"] if count == 3 else COLORS["dark_blue"]
            for count in data["feature_count"]
        ],
    )
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=7)
    axis.set_xlim(max(0, data["beta_sure_f1"].min() - 0.04), 1.0)
    axis.set_xlabel("Held-out Beta sure F1")
    axis.set_title(f"Top {top_n} equal-weight subsets")
    style_axis(axis, grid_axis="x", x_continuous=True, y_continuous=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return output_path


def plot_ablation_feature_evidence(feature_table: pd.DataFrame, output_path: Path) -> Path:
    """Show top-subset frequency beside paired marginal F1 effects."""

    apply_journal_style()
    data = feature_table.sort_values("feature_number")
    feature_labels = {
        1: "F1 Bridge",
        2: "F2 Roughness",
        3: "F3 Slope continuity",
        4: "F4 Duration",
        5: "F5 N-height",
        6: "F6 Solar strength",
        7: "F7 Solar alignment",
        8: "F8 Centered core",
        9: "F9 Ranked core",
    }
    data["feature_label"] = data["feature_number"].map(feature_labels)
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=True)
    axes[0].barh(
        data["feature_label"],
        data["top_25_frequency_pct"],
        color=COLORS["light_grey"],
    )
    axes[0].set_xlabel("Frequency in top 25 subsets (%)")
    axes[0].set_title("Top-subset frequency")
    effect_colors = [
        COLORS["orange"] if value >= 0 else COLORS["red"]
        for value in data["mean_paired_delta_f1"]
    ]
    axes[1].barh(
        data["feature_label"],
        data["mean_paired_delta_f1"],
        color=effect_colors,
    )
    axes[1].axvline(0, color=COLORS["grey"], linewidth=0.9)
    axes[1].set_xlabel("Mean paired change in Beta sure F1")
    axes[1].set_title("Marginal effect across matched subsets")
    style_axes(
        axes,
        grid_axis="x",
        x_continuous=True,
        y_continuous=False,
    )
    figure.tight_layout(w_pad=1.8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return output_path


def plot_weight_simplex(search_results: pd.DataFrame, output_path: Path) -> Path:
    """Plot three-feature simplex positions coloured by inner macro F1."""

    apply_journal_style()
    data = search_results.copy()
    data["simplex_x"] = data["weight_F3"] + 0.5 * data["weight_F4"]
    data["simplex_y"] = np.sqrt(3) / 2 * data["weight_F4"]
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, np.sqrt(3) / 2, 0]
    axis.plot(triangle_x, triangle_y, color=COLORS["dark_blue"], linewidth=1.2)
    points = axis.scatter(
        data["simplex_x"],
        data["simplex_y"],
        c=data["inner_macro_f1"],
        cmap=journal_colormap("weight_search"),
        s=18,
        alpha=0.75,
        edgecolor="none",
    )
    best = data.sort_values(
        ["inner_macro_f1", "inner_macro_precision"], ascending=False
    ).iloc[0]
    axis.scatter(
        [best["simplex_x"]],
        [best["simplex_y"]],
        marker="*",
        s=180,
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.8,
    )
    axis.annotate(
        "Selected full-Beta weight",
        (best["simplex_x"], best["simplex_y"]),
        xytext=(0.07, 0.74),
        textcoords="axes fraction",
        ha="left",
        va="center",
        fontsize=10,
        color=COLORS["dark_blue"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.5},
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["dark_blue"],
            "linewidth": 1.0,
            "connectionstyle": "arc3,rad=-0.12",
        },
    )
    axis.text(0.02, -0.02, "F1 bridge", ha="left", va="top")
    axis.text(0.98, -0.02, "F3 slope", ha="right", va="top")
    axis.text(0.5, np.sqrt(3) / 2 + 0.015, "F4 duration", ha="center", va="bottom")
    axis.set_title("Grid and random weight search on the positive simplex", pad=34)
    axis.set_aspect("equal")
    axis.set_xlim(-0.10, 1.10)
    axis.set_ylim(-0.08, 0.98)
    axis.set_xticks([])
    axis.set_yticks([])
    style_axis(axis, grid_axis=None, y_continuous=False)
    colorbar = figure.colorbar(
        points,
        ax=axis,
        shrink=0.72,
        label="Inner macro-substation F1",
    )
    style_colorbar(colorbar)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)
    return output_path


def plot_selected_weights(selected: pd.DataFrame, output_path: Path) -> Path:
    """Stack the optimised F1/F3/F4 weights selected in each outer fold."""

    apply_journal_style()
    data = selected.loc[selected["strategy"].eq("optimised")].sort_values(
        "heldout_substation"
    )
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    bottom = np.zeros(len(data))
    for column, label, color in [
        ("weight_F1", "F1 bridge", COLORS["dark_blue"]),
        ("weight_F3", "F3 slope", COLORS["orange"]),
        ("weight_F4", "F4 duration", COLORS["grey"]),
    ]:
        values = data[column].to_numpy(dtype=float)
        axis.bar(data["heldout_substation"], values, bottom=bottom, label=label, color=color)
        bottom += values
    axis.set_ylim(0, 1.18)
    axis.set_xlabel("Held-out Beta substation")
    axis.set_ylabel("Selected unit-sum weight")
    axis.set_title("Nested inner-LOSO weights used for each outer prediction")
    axis.legend(frameon=False, ncol=3, loc="upper center")
    style_axis(axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return output_path


def plot_physical_vs_ml(metrics: pd.DataFrame, output_path: Path) -> Path:
    """Compare pooled Beta-sure P/R/F1 for the physical and ML decisions."""

    apply_journal_style()
    data = metrics.copy().reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    centers = np.arange(len(data))
    width = 0.24
    for offset, (metric, color) in enumerate(
        zip(
            ["precision", "recall", "f1"],
            [COLORS["dark_blue"], COLORS["orange"], COLORS["grey"]],
            strict=True,
        )
    ):
        positions = centers + (offset - 1) * width
        bars = axis.bar(positions, data[metric], width=width, color=color, label=metric.title())
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=9, rotation=90)
    display_labels = data["display_label"].str.replace(" / ", "\n", regex=False)
    display_labels = display_labels.str.replace("Optimised physical", "Optimised\nphysical")
    axis.set_xticks(centers, display_labels)
    axis.set_ylim(0, 1.10)
    axis.set_ylabel("Held-out Beta sure score")
    axis.set_title("Physical model and Beta-only ML comparison")
    axis.legend(frameon=False, ncol=3, loc="upper center")
    style_axis(axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return output_path


def plot_final_confusion_matrices(day_metrics: pd.DataFrame, output_path: Path) -> Path:
    """Plot pooled day confusion matrices for Beta sure and Beta all."""

    apply_journal_style()
    pooled = day_metrics.loc[day_metrics["aggregation"].eq("pooled")].set_index(
        "confidence_scope"
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for axis, scope, title in zip(
        axes,
        ["beta_sure", "beta_all"],
        ["Beta sure", "Beta all"],
        strict=True,
    ):
        row = pooled.loc[scope]
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        axis.imshow(matrix, cmap=journal_colormap(f"confusion_{scope}"))
        for row_index in range(2):
            for column_index in range(2):
                axis.text(
                    column_index,
                    row_index,
                    f"{int(matrix[row_index, column_index]):,}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=(
                        "white"
                        if matrix[row_index, column_index] > matrix.max() / 2
                        else "black"
                    ),
                )
        axis.set_xticks([0, 1], ["Predicted no", "Predicted RPF"])
        axis.set_yticks([0, 1], ["True no", "True RPF"])
        axis.set_title(title)
        style_axis(axis, grid_axis=None, y_continuous=False)
    figure.suptitle("Nested outer-fold day decisions")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return output_path


def plot_window_iou_distribution(window_audit: pd.DataFrame, output_path: Path) -> Path:
    """Plot Beta-sure IoU for TP days and all truth-or-prediction event days."""

    apply_journal_style()
    sure = window_audit.loc[window_audit["confidence"].eq("sure")]
    tp = sure.loc[sure["true_day"] & sure["predicted_day"], "window_iou"]
    event = sure.loc[sure["true_day"] | sure["predicted_day"], "window_iou"]
    figure, axis = plt.subplots(figsize=(7.6, 4.3))
    bins = np.linspace(0, 1, 21)
    axis.hist(
        event,
        bins=bins,
        color=COLORS["light_grey"],
        alpha=0.9,
        label="Truth-or-prediction event days",
    )
    axis.hist(
        tp,
        bins=bins,
        histtype="step",
        linewidth=2.2,
        color=COLORS["orange"],
        label="True-positive days",
    )
    axis.set_xlabel("Candidate-window IoU")
    axis.set_ylabel("Beta sure substation-days")
    axis.set_title("Localisation of nested outer-fold correction windows")
    axis.legend(frameon=False)
    style_axis(axis, x_continuous=True)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return output_path


def plot_energy_summary(energy_metrics: pd.DataFrame, output_path: Path) -> Path:
    """Compare pooled full-day correction-energy metrics by confidence scope."""

    apply_journal_style()
    data = energy_metrics.loc[
        energy_metrics["aggregation"].eq("pooled")
        & energy_metrics["interval_scope"].eq("full_day")
    ].copy()
    data["scope_label"] = data["confidence_scope"].map(
        {"beta_sure": "Beta sure", "beta_all": "Beta all"}
    )
    metrics = ["energy_precision", "energy_recall", "energy_f1", "energy_iou"]
    colors = [
        COLORS["dark_blue"],
        COLORS["orange"],
        COLORS["grey"],
        COLORS["light_grey"],
    ]
    figure, axis = plt.subplots(figsize=(7.8, 4.4))
    centers = np.arange(len(data))
    width = 0.19
    for index, (metric, color) in enumerate(zip(metrics, colors, strict=True)):
        positions = centers + (index - 1.5) * width
        bars = axis.bar(positions, data[metric], width=width, color=color, label=metric[7:].title())
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
    axis.set_xticks(centers, data["scope_label"])
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Correction-energy score")
    axis.set_title("Pooled full-day correction-energy agreement")
    axis.legend(frameon=False, ncol=4, loc="upper center")
    style_axis(axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return output_path


def plot_auto_accept_burden(coverage: pd.DataFrame, output_path: Path) -> Path:
    """Dual-axis manual-review burden and stacked auto FP/FN errors."""

    apply_journal_style()
    data = coverage.sort_values("coverage_pct")
    x = data["coverage_pct"].to_numpy(dtype=float)
    figure, left_axis = plt.subplots(figsize=(8.2, 4.5))
    right_axis = left_axis.twinx()
    left_axis.plot(
        x,
        data["manual_review_days"],
        marker="o",
        linewidth=2.2,
        color=COLORS["dark_blue"],
        label="Days sent to manual review",
    )
    width = 3.2
    right_axis.bar(x, data["fp"], width=width, color=COLORS["red"], alpha=0.8, label="Auto FP")
    right_axis.bar(
        x,
        data["fn"],
        width=width,
        bottom=data["fp"],
        color=COLORS["orange"],
        alpha=0.85,
        label="Auto FN",
    )
    left_axis.set_xlabel("Auto-accepted Beta sure days (%)")
    left_axis.set_ylabel("Days remaining for manual review", color=COLORS["dark_blue"])
    right_axis.set_ylabel("Errors among auto-accepted days")
    left_axis.set_xticks(x)
    left_axis.set_title("Manual-review burden and auto-accepted errors")
    handles_left, labels_left = left_axis.get_legend_handles_labels()
    handles_right, labels_right = right_axis.get_legend_handles_labels()
    left_axis.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        frameon=False,
        ncol=3,
        loc="upper center",
    )
    align_twin_y_axes(left_axis, right_axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return output_path


def plot_coverage_scores(coverage: pd.DataFrame, output_path: Path) -> Path:
    """Plot precision, recall, and F1 within auto-accepted Beta sure days."""

    apply_journal_style()
    data = coverage.sort_values("coverage_pct")
    figure, axis = plt.subplots(figsize=(7.8, 4.3))
    for metric, color, marker in [
        ("precision", COLORS["dark_blue"], "o"),
        ("recall", COLORS["orange"], "s"),
        ("f1", COLORS["grey"], "^"),
    ]:
        axis.plot(
            data["coverage_pct"],
            data[metric],
            color=color,
            marker=marker,
            linewidth=2,
            label=metric.title(),
        )
    axis.set_ylim(0, 1.03)
    axis.set_xticks(data["coverage_pct"])
    axis.set_xlabel("Auto-accepted Beta sure days (%)")
    axis.set_ylabel("Score among auto-accepted days")
    axis.set_title("Confidence-margin operating points")
    axis.legend(frameon=False, ncol=3)
    style_axis(axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return output_path


def plot_gamma_example_week(series: pd.DataFrame, output_path: Path) -> Path:
    """Plot the September week with the largest manual correction magnitude."""

    apply_journal_style()
    data = series.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    september = data.loc[
        data["timestamp"].between(
            pd.Timestamp("2024-09-01", tz="UTC"),
            pd.Timestamp("2024-09-30 23:45:00", tz="UTC"),
            inclusive="both",
        )
    ].copy()
    september["week_start"] = (
        september["timestamp"].dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time
    )
    september["manual_change"] = (
        september["manually_corrected_MW"] - september["raw_uncorrected_MW"]
    ).abs()
    week_start = september.groupby("week_start")["manual_change"].sum().idxmax()
    week_end = week_start + pd.Timedelta(days=7)
    example = september.loc[
        september["timestamp"].dt.tz_localize(None).between(
            week_start, week_end, inclusive="left"
        )
    ]

    figure, axis = plt.subplots(figsize=(10.2, 4.2))
    axis.plot(
        example["timestamp"],
        example["raw_uncorrected_MW"],
        color=COLORS["grey"],
        linewidth=1.2,
        label="Raw uncorrected",
    )
    axis.plot(
        example["timestamp"],
        example["m9_pbm_corrected_MW"],
        color=COLORS["orange"],
        linewidth=1.6,
        label="m9_pbm corrected",
    )
    axis.plot(
        example["timestamp"],
        example["manually_corrected_MW"],
        color=COLORS["dark_blue"],
        linewidth=1.2,
        linestyle="--",
        label="Manually corrected reference",
    )
    axis.axhline(0, color=COLORS["dark_blue"], linewidth=0.7)
    axis.set_title(f"Gamma substation B: highest-impact September week ({week_start.date()})")
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Net load (MW)")
    axis.xaxis.set_major_locator(mdates.DayLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    series_min = float(
        example[["raw_uncorrected_MW", "m9_pbm_corrected_MW", "manually_corrected_MW"]]
        .min()
        .min()
    )
    series_max = float(
        example[["raw_uncorrected_MW", "m9_pbm_corrected_MW", "manually_corrected_MW"]]
        .max()
        .max()
    )
    series_span = max(series_max - series_min, 1.0)
    axis.set_ylim(series_min - 0.03 * series_span, series_max + 0.28 * series_span)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    style_axis(axis, x_dates=True)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_gamma_data_error(data_metrics: pd.DataFrame, output_path: Path) -> Path:
    """Compare raw and m9 data-error RMSE over the full year and test month."""

    apply_journal_style()
    order = ["full_gamma", "forecast_test_month"]
    labels = ["Full Gamma year", "September 2024"]
    raw = data_metrics.loc[data_metrics["data_condition"].eq("raw_uncorrected")]
    corrected = data_metrics.loc[data_metrics["data_condition"].eq("m9_pbm_corrected")]
    raw_values = raw.set_index("scope").loc[order, "rmse_MW"]
    corrected_values = corrected.set_index("scope").loc[order, "rmse_MW"]
    x = np.arange(len(order))
    width = 0.34

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(
        x - width / 2,
        raw_values,
        width,
        color=COLORS["grey"],
        label="Raw uncorrected",
    )
    axis.bar(
        x + width / 2,
        corrected_values,
        width,
        color=COLORS["orange"],
        label="m9_pbm corrected",
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Data-error RMSE (MW)")
    axis.set_title("Error against the manually corrected Gamma reference")
    axis.legend(frameon=False)
    style_axis(axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_gamma_forecast_rmse(metrics: pd.DataFrame, output_path: Path) -> Path:
    """Group forecast corrections with model variants inside each group."""

    apply_journal_style()
    models = ["seasonal_naive", "linear_regression", "xgboost"]
    conditions = ["raw_uncorrected", "m9_pbm_corrected", "manually_corrected"]
    model_labels = ["Seasonal naive", "Linear regression", "XGBoost"]
    condition_labels = ["Raw uncorrected", "m9_pbm corrected", "Manual reference"]
    colors = [COLORS["dark_blue"], COLORS["orange"], COLORS["grey"]]
    x = np.arange(len(conditions))
    width = 0.24

    figure, axis = plt.subplots(figsize=(8.4, 4.5))
    for index, (model, label, color) in enumerate(
        zip(models, model_labels, colors, strict=True)
    ):
        subset = metrics.loc[metrics["model"].eq(model)].set_index("data_condition")
        values = subset.loc[conditions, "rmse_MW"]
        axis.bar(x + (index - 1) * width, values, width, color=color, label=label)
    axis.set_xticks(x, condition_labels)
    axis.set_ylabel("Seven-day-ahead RMSE (MW)")
    axis.set_title("Direct September 2024 point forecasts")
    axis.set_ylim(0, float(metrics["rmse_MW"].max()) * 1.25)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    style_axis(axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_gamma_forecast_residuals(
    predictions: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot forecast residual distributions by model and data condition."""

    apply_journal_style()
    data = predictions.copy()
    data["residual_MW"] = data["y_pred"] - data["y_reference"]
    models = ["seasonal_naive", "linear_regression", "xgboost"]
    conditions = ["raw_uncorrected", "m9_pbm_corrected", "manually_corrected"]
    colors = [COLORS["grey"], COLORS["orange"], COLORS["dark_blue"]]
    labels = ["Raw", "m9_pbm", "Manual"]
    positions = []
    values = []
    box_colors = []
    tick_positions = []
    for model_index, model in enumerate(models):
        center = model_index * 4 + 2
        tick_positions.append(center)
        for condition_index, condition in enumerate(conditions):
            residual = data.loc[
                data["model"].eq(model) & data["data_condition"].eq(condition),
                "residual_MW",
            ].dropna()
            positions.append(center + condition_index - 1)
            values.append(residual.to_numpy())
            box_colors.append(colors[condition_index])

    figure, axis = plt.subplots(figsize=(9.4, 4.6))
    boxes = axis.boxplot(
        values,
        positions=positions,
        widths=0.72,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["dark_blue"], "linewidth": 1.0},
    )
    for box, color in zip(boxes["boxes"], box_colors, strict=True):
        box.set_facecolor(color)
        box.set_alpha(0.82)
    axis.axhline(0, color=COLORS["dark_blue"], linewidth=0.8)
    axis.set_xticks(tick_positions, ["Seasonal naive", "Linear regression", "XGBoost"])
    axis.set_ylabel("Forecast residual (MW)")
    axis.set_title("September forecast residuals against manual reference")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, label=label)
        for color, label in zip(colors, labels, strict=True)
    ]
    axis.legend(handles=handles, frameon=False, ncol=3, loc="upper center")
    style_axis(axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path

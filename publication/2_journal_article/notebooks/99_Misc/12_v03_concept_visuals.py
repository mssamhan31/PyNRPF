from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUTPUT_FOLDER_NAME = "12_v03_concept_visuals"
SLOTS_PER_DAY = 96
SLOTS_PER_HOUR = 4
CONTEXT_SLOTS = 4
WINDOW_START = 43  # 10:45
WINDOW_END = 55  # 13:45, inclusive

COLORS = {
    "uncorrected": "#eb932c",
    "corrected": "#22303d",
    "solar": "#5C7D99",
    "net": "#68737d",
    "bridge": "#3d4a53",
    "window": "#f2c879",
    "context": "#d9e1e8",
    "roughness": "#b45f06",
    "slope": "#2F4D67",
}


@dataclass(frozen=True)
class SyntheticDay:
    slots: np.ndarray
    hours: np.ndarray
    demand_true: np.ndarray
    solar: np.ndarray
    measured_net: np.ndarray
    u_no: np.ndarray
    u_corr: np.ndarray
    window_start: int
    window_end: int
    context_slots: int

    @property
    def window_start_hour(self) -> float:
        return self.window_start / SLOTS_PER_HOUR

    @property
    def window_end_hour(self) -> float:
        return (self.window_end + 1) / SLOTS_PER_HOUR

    @property
    def left_context(self) -> np.ndarray:
        return np.arange(self.window_start - self.context_slots, self.window_start)

    @property
    def right_context(self) -> np.ndarray:
        return np.arange(self.window_end + 1, self.window_end + 1 + self.context_slots)

    @property
    def window(self) -> np.ndarray:
        return np.arange(self.window_start, self.window_end + 1)


def output_dir() -> Path:
    return Path(__file__).resolve().parent / "outputs" / OUTPUT_FOLDER_NAME


def build_synthetic_day() -> SyntheticDay:
    slots = np.arange(SLOTS_PER_DAY)
    hours = slots / SLOTS_PER_HOUR

    morning = 0.42 * np.exp(-0.5 * ((hours - 7.4) / 1.45) ** 2)
    evening = 0.55 * np.exp(-0.5 * ((hours - 18.4) / 1.95) ** 2)
    midday_dip = -0.34 * np.exp(-0.5 * ((hours - 12.1) / 3.0) ** 2)
    slow_drift = 0.10 * np.sin((hours - 4.0) / 24.0 * 2.0 * np.pi)
    demand_true = 3.55 + morning + evening + midday_dip + slow_drift

    daylight = np.clip(np.sin((hours - 6.0) / 12.0 * np.pi), 0.0, None)
    solar = 5.25 * daylight**1.65

    true_net = demand_true - solar
    measured_net = true_net.copy()
    window = np.arange(WINDOW_START, WINDOW_END + 1)
    measured_net[window] = -true_net[window]

    u_no = solar + measured_net
    u_corr = u_no.copy()
    u_corr[window] = solar[window] - measured_net[window]

    return SyntheticDay(
        slots=slots,
        hours=hours,
        demand_true=demand_true,
        solar=solar,
        measured_net=measured_net,
        u_no=u_no,
        u_corr=u_corr,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        context_slots=CONTEXT_SLOTS,
    )


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#9aa5ad",
            "axes.labelcolor": "#26323a",
            "axes.titlecolor": "#26323a",
            "xtick.color": "#26323a",
            "ytick.color": "#26323a",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.color": "#d7dde2",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
        }
    )


def style_axis(ax: plt.Axes, title: str, show_ylabel: bool = True) -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlim(6.0, 18.0)
    ax.set_ylim(-2.4, 8.2)
    ax.set_xticks(np.arange(6, 19, 2))
    ax.set_xticklabels([f"{int(hour):02d}:00" for hour in np.arange(6, 19, 2)])
    ax.set_xlabel("Time of day")
    if show_ylabel:
        ax.set_ylabel("Power (MW)")
    else:
        ax.set_ylabel("")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def add_window(ax: plt.Axes, day: SyntheticDay, y: float = 7.55) -> None:
    ax.axvspan(
        day.window_start_hour,
        day.window_end_hour,
        color=COLORS["window"],
        alpha=0.26,
        lw=0,
        zorder=0,
    )
    ax.text(
        (day.window_start_hour + day.window_end_hour) / 2.0,
        y,
        "candidate window W",
        ha="center",
        va="center",
        color="#5b4420",
        fontsize=9,
    )


def add_context_regions(ax: plt.Axes, day: SyntheticDay) -> None:
    left_start = (day.window_start - day.context_slots) / SLOTS_PER_HOUR
    left_end = day.window_start / SLOTS_PER_HOUR
    right_start = (day.window_end + 1) / SLOTS_PER_HOUR
    right_end = (day.window_end + 1 + day.context_slots) / SLOTS_PER_HOUR
    ax.axvspan(left_start, left_end, color=COLORS["context"], alpha=0.22, lw=0, zorder=0)
    ax.axvspan(right_start, right_end, color=COLORS["context"], alpha=0.22, lw=0, zorder=0)


def plot_context(ax: plt.Axes, day: SyntheticDay, include_labels: bool = True) -> None:
    solar_label = "solar S" if include_labels else None
    net_label = "measured net y" if include_labels else None
    ax.plot(
        day.hours,
        day.solar,
        color=COLORS["solar"],
        lw=1.4,
        alpha=0.32,
        label=solar_label,
        zorder=1,
    )
    ax.plot(
        day.hours,
        day.measured_net,
        color=COLORS["net"],
        lw=1.1,
        ls=":",
        alpha=0.42,
        label=net_label,
        zorder=1,
    )


def plot_reconstructions(ax: plt.Axes, day: SyntheticDay, include_labels: bool = True) -> None:
    no_label = "U_no: uncorrected reconstruction" if include_labels else None
    corr_label = "U_corr: corrected in W" if include_labels else None
    ax.plot(
        day.hours,
        day.u_no,
        color=COLORS["uncorrected"],
        lw=2.5,
        label=no_label,
        zorder=4,
    )
    ax.plot(
        day.hours,
        day.u_corr,
        color=COLORS["corrected"],
        lw=2.7,
        label=corr_label,
        zorder=5,
    )


def add_legend(ax: plt.Axes, ncol: int = 2) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label and label not in seen:
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
    ax.legend(
        unique_handles,
        unique_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=ncol,
        frameon=False,
    )


def arrowprops(color: str) -> dict[str, object]:
    return {
        "arrowstyle": "->",
        "color": color,
        "lw": 1.3,
        "shrinkA": 3,
        "shrinkB": 3,
    }


def bridge_values(day: SyntheticDay) -> tuple[float, float, np.ndarray]:
    left_anchor = float(np.median(day.u_no[day.left_context]))
    right_anchor = float(np.median(day.u_no[day.right_context]))
    window = day.window
    length = len(window)
    j = np.arange(1, length + 1)
    bridge = left_anchor + (j / (length + 1)) * (right_anchor - left_anchor)
    return left_anchor, right_anchor, bridge


def plot_curve_comparison(
    ax: plt.Axes,
    day: SyntheticDay,
    standalone: bool = True,
    show_ylabel: bool = True,
) -> None:
    add_window(ax, day)
    plot_context(ax, day, include_labels=standalone)
    plot_reconstructions(ax, day, include_labels=standalone)
    style_axis(ax, "Comparing the two reconstructions", show_ylabel=show_ylabel)

    ax.annotate(
        "wrong-sign window\ncreates a reflected bump",
        xy=(12.15, day.u_no[49]),
        xytext=(8.15, 6.55),
        ha="left",
        color=COLORS["uncorrected"],
        arrowprops=arrowprops(COLORS["uncorrected"]),
    )
    ax.annotate(
        "sign flip recovers a\nsmooth demand path",
        xy=(12.9, day.u_corr[52]),
        xytext=(14.25, 3.05),
        ha="left",
        color=COLORS["corrected"],
        arrowprops=arrowprops(COLORS["corrected"]),
    )
    if standalone:
        add_legend(ax, ncol=2)


def plot_bridge(
    ax: plt.Axes,
    day: SyntheticDay,
    standalone: bool = True,
    show_ylabel: bool = True,
) -> None:
    add_context_regions(ax, day)
    add_window(ax, day)
    plot_context(ax, day, include_labels=False)
    plot_reconstructions(ax, day, include_labels=standalone)

    left_anchor, right_anchor, bridge = bridge_values(day)
    window_hours = day.hours[day.window]
    left_x = float(np.mean(day.hours[day.left_context]))
    right_x = float(np.mean(day.hours[day.right_context]))

    ax.plot(
        window_hours,
        bridge,
        color=COLORS["bridge"],
        lw=2.0,
        ls=(0, (5, 3)),
        label="linear bridge between anchors" if standalone else None,
        zorder=6,
    )
    ax.scatter(
        [left_x, right_x],
        [left_anchor, right_anchor],
        color=COLORS["bridge"],
        s=46,
        zorder=7,
        label="context anchors" if standalone else None,
    )
    for idx in day.window[::3]:
        pos = idx - day.window_start
        ax.vlines(
            day.hours[idx],
            bridge[pos],
            day.u_no[idx],
            color=COLORS["uncorrected"],
            lw=1.0,
            alpha=0.35,
            zorder=3,
        )

    style_axis(ax, "Bridge concept", show_ylabel=show_ylabel)
    ax.annotate(
        "anchors come from\npre/post context",
        xy=(left_x, left_anchor),
        xytext=(6.45, 4.8),
        ha="left",
        color=COLORS["bridge"],
        arrowprops=arrowprops(COLORS["bridge"]),
    )
    ax.annotate(
        "corrected curve stays\nnear the bridge",
        xy=(12.15, day.u_corr[49]),
        xytext=(14.1, 5.05),
        ha="left",
        color=COLORS["corrected"],
        arrowprops=arrowprops(COLORS["corrected"]),
    )
    if standalone:
        add_legend(ax, ncol=2)


def plot_roughness(
    ax: plt.Axes,
    day: SyntheticDay,
    standalone: bool = True,
    show_ylabel: bool = True,
) -> None:
    add_context_regions(ax, day)
    add_window(ax, day)
    plot_context(ax, day, include_labels=False)
    plot_reconstructions(ax, day, include_labels=standalone)

    local_start = day.window_start - day.context_slots
    local_end = day.window_end + day.context_slots
    local_hours = day.hours[local_start : local_end + 1]
    local_no = day.u_no[local_start : local_end + 1]
    ax.plot(
        local_hours,
        local_no,
        color=COLORS["roughness"],
        lw=0,
        marker="o",
        ms=3.5,
        alpha=0.65,
        label="local first differences" if standalone else None,
        zorder=8,
    )

    for left, right in [(day.window_start - 1, day.window_start), (day.window_end, day.window_end + 1)]:
        ax.annotate(
            "",
            xy=(day.hours[left], day.u_no[left]),
            xytext=(day.hours[right], day.u_no[right]),
            arrowprops={
                "arrowstyle": "<->",
                "color": COLORS["roughness"],
                "lw": 1.7,
            },
            zorder=9,
        )

    style_axis(ax, "Roughness concept", show_ylabel=show_ylabel)
    ax.annotate(
        "roughness checks\nlocal step sizes",
        xy=(10.75, day.u_no[43]),
        xytext=(6.65, 6.35),
        ha="left",
        color=COLORS["roughness"],
        arrowprops=arrowprops(COLORS["roughness"]),
    )
    ax.annotate(
        "correction reduces\nartificial jumps",
        xy=(14.0, day.u_corr[56]),
        xytext=(14.7, 3.85),
        ha="left",
        color=COLORS["corrected"],
        arrowprops=arrowprops(COLORS["corrected"]),
    )
    if standalone:
        add_legend(ax, ncol=2)


def fitted_segment(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coeff = np.polyfit(x, y, deg=1)
    y_fit = np.polyval(coeff, x)
    return x, y_fit


def draw_slope_segment(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    label: str | None = None,
) -> None:
    fit_x, fit_y = fitted_segment(x, y)
    ax.plot(
        fit_x,
        fit_y,
        color=color,
        lw=2.3,
        ls=(0, (4, 2)),
        label=label,
        zorder=8,
    )


def plot_slope_change(
    ax: plt.Axes,
    day: SyntheticDay,
    standalone: bool = True,
    show_ylabel: bool = True,
) -> None:
    add_context_regions(ax, day)
    add_window(ax, day)
    plot_context(ax, day, include_labels=False)
    plot_reconstructions(ax, day, include_labels=standalone)

    left_before = np.arange(day.window_start - day.context_slots, day.window_start)
    left_after = np.arange(day.window_start, day.window_start + day.context_slots)
    right_before = np.arange(day.window_end - day.context_slots + 1, day.window_end + 1)
    right_after = np.arange(day.window_end + 1, day.window_end + 1 + day.context_slots)

    draw_slope_segment(
        ax,
        day.hours[left_before],
        day.u_no[left_before],
        COLORS["uncorrected"],
        "uncorrected boundary slopes" if standalone else None,
    )
    draw_slope_segment(ax, day.hours[left_after], day.u_no[left_after], COLORS["uncorrected"])
    draw_slope_segment(
        ax,
        day.hours[right_before],
        day.u_no[right_before],
        COLORS["uncorrected"],
    )
    draw_slope_segment(ax, day.hours[right_after], day.u_no[right_after], COLORS["uncorrected"])

    draw_slope_segment(
        ax,
        day.hours[left_before],
        day.u_corr[left_before],
        COLORS["corrected"],
        "corrected boundary slopes" if standalone else None,
    )
    draw_slope_segment(ax, day.hours[left_after], day.u_corr[left_after], COLORS["corrected"])
    draw_slope_segment(
        ax,
        day.hours[right_before],
        day.u_corr[right_before],
        COLORS["corrected"],
    )
    draw_slope_segment(ax, day.hours[right_after], day.u_corr[right_after], COLORS["corrected"])

    style_axis(ax, "Slope-change concept", show_ylabel=show_ylabel)
    ax.annotate(
        "entry and exit slopes\nchange abruptly in U_no",
        xy=(10.75, day.u_no[43]),
        xytext=(6.55, 6.35),
        ha="left",
        color=COLORS["uncorrected"],
        arrowprops=arrowprops(COLORS["uncorrected"]),
    )
    ax.annotate(
        "corrected boundary\nslopes align better",
        xy=(13.75, day.u_corr[55]),
        xytext=(14.5, 3.05),
        ha="left",
        color=COLORS["corrected"],
        arrowprops=arrowprops(COLORS["corrected"]),
    )
    if standalone:
        add_legend(ax, ncol=2)


def save_standalone(
    day: SyntheticDay,
    path: Path,
    plotter,
    title: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    plotter(ax, day, standalone=True, show_ylabel=True)
    fig.suptitle(title, x=0.08, y=0.98, ha="left", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.28)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path


def save_overview(day: SyntheticDay, path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), sharex=True, sharey=True)
    plot_curve_comparison(axes[0, 0], day, standalone=False, show_ylabel=True)
    plot_bridge(axes[0, 1], day, standalone=False, show_ylabel=False)
    plot_roughness(axes[1, 0], day, standalone=False, show_ylabel=True)
    plot_slope_change(axes[1, 1], day, standalone=False, show_ylabel=False)

    overview_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["uncorrected"],
            lw=2.5,
            label="U_no: uncorrected reconstruction",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["corrected"],
            lw=2.7,
            label="U_corr: corrected in W",
        ),
        Line2D([0], [0], color=COLORS["solar"], lw=1.4, alpha=0.45, label="solar S"),
        Line2D(
            [0],
            [0],
            color=COLORS["net"],
            lw=1.2,
            ls=":",
            alpha=0.65,
            label="measured net y",
        ),
        Patch(facecolor=COLORS["window"], edgecolor="none", alpha=0.35, label="candidate window W"),
    ]
    fig.legend(
        handles=overview_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "v0.3 structured counterfactual concepts",
        x=0.05,
        y=0.985,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.91, bottom=0.11, wspace=0.09, hspace=0.22)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path


def write_figures() -> list[Path]:
    configure_matplotlib()
    day = build_synthetic_day()
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.png"):
        stale.unlink()

    figure_paths = [
        save_overview(day, out / "00_v03_concepts_overview.png"),
        save_standalone(
            day,
            out / "01_curve_comparison_uncorrected_vs_corrected.png",
            plot_curve_comparison,
            "Comparing uncorrected and corrected reconstructions",
        ),
        save_standalone(
            day,
            out / "02_bridge_concept.png",
            plot_bridge,
            "Bridge improvement concept",
        ),
        save_standalone(
            day,
            out / "03_roughness_concept.png",
            plot_roughness,
            "Roughness improvement concept",
        ),
        save_standalone(
            day,
            out / "04_slope_change_concept.png",
            plot_slope_change,
            "Slope-continuity improvement concept",
        ),
    ]
    return figure_paths


def main() -> None:
    figure_paths = write_figures()
    print(f"Wrote {len(figure_paths)} concept PNGs:")
    for path in figure_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()

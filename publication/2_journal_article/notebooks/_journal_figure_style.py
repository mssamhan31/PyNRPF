"""Shared visual contract for journal article figures."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import matplotlib
import matplotlib.dates as mdates
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "orange": "#eb932c",
    "dark_blue": "#22303d",
    "grey": "#2F4D67",
    "light_grey": "#5C7D99",
    "light_white": "#ebe3e3",
    "red": "#B64A4A",
}

BAR_COLORS = [
    COLORS["dark_blue"],
    COLORS["orange"],
    COLORS["grey"],
    COLORS["light_grey"],
]


def apply_journal_style() -> None:
    """Apply the article-wide typography, frame, and grid defaults."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["dark_blue"],
            "axes.labelcolor": COLORS["dark_blue"],
            "axes.titlecolor": COLORS["dark_blue"],
            "axes.axisbelow": True,
            "axes.grid": False,
            "axes.linewidth": 1.0,
            "grid.color": COLORS["light_white"],
            "grid.linewidth": 0.9,
            "grid.alpha": 0.9,
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "legend.fontsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "xtick.color": COLORS["dark_blue"],
            "ytick.color": COLORS["dark_blue"],
            "text.color": COLORS["dark_blue"],
            "savefig.facecolor": "white",
            "legend.frameon": False,
        }
    )


def journal_colormap(name: str = "journal_sequential") -> LinearSegmentedColormap:
    """Return a sequential map constructed only from the supplied palette."""

    return LinearSegmentedColormap.from_list(
        name,
        [
            COLORS["light_white"],
            COLORS["light_grey"],
            COLORS["orange"],
            COLORS["dark_blue"],
        ],
    )


def limit_continuous_ticks(
    axis: Any,
    *,
    x: bool = False,
    y: bool = False,
    x_dates: bool = False,
) -> None:
    """Limit selected continuous axes to at most five major tick positions."""

    if x_dates:
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
    elif x:
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    if y:
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))


def style_axis(
    axis: Any,
    *,
    grid_axis: str | None = "y",
    x_continuous: bool = False,
    y_continuous: bool = True,
    x_dates: bool = False,
) -> None:
    """Apply a complete frame, background grid, and optional tick limits."""

    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["dark_blue"])
    axis.grid(False)
    if grid_axis is not None:
        axis.grid(
            True,
            axis=grid_axis,
            color=COLORS["light_white"],
            linewidth=0.9,
            alpha=0.9,
            zorder=0,
        )
    limit_continuous_ticks(
        axis,
        x=x_continuous,
        y=y_continuous,
        x_dates=x_dates,
    )


def style_colorbar(colorbar: Any) -> None:
    """Apply article typography and a five-tick cap to a colorbar."""

    colorbar.locator = MaxNLocator(nbins=4)
    colorbar.update_ticks()
    colorbar.ax.tick_params(labelsize=13, colors=COLORS["dark_blue"])
    colorbar.set_label(colorbar.ax.get_ylabel(), fontsize=14, color=COLORS["dark_blue"])


def align_twin_y_axes(left_axis: Any, right_axis: Any) -> None:
    """Align two non-negative y scales at five shared normalized positions."""

    def nice_upper(axis: Any) -> float:
        upper = max(0.0, float(axis.get_ylim()[1]))
        ticks = MaxNLocator(nbins=4).tick_values(0.0, upper)
        positive = ticks[ticks > 0]
        return float(positive[-1]) if len(positive) else 1.0

    left_upper = nice_upper(left_axis)
    right_upper = nice_upper(right_axis)
    left_axis.set_ylim(0.0, left_upper)
    right_axis.set_ylim(0.0, right_upper)
    left_axis.set_yticks(np.linspace(0.0, left_upper, 5))
    right_axis.set_yticks(np.linspace(0.0, right_upper, 5))
    style_axis(left_axis, grid_axis="y", y_continuous=False)
    style_axis(right_axis, grid_axis=None, y_continuous=False)


def style_axes(axes: Iterable[Any], **kwargs: Any) -> None:
    """Apply :func:`style_axis` to a collection of axes."""

    for axis in axes:
        style_axis(axis, **kwargs)
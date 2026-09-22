"""The journal figure style: typography, a colour-blind-safe palette, frames and a saver.

Inputs:  none; ``apply_journal_style`` sets matplotlib's rcParams for the process.
Outputs: colour constants, axis helpers and ``save_figure``, which writes PNG (and PDF)
         files with no embedded dates so a regenerated file hashes the same.
Key steps: an Arial-like sans-serif (Arial, then Liberation Sans, then DejaVu Sans),
         sizes that read at a single journal column, TrueType fonts embedded in PDF,
         the Okabe–Ito palette for series and a dark slate for text and frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# Okabe–Ito series colours (distinguishable under the common colour-vision deficiencies)
# plus the neutral tones used for text, frames and grids.
COLORS = {
    "orange": "#E69F00", "blue": "#0072B2", "sky": "#56B4E9", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "grey": "#6E6E6E", "light_grey": "#B0B0B0",
    "dark_blue": "#22303D", "light_white": "#EBEBEB",
}
METHOD_COLORS = {"m7": COLORS["grey"], "m8": COLORS["blue"], "m9": COLORS["orange"]}
GROUP_LABELS = {"combined": "Combined", "alpha": "Alpha", "beta": "Beta sure"}
COHORT_LABELS = {"alpha": "Alpha", "beta": "Beta 'sure'"}


def apply_journal_style() -> None:
    """Article-wide typography, frame and grid defaults."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": COLORS["dark_blue"],
        "axes.labelcolor": COLORS["dark_blue"], "axes.titlecolor": COLORS["dark_blue"], "axes.axisbelow": True,
        "axes.grid": False, "axes.linewidth": 1.0, "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10, "xtick.color": COLORS["dark_blue"],
        "ytick.color": COLORS["dark_blue"], "text.color": COLORS["dark_blue"], "savefig.facecolor": "white",
        "legend.frameon": False,
    })


def style_axis(axis: Any, grid_axis: str | None = "y", y_continuous: bool = True) -> None:
    """Complete frame, light horizontal grid, at most five major y ticks."""
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["dark_blue"])
    axis.grid(False)
    if grid_axis is not None:
        axis.grid(True, axis=grid_axis, color=COLORS["light_white"], linewidth=0.9, alpha=0.9, zorder=0)
    if y_continuous:
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))


def align_twin_y_axes(left: Any, right: Any) -> None:
    """Align two non-negative y scales at five shared positions."""
    def nice_upper(axis: Any) -> float:
        upper = max(0.0, float(axis.get_ylim()[1]))
        ticks = MaxNLocator(nbins=4).tick_values(0.0, upper)
        positive = ticks[ticks > 0]
        return float(positive[-1]) if len(positive) else 1.0
    lu, ru = nice_upper(left), nice_upper(right)
    left.set_ylim(0.0, lu)
    right.set_ylim(0.0, ru)
    left.set_yticks(np.linspace(0.0, lu, 5))
    right.set_yticks(np.linspace(0.0, ru, 5))
    style_axis(left, grid_axis="y", y_continuous=False)
    style_axis(right, grid_axis=None, y_continuous=False)


def save_figure(figure: Any, path: Path, formats: tuple[str, ...] = ("png",), dpi: int = 200) -> list[Path]:
    """Write a figure as ``path`` with each of ``formats`` as suffix, then close it.

    Args:
        figure: the matplotlib figure.
        path: the file to write; its suffix is replaced by each format.
        formats: file formats, ``png`` and/or ``pdf``.
        dpi: raster resolution for PNG.

    Returns:
        The paths written. No creation date is embedded, so a rerun gives the same bytes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    written = []
    for fmt in formats:
        target = path.with_suffix(f".{fmt}")
        metadata = {"CreationDate": None, "ModDate": None} if fmt == "pdf" else {"Software": None}
        figure.savefig(target, dpi=dpi, bbox_inches="tight", metadata=metadata)
        written.append(target)
    plt.close(figure)
    return written

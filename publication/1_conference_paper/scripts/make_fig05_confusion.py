"""Fig. 5 of the conference paper: test-set confusion matrices of m7_dtr and m8_xgb, day and interval level.

Purpose: draw Fig. 5 at its final printed size (7.0 in x 1.25 in, full page width, Times New Roman, no text
         below 8 pt), so it is readable on the page. The look is the original one: viridis on each panel's
         counts with a colour bar per panel, bold counts as 1.3k, P/R/F1 under each panel title, and the
         Day-Level and Interval-Level groups.
Inputs:  outputs/metrics__local_dev.json, written by notebook 01 (keys m7_threshold | m8_xgb ->
         day | interval_tp_days_only -> test -> tp, fp, fn, tn).
Outputs: outputs/publication_figures/fig05_confusion_matrices_1x4.png (600 dpi) and .svg (text kept as text).
Key steps:
    1. Read the sixteen test-set counts.
    2. Recompute precision, recall and F1 and stop if they differ from the values printed in the paper.
    3. Draw two groups (Day-Level, Interval-Level) of two matrices each and save at the exact size.

Day level counts site-days. Interval level counts daytime 15-minute slots (06:00 to 18:00) on the days the
method correctly flagged at day level (the tp_days_only scope of ``src/evaluate.py``).

Run from this folder's parent:  python scripts/make_fig05_confusion.py
Notebook 02 calls ``save_fig05`` in its Figure 5 cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
METRICS_JSON = ROOT / "outputs" / "metrics__local_dev.json"
FIG05_PATH = ROOT / "outputs" / "publication_figures" / "fig05_confusion_matrices_1x4.png"

METHODS = [("m7_dtr", "m7_threshold"), ("m8_xgb", "m8_xgb")]        # (name in the paper, key in the JSON)
LEVELS = [("Day-Level", "Day-Level", "day"),
          ("Interval-Level", "Interval (TP days only)", "interval_tp_days_only")]   # (header, title, key)

# Precision, recall and F1 as printed in the paper's results table; the figure must agree with them.
PAPER_PRF = {
    ("m7_threshold", "day"): (0.912, 0.963, 0.937),
    ("m8_xgb", "day"): (0.954, 0.958, 0.956),
    ("m7_threshold", "interval_tp_days_only"): (0.869, 0.796, 0.831),
    ("m8_xgb", "interval_tp_days_only"): (0.928, 0.796, 0.857),
}

FIG_W, FIG_H = 7.0, 1.25          # final size in the paper (full page width); the paper is at its page limit
TEXT_PT, COUNT_PT = 8, 9          # every label, tick and title; the counts in the cells
STYLE = {
    "font.family": "Times New Roman", "font.size": TEXT_PT, "axes.titlesize": TEXT_PT,
    "axes.labelsize": TEXT_PT, "xtick.labelsize": TEXT_PT, "ytick.labelsize": TEXT_PT,
    "axes.linewidth": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.titlepad": 2, "axes.labelpad": 1,
    "svg.fonttype": "none", "savefig.facecolor": "white",
}


# ----------------------------------------------------------------------------- data

def load_counts(path: Path = METRICS_JSON) -> dict[tuple[str, str], np.ndarray]:
    """Test-set confusion matrices ``{(method key, level key): [[tp, fn], [fp, tn]]}``.

    Rows are Actual (Pos, Neg) and columns Predicted (Pos, Neg); Pos is an RPF sign error.
    """
    metrics = json.loads(path.read_text(encoding="utf-8"))
    counts = {}
    for _, method in METHODS:
        for _, _, level in LEVELS:
            t = metrics[method][level]["test"]
            counts[(method, level)] = np.array([[t["tp"], t["fn"]], [t["fp"], t["tn"]]], dtype=int)
    return counts


def precision_recall_f1(cm: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall and F1 of the positive class from ``[[tp, fn], [fp, tn]]``."""
    (tp, fn), (fp, _) = cm
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return precision, recall, 2 * precision * recall / (precision + recall)


def check_against_paper(counts: dict[tuple[str, str], np.ndarray]) -> None:
    """Stop before drawing if any recomputed P, R or F1 (3 decimals) differs from the paper's table."""
    problems = [key for key, printed in PAPER_PRF.items()
                if tuple(round(float(v), 3) for v in precision_recall_f1(counts[key])) != printed]
    if problems:
        raise ValueError(f"Counts do not reproduce the paper's P/R/F1 for {problems}; Fig. 5 not drawn.")


# ----------------------------------------------------------------------------- drawing

def format_k(value: float) -> str:
    """Counts as printed in the cells and on the colour bars: 49, 124, 1.3k, 16k."""
    if abs(value) >= 10_000:
        return f"{value / 1000:.0f}k"
    if abs(value) >= 1_000:
        return f"{value / 1000:.1f}k"
    return f"{value:.0f}"


def draw_matrix(axis, cm: np.ndarray, title: str, first: bool):
    """One matrix: viridis on its own counts, bold count in each cell, P/R/F1 under the title."""
    image = axis.imshow(cm, cmap="viridis", interpolation="nearest", aspect="auto")
    for i in range(2):
        for j in range(2):
            value = int(cm[i, j])
            colour = "white" if value < cm.max() * 0.5 else "black"           # light text on the dark cells
            axis.text(j, i, format_k(value), ha="center", va="center", fontsize=COUNT_PT, fontweight="bold",
                      color=colour)
    p, r, f1 = precision_recall_f1(cm)
    axis.set_title(f"{title}\nP={p:.3f}  R={r:.3f}  F1={f1:.3f}", linespacing=1.25)
    axis.set_xticks([0, 1], ["Pos", "Neg"])
    axis.set_yticks([0, 1], ["Pos", "Neg"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual" if first else "")
    axis.grid(False)
    return image


def colour_bar(figure, image, axis) -> None:
    bar = figure.colorbar(image, ax=axis, fraction=0.06, pad=0.03, aspect=10)
    bar.set_label("Count", labelpad=1)
    bar.ax.yaxis.set_major_locator(mtick.MaxNLocator(3))
    bar.ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: format_k(v)))
    bar.ax.tick_params(length=1.5, pad=1, width=0.5)
    bar.outline.set_linewidth(0.5)


def draw_fig05(counts: dict[tuple[str, str], np.ndarray]):
    """Two subfigures (Day-Level, Interval-Level), a matrix and its colour bar per method, at 7.0 x 1.25 in."""
    figure = plt.figure(figsize=(FIG_W, FIG_H), layout="constrained")
    figure.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02)
    for group, (header, title, level) in zip(figure.subfigures(1, 2, wspace=0.04), LEVELS, strict=True):
        group.suptitle(header, fontsize=TEXT_PT, fontweight="bold")
        for k, (axis, (name, method)) in enumerate(zip(group.subplots(1, 2), METHODS, strict=True)):
            image = draw_matrix(axis, counts[(method, level)], f"{name}  {title}", first=(k == 0))
            colour_bar(group, image, axis)
    return figure


def save_fig05(metrics_json: Path = METRICS_JSON, out_png: Path = FIG05_PATH):
    """Check the counts, draw Fig. 5 and save it as 600 dpi PNG and SVG; returns the figure.

    The style is applied inside a context, so a notebook's own settings are untouched afterwards.
    """
    font_manager.findfont("Times New Roman", fallback_to_default=False)     # fail loudly if missing
    counts = load_counts(metrics_json)
    check_against_paper(counts)
    with plt.rc_context(STYLE):
        figure = draw_fig05(counts)
        figure.savefig(out_png, dpi=600)        # no bbox_inches="tight": the size stays 7.0 x 1.25 in
        figure.savefig(out_png.with_suffix(".svg"))
    return figure


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    save_fig05()
    print("Saved:", FIG05_PATH, "and", FIG05_PATH.with_suffix(".svg"))

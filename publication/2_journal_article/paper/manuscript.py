"""The manuscript's figures and tables, numbered as in the paper.

Inputs:  the reference run under ``results/`` (through ``results``), the Alpha and Beta datasets
         for the day-level figures, and the release calibration of ``pynrpf``.
Outputs: ``FIGURES`` (``fig01`` to ``fig08``, each ``settings -> Figure``) and ``TABLES``
         (``table01`` to ``table03``, each ``settings -> DataFrame``); ``figures.write_all`` and
         ``tables.write_all`` save them under ``results/paper/``.
Key steps: every figure is drawn at its final size for a two-column IEEE Transactions page
         (3.5 in single column, 7.16 in double) with 8 pt type; multi-panel figures stack their
         panels in rows; legends sit below the panels so no text overlaps a curve.

    fig01  expected versus observed reading under a wrong sign (Alpha day, no station identifier)
    fig02  the two questions: is the day wrong, which slots flip (Beta worked day)
    fig03  the energy metrics on a synthetic day, and the station-held-out design
    fig04  the method on the worked day: two stories, evidence surface, calibration and decision bands
    fig05  headline metrics, three methods, Beta sure first
    fig06  per station: Beta with three methods, Alpha with M9 only
    fig07  the review trade-off: the precision frontier, the review burden and the remaining errors against c
    fig08  forecast impact on Gamma: the highest-impact week and the RMSE per model (double column)
    table01  the three datasets by role
    table02  headline metrics with bootstrap intervals, Beta sure first
    table03  Energy IoU and energy precision per held-out station and method (Fig. 6 in numbers)
"""

from __future__ import annotations

from typing import Any, Callable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pynrpf.m9 import RELEASE_CALIBRATION, bridge, edges, evidence, misfit, stories, windows, winner

from . import results
from .config import Settings
from .data import METHOD_LABELS, load_cohort
from .gamma import CONDITIONS, MODEL_LABELS, MODELS, _utc
from .style import COLORS, COLUMN_WIDTH, DOUBLE_WIDTH, apply_paper_style, legend_below, panel_label, style_axis

FigureFn = Callable[[Settings], Any]
TableFn = Callable[[Settings], pd.DataFrame]

# The worked day of the method section (the deck and the documentation walk through it) and
# the Alpha day of the introduction figure.
WORKED_DAY = ("beta", "beta_B", "2024-09-20")
SIGN_ERROR_DAY = ("alpha", "alpha_F", "2024-02-17")
METHOD_COLORS = {"m7": COLORS["grey"], "m8": COLORS["blue"], "m9": COLORS["orange"]}
GROUPS = [("beta", "Beta sure"), ("alpha", "Alpha"), ("combined", "Both")]
HOURS = np.arange(96) / 4.0
NOTE_BOX = dict(boxstyle="round,pad=0.3", fc="#EBEBEB", ec="none")
LABEL_BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)


# ----------------------------------------------------------------------------- shared pieces

def day(settings: Settings, cohort: str, station: str, date: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(y, s, truth)`` of one site-day, 96 slots each, from the dataset."""
    df = load_cohort(settings, cohort)
    g = df[(df["station"] == station) & (df["date"] == date)].sort_values("slot")
    return g["y"].to_numpy(float), g["s"].to_numpy(float), g["truth"].to_numpy(bool)


def evidence_surface(y: np.ndarray, s: np.ndarray, phi: float) -> tuple[np.ndarray, Any]:
    """The evidence matrix ``r[start, end]`` (steps 2 to 6) and the winner (step 7) of one day."""
    admissible = windows.admissible(y, s)
    start_ok, end_ok = edges.valid_edges(y, s)
    admissible &= start_ok[:, None] & end_ok[None, :]
    kept, flipped = stories.demand_if_kept(y, s), stories.demand_if_flipped(y, s)
    before, after = bridge.anchors(kept)
    rss_u, rss_c, length = misfit.residual_matrices(kept, flipped, before, after)
    r = evidence.evidence_matrix(rss_u, rss_c, length, phi, admissible)
    best, _ = winner.rank(r)
    return r, best


def hour_axis(axis: Any) -> None:
    axis.set_xlim(0, 24)
    axis.set_xticks(range(0, 25, 6))
    axis.axhline(0, color=COLORS["dark_blue"], lw=0.6)


def span(mask: np.ndarray) -> tuple[float, float]:
    idx = np.flatnonzero(mask)
    return idx[0] / 4.0, (idx[-1] + 1) / 4.0


# ----------------------------------------------------------------------------- fig01

def fig01_sign_error(settings: Settings) -> Any:
    """Expected and observed readings of one day whose export was stored as an import."""
    apply_paper_style()
    y, _, truth = day(settings, *SIGN_ERROR_DAY)
    expected = np.where(truth, -y, y)          # recorded = |true|, so the true value is −y where labelled
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.4), sharex=True, sharey=True)
    lo, hi = span(truth)
    panels = ((top, expected, "a", "Expected reading (correct sign)"),
              (bottom, y, "b", "Observed reading (wrong positive sign)"))
    for axis, series, letter, title in panels:
        axis.axvspan(lo, hi, color=COLORS["orange"], alpha=0.18, lw=0, label="reverse power flow")
        axis.plot(HOURS, series, color=COLORS["dark_blue"], lw=1.1, label="net load")
        hour_axis(axis)
        axis.set_ylabel("Net load (MW)")
        panel_label(axis, letter, title)
        style_axis(axis)
    bottom.set_xlabel("Hour of day")
    legend_below(figure, top, ncol=2)
    return figure


# ----------------------------------------------------------------------------- fig02

def fig02_two_questions(settings: Settings) -> Any:
    """The site-day question and the interval question on one reviewed Beta day."""
    apply_paper_style()
    y, s, truth = day(settings, *WORKED_DAY)
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.5), sharex=True, sharey=True)
    for axis in (top, bottom):
        axis.plot(HOURS, y, color=COLORS["dark_blue"], lw=1.1, label="recorded net load")
        axis.plot(HOURS, s, color=COLORS["orange"], lw=0.9, label="solar estimate")
        hour_axis(axis)
        axis.set_ylabel("MW")
        style_axis(axis)
    panel_label(top, "a", "Site-day: does this day carry a wrong sign?")
    top.text(0.98, 0.93, "one answer per day", transform=top.transAxes, ha="right", va="top", fontsize=7.5,
             bbox=NOTE_BOX)
    panel_label(bottom, "b", "Intervals: which slots should be flipped?")
    lo, hi = span(truth)
    first, last = int(np.flatnonzero(truth)[0]), int(np.flatnonzero(truth)[-1])
    bottom.axvspan(lo, hi, color=COLORS["red"], alpha=0.14, lw=0, label="reviewers' span")
    bottom.plot(HOURS[truth], np.full(int(truth.sum()), -0.5), "|", color=COLORS["red"], ms=6, mew=1.0)
    bottom.text(0.98, 0.93, f"slots {first} to {last}", transform=bottom.transAxes, ha="right", va="top",
                fontsize=7.5, bbox=NOTE_BOX)
    bottom.set_xlabel("Hour of day")
    bottom.set_ylim(-1.2, max(float(np.nanmax(y)), float(np.nanmax(s))) * 1.25)
    legend_below(figure, bottom, ncol=3)
    return figure


# ----------------------------------------------------------------------------- fig03

def draw_energy_metrics(axis: Any) -> None:
    """A synthetic recorded trace with the reference span and a shifted proposed span: the three energies."""
    t = np.linspace(6, 20, 561)
    true = 6 - 11 * np.exp(-((t - 12.5) / 2.8) ** 2)
    recorded = np.abs(true)
    reference = true < 0
    proposed = (t >= 11.4) & (t <= 16.4)
    axis.plot(t, recorded, color=COLORS["dark_blue"], lw=1.2, label="recorded net load")
    axis.fill_between(t, 0, recorded, where=reference & ~proposed, color=COLORS["red"], alpha=0.35, lw=0,
                      label="reference only (missed)")
    axis.fill_between(t, 0, recorded, where=proposed & ~reference, color=COLORS["grey"], alpha=0.45, lw=0,
                      label="proposed only (unsupported)")
    axis.fill_between(t, 0, recorded, where=proposed & reference, color=COLORS["orange"], alpha=0.75, lw=0,
                      label="overlap")
    axis.axhline(0, color=COLORS["dark_blue"], lw=0.6)
    axis.set_xlim(6, 20)
    axis.set_xticks(range(6, 21, 2))
    axis.set_yticks([])
    axis.set_xlabel("Hour of day")
    axis.set_ylabel("|y| (MW)")
    style_axis(axis, grid_axis=None, y_continuous=False)


def draw_holdout_design(axis: Any) -> None:
    """One fold of eighteen: Beta C held out, the other Beta stations fit, Alpha scored only."""
    stations = [("alpha", c) for c in "ABCDEFGHIJ"] + [("beta", c) for c in "ABCDEFGH"]
    held = ("beta", "C")
    for i, (cohort, letter) in enumerate(stations):
        if (cohort, letter) == held:
            colour = COLORS["orange"]
        else:
            colour = COLORS["dark_blue"] if cohort == "beta" else COLORS["grey"]
        axis.add_patch(plt.Rectangle((i, 0.5), 0.9, 0.9, color=colour, lw=0))
        axis.text(i + 0.45, 0.95, letter, ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    axis.text(4.95, 0.15, "Alpha: scored only, never fitted", ha="center", va="center", fontsize=7,
              color=COLORS["grey"])
    axis.text(13.95, 0.15, "Beta: one held out, seven fit", ha="center", va="center", fontsize=7,
              color=COLORS["dark_blue"])
    axis.text(0, 1.75, "one fold of eighteen; every station is held out once", fontsize=7,
              color=COLORS["dark_blue"])
    axis.set_xlim(-0.2, 18.2)
    axis.set_ylim(-0.15, 2.1)
    axis.axis("off")


def fig03_metrics_design(settings: Settings) -> Any:
    """(a) Energy IoU and energy precision on one day; (b) the station-held-out design."""
    apply_paper_style()
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.9), gridspec_kw={"height_ratios": [3, 2]})
    draw_energy_metrics(top)
    panel_label(top, "a", "Energy IoU and energy precision on one day")
    legend_below(top, top, ncol=2)
    draw_holdout_design(bottom)
    panel_label(bottom, "b", "Station-held-out evaluation, fitted on Beta only")
    return figure


# ----------------------------------------------------------------------------- fig04

def draw_two_stories(axis: Any, y: np.ndarray, s: np.ndarray, truth: np.ndarray, best: Any) -> None:
    kept = stories.demand_if_kept(y, s)
    flipped = np.where(best.mask(96), stories.demand_if_flipped(y, s), kept)
    axis.plot(HOURS, y, color="black", lw=1.1, label="recorded net load $y$")
    axis.plot(HOURS, s, color=COLORS["orange"], lw=0.9, label="solar estimate $s$")
    axis.plot(HOURS, kept, color=COLORS["grey"], lw=0.9, ls="--", label="demand, sign kept ($s+y$)")
    axis.plot(HOURS, flipped, color=COLORS["blue"], lw=1.1, label="demand, sign flipped in $W^*$")
    if not best.is_null:
        axis.axvspan(best.start / 4, (best.end + 1) / 4, color=COLORS["blue"], alpha=0.10, lw=0,
                     label="best window $W^*$")
    lo, hi = span(truth)
    axis.plot([lo, hi], [-0.9, -0.9], color=COLORS["red"], lw=4, solid_capstyle="butt", label="reviewers' span")
    hour_axis(axis)
    axis.set_ylim(-2.0, float(np.nanmax(kept)) * 1.08)
    axis.set_ylabel("MW")
    axis.set_xlabel("Hour of day")
    style_axis(axis)


def draw_evidence_surface(figure: Any, axis: Any, r: np.ndarray, best: Any) -> None:
    surface = np.where(np.isfinite(r), r, np.nan)
    extent = (windows.SCAN_START / 4, windows.SCAN_END / 4, windows.SCAN_START / 4, windows.SCAN_END / 4)
    image = axis.imshow(np.ma.masked_invalid(surface), origin="lower", cmap="viridis", aspect="auto", extent=extent)
    if not best.is_null:
        label = f"$W^*$: {best.start / 4:.2f} to {(best.end + 1) / 4:.2f} h, $r^*$ = {best.evidence:.1f}"
        axis.plot((best.end + 0.5) / 4, (best.start + 0.5) / 4, marker="o", ms=7, mfc="none", mec=COLORS["red"],
                  mew=1.4, ls="none", label=label)
    axis.set_xlabel("Window end (hour of day)")
    axis.set_ylabel("Window start (hour)")
    axis.legend(loc="upper left", fontsize=7)          # the region above the diagonal is empty
    bar = figure.colorbar(image, ax=axis, shrink=0.9, pad=0.02)
    bar.set_label("evidence $r(W)$")
    bar.ax.tick_params(labelsize=7)
    style_axis(axis, grid_axis=None, y_continuous=False)


def draw_decision_bands(axis: Any, c: float, day_evidence: float) -> None:
    """The release calibration curve p(r*) with the three outcome bands and the worked day marked."""
    r = np.concatenate([-np.logspace(1.5, -2, 120), [0.0], np.logspace(-2, 3, 200)])
    p = RELEASE_CALIBRATION.probability(r)
    axis.axhspan(0, 1 - c, color=COLORS["light_grey"], alpha=0.35, lw=0)
    axis.axhspan(1 - c, c, color=COLORS["orange"], alpha=0.18, lw=0)
    axis.axhspan(c, 1, color=COLORS["blue"], alpha=0.16, lw=0)
    axis.plot(r, p, color=COLORS["dark_blue"], lw=1.2, label="$p$ under the release calibration")
    p_day = float(RELEASE_CALIBRATION.probability(day_evidence))
    axis.plot([day_evidence], [p_day], marker="o", ms=5, color=COLORS["red"], ls="none",
              label=f"worked day: $r^*$ = {day_evidence:.1f}, $p$ = {p_day:.3f}")
    notes = ((c + 0.13, f"AUTO_CORRECT ($p \\geq {c:.1f}$)"), (0.5, "UNCERTAIN: review"),
             ((1 - c) / 2, f"AUTO_KEEP ($p \\leq {1 - c:.1f}$)"))
    for value, text in notes:
        axis.text(-25, value, text, fontsize=7, va="center", ha="left", color=COLORS["dark_blue"])
    axis.set_xscale("symlog", linthresh=1.0)
    axis.set_xlim(-30, 1000)
    axis.set_ylim(0, 1)
    axis.set_yticks([0, 1 - c, 0.5, c, 1])
    axis.set_xlabel("evidence of the best window $r^*$ (symmetric log scale)")
    axis.set_ylabel("probability $p$")
    legend_below(axis, axis, ncol=1, pad=0.22)
    style_axis(axis, grid_axis=None, y_continuous=False)


def fig04_method(settings: Settings) -> Any:
    """The method on the worked day: (a) two stories, (b) evidence surface, (c) calibration and decision."""
    apply_paper_style()
    cohort, station, date = WORKED_DAY
    y, s, truth = day(settings, cohort, station, date)
    phi = float(results.scores(settings, cohort)["sigma_floor"].iloc[0])
    r, best = evidence_surface(y, s, phi)
    figure, (top, middle, bottom) = plt.subplots(3, 1, figsize=(COLUMN_WIDTH, 7.8),
                                                 gridspec_kw={"height_ratios": [3, 3, 2.4]})
    draw_two_stories(top, y, s, truth, best)
    panel_label(top, "a", "Two stories of demand on the worked day")
    legend_below(top, top, ncol=2)
    draw_evidence_surface(figure, middle, r, best)
    panel_label(middle, "b", "Evidence of every admissible window")
    draw_decision_bands(bottom, float(settings["m9"]["c"]), best.evidence)
    panel_label(bottom, "c", "From evidence to probability to decision")
    return figure


# ----------------------------------------------------------------------------- fig05

HEADLINE = [("energy_iou", "Energy IoU"), ("energy_precision", "Energy precision"),
            ("day_f1", "Site-day F1"), ("day_precision", "Site-day precision")]


def fig05_headline(settings: Settings) -> Any:
    """Four headline metrics, one panel each; groups Beta sure, Alpha, Both; one bar per method."""
    apply_paper_style()
    pooled = results.metric(settings, "pooled")
    gate = float(settings["gate"]["energy_precision_min"])
    figure, axes = plt.subplots(2, 2, figsize=(COLUMN_WIDTH, 3.3), sharey=True)
    x = np.arange(len(GROUPS))
    width = 0.26
    for axis, (key, label) in zip(axes.ravel(), HEADLINE, strict=True):
        for k, method in enumerate(("m7", "m8", "m9")):
            values = [float(pooled[(pooled["method"] == method) & (pooled["group"] == g)][key].iloc[0])
                      for g, _ in GROUPS]
            axis.bar(x + (k - 1) * width, values, width, color=METHOD_COLORS[method], label=METHOD_LABELS[method],
                     lw=0)
        if key == "energy_precision":
            axis.axhline(gate, color=COLORS["red"], lw=0.8, ls="--", label=f"precision gate ({gate:.2f})")
        axis.set_xticks(x, [name for _, name in GROUPS])
        axis.set_ylim(0, 1.05)
        axis.set_yticks([0, 0.5, 1.0])
        axis.set_title(label, fontsize=8.5)
        style_axis(axis, y_continuous=False)
    legend_below(figure, axes, ncol=4)
    return figure


# ----------------------------------------------------------------------------- fig06

def fig06_per_station(settings: Settings) -> Any:
    """Energy IoU and energy precision per held-out station: Beta with three methods, Alpha with M9 only.

    Stations without a wrong-sign day (Alpha A, D, H and Beta C) have no defined Energy IoU and are left out.
    """
    apply_paper_style()
    stations = results.metric(settings, "stations")
    stations = stations[stations["n_rpf"] > 0]
    figure, axes = plt.subplots(2, 2, figsize=(COLUMN_WIDTH, 3.5), sharey=True,
                                gridspec_kw={"width_ratios": [3, 2]})
    width = 0.27
    beta = stations[stations["cohort"] == "beta"]
    names = sorted(beta["station"].unique())
    x = np.arange(len(names))
    alpha = stations[(stations["cohort"] == "alpha") & (stations["method"] == "m9")].sort_values("station")
    xa = np.arange(len(alpha))
    for row, (key, label) in enumerate((("energy_iou", "Energy IoU"), ("energy_precision", "Energy precision"))):
        for k, method in enumerate(("m7", "m8", "m9")):
            values = [float(beta[(beta["method"] == method) & (beta["station"] == st)][key].iloc[0]) for st in names]
            axes[row, 0].bar(x + (k - 1) * width, values, width, color=METHOD_COLORS[method],
                             label=METHOD_LABELS[method], lw=0)
        axes[row, 0].set_xticks(x, [n.split("_")[1] for n in names])
        axes[row, 0].set_ylabel(label)
        axes[row, 1].bar(xa, alpha[key].fillna(0.0).to_numpy(float), 0.6, color=METHOD_COLORS["m9"], lw=0)
        axes[row, 1].set_xticks(xa, [n.split("_")[1] for n in alpha["station"]])
        for axis in axes[row]:
            axis.set_ylim(0, 1.05)
            axis.set_yticks([0, 0.5, 1.0])
            style_axis(axis, y_continuous=False)
    panel_label(axes[0, 0], "a", "Beta stations, sure days")
    panel_label(axes[0, 1], "b", "Alpha, M9 only")
    axes[1, 0].set_xlabel("Beta station")
    axes[1, 1].set_xlabel("Alpha station")
    legend_below(figure, axes[0, 0], ncol=3)
    return figure


# ----------------------------------------------------------------------------- fig07

def draw_precision_frontier(axis: Any, front: pd.DataFrame, targets: pd.DataFrame, c_release: float) -> None:
    c = front["c_correct"].to_numpy(float)
    axis.plot(c, front["energy_precision"], color=COLORS["orange"], lw=1.4, label="energy precision")
    axis.plot(c, front["day_precision"], color=COLORS["dark_blue"], lw=1.2, label="site-day precision")
    axis.plot(c, front["day_recall"], color=COLORS["grey"], lw=1.2, ls="--", label="site-day recall")
    axis.axvline(c_release, color=COLORS["red"], lw=0.8, ls=":")
    axis.text(c_release + 0.006, 0.44, f"release $c$ = {c_release:.2f}", rotation=90, fontsize=7, va="bottom",
              color=COLORS["red"])
    for _, m in targets.iterrows():
        axis.plot([m["c_correct_mean"]], [m["target"]], marker="D", ms=4, color=COLORS["orange"],
                  mec=COLORS["dark_blue"], mew=0.6, ls="none")
        left_of_line = c_release - 0.05 < m["c_correct_mean"] < c_release      # keep clear of the release line
        axis.annotate(f"{m['target']:.3f}", (m["c_correct_mean"], m["target"]), textcoords="offset points",
                      xytext=(-2 if left_of_line else 4, -10), ha="right" if left_of_line else "left", fontsize=6.5,
                      color=COLORS["dark_blue"], bbox=LABEL_BOX)
    axis.set_xlim(0.5, 1.0)
    axis.set_ylim(0.4, 1.02)
    axis.set_xlabel("correction threshold $c$ (AUTO_CORRECT when $p \\geq c$)")
    axis.set_ylabel("held-out score, Beta sure")
    style_axis(axis)


def draw_review_days(axis: Any, coverage: pd.DataFrame, targets: pd.DataFrame, c_release: float) -> None:
    """Days sent to review against c; the release point and the three precision targets are labelled."""
    data = coverage[coverage["cohort"] == "beta"].sort_values("c")
    axis.plot(data["c"], data["review_days"], marker="o", ms=3.2, lw=1.3, color=COLORS["dark_blue"])
    # The curve hugs the bottom until c = 0.85, so the upper-left of the panel is empty: the four labels are
    # stacked there and joined to their points by thin leader lines.
    labelled = []
    release = data[np.isclose(data["c"], c_release)]
    if len(release):
        days = int(release["review_days"].iloc[0])
        axis.plot([c_release], [days], marker="o", ms=5, color=COLORS["red"], ls="none")
        labelled.append((f"release $c$ = {c_release:.2f}: {days} d", (c_release, float(days)), COLORS["red"]))
    for _, m in targets.iterrows():
        axis.plot([m["c_correct_mean"]], [m["review_days"]], marker="D", ms=4.5, color=COLORS["orange"],
                  mec=COLORS["dark_blue"], mew=0.6, ls="none")
        labelled.append((f"P ≥ {m['target']:.3f}: {int(m['review_days'])} d",
                         (float(m["c_correct_mean"]), float(m["review_days"])), COLORS["dark_blue"]))
    labelled.sort(key=lambda item: item[1][0])                 # left-most point at the bottom of the stack
    for k, (text, point, colour) in enumerate(labelled):
        axis.annotate(text, point, xytext=(0.10, 0.42 + 0.16 * k), textcoords="axes fraction", fontsize=7,
                      ha="left", va="center", color=colour,
                      arrowprops=dict(arrowstyle="-", lw=0.6, color=COLORS["grey"], shrinkA=2, shrinkB=3,
                                      relpos=(0.0 if point[0] < 0.55 else 0.5, 0.0)))
    axis.set_xlim(0.48, 1.0)
    axis.set_ylabel("days sent to review")
    style_axis(axis)


def draw_remaining_errors(axis: Any, coverage: pd.DataFrame, c_release: float) -> None:
    """False corrections and missed error days among the automatic decisions, stacked, against c."""
    data = coverage[coverage["cohort"] == "beta"].sort_values("c")
    c = data["c"].to_numpy(float)
    axis.bar(c, data["auto_fp"], width=0.03, color=COLORS["red"], alpha=0.75, lw=0, label="false corrections")
    axis.bar(c, data["auto_fn"], width=0.03, bottom=data["auto_fp"], color=COLORS["orange"], alpha=0.85, lw=0,
             label="missed error days (kept)")
    axis.axvline(c_release, color=COLORS["red"], lw=0.8, ls=":")
    axis.set_xlim(0.48, 1.0)
    axis.set_xlabel("correction threshold $c$")
    axis.set_ylabel("errors among\nautomatic decisions")
    style_axis(axis)


def fig07_review_tradeoff(settings: Settings) -> Any:
    """(a) precision and recall against c with the targets marked; (b) days sent to review; (c) remaining errors."""
    apply_paper_style()
    c_release = float(settings["m9"]["c"])
    front = results.operating_point(settings, "frontier")
    front = front[front["cohort"] == "beta"]
    targets = results.operating_point(settings, "targets")
    rule = settings["operating_points"]["recommended_rule"]
    targets = targets[(targets["cohort"] == "beta") & (targets["applies"] == "both") & (targets["rule"] == rule)
                      & targets["target"].isin([0.9, 0.925, 0.95])].sort_values("target")
    coverage = results.metric(settings, "coverage")
    figure, (top, middle, bottom) = plt.subplots(3, 1, figsize=(COLUMN_WIDTH, 6.9),
                                                 gridspec_kw={"height_ratios": [3, 2.6, 2]})
    draw_precision_frontier(top, front, targets, c_release)
    panel_label(top, "a", "Precision and recall of automatic corrections")
    legend_below(top, top, ncol=3)
    draw_review_days(middle, coverage, targets, c_release)
    panel_label(middle, "b", "Days sent to review, Beta sure")
    draw_remaining_errors(bottom, coverage, c_release)
    panel_label(bottom, "c", "Errors left among the automatic decisions")
    legend_below(bottom, bottom, ncol=2)
    return figure


# ----------------------------------------------------------------------------- fig08

def draw_example_week(axis: Any, series: pd.DataFrame, test_start: str, test_end: str) -> pd.Timestamp:
    in_month = series["timestamp"].between(_utc(test_start), _utc(test_end, end_of_day=True), inclusive="both")
    month = series[in_month].copy()
    naive = month["timestamp"].dt.tz_localize(None)
    month["week_start"] = naive.dt.to_period("W-SUN").dt.start_time
    month["manual_change"] = (month["manually_corrected_MW"] - month["raw_uncorrected_MW"]).abs()
    week_start = month.groupby("week_start")["manual_change"].sum().idxmax()
    week = month[naive.between(week_start, week_start + pd.Timedelta(days=7), inclusive="left")]
    axis.plot(week["timestamp"], week["raw_uncorrected_MW"], color=COLORS["grey"], lw=1.0, label="raw")
    axis.plot(week["timestamp"], week["m9_corrected_MW"], color=COLORS["orange"], lw=1.3, label="M9 corrected")
    axis.plot(week["timestamp"], week["manually_corrected_MW"], color=COLORS["dark_blue"], lw=0.9, ls="--",
              label="manual reference")
    axis.axhline(0, color=COLORS["dark_blue"], lw=0.6)
    axis.set_ylabel("Net load (MW)")
    axis.xaxis.set_major_locator(mdates.DayLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    style_axis(axis)
    return week_start


def draw_forecast_rmse(axis: Any, metrics: pd.DataFrame) -> None:
    colours = [COLORS["dark_blue"], COLORS["orange"], COLORS["grey"]]
    labels = {"raw_uncorrected": "raw", "m9_corrected": "M9 corrected", "manually_corrected": "manual"}
    x, width = np.arange(len(CONDITIONS)), 0.26
    for k, (model, colour) in enumerate(zip(MODELS, colours, strict=True)):
        values = metrics[metrics["model"] == model].set_index("data_condition").loc[CONDITIONS, "rmse_MW"]
        bars = axis.bar(x + (k - 1) * width, values, width, color=colour, label=MODEL_LABELS[model], lw=0)
        axis.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)
    axis.set_xticks(x, [labels[c] for c in CONDITIONS])
    axis.set_ylabel("7-day-ahead RMSE (MW)")
    axis.set_ylim(0, float(metrics["rmse_MW"].max()) * 1.2)
    style_axis(axis)


def fig08_forecast_impact(settings: Settings) -> Any:
    """Gamma: the highest-impact week of the test month and the seven-day-ahead RMSE per model (double column)."""
    apply_paper_style()
    g = settings["gamma"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(DOUBLE_WIDTH, 2.7), gridspec_kw={"width_ratios": [5, 3]})
    figure.get_layout_engine().set(wspace=0.14)         # the two panels read as one without this gap
    week_start = draw_example_week(left, results.gamma_series(settings), g["forecast_test_start"],
                                   g["forecast_test_end"])
    panel_label(left, "a", f"Highest-impact week of the test month, from {week_start:%d %b %Y}")
    legend_below(left, left, ncol=3)
    draw_forecast_rmse(right, results.gamma_table(settings, "gamma_forecast_metrics"))
    panel_label(right, "b", "Forecast error by training data")
    legend_below(right, right, ncol=3)
    return figure


# ----------------------------------------------------------------------------- tables

DATASET_FACTS = {
    "alpha": ("Placement check",
              "Absolute value of correctly signed readings; every negative interval is a label",
              "Scored only; never fitted"),
    "beta": ("Real-world calibration and validation",
             "Manual review of every site-day: span to flip and confidence, sure or unsure",
             "Per-fold fitting on the other Beta stations; sure days are the headline, unsure days a sensitivity"),
}


def table01_datasets(settings: Settings) -> pd.DataFrame:
    """The three datasets by their role in the paper: size, labels and what each may be used for."""
    rows = []
    for cohort in ("alpha", "beta"):
        idx = results.index(settings, cohort)
        complete = idx[idx["complete"]]
        sure, unsure = complete[complete["headline"]], complete[~complete["headline"]]
        role, labels, use = DATASET_FACTS[cohort]
        if cohort == "alpha":
            days = f"{len(complete):,}"
            wrong = f"{int(sure['rpf'].sum()):,} (by construction)"
        else:
            days = f"{len(complete):,} ({len(sure):,} sure, {len(unsure):,} unsure)"
            wrong = f"{int(sure['rpf'].sum()):,} sure, {int(unsure['rpf'].sum()):,} unsure"
        rows.append({"Dataset": cohort.capitalize(), "Role": role, "Stations": str(int(idx["station"].nunique())),
                     "Complete site-days": days, "Wrong-sign days": wrong, "Label source": labels,
                     "Allowed use": use})
    gamma = pd.read_parquet(settings.dataset("gamma"))
    per_day = gamma.groupby("date").agg(label=("label_day", "any"), confidence=("confidence", "first"))
    n_sure = int((per_day["confidence"] == "sure").sum())
    n_unsure = int((per_day["confidence"] == "unsure").sum())
    rows.append({"Dataset": "Gamma", "Role": "Forecast-impact case study",
                 "Stations": "1 (Beta " + settings["gamma"]["station"].split("_")[1] + ")",
                 "Complete site-days": f"{len(per_day):,} ({n_sure} sure, {n_unsure} unsure)",
                 "Wrong-sign days": f"{int(per_day['label'].sum()):,} labelled", "Label source": "As Beta",
                 "Allowed use": "Forecast targets on raw, M9-corrected and manually corrected history"})
    return pd.DataFrame(rows)


def table02_headline(settings: Settings) -> pd.DataFrame:
    """Headline metrics with station-bootstrap intervals, Beta sure first; the energy-precision gate on Beta sure.

    Plain ASCII throughout (intervals as "(low, high)"), so the CSV opens cleanly in a spreadsheet.
    """
    pooled = results.metric(settings, "pooled")
    boot = results.metric(settings, "bootstrap")
    gate = float(settings["gate"]["energy_precision_min"])
    rows = []
    for group, group_label in GROUPS:
        for method in ("m9", "m8", "m7"):
            r = pooled[(pooled["method"] == method) & (pooled["group"] == group)].iloc[0]
            row = {"Group": group_label, "Method": METHOD_LABELS[method], "Site-days": int(r["n_days"])}
            for key, label in HEADLINE:
                ci = boot[(boot["method"] == method) & (boot["group"] == group) & (boot["metric"] == key)]
                interval = f" ({ci['ci_low'].iloc[0]:.3f}, {ci['ci_high'].iloc[0]:.3f})" if len(ci) else ""
                row[f"{label} (95% CI)"] = f"{float(r[key]):.3f}{interval}"
                if key == "energy_precision":
                    passed = "pass" if float(r[key]) >= gate else "fail"
                    row[f"Gate {gate:.2f}"] = passed if group == "beta" else "n/a"
            rows.append(row)
    return pd.DataFrame(rows)


PER_STATION = [("energy_iou", "Energy IoU"), ("energy_precision", "Energy precision")]


def table03_per_station(settings: Settings) -> pd.DataFrame:
    """Energy IoU and energy precision per held-out station and method: the numbers behind Fig. 6, with the
    M7 and M8 Alpha columns the figure leaves out. n/a marks an undefined value: a station without a
    wrong-sign day has no Energy IoU, and a method that flipped no energy there has no precision."""
    stations = results.metric(settings, "stations")
    rows = []
    for cohort, cohort_label in (("beta", "Beta sure"), ("alpha", "Alpha")):
        block = stations[stations["cohort"] == cohort]
        for station in sorted(block["station"].unique()):
            per = block[block["station"] == station].set_index("method")
            n_rpf = int(per["n_rpf"].iloc[0])
            row = {"Group": cohort_label, "Station": station.split("_")[1], "Site-days": int(per["n_days"].iloc[0]),
                   "Wrong-sign days": n_rpf}
            for key, label in PER_STATION:
                for method in ("m7", "m8", "m9"):
                    value = float(per.loc[method, key]) if method in per.index else float("nan")
                    defined = n_rpf > 0 and np.isfinite(value)
                    row[f"{label} {METHOD_LABELS[method]}"] = f"{value:.3f}" if defined else "n/a"
            rows.append(row)
    return pd.DataFrame(rows)


FIGURES: dict[str, FigureFn] = {
    "fig01_sign_error": fig01_sign_error,
    "fig02_two_questions": fig02_two_questions,
    "fig03_metrics_design": fig03_metrics_design,
    "fig04_method": fig04_method,
    "fig05_headline": fig05_headline,
    "fig06_per_station": fig06_per_station,
    "fig07_review_tradeoff": fig07_review_tradeoff,
    "fig08_forecast_impact": fig08_forecast_impact,
}
TABLES: dict[str, TableFn] = {
    "table01_datasets": table01_datasets,
    "table02_headline": table02_headline,
    "table03_per_station": table03_per_station,
}

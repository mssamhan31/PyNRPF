"""The manuscript's figures and tables, numbered as in the paper.

Inputs:  the reference run under ``results/`` (through ``results``), the Alpha and Beta datasets
         for the day-level figures, and the release calibration of ``pynrpf``.
Outputs: ``FIGURES`` (``fig01`` to ``fig12``, each ``settings -> Figure``) and ``TABLES``
         (``table01``, ``table02``, each ``settings -> DataFrame``); ``figures.write_all`` and
         ``tables.write_all`` save them under ``results/paper/``.
Key steps: every figure is drawn at its final size for one column of a two-column IEEE
         Transactions page (3.5 in wide) with 8 pt type; every figure is one panel without a title
         (the caption in the manuscript names it); legends sit below the axes so no text overlaps a curve.

    fig01  expected versus observed reading under a wrong sign (Alpha day, no station identifier)
    fig02  the two tasks on one reviewed Beta day: site-day detection and interval correction
    fig03  Energy IoU on one synthetic day: overlap over union
    fig04  the evaluation design: one of the eighteen station-held-out folds, fitted on Beta only
    fig05  the worked day under the keep and flip hypotheses of demand
    fig06  the evidence of every admissible window of the worked day
    fig07  the calibrated probability and the three decision bands at the control c
    fig08  precision and recall of automatic corrections against c (Beta-sure)
    fig09  days sent to review against c
    fig10  errors left among the automatic decisions against c
    fig11  Gamma: the highest-impact week of the test month
    fig12  Gamma: one-week-ahead forecast RMSE by model and training data
    table01  the three datasets by role
    table02  headline metrics with bootstrap intervals, Beta-sure first

The headline bar chart, the per-station figure and the per-station table the draft no longer
prints live in the supplementary registries of ``figures`` and ``tables``.
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
from .data import load_cohort
from .gamma import CONDITIONS, MODEL_LABELS, MODELS, _utc
from .style import COLORS, COLUMN_WIDTH, apply_paper_style, legend_below, panel_label, style_axis

FigureFn = Callable[[Settings], Any]
TableFn = Callable[[Settings], pd.DataFrame]

# The worked day of the method section (the deck and the documentation walk through it) and
# the Alpha day of the introduction figure.
WORKED_DAY = ("beta", "beta_B", "2024-09-20")
SIGN_ERROR_DAY = ("alpha", "alpha_F", "2024-02-17")
METHOD_COLORS = {"m7": COLORS["grey"], "m8": COLORS["blue"], "m9": COLORS["orange"]}
GROUPS = [("beta", "Beta-sure"), ("alpha", "Alpha"), ("combined", "Both")]
# The paper's method names; the pipeline and the reference run keep M7, M8 and M9.
PAPER_METHOD_NAMES = {"m7": "m7_dtr", "m8": "m8_xgb", "m9": "m9_pbm"}
PAPER_MODEL_LABELS = {**MODEL_LABELS, "seasonal_naive": "Seasonal naïve"}
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


# ----------------------------------------------------------------------------- fig03 and the drawing of fig04

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


def fig03_energy_iou(settings: Settings) -> Any:
    """Energy IoU on one synthetic day: reference only, proposed only and their overlap under the recorded trace."""
    apply_paper_style()
    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.3))
    draw_energy_metrics(axis)
    legend_below(figure, axis, ncol=2)
    return figure


# ----------------------------------------------------------------------------- fig04

def fig04_holdout_design(settings: Settings) -> Any:
    """One of the eighteen station-held-out folds: Beta C held out, the other Beta stations fit, Alpha scored."""
    apply_paper_style()
    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH, 1.1))
    draw_holdout_design(axis)
    return figure


# ----------------------------------------------------------------------------- fig05 to fig07: the worked day

def draw_two_hypotheses(axis: Any, y: np.ndarray, s: np.ndarray, truth: np.ndarray, best: Any) -> None:
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
    style_axis(axis, grid_axis=None, y_continuous=False)


def worked_day_evidence(settings: Settings) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """``(y, s, truth, r, best)`` of the worked day: its readings, the evidence surface and the winning window."""
    cohort, station, date = WORKED_DAY
    y, s, truth = day(settings, cohort, station, date)
    phi = float(results.scores(settings, cohort)["sigma_floor"].iloc[0])
    r, best = evidence_surface(y, s, phi)
    return y, s, truth, r, best


def fig05_two_hypotheses(settings: Settings) -> Any:
    """The worked day under the keep and flip hypotheses of demand, with the best window and the reviewers' span."""
    apply_paper_style()
    y, s, truth, _, best = worked_day_evidence(settings)
    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH, 3.0))
    draw_two_hypotheses(axis, y, s, truth, best)
    legend_below(figure, axis, ncol=2)
    return figure


def fig06_evidence_surface(settings: Settings) -> Any:
    """The evidence r(W) of every admissible window of the worked day, with the best window circled."""
    apply_paper_style()
    _, _, _, r, best = worked_day_evidence(settings)
    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.7))
    draw_evidence_surface(figure, axis, r, best)
    return figure


def fig07_decision_bands(settings: Settings) -> Any:
    """The release calibration p(r*) with the three decision bands at the control c, and the worked day marked."""
    apply_paper_style()
    _, _, _, _, best = worked_day_evidence(settings)
    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.8))
    draw_decision_bands(axis, float(settings["m9"]["c"]), best.evidence)
    legend_below(figure, axis, ncol=1)
    return figure


# ----------------------------------------------------------------------------- fig08 to fig10: the review trade-off

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
    axis.set_xlabel("control threshold $c$ (AUTO_CORRECT when $p \\geq c$)")
    axis.set_ylabel("held-out score, Beta-sure")
    style_axis(axis)


def draw_review_days(axis: Any, coverage: pd.DataFrame, targets: pd.DataFrame, c_release: float) -> None:
    """Days sent to review against c; the release point and the three precision targets are labelled."""
    data = coverage[coverage["cohort"] == "beta"].sort_values("c")
    axis.plot(data["c"], data["review_days"], marker="o", ms=3.2, lw=1.3, color=COLORS["dark_blue"])
    # The curve hugs the bottom until c = 0.85, so the labels sit in the empty space above it. Each label is
    # right-aligned just left of its own point, one row higher than the label of the point before it, and its
    # leader drops from the label's right end; the left-most point (c near 0.5) is labelled from the top-left
    # corner instead. No leader then passes through another label's text.
    labelled = []
    release = data[np.isclose(data["c"], c_release)]
    if len(release):
        days = int(release["review_days"].iloc[0])
        axis.plot([c_release], [days], marker="o", ms=5, color=COLORS["red"], ls="none")
        labelled.append((f"release $c$ = {c_release:.2f}: {days} d", (c_release, float(days)), COLORS["red"]))
    for _, m in targets.iterrows():
        axis.plot([m["c_correct_mean"]], [m["review_days"]], marker="D", ms=4.5, color=COLORS["orange"],
                  mec=COLORS["dark_blue"], mew=0.6, ls="none")
        labelled.append((f"$P_E \\geq$ {m['target']:.3f}: {int(m['review_days'])} d",
                         (float(m["c_correct_mean"]), float(m["review_days"])), COLORS["dark_blue"]))
    labelled.sort(key=lambda item: item[1][0])
    top = float(data["review_days"].max())
    leader = dict(arrowstyle="-", lw=0.6, color=COLORS["grey"], shrinkA=1, shrinkB=3)
    first_text, first_point, first_colour = labelled[0]
    axis.annotate(first_text, first_point, xytext=(first_point[0] - 0.015, 0.80 * top), textcoords="data",
                  fontsize=7, ha="left", va="bottom", color=first_colour, arrowprops={**leader, "relpos": (0.0, 0.0)})
    for k, (text, point, colour) in enumerate(labelled[1:]):
        axis.annotate(text, point, xytext=(point[0] - 0.018, (0.25 + 0.18 * k) * top), textcoords="data",
                      fontsize=7, ha="right", va="bottom", color=colour, arrowprops={**leader, "relpos": (1.0, 0.0)})
    axis.set_xlim(0.48, 1.0)
    axis.set_xlabel("control threshold $c$")
    axis.set_ylabel("days sent to review, Beta-sure")
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
    axis.set_xlabel("control threshold $c$")
    axis.set_ylabel("errors among\nautomatic decisions")
    style_axis(axis)


def review_inputs(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """``(frontier, targets, coverage, c_release)`` on Beta-sure days: the inputs of the three review figures."""
    c_release = float(settings["m9"]["c"])
    front = results.operating_point(settings, "frontier")
    front = front[front["cohort"] == "beta"]
    targets = results.operating_point(settings, "targets")
    rule = settings["operating_points"]["recommended_rule"]
    targets = targets[(targets["cohort"] == "beta") & (targets["applies"] == "both") & (targets["rule"] == rule)
                      & targets["target"].isin([0.9, 0.925, 0.95])].sort_values("target")
    return front, targets, results.metric(settings, "coverage"), c_release


def fig08_precision_frontier(settings: Settings) -> Any:
    """Precision and recall of automatic corrections against c, with the precision targets and the release c."""
    apply_paper_style()
    front, targets, _, c_release = review_inputs(settings)
    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.8))
    draw_precision_frontier(axis, front, targets, c_release)
    legend_below(figure, axis, ncol=3)
    return figure


def fig09_review_days(settings: Settings) -> Any:
    """Days sent to review against c, with the release c and the three precision targets labelled."""
    apply_paper_style()
    _, targets, coverage, c_release = review_inputs(settings)
    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    draw_review_days(axis, coverage, targets, c_release)
    return figure


def fig10_remaining_errors(settings: Settings) -> Any:
    """False corrections and missed error days left among the automatic decisions, against c."""
    apply_paper_style()
    _, _, coverage, c_release = review_inputs(settings)
    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.4))
    draw_remaining_errors(axis, coverage, c_release)
    legend_below(figure, axis, ncol=2)
    return figure


# ----------------------------------------------------------------------------- fig11 and fig12: forecast impact

def draw_example_week(axis: Any, series: pd.DataFrame, test_start: str, test_end: str) -> pd.Timestamp:
    in_month = series["timestamp"].between(_utc(test_start), _utc(test_end, end_of_day=True), inclusive="both")
    month = series[in_month].copy()
    naive = month["timestamp"].dt.tz_localize(None)
    month["week_start"] = naive.dt.to_period("W-SUN").dt.start_time
    month["manual_change"] = (month["manually_corrected_MW"] - month["raw_uncorrected_MW"]).abs()
    week_start = month.groupby("week_start")["manual_change"].sum().idxmax()
    week = month[naive.between(week_start, week_start + pd.Timedelta(days=7), inclusive="left")]
    axis.plot(week["timestamp"], week["raw_uncorrected_MW"], color=COLORS["grey"], lw=1.0, label="raw")
    axis.plot(week["timestamp"], week["m9_corrected_MW"], color=COLORS["orange"], lw=1.3, label="m9_pbm-corrected")
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
    labels = {"raw_uncorrected": "raw", "m9_corrected": "m9_pbm-corrected", "manually_corrected": "manual"}
    x, width = np.arange(len(CONDITIONS)), 0.26
    for k, (model, colour) in enumerate(zip(MODELS, colours, strict=True)):
        values = metrics[metrics["model"] == model].set_index("data_condition").loc[CONDITIONS, "rmse_MW"]
        bars = axis.bar(x + (k - 1) * width, values, width, color=colour, label=PAPER_MODEL_LABELS[model], lw=0)
        axis.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)
    axis.set_xticks(x, [labels[c] for c in CONDITIONS])
    axis.set_ylabel("One-week-ahead RMSE (MW)")
    axis.set_ylim(0, float(metrics["rmse_MW"].max()) * 1.2)
    style_axis(axis)


def fig11_forecast_week(settings: Settings) -> Any:
    """Gamma: the highest-impact week of the test month, raw, m9_pbm-corrected and manually corrected."""
    apply_paper_style()
    g = settings["gamma"]
    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.6))
    week_start = draw_example_week(axis, results.gamma_series(settings), g["forecast_test_start"],
                                   g["forecast_test_end"])
    # One tick per day at column width: the day of the month, with the month and year in the axis label.
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    axis.set_xlabel(f"Day of {week_start:%B %Y}")
    legend_below(figure, axis, ncol=3)
    return figure


def fig12_forecast_rmse(settings: Settings) -> Any:
    """Gamma: one-week-ahead forecast RMSE by model and by the history the model was trained on."""
    apply_paper_style()
    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.6))
    draw_forecast_rmse(axis, results.gamma_table(settings, "gamma_forecast_metrics"))
    legend_below(figure, axis, ncol=3)
    return figure


# ----------------------------------------------------------------------------- tables

DATASET_FACTS = {
    "alpha": ("Placement check",
              "Absolute value of correctly signed readings; every negative interval is a label",
              "Scored only; never fitted"),
    "beta": ("Real-world calibration and validation",
             "Manual review of every site-day: span to flip and confidence, sure or unsure",
             "Per-fold fitting on the other Beta stations; results reported on sure days"),
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
                 "Allowed use": "Forecast targets on raw, m9_pbm-corrected and manually corrected history"})
    return pd.DataFrame(rows)


HEADLINE = [("energy_iou", "Energy IoU"), ("energy_precision", "Energy precision"),
            ("day_f1", "Site-day F1"), ("day_precision", "Site-day precision")]


def table02_headline(settings: Settings) -> pd.DataFrame:
    """Headline metrics with station-bootstrap intervals, Beta-sure first: group, method and four metrics.

    Plain ASCII throughout (intervals as "(low, high)"), so the CSV opens cleanly in a spreadsheet.
    """
    pooled = results.metric(settings, "pooled")
    boot = results.metric(settings, "bootstrap")
    rows = []
    for group, group_label in GROUPS:
        for method in ("m9", "m8", "m7"):
            r = pooled[(pooled["method"] == method) & (pooled["group"] == group)].iloc[0]
            row = {"Group": group_label, "Method": PAPER_METHOD_NAMES[method]}
            for key, label in HEADLINE:
                ci = boot[(boot["method"] == method) & (boot["group"] == group) & (boot["metric"] == key)]
                interval = f" ({ci['ci_low'].iloc[0]:.3f}, {ci['ci_high'].iloc[0]:.3f})" if len(ci) else ""
                row[f"{label} (95% CI)"] = f"{float(r[key]):.3f}{interval}"
            rows.append(row)
    return pd.DataFrame(rows)


FIGURES: dict[str, FigureFn] = {
    "fig01_sign_error": fig01_sign_error,
    "fig02_two_questions": fig02_two_questions,
    "fig03_energy_iou": fig03_energy_iou,
    "fig04_holdout_design": fig04_holdout_design,
    "fig05_two_hypotheses": fig05_two_hypotheses,
    "fig06_evidence_surface": fig06_evidence_surface,
    "fig07_decision_bands": fig07_decision_bands,
    "fig08_precision_frontier": fig08_precision_frontier,
    "fig09_review_days": fig09_review_days,
    "fig10_remaining_errors": fig10_remaining_errors,
    "fig11_forecast_week": fig11_forecast_week,
    "fig12_forecast_rmse": fig12_forecast_rmse,
}
TABLES: dict[str, TableFn] = {
    "table01_datasets": table01_datasets,
    "table02_headline": table02_headline,
}

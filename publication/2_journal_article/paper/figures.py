"""Paper figures: the manuscript registry from ``manuscript`` and the supplementary figures drawn here.

Inputs:  the files of the finished stages under ``results/`` (read through ``results``)
         and, for the day-level figures, the Beta dataset.
Outputs: matplotlib figures; ``write_all`` saves every manuscript entry under
         ``results/paper/figures/`` and every supplementary entry under
         ``results/paper/supplementary/figures/``, each as PNG and PDF.
Key steps: every registry entry is ``name -> function(settings) -> Figure``. A notebook
         cell calls ``show("fig05_headline", settings)``; adding a figure later is one
         function and one registry line.

    manuscript (``manuscript.FIGURES``): fig01 to fig08, drawn at journal column width
    supplementary (``SUPPLEMENTARY``):
        supp_fig01  the worked day with the evidence surface, at report size
        supp_fig02  M9 calibration on held-out stations (reliability diagram)
        supp_fig03  M9 coverage scores against c (Beta)
        supp_fig04  sample site-days, M7    supp_fig05  M8    supp_fig06  M9   (27 panels each, Beta sure)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

from pynrpf.m9 import AUTO_CORRECT, AUTO_KEEP, UNCERTAIN, bridge, edges, evidence, misfit, stories, windows, winner

from . import manuscript, results
from .config import Settings
from .data import KEY, METHOD_LABELS, load_cohort
from .metrics import methods_present
from .style import COHORT_LABELS, COLORS, apply_journal_style, save_figure, style_axis

FigureFn = Callable[[Settings], Any]
# ----------------------------------------------------------------------------- the worked day

def day_arrays(settings: Settings, cohort: str, station: str, date: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y, s, truth) of one site-day from the dataset, 96 slots each."""
    df = load_cohort(settings, cohort)
    g = df[(df["station"] == station) & (df["date"] == date)].sort_values("slot")
    return g["y"].to_numpy(float), g["s"].to_numpy(float), g["truth"].to_numpy(bool)


def evidence_surface(y: np.ndarray, s: np.ndarray, phi: float) -> tuple[np.ndarray, Any, Any]:
    """Steps 2 to 7 of M9 as ``pynrpf.m9.score_siteday`` composes them, keeping the evidence matrix.

    Returns the matrix ``r[start, end]`` over the 48 scan slots (``-inf`` where a window
    is inadmissible), the winner and the runner-up.
    """
    admissible = windows.admissible(y, s)
    start_ok, end_ok = edges.valid_edges(y, s)
    admissible &= start_ok[:, None] & end_ok[None, :]
    kept, flipped = stories.demand_if_kept(y, s), stories.demand_if_flipped(y, s)
    before, after = bridge.anchors(kept)
    rss_u, rss_c, length = misfit.residual_matrices(kept, flipped, before, after)
    r = evidence.evidence_matrix(rss_u, rss_c, length, phi, admissible)
    best, runner = winner.rank(r)
    return r, best, runner


def supp_worked_day(settings: Settings) -> Any:
    """The manuscript's worked day at report size: the two stories (left), the evidence surface (right)."""
    apply_journal_style()
    cohort, station, date = manuscript.WORKED_DAY
    y, s, truth = day_arrays(settings, cohort, station, date)
    phi = float(results.scores(settings, "beta")["sigma_floor"].iloc[0])
    r, best, _ = evidence_surface(y, s, phi)
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.4, 4.6), gridspec_kw={"width_ratios": [3, 2]})
    t = np.arange(96) / 4
    kept = stories.demand_if_kept(y, s)
    corrected = np.where(best.mask(96), stories.demand_if_flipped(y, s), kept)
    left.plot(t, y, color="black", lw=1.3, label="recorded net load y")
    left.plot(t, s, color=COLORS["orange"], lw=1.0, label="solar estimate s")
    left.plot(t, kept, color=COLORS["grey"], lw=1.0, ls="--", label="demand if the sign is kept, s + y")
    left.plot(t, corrected, color=COLORS["blue"], lw=1.3, label="demand if flipped inside the window")
    if not best.is_null:
        left.axvspan(best.start / 4, (best.end + 1) / 4, color=COLORS["blue"], alpha=0.12, label="best window")
    idx = np.flatnonzero(truth)
    if idx.size:
        # The reviewers' span as a bar along the bottom edge, so it stays legible when it coincides with the window.
        edge = blended_transform_factory(left.transData, left.transAxes)
        left.plot([idx[0] / 4, (idx[-1] + 1) / 4], [0.015, 0.015], color=COLORS["red"], lw=6, solid_capstyle="butt",
                  transform=edge, label="reviewers' span")
    left.axhline(0, color="black", lw=0.5)
    left.set_xlim(0, 24)
    left.set_xticks(range(0, 25, 6))
    left.set_xlabel("Hour of day")
    left.set_ylabel("MW")
    left.set_title(f"{station}, {date}: the two stories of demand", fontsize=11)
    left.legend(fontsize=8, loc="upper left")
    style_axis(left)
    surface = np.where(np.isfinite(r), r, np.nan)
    extent = (windows.SCAN_START / 4, windows.SCAN_END / 4, windows.SCAN_START / 4, windows.SCAN_END / 4)
    image = right.imshow(np.ma.masked_invalid(surface), origin="lower", cmap="viridis", aspect="auto", extent=extent)
    if not best.is_null:
        label = f"best window {best.start / 4:.2f}–{(best.end + 1) / 4:.2f} h, r* = {best.evidence:.1f}"
        right.plot((best.end + 0.5) / 4, (best.start + 0.5) / 4, marker="o", ms=9, mfc="none", mec=COLORS["red"], mew=2,
                   ls="none", label=label)
    right.set_xlabel("Window end (hour of day)")
    right.set_ylabel("Window start (hour of day)")
    right.set_title("Evidence r(W) for every admissible window", fontsize=11)
    right.legend(fontsize=8, loc="upper left")   # the region above the diagonal (start > end) is empty
    figure.colorbar(image, ax=right, shrink=0.85, label="evidence r(W)")
    style_axis(right, grid_axis=None, y_continuous=False)
    return figure


# ----------------------------------------------------------------------------- headline and stations

# ----------------------------------------------------------------------------- review burden and frontier

def supp_coverage_scores(settings: Settings) -> Any:
    """Energy IoU, energy precision and sure-day recall of the automatic decisions against c (Beta)."""
    apply_journal_style()
    coverage = results.metric(settings, "coverage")
    data = coverage[coverage["cohort"] == "beta"].sort_values("c")
    figure, axis = plt.subplots(figsize=(7.6, 4.2))
    for key, label, color, marker in [("energy_iou", "Energy IoU", COLORS["dark_blue"], "o"),
                                      ("energy_precision", "Energy precision", COLORS["orange"], "s"),
                                      ("sure_day_recall", "Sure-day recall", COLORS["grey"], "^"),
                                      ("auto_decided_share", "Auto-decided share", COLORS["light_grey"], "d")]:
        axis.plot(data["c"], data[key], color=color, marker=marker, linewidth=2, label=label)
    axis.set_ylim(0, 1.03)
    axis.set_xlabel("Confidence control c")
    axis.set_ylabel("Score")
    axis.set_title(f"M9 operating points on {COHORT_LABELS['beta']}")
    axis.legend(ncol=2, loc="lower left", fontsize=9)
    style_axis(axis)
    return figure


# ----------------------------------------------------------------------------- Gamma

# ----------------------------------------------------------------------------- calibration

def supp_calibration(settings: Settings, bins: int = 10) -> Any:
    """Reliability diagram of the M9 probability per cohort (equal-count bins)."""
    apply_journal_style()
    days = results.site_days(settings)
    t = days[(days["method"] == "m9") & days["headline"] & days["prob_day"].notna()]
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    for cohort, color in (("alpha", COLORS["grey"]), ("beta", COLORS["orange"])):
        g = t[t["cohort"] == cohort].sort_values("prob_day")
        chunks = np.array_split(np.arange(len(g)), bins)
        xs = [g["prob_day"].iloc[i].mean() for i in chunks if len(i)]
        ys = [g["rpf"].iloc[i].mean() for i in chunks if len(i)]
        axis.plot(xs, ys, marker="o", color=color, label=f"{COHORT_LABELS[cohort]} (n = {len(g):,})")
    axis.plot([0, 1], [0, 1], color=COLORS["light_grey"], lw=0.8, ls="--")
    axis.set_xlabel("Calibrated probability of RPF (held-out)")
    axis.set_ylabel("Observed share of RPF days")
    axis.set_title("M9 calibration on held-out stations")
    axis.legend(loc="lower right")
    style_axis(axis)
    return figure


# ----------------------------------------------------------------------------- sample site-days

def _runs(idx: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of sorted slot indices as (start, end) pairs."""
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _panel(axis: Any, day: pd.DataFrame, row: pd.Series) -> None:
    """One site-day: recorded net load, solar, underlying load kept and corrected, windows."""
    t = np.arange(len(day)) / 4
    y = day["y"].to_numpy(float)
    s = day["s"].to_numpy(float)
    truth = day["truth"].to_numpy(bool)
    candidate = day["pred_interval"].to_numpy(bool)
    applied = row["outcome"] == AUTO_CORRECT
    u0 = s + y
    axis.plot(t, y, color="black", lw=1.2, label="recorded net load")
    axis.plot(t, s, color=COLORS["orange"], lw=0.9, label="solar estimate")
    axis.plot(t, u0, color="grey", lw=0.9, ls="--", label="underlying load, sign kept")
    if candidate.any():
        uc = np.where(candidate, s - y, u0)
        axis.plot(t, uc, color=COLORS["blue"], lw=1.1, label="underlying load, corrected")
        idx = np.flatnonzero(candidate)
        # Shade each contiguous run so non-contiguous M7/M8 flags are shown faithfully.
        for start, end in _runs(idx):
            label = ("window applied" if applied else "window proposed, not applied") if start == idx[0] else None
            axis.axvspan(start / 4, (end + 1) / 4, color=COLORS["blue"] if applied else COLORS["purple"], alpha=0.15,
                         label=label)
    if truth.any():
        first = np.flatnonzero(truth)[0]
        for start, end in _runs(np.flatnonzero(truth)):
            axis.axvspan(start / 4, (end + 1) / 4, color=COLORS["red"], alpha=0.10,
                         label="reference RPF" if start == first else None)
    axis.axhline(0, color="k", lw=0.5)
    axis.set_xlim(0, 24)
    axis.set_xticks(range(0, 25, 6))
    axis.set_ylabel("MW", fontsize=8)
    axis.tick_params(labelsize=7)
    score = f"p = {row['prob_day']:.2f}" if np.isfinite(row["prob_day"]) else "deterministic"
    axis.set_title(f"{row['station']}  {row['date']}   {row['outcome']}\n"
                   f"{score}   reference {row['required_mwh']:.1f} MWh   proposed {row['candidate_mwh']:.1f} MWh",
                   fontsize=8)


def select_samples(site_days: pd.DataFrame, method: str, settings: Settings) -> pd.DataFrame:
    """The 27 Beta 'sure' panels of one method: TP 9, FN 6, FP 6, TN 6, by a seeded rule.

    TP: three largest proposed MWh, three nearest the decision threshold (or three
    smallest proposed MWh for M7, which has no score), three at random. FN: six largest
    reference MWh (M9: three UNCERTAIN then three AUTO_KEEP). FP: six largest proposed
    MWh. TN: three nearest the threshold (M7: random) and three at random.
    """
    fig = settings["figures"]
    rng = np.random.default_rng(int(fig["sample_seed"]))
    n = fig["samples"]
    t = site_days[(site_days["method"] == method) & (site_days["cohort"] == "beta") & site_days["headline"]].copy()
    threshold = float(settings["m9"]["c"]) if method == "m9" else float(settings["m8"]["thresholds"]["xgb1_day"])
    t["dist"] = (t["prob_day"] - threshold).abs() if method != "m7" else t["candidate_mwh"]
    corrected = t["outcome"] == AUTO_CORRECT
    tp, fn = t[corrected & (t["rpf"] == 1)], t[~corrected & (t["rpf"] == 1)]
    fp, tn = t[corrected & (t["rpf"] == 0)], t[~corrected & (t["rpf"] == 0)]

    def random(df: pd.DataFrame, k: int) -> pd.DataFrame:
        return df.iloc[rng.choice(len(df), size=min(k, len(df)), replace=False)] if len(df) else df

    k = n["TP"] // 3
    tp_a = tp.nlargest(k, "proposed_mwh")
    tp_b = tp.drop(tp_a.index).nsmallest(k, "dist")
    tp_sel = pd.concat([tp_a, tp_b, random(tp.drop(tp_a.index).drop(tp_b.index), n["TP"] - len(tp_a) - len(tp_b))])
    if method == "m9":
        fn_sel = pd.concat([fn[fn["outcome"] == UNCERTAIN].nlargest(n["FN"] // 2, "required_mwh"),
                            fn[fn["outcome"] == AUTO_KEEP].nlargest(n["FN"] - n["FN"] // 2, "required_mwh")])
    else:
        fn_sel = fn.nlargest(n["FN"], "required_mwh")
    fp_sel = fp.nlargest(n["FP"], "proposed_mwh")
    tn_a = tn.nsmallest(n["TN"] // 2, "dist") if method != "m7" else random(tn, n["TN"] // 2)
    tn_sel = pd.concat([tn_a, random(tn.drop(tn_a.index), n["TN"] - len(tn_a))])
    return pd.concat([tp_sel.assign(kind="TP"), fn_sel.assign(kind="FN"), fp_sel.assign(kind="FP"),
                      tn_sel.assign(kind="TN")])


def sample_index(settings: Settings) -> pd.DataFrame:
    """The selected sample days of every method with their decisions and energies."""
    site_days = results.site_days(settings)
    parts = [select_samples(site_days, method, settings).assign(method=method) for method in methods_present(site_days)]
    columns = ["method", "kind", "cohort", "station", "date", "outcome", "prob_day", "pred_start", "pred_end",
               "true_start", "true_end", "required_mwh", "candidate_mwh", "proposed_mwh", "correct_mwh"]
    return pd.concat(parts, ignore_index=True)[columns]


def plot_samples(selected: pd.DataFrame, intervals: pd.DataFrame, method: str) -> Any:
    """One figure of every selected panel of a method, three per row, grouped by kind (TP, FN, FP, TN)."""
    apply_journal_style()
    titles = {"TP": "corrected, reference agrees", "FN": "not corrected, reference says RPF",
              "FP": "corrected, reference says no RPF", "TN": "kept, reference agrees"}
    by_day = {k: g for k, g in intervals.groupby(KEY, sort=False)}
    ncol = 3
    nrow = int(sum(np.ceil(len(rows) / ncol) for _, rows in selected.groupby("kind", sort=False)))
    figure, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 2.7 * nrow), squeeze=False)
    row_at = 0
    for kind, rows in selected.groupby("kind", sort=False):
        axes[row_at, 0].annotate(f"{kind}: {titles[kind]}", (0.0, 1.32), xycoords="axes fraction", fontsize=10,
                                 fontweight="bold", color=COLORS["dark_blue"])
        for k, (_, row) in enumerate(rows.iterrows()):
            _panel(axes[row_at + k // ncol, k % ncol], by_day[(row["cohort"], row["station"], row["date"])], row)
        for k in range(len(rows), int(np.ceil(len(rows) / ncol)) * ncol):
            axes[row_at + k // ncol, k % ncol].axis("off")
        row_at += int(np.ceil(len(rows) / ncol))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=6, fontsize=8)
    figure.suptitle(f"{METHOD_LABELS[method]} sample site-days (Beta 'sure')", fontsize=12)
    figure.tight_layout(rect=(0, 0.02, 1, 0.985), h_pad=2.4)
    return figure


def _samples_figure(method: str) -> FigureFn:
    def draw(settings: Settings) -> Any:
        site_days = results.site_days(settings)
        selected = select_samples(site_days, method, settings)
        beta = load_cohort(settings, "beta")[KEY + ["slot", "y", "s", "truth"]]
        table = results.intervals(settings, method)
        table = table[table["cohort"] == "beta"].merge(selected[KEY], on=KEY, how="inner")
        joined = table.merge(beta, on=KEY + ["slot"], how="left", validate="one_to_one")
        return plot_samples(selected, joined, method)
    draw.__doc__ = f"The 27 sample site-days of {METHOD_LABELS[method]} on Beta 'sure' days."
    return draw


# ----------------------------------------------------------------------------- the registry

# Raster resolution per entry; the sample grids are large and are read on screen, not in print.


# ----------------------------------------------------------------------------- the registries

FIGURES: dict[str, FigureFn] = dict(manuscript.FIGURES)
SUPPLEMENTARY: dict[str, FigureFn] = {
    "supp_fig01_worked_day": supp_worked_day,
    "supp_fig02_calibration": supp_calibration,
    "supp_fig03_coverage_scores": supp_coverage_scores,
    "supp_fig04_samples_m7": _samples_figure("m7"),
    "supp_fig05_samples_m8": _samples_figure("m8"),
    "supp_fig06_samples_m9": _samples_figure("m9"),
}
# Raster resolution: manuscript figures at print resolution; the sample grids are read on screen.
DPI = {name: 600 for name in FIGURES}
DPI.update({"supp_fig04_samples_m7": 110, "supp_fig05_samples_m8": 110, "supp_fig06_samples_m9": 110})


def show(name: str, settings: Settings) -> Any:
    """Draw one registry entry (manuscript or supplementary) and return the figure."""
    return {**FIGURES, **SUPPLEMENTARY}[name](settings)


def write_all(settings: Settings, names: list[str] | None = None) -> list[Path]:
    """Save every registry entry as PNG and PDF; manuscript under ``paper/figures``, the rest under
    ``paper/supplementary/figures``. Returns the paths written."""
    written: list[Path] = []
    for registry, kind in ((FIGURES, "figures"), (SUPPLEMENTARY, "supplementary/figures")):
        folder = results.paper_dir(settings, kind)
        for name, draw in registry.items():
            if names and name not in names:
                continue
            written += save_figure(draw(settings), folder / f"{name}.png", formats=("png", "pdf"),
                                   dpi=DPI.get(name, 200))
    return written

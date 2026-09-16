"""Paper figures: headline comparison, per-station bars, sample site-days, review burden.

Inputs:  the pooled, per-station and coverage tables; the site-day table; the interval
         tables joined with the recorded net load, solar and labels (for sample panels).
Outputs: PNG files under outputs/01_final_evaluation/08_figures/.
Key steps: one visual contract (the journal palette and axis style carried over from
         the archived pipeline); 27 sample panels per method on Beta 'sure' days, chosen
         by a seeded, reproducible rule; the manual-review-burden figure reads the
         confidence-versus-coverage table (M9 only, since M7 and M8 have no review band).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from .common import AUTO_CORRECT, AUTO_KEEP, METHOD_LABELS, METHODS, UNCERTAIN
from .config import Settings
from .data import KEY

COLORS = {"orange": "#eb932c", "dark_blue": "#22303d", "grey": "#2F4D67", "light_grey": "#5C7D99",
          "light_white": "#ebe3e3", "red": "#B64A4A"}
METHOD_COLORS = {"m7": COLORS["grey"], "m8": COLORS["light_grey"], "m9": COLORS["orange"]}
GROUP_LABELS = {"combined": "Combined", "alpha": "Alpha", "beta": "Beta sure"}


# ----------------------------------------------------------------------------- style

def apply_journal_style() -> None:
    """Article-wide typography, frame and grid defaults."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": COLORS["dark_blue"],
        "axes.labelcolor": COLORS["dark_blue"], "axes.titlecolor": COLORS["dark_blue"], "axes.axisbelow": True,
        "axes.grid": False, "axes.linewidth": 1.0, "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
        "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11, "xtick.color": COLORS["dark_blue"],
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


def _save(figure: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


# ----------------------------------------------------------------------------- headline

def plot_headline(pooled: pd.DataFrame, path: Path) -> Path:
    """Four headline metrics, grouped bars by evaluation group, one bar per method."""
    apply_journal_style()
    metrics = [("energy_iou", "Reference Energy IoU"), ("energy_precision", "Reference energy precision"),
               ("day_f1", "Site-day F1"), ("day_precision", "Site-day precision")]
    groups = list(GROUP_LABELS)
    figure, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    x = np.arange(len(groups))
    width = 0.26
    methods = [m for m in METHODS if m in set(pooled["method"])]
    for axis, (key, label) in zip(axes.ravel(), metrics, strict=True):
        for k, method in enumerate(methods):
            vals = [float(pooled[(pooled["method"] == method) & (pooled["group"] == g)][key].iloc[0]) for g in groups]
            bars = axis.bar(x + (k - 1) * width, vals, width, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=1)
        axis.set_xticks(x, [GROUP_LABELS[g] for g in groups])
        axis.set_ylim(0, 1.12)
        axis.set_title(label)
        style_axis(axis)
    axes[0, 0].legend(ncol=3, loc="upper left")
    figure.suptitle("Station-held-out evaluation: headline metrics on headline-confidence site-days", fontsize=12)
    return _save(figure, path)


def plot_stations(stations: pd.DataFrame, metric: str, label: str, path: Path) -> Path:
    """One metric per station, one bar per method, Alpha on the left and Beta on the right."""
    apply_journal_style()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [10, 8]})
    width = 0.26
    methods = [m for m in METHODS if m in set(stations["method"])]
    for axis, cohort in zip(axes, ("alpha", "beta"), strict=True):
        t = stations[stations["cohort"] == cohort]
        names = sorted(t["station"].unique())
        x = np.arange(len(names))
        for k, method in enumerate(methods):
            vals = [float(t[(t["method"] == method) & (t["station"] == s)][metric].iloc[0]) for s in names]
            axis.bar(x + (k - 1) * width, vals, width, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axis.set_xticks(x, [n.replace(f"{cohort}_", "") for n in names])
        axis.set_xlabel(f"{cohort.capitalize()} station")
        axis.set_ylim(0, 1.05)
        style_axis(axis)
    axes[0].set_ylabel(label)
    axes[0].legend(ncol=3, loc="lower left")
    figure.suptitle(f"{label} per held-out station", fontsize=12)
    return _save(figure, path)


# ----------------------------------------------------------------------------- review burden

def plot_review_burden(coverage: pd.DataFrame, cohort: str, path: Path) -> Path:
    """Manual-review burden and auto-accepted errors across the confidence control c.

    Left axis: days sent to review (UNCERTAIN). Right axis: errors among automatically
    decided days, stacked (auto FP = corrected non-RPF day, auto FN = kept RPF day).
    The x axis is the share of days decided automatically; each point is one value of c.
    """
    apply_journal_style()
    data = coverage[coverage["cohort"] == cohort].sort_values("auto_decided_share")
    x = 100 * data["auto_decided_share"].to_numpy(float)
    figure, left = plt.subplots(figsize=(8.4, 4.6))
    right = left.twinx()
    left.plot(x, data["review_days"], marker="o", linewidth=2.2, color=COLORS["dark_blue"], label="Days sent to manual review")
    width = max(0.8, 0.6 * np.min(np.diff(np.unique(x))) if len(np.unique(x)) > 1 else 1.0)
    right.bar(x, data["auto_fp"], width=width, color=COLORS["red"], alpha=0.8, label="Auto FP (corrected, not RPF)")
    right.bar(x, data["auto_fn"], width=width, bottom=data["auto_fp"], color=COLORS["orange"], alpha=0.85, label="Auto FN (kept, RPF)")
    # Alternate the label offsets so neighbouring points at high coverage stay legible.
    for k, (xi, c) in enumerate(zip(x, data["c"], strict=True)):
        left.annotate(f"c={c:.2f}", (xi, float(data.loc[data['c'] == c, 'review_days'].iloc[0])), textcoords="offset points",
                      xytext=(0, 7 if k % 2 == 0 else -13), ha="center", fontsize=7, color=COLORS["dark_blue"])
    cohort_label = "Beta sure" if cohort == "beta" else "Alpha"
    left.set_xlabel(f"Automatically decided {cohort_label} site-days (%)")
    left.set_ylabel("Days remaining for manual review", color=COLORS["dark_blue"])
    right.set_ylabel("Errors among automatically decided days")
    left.set_title(f"M9 manual-review burden and auto-accepted errors ({cohort_label})")
    h1, l1 = left.get_legend_handles_labels()
    h2, l2 = right.get_legend_handles_labels()
    left.legend(h1 + h2, l1 + l2, ncol=3, loc="upper center", fontsize=9)
    align_twin_y_axes(left, right)
    return _save(figure, path)


def plot_coverage_scores(coverage: pd.DataFrame, cohort: str, path: Path) -> Path:
    """Energy IoU, energy precision and sure-day recall of the automatic decisions against c."""
    apply_journal_style()
    data = coverage[coverage["cohort"] == cohort].sort_values("c")
    figure, axis = plt.subplots(figsize=(7.6, 4.2))
    for key, label, color, marker in [("energy_iou", "Energy IoU", COLORS["dark_blue"], "o"),
                                      ("energy_precision", "Energy precision", COLORS["orange"], "s"),
                                      ("sure_day_recall", "Sure-day recall", COLORS["grey"], "^"),
                                      ("auto_decided_share", "Auto-decided share", COLORS["light_grey"], "d")]:
        axis.plot(data["c"], data[key], color=color, marker=marker, linewidth=2, label=label)
    axis.set_ylim(0, 1.03)
    axis.set_xlabel("Confidence control c")
    axis.set_ylabel("Score")
    axis.set_title(f"M9 operating points on {'Beta sure' if cohort == 'beta' else 'Alpha'}")
    axis.legend(ncol=2, loc="lower left", fontsize=9)
    style_axis(axis)
    return _save(figure, path)


# ----------------------------------------------------------------------------- samples

def _panel(axis: Any, day: pd.DataFrame, row: pd.Series, method: str, c: float) -> None:
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
        axis.plot(t, uc, color="tab:blue", lw=1.1, label="underlying load, corrected")
        idx = np.flatnonzero(candidate)
        # Shade each contiguous run so non-contiguous M7/M8 flags are shown faithfully.
        for start, end in _runs(idx):
            axis.axvspan(start / 4, (end + 1) / 4, color="tab:blue" if applied else "tab:purple", alpha=0.15,
                         label=("window applied" if applied else "window proposed, not applied") if start == idx[0] else None)
    if truth.any():
        for start, end in _runs(np.flatnonzero(truth)):
            axis.axvspan(start / 4, (end + 1) / 4, color="tab:red", alpha=0.10,
                         label="reference RPF" if start == np.flatnonzero(truth)[0] else None)
    axis.axhline(0, color="k", lw=0.5)
    axis.set_xlim(0, 24)
    axis.set_xticks(range(0, 25, 6))
    axis.set_ylabel("MW", fontsize=8)
    axis.tick_params(labelsize=7)
    score = f"p = {row['prob_day']:.2f}" if np.isfinite(row["prob_day"]) else "deterministic"
    axis.set_title(f"{row['station']}  {row['date']}   {row['outcome']}\n{score}   reference {row['required_mwh']:.1f} MWh   "
                   f"proposed {row['candidate_mwh']:.1f} MWh", fontsize=8)


def _runs(idx: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of sorted slot indices as (start, end) pairs."""
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


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
    return pd.concat([tp_sel.assign(kind="TP"), fn_sel.assign(kind="FN"), fp_sel.assign(kind="FP"), tn_sel.assign(kind="TN")])


def plot_samples(selected: pd.DataFrame, intervals: pd.DataFrame, method: str, settings: Settings, folder: Path) -> list[Path]:
    """Four grids (TP, FN, FP, TN) of sample panels for one method; returns the paths."""
    apply_journal_style()
    titles = {"TP": "corrected, reference agrees", "FN": "not corrected, reference says RPF",
              "FP": "corrected, reference says no RPF", "TN": "kept, reference agrees"}
    c = float(settings["m9"]["c"])
    by_day = {k: g for k, g in intervals.groupby(KEY, sort=False)}
    paths = []
    for kind, rows in selected.groupby("kind", sort=False):
        ncol = 3
        nrow = max(1, int(np.ceil(len(rows) / ncol)))
        figure, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 2.9 * nrow), squeeze=False)
        for k, (_, row) in enumerate(rows.iterrows()):
            _panel(axes[k // ncol, k % ncol], by_day[(row["cohort"], row["station"], row["date"])], row, method, c)
        for k in range(len(rows), nrow * ncol):
            axes[k // ncol, k % ncol].axis("off")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=6, fontsize=8)
        figure.suptitle(f"{METHOD_LABELS[method]} — {kind}: {titles[kind]} (Beta 'sure' site-days)", fontsize=11)
        figure.tight_layout(rect=(0, 0.04, 1, 0.97))
        path = folder / f"samples_{method}_{kind}.png"
        figure.savefig(path, dpi=120)
        plt.close(figure)
        paths.append(path)
    return paths


def plot_calibration(m9_site_days: pd.DataFrame, path: Path, bins: int = 10) -> Path:
    """Reliability diagram of the M9 probability per cohort (equal-count bins)."""
    apply_journal_style()
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    t = m9_site_days[m9_site_days["headline"] & m9_site_days["prob_day"].notna()]
    for cohort, color in (("alpha", COLORS["grey"]), ("beta", COLORS["orange"])):
        g = t[t["cohort"] == cohort].sort_values("prob_day")
        chunks = np.array_split(np.arange(len(g)), bins)
        xs = [g["prob_day"].iloc[i].mean() for i in chunks if len(i)]
        ys = [g["rpf"].iloc[i].mean() for i in chunks if len(i)]
        axis.plot(xs, ys, marker="o", color=color, label=f"{'Beta sure' if cohort == 'beta' else 'Alpha'} (n = {len(g):,})")
    axis.plot([0, 1], [0, 1], color=COLORS["light_grey"], lw=0.8, ls="--")
    axis.set_xlabel("Calibrated probability of RPF (held-out)")
    axis.set_ylabel("Observed share of RPF days")
    axis.set_title("M9 calibration on held-out stations")
    axis.legend(loc="upper left")
    style_axis(axis)
    return _save(figure, path)


FigureFn = Callable[..., Path]

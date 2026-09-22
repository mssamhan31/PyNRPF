"""Operating-point study: precision targets, the two-control decision policy and its costs.

Inputs:  the M9 rows of the site-day table (held-out probability, candidate energies,
         labels), the fold manifest (calibration stations per fold), the settings block
         ``operating_points``.
Outputs: the frontier over c_correct, the (c_correct, c_keep) grid, the target table
         (threshold per fold, achieved held-out precision, recall, Energy IoU, false
         corrections per station-year) for both selection rules, the per-station table
         at chosen points, station-bootstrap intervals, and six figures.
Key steps: the policy has two controls, AUTO_CORRECT at p >= c_correct and AUTO_KEEP at
         p <= c_keep; the locked policy is c_keep = 1 - c_correct. For a precision target
         the threshold is chosen per fold on the calibration stations only, by rule L
         (expected precision from the calibrated probabilities, label-free) or rule E
         (observed precision on the calibration stations' labels), and applied once to
         the held-out station; held-out results are pooled. Nothing here alters the
         headline result at the locked c = 0.7.

Only the correction threshold moves precision, recall and the energy metrics; the keep
threshold moves only the review load and the number of RPF days kept silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import AUTO_CORRECT
from .config import Settings
from .figures import COLORS, _save, apply_journal_style, style_axis
from .folds import Fold

DAYS_PER_YEAR = 365.25
RULES = ("L", "E")
RULE_LABELS = {"L": "rule L: expected precision from p (label-free)", "E": "rule E: observed precision on calibration stations"}


# ----------------------------------------------------------------------------- one operating point

def _prepare(m9_days: pd.DataFrame) -> pd.DataFrame:
    """Headline days with a finite probability; the population every point is measured on."""
    t = m9_days[(m9_days["method"] == "m9") & m9_days["headline"] & m9_days["prob_day"].notna()].copy()
    t["has_window"] = t["pred_start"] >= 0
    return t


def evaluate_point(t: pd.DataFrame, c_correct: float, c_keep: float) -> dict[str, Any]:
    """All metrics of one (c_correct, c_keep) policy on a prepared table.

    Rates per station-year divide counts by the number of evaluated days over 365.25,
    so cohorts of different length are comparable.
    """
    p = t["prob_day"].to_numpy(float)
    rpf = t["rpf"].to_numpy().astype(bool)
    correct = (p >= c_correct) & t["has_window"].to_numpy()
    keep = ~correct & (p <= c_keep)
    review = ~correct & ~keep
    cand = t["candidate_mwh"].to_numpy(float)
    cand_ok = t["candidate_correct_mwh"].to_numpy(float)
    proposed, good, required = cand[correct].sum(), cand_ok[correct].sum(), t["required_mwh"].sum()
    tp, fp, fn_auto = int((correct & rpf).sum()), int((correct & ~rpf).sum()), int((keep & rpf).sum())
    station_years = len(t) / DAYS_PER_YEAR
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / rpf.sum() if rpf.sum() else np.nan
    return dict(
        c_correct=c_correct, c_keep=c_keep, n_days=len(t), n_rpf=int(rpf.sum()),
        auto_share=float((~review).mean()), review_days=int(review.sum()), review_share=float(review.mean()),
        review_yield=float(rpf[review].mean()) if review.any() else np.nan,
        corrected_days=int(correct.sum()), auto_fp=fp, auto_fn=fn_auto, tp=tp,
        day_precision=prec, day_recall=rec,
        day_f1=(2 * prec * rec / (prec + rec)) if (prec and rec and np.isfinite(prec + rec)) else 0.0,
        energy_precision=(good / proposed) if proposed else np.nan,
        energy_iou=(good / (proposed + required - good)) if (proposed + required - good) else np.nan,
        fp_per_station_year=fp / station_years, auto_fn_per_station_year=fn_auto / station_years,
        review_days_per_station_year=int(review.sum()) / station_years,
        expected_day_precision=float(p[correct].mean()) if correct.any() else np.nan,
        expected_fp=float((1 - p[correct]).sum()),
    )


def frontier(t: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Metrics against c_correct on the configured grid with the symmetric keep rule."""
    g = settings["operating_points"]["c_correct_grid"]
    grid = np.round(np.arange(g["start"], g["stop"] + 1e-9, g["step"]), 4)
    return pd.DataFrame([evaluate_point(t, float(c), float(1 - c)) for c in grid])


def heatmap_grid(t: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Review load and silently kept RPF days over (c_correct, c_keep)."""
    op = settings["operating_points"]
    rows = []
    for cc in op["heatmap_c_correct"]:
        for ck in op["heatmap_c_keep"] + [cc]:
            if ck <= cc:
                rows.append(dict(**evaluate_point(t, float(cc), float(ck)), no_review_band=bool(ck == cc)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- threshold selection

def _thresholds(settings: Settings) -> np.ndarray:
    g = settings["operating_points"]["c_correct_grid"]
    return np.round(np.arange(g["start"], g["stop"] + 1e-9, g["step"]), 4)


def precision_curve(cal: pd.DataFrame, thresholds: np.ndarray, rule: str) -> pd.DataFrame:
    """Day and energy precision of corrections at each threshold on a calibration set.

    Rule L: expected values from the calibrated probability (each corrected day is right
    with probability p; its candidate energy is right with the same probability).
    Rule E: observed values from the calibration stations' labels.
    Both curves are made monotone from the top (running minimum over higher thresholds)
    so a target is met at t and at every threshold above it.
    """
    p = cal["prob_day"].to_numpy(float)
    ok = cal["has_window"].to_numpy()
    cand = cal["candidate_mwh"].to_numpy(float)
    rows = []
    for t in thresholds:
        sel = (p >= t) & ok
        if not sel.any():
            rows.append(dict(threshold=t, day_precision=np.nan, energy_precision=np.nan, n_corrected=0))
            continue
        if rule == "L":
            day = float(p[sel].mean())
            energy = float((p[sel] * cand[sel]).sum() / cand[sel].sum()) if cand[sel].sum() else np.nan
        else:
            rpf = cal["rpf"].to_numpy().astype(bool)
            day = float(rpf[sel].mean())
            good = cal["candidate_correct_mwh"].to_numpy(float)
            energy = float(good[sel].sum() / cand[sel].sum()) if cand[sel].sum() else np.nan
        rows.append(dict(threshold=t, day_precision=day, energy_precision=energy, n_corrected=int(sel.sum())))
    curve = pd.DataFrame(rows)
    for col in ("day_precision", "energy_precision"):
        vals = curve[col].to_numpy(float)
        # Running minimum from the highest threshold downwards; NaN (nothing corrected) is treated as met.
        filled = np.where(np.isfinite(vals), vals, 1.0)
        curve[col + "_envelope"] = np.minimum.accumulate(filled[::-1])[::-1]
    return curve


def choose_threshold(curve: pd.DataFrame, target: float, column: str) -> float:
    """Smallest threshold whose monotone precision envelope reaches the target; the top of the grid if none."""
    met = curve[curve[column + "_envelope"] >= target]
    return float(met["threshold"].iloc[0]) if len(met) else float(curve["threshold"].iloc[-1])


def select_fold_threshold(t: pd.DataFrame, fold: Fold, target_energy: float | None, target_day: float | None,
                          rule: str, thresholds: np.ndarray) -> dict[str, Any]:
    """The operating threshold for one fold: chosen on its calibration stations only.

    Args:
        t: prepared M9 site-days to draw the calibration stations from. It must hold
            every station named by ``fold.m9_calibration``, which under ``beta_only``
            lie in the other cohort, so callers pass the full prepared table.
        fold: the fold; ``m9_calibration`` names the stations whose labels may be read.
        target_energy: energy-precision target in [0, 1], or None for no constraint.
        target_day: day-precision target in [0, 1], or None for no constraint.
        rule: "L" (expected precision from the calibrated probabilities) or "E"
            (observed precision on the calibration labels).
        thresholds: the c_correct grid.

    Returns:
        The fold's selection: the chosen ``c_correct``, the threshold each target
        demanded, which one binds, the number of calibration days and whether the
        target was attainable within the grid.
    """
    cal = t[t["station"].isin(fold.m9_calibration)]
    if fold.held_out in set(cal["station"]):
        raise ValueError(f"Fold {fold.fold_id}: held-out station inside the calibration set.")
    if cal.empty:
        raise ValueError(f"Fold {fold.fold_id}: none of its calibration stations {fold.m9_calibration} is present.")
    curve = precision_curve(cal, thresholds, rule)
    t_energy = choose_threshold(curve, target_energy, "energy_precision") if target_energy is not None else -np.inf
    t_day = choose_threshold(curve, target_day, "day_precision") if target_day is not None else -np.inf
    c_correct = max(t_energy, t_day)
    binding = "energy" if t_energy >= t_day else "day"
    if target_energy is not None and target_day is not None and abs(t_energy - t_day) < 1e-9:
        binding = "both"
    return dict(fold_id=fold.fold_id, cohort=fold.cohort, held_out=fold.held_out, rule=rule,
                target_energy=target_energy, target_day=target_day, c_correct=c_correct,
                threshold_energy=t_energy if np.isfinite(t_energy) else np.nan,
                threshold_day=t_day if np.isfinite(t_day) else np.nan, binding=binding,
                n_calibration_days=len(cal), attainable=bool(c_correct < thresholds[-1]))


def apply_selected(t: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    """Held-out decisions under per-fold thresholds; returns the table with outcome columns added."""
    out = t.merge(selections[["fold_id", "c_correct"]], on="fold_id", how="inner", validate="many_to_one")
    p = out["prob_day"].to_numpy(float)
    out["correct"] = (p >= out["c_correct"].to_numpy()) & out["has_window"].to_numpy()
    out["keep"] = ~out["correct"] & (p <= 1 - out["c_correct"].to_numpy())
    return out


def pooled_from_applied(applied: pd.DataFrame) -> dict[str, Any]:
    """Pooled held-out metrics of a per-fold policy (thresholds differ across folds)."""
    rpf = applied["rpf"].to_numpy().astype(bool)
    correct, keep = applied["correct"].to_numpy(), applied["keep"].to_numpy()
    review = ~correct & ~keep
    cand, good = applied["candidate_mwh"].to_numpy(float), applied["candidate_correct_mwh"].to_numpy(float)
    proposed, ok, required = cand[correct].sum(), good[correct].sum(), applied["required_mwh"].sum()
    tp, fp, fn_auto = int((correct & rpf).sum()), int((correct & ~rpf).sum()), int((keep & rpf).sum())
    station_years = len(applied) / DAYS_PER_YEAR
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / rpf.sum() if rpf.sum() else np.nan
    return dict(n_days=len(applied), corrected_days=int(correct.sum()), auto_fp=fp, auto_fn=fn_auto, tp=tp,
                day_precision=prec, day_recall=rec,
                energy_precision=(ok / proposed) if proposed else np.nan,
                energy_iou=(ok / (proposed + required - ok)) if (proposed + required - ok) else np.nan,
                fp_per_station_year=fp / station_years, review_days=int(review.sum()),
                review_days_per_station_year=int(review.sum()) / station_years, auto_share=float((~review).mean()))


def target_table(t: pd.DataFrame, folds: list[Fold], settings: Settings,
                 calibration_pool: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (target, applies-to, rule): the fold thresholds and the pooled held-out result.

    Args:
        t: prepared M9 site-days of the held-out cohort (every point is measured on it).
        folds: all folds; those of the cohort are used.
        settings: the evaluation settings.
        calibration_pool: prepared M9 site-days the thresholds are chosen from; defaults
            to ``t``. Under ``beta_only`` an Alpha fold calibrates on Beta stations, so
            the caller passes the table of both cohorts.

    Returns:
        (summary, selections). 'applies' is energy, day or both (same target on each).
    """
    op = settings["operating_points"]
    thresholds = _thresholds(settings)
    cohort = t["cohort"].iloc[0]
    fold_list = [f for f in folds if f.cohort == cohort]
    pool = t if calibration_pool is None else calibration_pool
    summaries, selections = [], []
    for target in op["targets"]:
        for applies in ("energy", "day", "both"):
            te = float(target) if applies in ("energy", "both") else None
            td = float(target) if applies in ("day", "both") else None
            for rule in RULES:
                sel = pd.DataFrame([select_fold_threshold(pool, f, te, td, rule, thresholds) for f in fold_list])
                sel["applies"] = applies
                sel["target"] = float(target)
                selections.append(sel)
                pooled = pooled_from_applied(apply_selected(t, sel))
                summaries.append(dict(cohort=cohort, target=float(target), applies=applies, rule=rule,
                                      c_correct_mean=float(sel["c_correct"].mean()), c_correct_min=float(sel["c_correct"].min()),
                                      c_correct_max=float(sel["c_correct"].max()), binding=sel["binding"].mode().iloc[0],
                                      all_attainable=bool(sel["attainable"].all()), **pooled))
    return pd.DataFrame(summaries), pd.concat(selections, ignore_index=True)


def bootstrap_targets(t: pd.DataFrame, selections: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Station-bootstrap intervals of achieved precision for each (target, applies, rule)."""
    bs = settings["metrics"]["bootstrap"]
    rng = np.random.default_rng(int(bs["seed"]))
    lo, hi = (float(v) for v in bs["percentiles"])
    draws = int(bs["draws"])
    rows = []
    for (target, applies, rule), sel in selections.groupby(["target", "applies", "rule"], sort=True):
        applied = apply_selected(t, sel)
        by_station = {s: g for s, g in applied.groupby("station")}
        stations = sorted(by_station)
        samples = {"energy_precision": [], "day_precision": [], "day_recall": []}
        for _ in range(draws):
            pick = pd.concat([by_station[s] for s in rng.choice(stations, size=len(stations), replace=True)], ignore_index=True)
            m = pooled_from_applied(pick)
            for k in samples:
                samples[k].append(m[k])
        point = pooled_from_applied(applied)
        for k, vals in samples.items():
            arr = np.asarray(vals, dtype=float)
            rows.append(dict(cohort=t["cohort"].iloc[0], target=target, applies=applies, rule=rule, metric=k, point=point[k],
                             ci_low=float(np.nanpercentile(arr, lo)), ci_high=float(np.nanpercentile(arr, hi))))
    return pd.DataFrame(rows)


def per_station_points(t: pd.DataFrame, folds: list[Fold], selections: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Per-station precision and recall at the configured operating points."""
    op = settings["operating_points"]
    c_locked = float(settings["m9"]["c"])
    rows = []
    for point in op["per_station_points"]:
        if point["rule"] == "locked":
            applied = t.assign(c_correct=c_locked)
            p = applied["prob_day"].to_numpy(float)
            applied["correct"] = (p >= c_locked) & applied["has_window"].to_numpy()
            applied["keep"] = ~applied["correct"] & (p <= 1 - c_locked)
        else:
            sel = selections[(selections["rule"] == point["rule"]) & (selections["applies"] == "both") & (selections["target"] == float(point["target"]))]
            applied = apply_selected(t, sel)
        for station, g in applied.groupby("station", sort=True):
            rows.append(dict(point=point["label"], cohort=t["cohort"].iloc[0], station=station, **pooled_from_applied(g)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures

def _cohort_label(cohort: str) -> str:
    return "Beta 'sure'" if cohort == "beta" else "Alpha"


def plot_frontier(front: pd.DataFrame, cohort: str, settings: Settings, marks: pd.DataFrame | None, path: Path) -> Path:
    """Fig A: precision and recall of automatic corrections against c_correct, targets band shaded."""
    apply_journal_style()
    figure, axis = plt_subplots((8.6, 4.8))
    c = front["c_correct"]
    axis.plot(c, front["energy_precision"], color=COLORS["orange"], lw=2.2, label="Reference energy precision")
    axis.plot(c, front["day_precision"], color=COLORS["dark_blue"], lw=2.2, label="Site-day precision")
    axis.plot(c, front["day_recall"], color=COLORS["grey"], lw=2.2, ls="--", label="Site-day recall of automatic corrections")
    targets = settings["operating_points"]["targets"]
    axis.axhspan(min(targets), max(targets), color=COLORS["light_white"], alpha=0.8, label=f"precision targets {min(targets):.2f}–{max(targets):.2f}")
    axis.axvline(float(settings["m9"]["c"]), color=COLORS["light_grey"], lw=1, ls=":")
    axis.text(float(settings["m9"]["c"]) + 0.004, 0.42, f"locked c = {settings['m9']['c']:.2f}", rotation=90, fontsize=8, color=COLORS["light_grey"], va="bottom")
    if marks is not None:
        for _, m in marks.iterrows():
            axis.axvline(m["c_correct_mean"], color=COLORS["orange"], lw=0.8, ls=":")
            axis.text(m["c_correct_mean"] + 0.004, 0.42, f"rule L, energy ≥ {m['target']:.3g}", rotation=90, fontsize=7, color=COLORS["orange"], va="bottom")
    axis.set_xlabel("Correction threshold c_correct (AUTO_CORRECT when p ≥ c_correct)")
    axis.set_ylabel(f"Held-out score, {_cohort_label(cohort)} site-days")
    axis.set_ylim(0.4, 1.02)
    axis.set_title("Precision and recall of M9 automatic corrections against the correction threshold")
    axis.legend(loc="lower left", fontsize=9)
    style_axis(axis)
    return _save(figure, path)


def plot_target_vs_achieved(summary: pd.DataFrame, boot: pd.DataFrame, cohort: str, path: Path) -> Path:
    """Fig B: target precision against achieved held-out precision, both rules, both precision types."""
    apply_journal_style()
    figure, axes = plt_subplots((10.4, 4.6), ncols=2, sharey=True)
    for axis, (applies, metric, label) in zip(axes, [("energy", "energy_precision", "Reference energy precision"), ("day", "day_precision", "Site-day precision")], strict=True):
        lo_x = min(summary["target"]) - 0.02
        axis.plot([lo_x, 1.0], [lo_x, 1.0], color=COLORS["light_grey"], ls="--", lw=1, label="achieved = target")
        for rule, color in (("L", COLORS["orange"]), ("E", COLORS["dark_blue"])):
            s = summary[(summary["applies"] == applies) & (summary["rule"] == rule)].sort_values("target")
            b = boot[(boot["applies"] == applies) & (boot["rule"] == rule) & (boot["metric"] == metric)].set_index("target").loc[s["target"]]
            axis.errorbar(s["target"], s[metric], yerr=[s[metric].to_numpy() - b["ci_low"].to_numpy(), b["ci_high"].to_numpy() - s[metric].to_numpy()],
                          fmt="o-", color=color, capsize=3, lw=1.8, label=RULE_LABELS[rule])
        axis.set_xlabel("Target minimum precision")
        axis.set_title(label, fontsize=11)
        style_axis(axis)
    axes[0].set_ylabel(f"Achieved on held-out {_cohort_label(cohort)} days")
    axes[0].legend(fontsize=8, loc="upper left")
    figure.suptitle("Does asking for a precision deliver it on unseen stations? (thresholds set per fold on other stations; station-bootstrap 95% CI)", fontsize=10.5)
    return _save(figure, path)


def plot_cost(summary: pd.DataFrame, cohort: str, rule: str, path: Path) -> Path:
    """Fig C: recall, Energy IoU and remaining false corrections at each target (both precisions, one rule)."""
    apply_journal_style()
    s = summary[(summary["applies"] == "both") & (summary["rule"] == rule)].sort_values("target")
    figure, left = plt_subplots((8.6, 4.6))
    right = left.twinx()
    step = float(np.min(np.diff(s["target"]))) if len(s) > 1 else 0.02
    right.bar(s["target"], s["fp_per_station_year"], width=0.45 * step, color=COLORS["red"], alpha=0.55, label="False corrections per station-year")
    left.plot(s["target"], s["day_recall"], "o-", color=COLORS["grey"], lw=2, label="Site-day recall")
    left.plot(s["target"], s["energy_iou"], "s-", color=COLORS["orange"], lw=2, label="Reference Energy IoU")
    left.plot(s["target"], s["energy_precision"], "^-", color=COLORS["dark_blue"], lw=2, label="Achieved energy precision")
    for x, y, c in zip(s["target"], s["day_recall"], s["c_correct_mean"], strict=True):
        left.annotate(f"c≈{c:.2f}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=7, color=COLORS["grey"])
    left.set_xlabel("Target minimum precision, applied to energy and site-day precision (rule " + rule + ")")
    left.set_ylabel(f"Held-out score, {_cohort_label(cohort)}")
    left.set_ylim(0.0, 1.05)
    right.set_ylabel("False corrections per station-year")
    right.set_ylim(0, max(1.0, float(s["fp_per_station_year"].max()) * 1.6))
    h1, l1 = left.get_legend_handles_labels()
    h2, l2 = right.get_legend_handles_labels()
    left.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    left.set_title("What each precision target costs and leaves behind")
    style_axis(left)
    style_axis(right, grid_axis=None)
    return _save(figure, path)


def plot_heatmaps(grid: pd.DataFrame, cohort: str, settings: Settings, path: Path) -> Path:
    """Fig D: review days and silently kept RPF days per station-year over (c_correct, c_keep)."""
    apply_journal_style()
    op = settings["operating_points"]
    cc, ck = list(op["heatmap_c_correct"]), list(op["heatmap_c_keep"]) + ["no band"]
    figure, axes = plt_subplots((11.5, 4.8), ncols=2)
    for axis, key, title, cmap in [(axes[0], "review_days_per_station_year", "Review days per station-year", "Blues"),
                                   (axes[1], "auto_fn_per_station_year", "RPF days kept without review, per station-year", "Oranges")]:
        z = np.full((len(cc), len(ck)), np.nan)
        for i, c_ in enumerate(cc):
            for j, k in enumerate(ck):
                row = grid[(grid["c_correct"] == c_) & ((grid["no_review_band"]) if k == "no band" else (grid["c_keep"] == k) & ~grid["no_review_band"])]
                if len(row):
                    z[i, j] = float(row[key].iloc[0])
        im = axis.imshow(np.ma.masked_invalid(z), cmap=cmap, origin="lower", aspect="auto")
        axis.set_xticks(range(len(ck)), [k if isinstance(k, str) else f"{k:.2f}" for k in ck])
        axis.set_yticks(range(len(cc)), [f"{c_:.2f}" for c_ in cc])
        axis.set_xlabel("Keep threshold c_keep (AUTO_KEEP when p ≤ c_keep)")
        axis.set_ylabel("Correction threshold c_correct")
        axis.set_title(title, fontsize=10)
        for i in range(len(cc)):
            for j in range(len(ck)):
                axis.text(j, i, "–" if not np.isfinite(z[i, j]) else f"{z[i, j]:.0f}", ha="center", va="center", fontsize=8)
        figure.colorbar(im, ax=axis, shrink=0.85)
    figure.suptitle(f"Review load and missed RPF days across both thresholds ({_cohort_label(cohort)}); 'no band' = c_keep = c_correct", fontsize=10)
    return _save(figure, path)


def plot_per_station(stations: pd.DataFrame, cohort: str, settings: Settings, path: Path) -> Path:
    """Fig E: per-station energy precision and recall at the configured operating points."""
    apply_journal_style()
    points = [p["label"] for p in settings["operating_points"]["per_station_points"]]
    figure, axes = plt_subplots((4.2 * len(points), 4.3), ncols=len(points), sharey=True)
    axes = np.atleast_1d(axes)
    gate = float(settings["gate"]["energy_precision_min"])
    for axis, label in zip(axes, points, strict=True):
        s = stations[stations["point"] == label].sort_values("station")
        x = np.arange(len(s))
        axis.bar(x - 0.2, s["energy_precision"], 0.4, color=COLORS["orange"], label="energy precision")
        axis.bar(x + 0.2, s["day_recall"], 0.4, color=COLORS["grey"], label="site-day recall")
        axis.axhline(gate, color=COLORS["red"], lw=1, ls=":")
        axis.set_xticks(x, [n.split("_")[-1] for n in s["station"]])
        axis.set_xlabel(f"{cohort.capitalize()} station")
        axis.set_title(label, fontsize=10)
        axis.set_ylim(0, 1.05)
        style_axis(axis)
    axes[0].set_ylabel("Held-out score per station")
    axes[0].legend(fontsize=8, loc="lower left")
    figure.suptitle(f"Per-station consistency at candidate operating points (dotted line: {gate:.2f} gate)", fontsize=11)
    return _save(figure, path)


def plot_ausgrid(summary: pd.DataFrame, cohort: str, rule: str, path: Path) -> Path:
    """Fig F: one plain figure of what a precision requirement buys and costs."""
    apply_journal_style()
    s = summary[(summary["applies"] == "both") & (summary["rule"] == rule)].sort_values("target")
    figure, axis = plt_subplots((8.6, 4.6))
    axis.plot(s["target"], s["energy_precision"], "^-", color=COLORS["orange"], lw=2, label="Achieved energy precision on unseen stations")
    axis.plot(s["target"], s["day_recall"], "o-", color=COLORS["grey"], lw=2, label="RPF days corrected automatically (recall)")
    axis.plot(s["target"], s["auto_share"], "s-", color=COLORS["dark_blue"], lw=2, label="Days decided without review")
    axis.set_xlabel("Minimum precision asked for")
    axis.set_ylabel("Share")
    axis.set_ylim(0.0, 1.05)
    axis.set_title(f"What a precision requirement buys and costs ({_cohort_label(cohort)})")
    axis.legend(loc="lower left", fontsize=9)
    style_axis(axis)
    return _save(figure, path)


def plt_subplots(figsize: tuple[float, float], ncols: int = 1, sharey: bool = False):
    import matplotlib.pyplot as plt

    return plt.subplots(1, ncols, figsize=figsize, sharey=sharey)


# ----------------------------------------------------------------------------- the study

def run_study(m9_days: pd.DataFrame, folds: list[Fold], settings: Settings, folder: Path) -> dict[str, Any]:
    """Everything for both cohorts; Beta is primary, Alpha the appendix. Returns tables and paths."""
    folder.mkdir(parents=True, exist_ok=True)
    op = settings["operating_points"]
    rule = op["recommended_rule"]
    tables: dict[str, list[pd.DataFrame]] = {k: [] for k in ("frontier", "heatmap", "targets", "selections", "bootstrap", "stations")}
    figures: list[Path] = []
    prepared = _prepare(m9_days)
    for cohort in ("beta", "alpha"):
        t = prepared[prepared["cohort"] == cohort].copy()
        if t.empty:
            continue
        front = frontier(t, settings).assign(cohort=cohort)
        grid = heatmap_grid(t, settings).assign(cohort=cohort)
        summary, selections = target_table(t, folds, settings, calibration_pool=prepared)
        boot = bootstrap_targets(t, selections, settings)
        stations = per_station_points(t, folds, selections, settings)
        for k, v in (("frontier", front), ("heatmap", grid), ("targets", summary), ("selections", selections.assign(cohort=cohort)),
                     ("bootstrap", boot), ("stations", stations)):
            tables[k].append(v)
        marks = summary[(summary["applies"] == "energy") & (summary["rule"] == rule) & (summary["target"].isin([0.95, 0.99]))]
        figures += [
            plot_frontier(front, cohort, settings, marks, folder / f"figA_precision_recall_frontier_{cohort}.png"),
            plot_target_vs_achieved(summary, boot, cohort, folder / f"figB_target_vs_achieved_{cohort}.png"),
            plot_cost(summary, cohort, rule, folder / f"figC_cost_of_precision_target_{cohort}.png"),
            plot_heatmaps(grid, cohort, settings, folder / f"figD_review_side_heatmaps_{cohort}.png"),
            plot_per_station(stations, cohort, settings, folder / f"figE_per_station_operating_points_{cohort}.png"),
            plot_ausgrid(summary, cohort, rule, folder / f"figF_ausgrid_one_pager_{cohort}.png"),
        ]
    outputs = []
    result: dict[str, Any] = {}
    for k, parts in tables.items():
        frame = pd.concat(parts, ignore_index=True)
        path = folder / f"{k}.csv"
        frame.to_csv(path, index=False)
        outputs.append(path)
        result[k] = frame
    result["outputs"] = outputs + figures
    return result


__all__ = ["AUTO_CORRECT", "evaluate_point", "frontier", "heatmap_grid", "precision_curve", "choose_threshold",
           "select_fold_threshold", "apply_selected", "pooled_from_applied", "target_table", "bootstrap_targets",
           "per_station_points", "run_study"]

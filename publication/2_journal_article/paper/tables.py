"""Paper tables: the registry ``TABLES`` and the functions behind it, written as CSV and Markdown.

Inputs:  the files of the finished stages under ``results/`` (read through ``results``)
         and the configuration.
Outputs: pandas frames; ``write_all`` saves each registry entry as
         ``results/paper/tables/<name>.csv`` and ``.md``.
Key steps: every registry entry is ``name -> function(settings) -> DataFrame``. A
         notebook cell calls ``show("tab02_headline", settings)``; adding a table later
         is one function and one registry line. A small Markdown writer keeps the
         dependency list short.

    tab01  datasets: stations, site-days, labelled days, sure and unsure
    tab02  headline metrics with station-bootstrap intervals
    tab03  per station, every method
    tab04  operating points: precision target, c, review days, false corrections, recall
    tab05  Gamma forecast error, raw versus corrected versus manual
    tab06  method summary: settings, fitted numbers, release calibration
    tab07  M9 confidence versus coverage      tab08  Beta 'unsure' sensitivity
    tab09  M9 calibration fits per fold       tab10  pooled and macro results with every station
    tab11  method comparison, all supporting metrics     tab12  the sample site-days of the figures
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import pynrpf
from pynrpf.m9 import RELEASE_CALIBRATION, RELEASE_PHI

from . import results
from .config import Settings
from .data import METHOD_LABELS
from .figures import sample_index
from .style import GROUP_LABELS

TableFn = Callable[[Settings], pd.DataFrame]
HEADLINE = [("energy_iou", "Reference Energy IoU (main)"), ("energy_precision", "Reference energy precision"),
            ("day_f1", "Site-day F1"), ("day_precision", "Site-day precision")]
# Metrics shown side by side as pooled and macro in the results table.
POOLED_MACRO = [("energy_iou", "Energy IoU"), ("energy_precision", "Energy precision"), ("day_f1", "Day F1"),
                ("day_precision", "Day precision"), ("day_recall", "Day recall")]


# ----------------------------------------------------------------------------- writing

def _fmt(value, digits: int = 3) -> str:
    if isinstance(value, (float, np.floating)):
        return "" if not np.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    """Render a frame as a GitHub-flavoured Markdown table without extra dependencies."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, digits) for v in row) + " |")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, folder: Path, name: str, digits: int = 3) -> list[Path]:
    """Write ``name.csv`` and ``name.md`` and return both paths."""
    csv = folder / f"{name}.csv"
    md = folder / f"{name}.md"
    df.to_csv(csv, index=False)
    md.write_text(to_markdown(df, digits), encoding="utf-8")
    return [csv, md]


# ----------------------------------------------------------------------------- the tables

def datasets_table(settings: Settings) -> pd.DataFrame:
    """tab01: per cohort, the stations, site-days, complete days and labelled wrong-sign days by confidence."""
    rows = []
    for cohort in settings["population"]["cohorts"]:
        idx = results.index(settings, cohort)
        complete = idx[idx["complete"]]
        sure, unsure = complete[complete["headline"]], complete[~complete["headline"]]
        rows.append({
            "Cohort": cohort.capitalize(), "Stations": int(idx["station"].nunique()),
            "Site-days": int(len(idx)), "Complete site-days": int(len(complete)),
            "Dropped (incomplete)": int((~idx["complete"]).sum()),
            "Sure or controlled days": int(len(sure)), "Unsure days": int(len(unsure)),
            "Wrong-sign days (sure)": int(sure["rpf"].sum()), "Wrong-sign days (unsure)": int(unsure["rpf"].sum()),
        })
    return pd.DataFrame(rows)


def headline_table(pooled: pd.DataFrame, bootstrap: pd.DataFrame | None = None) -> pd.DataFrame:
    """tab02: method × evaluation group, the four headline metrics with station-bootstrap intervals."""
    rows = []
    for _, r in pooled.iterrows():
        row = {"Method": METHOD_LABELS[r["method"]], "Evaluation group": GROUP_LABELS[r["group"]],
               "Held-out stations": int(r["n_stations"]), "Evaluated site-days": int(r["n_days"])}
        for key, label in HEADLINE:
            row[label] = float(r[key])
            if bootstrap is not None:
                ci = bootstrap[(bootstrap["method"] == r["method"]) & (bootstrap["group"] == r["group"])
                               & (bootstrap["metric"] == key)]
                if len(ci):
                    row[f"{label} 95% CI"] = f"{ci['ci_low'].iloc[0]:.3f}–{ci['ci_high'].iloc[0]:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def station_all_methods(stations: pd.DataFrame) -> pd.DataFrame:
    """tab03: one row per station; per method its Energy IoU, energy precision and day F1."""
    metrics = [("energy_iou", "Energy IoU"), ("energy_precision", "energy precision"), ("day_f1", "day F1")]
    methods = [m for m in METHOD_LABELS if m in set(stations["method"])]
    counts = stations[stations["method"] == methods[0]].set_index(["cohort", "station"])
    rows = []
    for (cohort, station), c in counts.iterrows():
        row = {"Cohort": cohort.capitalize(), "Station": station, "Site-days": int(c["n_days"]),
               "RPF days": int(c["n_rpf"]), "Required MWh": float(c["required_mwh"])}
        for method in methods:
            r = stations[(stations["method"] == method) & (stations["station"] == station)].iloc[0]
            for key, label in metrics:
                row[f"{METHOD_LABELS[method]} {label}"] = float(r[key])
        rows.append(row)
    return pd.DataFrame(rows)


def operating_points_table(targets: pd.DataFrame, rule: str, cohort: str = "beta") -> pd.DataFrame:
    """tab04: per precision target (on both precisions, one rule): the threshold, review load, errors and recall."""
    s = targets[(targets["cohort"] == cohort) & (targets["applies"] == "both") & (targets["rule"] == rule)]
    s = s.sort_values("target")
    return pd.DataFrame({
        "Precision target": s["target"].to_numpy(),
        "c_correct (mean over folds)": s["c_correct_mean"].to_numpy(),
        "c_correct (min)": s["c_correct_min"].to_numpy(),
        "c_correct (max)": s["c_correct_max"].to_numpy(),
        "Review days": s["review_days"].astype(int).to_numpy(),
        "Review days per station-year": s["review_days_per_station_year"].to_numpy(),
        "False corrections": s["auto_fp"].astype(int).to_numpy(),
        "False corrections per station-year": s["fp_per_station_year"].to_numpy(),
        "Site-day recall": s["day_recall"].to_numpy(),
        "Achieved energy precision": s["energy_precision"].to_numpy(),
        "Energy IoU": s["energy_iou"].to_numpy(),
    })


def gamma_table(impact: pd.DataFrame) -> pd.DataFrame:
    """tab05: seven-day-ahead RMSE per forecast model on raw, M9-corrected and manually corrected data."""
    return pd.DataFrame({
        "Forecast model": impact["model_label"].to_numpy(),
        "Raw RMSE (MW)": impact["raw_rmse_MW"].to_numpy(),
        "M9-corrected RMSE (MW)": impact["m9_corrected_rmse_MW"].to_numpy(),
        "Manually corrected RMSE (MW)": impact["manually_corrected_rmse_MW"].to_numpy(),
        "M9 reduction vs raw (MW)": impact["m9_rmse_reduction_vs_raw_MW"].to_numpy(),
        "M9 reduction vs raw (%)": impact["m9_rmse_reduction_vs_raw_pct"].to_numpy(),
        "Remaining gap to manual (MW)": impact["remaining_gap_to_manual_rmse_MW"].to_numpy(),
    })


def method_summary(settings: Settings) -> pd.DataFrame:
    """tab06: the released M9 settings, the fitted numbers of the reference run and the release calibration."""
    m9 = settings["m9"]
    fits = results.calibration_fits(settings)
    beta_fits, alpha_fits = fits[fits["cohort"] == "beta"], fits[fits["cohort"] == "alpha"]
    floors = {c: float(results.scores(settings, c)["sigma_floor"].iloc[0]) for c in settings["population"]["cohorts"]}
    c = float(m9["c"])
    rows = [
        ("Bridge misfit", "sum of squared residuals against the straight bridge between the nearest finite anchors"),
        ("Evidence statistic", "r(W) = (L/2) log((RSS_u + λ) / (RSS_c + λ)), λ = L φ²"),
        ("Evidence floor φ, Beta (MW)", f"{floors.get('beta', float('nan')):.6g}"),
        ("Evidence floor φ, Alpha (MW)", f"{floors.get('alpha', float('nan')):.6g}"),
        ("Release φ (MW)", f"{RELEASE_PHI:.6g}"),
        ("Missing readings", "windows touching a missing reading are inadmissible; the day is scored on the rest"),
        ("Window edges",
         "local minima of net load; start at or one slot after, end at or one slot before; gap exemption"),
        ("Candidate windows", "1,176 windows between 06:00 and 18:00, plus no correction at evidence zero"),
        ("Tie order", "no correction, then the shorter window, then the earlier window"),
        ("Calibration", "p = logistic(a + b z), z = sign(r*) log(1 + |r*|); a and b by logistic regression per fold"),
        ("Control c", f"{c:.2f}: AUTO_CORRECT at p ≥ {c:.2f}, AUTO_KEEP at p ≤ {1 - c:.2f}, UNCERTAIN between"),
        ("Fitting scope", f"M8 {settings['folds']['m8_training_scope']}, "
                          f"M9 {settings['folds']['m9_calibration_scope']}"),
        ("Beta folds: intercept a",
         f"{beta_fits['cal_intercept'].min():.3f} to {beta_fits['cal_intercept'].max():.3f}"),
        ("Beta folds: slope b", f"{beta_fits['cal_slope'].min():.3f} to {beta_fits['cal_slope'].max():.3f}"),
        ("Alpha folds (all eight Beta stations): a, b",
         f"{alpha_fits['cal_intercept'].iloc[0]:.4f}, {alpha_fits['cal_slope'].iloc[0]:.4f}"),
        ("Release calibration a, b", f"{RELEASE_CALIBRATION.intercept:.4f}, {RELEASE_CALIBRATION.slope:.4f}"),
        ("Release calibration provenance", RELEASE_CALIBRATION.provenance),
        ("Evidence at p = c under the release pair", f"{RELEASE_CALIBRATION.evidence_at(c):.3f}"),
        ("Evidence at p = 1 - c under the release pair", f"{RELEASE_CALIBRATION.evidence_at(1 - c):.3f}"),
        ("Gate", f"energy precision ≥ {float(settings['gate']['energy_precision_min']):.2f} "
                 f"on {settings['gate']['gated_cohort']}"),
        ("Bootstrap", f"{int(settings['metrics']['bootstrap']['draws'])} station resamples, "
                      f"seed {settings['metrics']['bootstrap']['seed']}"),
        ("Package", f"pynrpf {pynrpf.__version__}"),
    ]
    return pd.DataFrame(rows, columns=["Setting", "Value"])


def comparison_table(pooled: pd.DataFrame) -> pd.DataFrame:
    """tab11 (and tab08): method comparison with the supporting metrics, long form."""
    cols = ["method", "group", "n_stations", "n_days", "n_rpf", "required_mwh", "applied_mwh", "false_mwh",
            "energy_iou", "energy_precision", "day_f1", "day_precision", "day_recall", "sure_day_recall",
            "sure_day_uncertain_rate", "tp", "fp", "fn", "rate_auto_correct", "rate_auto_keep", "rate_uncertain",
            "interval_precision", "interval_recall", "interval_f1", "window_iou_mean", "start_error_mae",
            "end_error_mae",
            "min_change_days", "min_change_mean_mw", "reference_min_change_days", "reference_min_change_mean_mw"]
    out = pooled[cols].copy()
    out["method"] = out["method"].map(METHOD_LABELS)
    out["group"] = out["group"].map(lambda g: GROUP_LABELS.get(g, g))
    return out


def station_wide(stations: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-station appendix: one row per station, one column per method, for one metric."""
    wide = stations.pivot_table(index=["cohort", "station"], columns="method", values=metric).reset_index()
    wide.columns = [METHOD_LABELS.get(c, c) for c in wide.columns]
    first = stations["method"].iloc[0]
    counts = stations[stations["method"] == first].set_index(["cohort", "station"])[["n_days", "n_rpf", "required_mwh"]]
    wide = wide.merge(counts.reset_index(), on=["cohort", "station"], how="left")
    present = [METHOD_LABELS[m] for m in METHOD_LABELS if METHOD_LABELS[m] in wide.columns]
    return wide[["cohort", "station", "n_days", "n_rpf", "required_mwh"] + present]


def fits_table(fits: pd.DataFrame) -> pd.DataFrame:
    """tab09: the calibration pair and the evidence thresholds it implies, per fold."""
    return fits[["fold_id", "cohort", "held_out", "n_train", "n_train_rpf", "cal_intercept", "cal_slope",
                 "raw_threshold_correct", "raw_threshold_keep"]]


def pooled_macro_table(pooled: pd.DataFrame, macro: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """tab10: overall, Alpha and Beta rows with pooled and macro columns, then every station.

    Args:
        pooled: ``metrics.pooled_table`` (site-days and MWh weighted equally).
        macro: ``metrics.macro_table`` (every station weighted equally).
        stations: ``metrics.station_table``.

    Returns:
        Long form, one block per method. The three group rows carry ``Pooled <metric>``
        and ``Macro <metric>`` for each entry of ``POOLED_MACRO``; the station rows that
        follow carry the station's own values in the pooled columns and leave the macro
        columns empty, because a single station has no mean over stations.
    """
    rows = []
    for method in [m for m in METHOD_LABELS if m in set(pooled["method"])]:
        for group in ("combined", "alpha", "beta"):
            p = pooled[(pooled["method"] == method) & (pooled["group"] == group)].iloc[0]
            m = macro[(macro["method"] == method) & (macro["group"] == group)].iloc[0]
            row = {"Method": METHOD_LABELS[method], "Group": "Overall" if group == "combined" else GROUP_LABELS[group],
                   "Stations": int(p["n_stations"]), "Site-days": int(p["n_days"]), "RPF days": int(p["n_rpf"])}
            for key, label in POOLED_MACRO:
                row[f"Pooled {label}"] = float(p[key])
            for key, label in POOLED_MACRO:
                row[f"Macro {label}"] = float(m[key])
            rows.append(row)
        st = stations[stations["method"] == method].sort_values(["cohort", "station"])
        for _, r in st.iterrows():
            row = {"Method": METHOD_LABELS[method], "Group": r["station"], "Stations": 1,
                   "Site-days": int(r["n_days"]), "RPF days": int(r["n_rpf"])}
            for key, label in POOLED_MACRO:
                row[f"Pooled {label}"] = float(r[key])
            for _, label in POOLED_MACRO:
                row[f"Macro {label}"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- the registry

TABLES: dict[str, TableFn] = {
    "tab01_datasets": datasets_table,
    "tab02_headline": lambda s: headline_table(results.metric(s, "pooled"), results.metric(s, "bootstrap")),
    "tab03_per_station": lambda s: station_all_methods(results.metric(s, "stations")),
    "tab04_operating_points": lambda s: operating_points_table(results.operating_point(s, "targets"),
                                                               s["operating_points"]["recommended_rule"]),
    "tab05_gamma_forecast": lambda s: gamma_table(results.gamma_table(s, "gamma_forecast_impact")),
    "tab06_method_summary": method_summary,
    "tab07_coverage": lambda s: results.metric(s, "coverage"),
    "tab08_sensitivity": lambda s: comparison_table(results.metric(s, "sensitivity")),
    "tab09_calibration_fits": lambda s: fits_table(results.calibration_fits(s)),
    "tab10_pooled_macro_stations": lambda s: pooled_macro_table(results.metric(s, "pooled"), results.metric(s, "macro"),
                                                                results.metric(s, "stations")),
    "tab11_method_comparison": lambda s: comparison_table(results.metric(s, "pooled")),
    "tab12_sample_days": sample_index,
}
DIGITS = {"tab09_calibration_fits": 4}


def show(name: str, settings: Settings) -> pd.DataFrame:
    """Build one registry entry and return the frame (a notebook cell displays it)."""
    return TABLES[name](settings)


def write_all(settings: Settings, names: list[str] | None = None) -> list[Path]:
    """Write every registry entry as CSV and Markdown under ``results/paper/tables/``; returns the paths written."""
    folder = results.paper_dir(settings, "tables")
    written: list[Path] = []
    for name in names or list(TABLES):
        written += write_table(TABLES[name](settings), folder, name, digits=DIGITS.get(name, 3))
    return written

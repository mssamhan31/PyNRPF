"""Paper-facing tables from the frozen metric tables, written as CSV and Markdown.

Inputs:  the pooled, per-station, macro, bootstrap, coverage, sensitivity and fits
         tables and the gate decision (metrics.py, m9.py).
Outputs: the section 04 headline table, the Ausgrid presentation table, the
         method-comparison table with supporting metrics and bootstrap intervals, the
         per-station appendix table, the coverage table, the sensitivity table, the
         calibration-fit table and the pooled-and-macro results table, each as ``.csv``
         and ``.md``.
Key steps: select and rename columns; format intervals as text; a small Markdown
         writer so no optional dependency is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import METHOD_LABELS

GROUP_LABELS = {"combined": "Combined", "alpha": "Alpha", "beta": "Beta sure"}
HEADLINE = [("energy_iou", "Reference Energy IoU (main)"), ("energy_precision", "Reference energy precision"),
            ("day_f1", "Site-day F1"), ("day_precision", "Site-day precision")]
# Metrics shown side by side as pooled and macro in the results table.
POOLED_MACRO = [("energy_iou", "Energy IoU"), ("energy_precision", "Energy precision"), ("day_f1", "Day F1"),
                ("day_precision", "Day precision"), ("day_recall", "Day recall")]


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


def headline_table(pooled: pd.DataFrame, bootstrap: pd.DataFrame | None = None) -> pd.DataFrame:
    """The locked paper headline table: method × evaluation group, four headline metrics."""
    rows = []
    for _, r in pooled.iterrows():
        row = {"Method": METHOD_LABELS[r["method"]], "Evaluation group": GROUP_LABELS[r["group"]],
               "Held-out stations": int(r["n_stations"]), "Evaluated site-days": int(r["n_days"])}
        for key, label in HEADLINE:
            row[label] = float(r[key])
            if bootstrap is not None:
                ci = bootstrap[(bootstrap["method"] == r["method"]) & (bootstrap["group"] == r["group"]) & (bootstrap["metric"] == key)]
                if len(ci):
                    row[f"{label} 95% CI"] = f"{ci['ci_low'].iloc[0]:.3f}–{ci['ci_high'].iloc[0]:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def ausgrid_table(pooled: pd.DataFrame, gate: dict) -> pd.DataFrame:
    """The locked Ausgrid presentation table with the energy-precision gate per cohort."""
    rows = []
    for method, g in gate["methods"].items():
        p = pooled[pooled["method"] == method].set_index("group")
        rows.append({
            "Method": METHOD_LABELS[method],
            "Combined Energy IoU": float(p.loc["combined", "energy_iou"]),
            "Alpha Energy IoU": float(p.loc["alpha", "energy_iou"]),
            "Beta sure Energy IoU": float(p.loc["beta", "energy_iou"]),
            "Alpha/Beta energy-precision gate": f"{'Pass' if g['alpha_gate_pass'] else 'Fail'} / {'Pass' if g['beta_gate_pass'] else 'Fail'}",
            "Alpha energy precision": g["alpha_energy_precision"],
            "Beta energy precision": g["beta_energy_precision"],
            "Site-day F1": float(p.loc["combined", "day_f1"]),
            "Site-day precision": float(p.loc["combined", "day_precision"]),
            "Decision": ("Default" if (method == "m9" and gate["m9_default"]) else
                         ("Best validated, default" if (method == gate["strongest_combined_energy_iou"] and not gate["m9_default"]) else "Not default")),
        })
    return pd.DataFrame(rows)


def comparison_table(pooled: pd.DataFrame) -> pd.DataFrame:
    """Method comparison with the supporting metrics, long form, three decimals."""
    cols = ["method", "group", "n_stations", "n_days", "n_rpf", "required_mwh", "applied_mwh", "false_mwh",
            "energy_iou", "energy_precision", "day_f1", "day_precision", "day_recall", "sure_day_recall",
            "sure_day_uncertain_rate", "tp", "fp", "fn", "rate_auto_correct", "rate_auto_keep", "rate_uncertain",
            "interval_precision", "interval_recall", "interval_f1", "window_iou_mean", "start_error_mae", "end_error_mae",
            "min_change_days", "min_change_mean_mw", "reference_min_change_days", "reference_min_change_mean_mw"]
    out = pooled[cols].copy()
    out["method"] = out["method"].map(METHOD_LABELS)
    out["group"] = out["group"].map(GROUP_LABELS)
    return out


def station_wide(stations: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-station appendix: one row per station, one column per method, for one metric."""
    wide = stations.pivot_table(index=["cohort", "station"], columns="method", values=metric).reset_index()
    wide.columns = [METHOD_LABELS.get(c, c) for c in wide.columns]
    counts = stations[stations["method"] == stations["method"].iloc[0]].set_index(["cohort", "station"])[["n_days", "n_rpf", "required_mwh"]]
    wide = wide.merge(counts.reset_index(), on=["cohort", "station"], how="left")
    present = [METHOD_LABELS[m] for m in METHOD_LABELS if METHOD_LABELS[m] in wide.columns]
    return wide[["cohort", "station", "n_days", "n_rpf", "required_mwh"] + present]


def fits_table(fits: pd.DataFrame) -> pd.DataFrame:
    return fits[["fold_id", "cohort", "held_out", "n_train", "n_train_rpf", "cal_intercept", "cal_slope",
                 "raw_threshold_correct", "raw_threshold_keep"]]


def pooled_macro_table(pooled: pd.DataFrame, macro: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Results table: overall, Alpha and Beta rows with pooled and macro columns, then every station.

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

"""Headline and supporting metrics, pooled and per station, with bootstrap intervals.

Inputs:  the site-day decision-and-impact table of all methods (impact.py).
Outputs: pooled table (method × evaluation group), per-station table, macro table (the
         unweighted mean over stations of the per-station metrics), station-level
         bootstrap intervals, the M9 confidence-versus-coverage table, the release-gate
         decision, the Beta 'unsure' sensitivity table and M9 calibration reliability.
Key steps: the four headline metrics come from the locked definitions in
         ``m9_dev.m9_metrics`` (energies pooled by summing before dividing); supporting
         metrics are interval precision / recall / F1, mean window IoU, boundary error,
         confusion counts, outcome rates and minimum-demand impact. Headline groups use
         headline-confidence days only; Beta 'unsure' appears in the sensitivity table
         and nowhere else.
"""

from __future__ import annotations

import m9_metrics as mx  # frozen definitions; m9_dev is on sys.path via final_eval/__init__
import numpy as np
import pandas as pd

from .common import AUTO_CORRECT, AUTO_KEEP, METHODS, UNCERTAIN
from .config import Settings

GROUPS = ("combined", "alpha", "beta")
HEADLINE_METRICS = ("energy_iou", "energy_precision", "day_f1", "day_precision")
# Ratio metrics the macro table averages over stations; counts are summed instead.
MACRO_METRICS = ("energy_iou", "energy_precision", "day_f1", "day_precision", "day_recall",
                 "sure_day_recall", "rate_uncertain", "interval_precision", "interval_recall",
                 "interval_f1", "window_iou_mean")


def methods_present(site_days: pd.DataFrame) -> list[str]:
    """The methods in a site-day table, in the canonical order (a partial run is allowed while debugging)."""
    present = set(site_days["method"].unique())
    return [m for m in METHODS if m in present]


def _group_rows(site_days: pd.DataFrame, group: str) -> pd.DataFrame:
    t = site_days[site_days["headline"]]
    return t if group == "combined" else t[t["cohort"] == group]


def supporting(t: pd.DataFrame) -> dict:
    """Interval-level, window and minimum-demand metrics for one group of site-days."""
    tp, fp, fn = (int(t[c].sum()) for c in ("slot_tp", "slot_fp", "slot_fn"))
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    rpf = t[t["rpf"] == 1]
    applied = t[t["outcome"] == AUTO_CORRECT]
    changed = applied[applied["min_change_mw"].abs() > 0]
    ref_changed = rpf[rpf["reference_min_change_mw"].abs() > 0]
    return dict(
        n_stations=int(t["station"].nunique()),
        interval_precision=prec, interval_recall=rec,
        interval_f1=(2 * prec * rec / (prec + rec)) if (prec and rec and np.isfinite(prec + rec)) else 0.0,
        slot_tp=tp, slot_fp=fp, slot_fn=fn,
        window_iou_mean=float(rpf["window_iou"].fillna(0.0).mean()) if len(rpf) else np.nan,
        start_error_mae=float(t["start_error"].abs().mean()) if t["start_error"].notna().any() else np.nan,
        end_error_mae=float(t["end_error"].abs().mean()) if t["end_error"].notna().any() else np.nan,
        false_mwh=float(applied["false_mwh"].sum()),
        min_change_days=len(changed), min_change_mean_mw=float(changed["min_change_mw"].mean()) if len(changed) else np.nan,
        min_change_total_mw=float(changed["min_change_mw"].sum()),
        reference_min_change_days=len(ref_changed),
        reference_min_change_mean_mw=float(ref_changed["reference_min_change_mw"].mean()) if len(ref_changed) else np.nan,
    )


def summarise(t: pd.DataFrame) -> dict:
    """Headline (locked) plus supporting metrics for one group of site-days."""
    return {**mx.summarise(t), **supporting(t)}


def pooled_table(site_days: pd.DataFrame) -> pd.DataFrame:
    """Method × {combined, alpha, beta} on headline-confidence days, energies pooled."""
    rows = []
    for method in methods_present(site_days):
        sd = site_days[site_days["method"] == method]
        for group in GROUPS:
            rows.append(dict(method=method, group=group, **summarise(_group_rows(sd, group))))
    return pd.DataFrame(rows)


def station_table(site_days: pd.DataFrame) -> pd.DataFrame:
    """One row per method × cohort × station on headline-confidence days."""
    rows = []
    t = site_days[site_days["headline"]]
    for (method, cohort, station), g in t.groupby(["method", "cohort", "station"], sort=True):
        rows.append(dict(method=method, cohort=cohort, station=station, **summarise(g)))
    return pd.DataFrame(rows)


def macro_table(site_days: pd.DataFrame) -> pd.DataFrame:
    """Method × {combined, alpha, beta}: the unweighted mean over stations of the station metrics.

    The pooled table weights every site-day (and every MWh) equally, so the large Alpha
    stations dominate the combined row; the macro table gives every station one vote,
    which is the unit the evaluation generalises over. Each ``MACRO_METRICS`` value is
    the mean of the ``station_table`` values over the stations of the group. A station
    whose value is undefined is left out of that mean: energy and day precision need at
    least one applied correction, day recall needs at least one RPF day. Energy IoU of
    a station without RPF days is 0 by the frozen definition and stays in the mean.

    Args:
        site_days: the site-day decision-and-impact table of all methods (headline flag,
            outcome, energies in MWh, slot counts).

    Returns:
        One row per method × group with ``n_stations``, ``n_stations_with_rpf``,
        ``n_stations_with_correction``, the summed ``n_days`` and ``n_rpf`` and the
        station means of ``MACRO_METRICS`` (dimensionless ratios).
    """
    stations = station_table(site_days)
    rows = []
    for method in methods_present(site_days):
        for group in GROUPS:
            st = stations[stations["method"] == method]
            if group != "combined":
                st = st[st["cohort"] == group]
            row = dict(method=method, group=group, n_stations=int(len(st)),
                       n_stations_with_rpf=int((st["n_rpf"] > 0).sum()),
                       n_stations_with_correction=int((st["rate_auto_correct"] > 0).sum()),
                       n_days=int(st["n_days"].sum()), n_rpf=int(st["n_rpf"].sum()))
            for metric in MACRO_METRICS:
                row[metric] = float(st[metric].mean(skipna=True)) if st[metric].notna().any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_stations(site_days: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Station-level bootstrap of the headline metrics: resample stations with replacement.

    Stations are the unit of generalisation, so they, not site-days, are resampled;
    for the combined group each cohort is resampled separately so the cohort mix is kept.
    """
    bs = settings["metrics"]["bootstrap"]
    rng = np.random.default_rng(int(bs["seed"]))
    lo, hi = (float(v) for v in bs["percentiles"])
    draws = int(bs["draws"])
    rows = []
    for method in methods_present(site_days):
        sd = site_days[(site_days["method"] == method) & site_days["headline"]]
        by_station = {k: g for k, g in sd.groupby(["cohort", "station"])}
        cohorts = {c: sorted(s for (cc, s) in by_station if cc == c) for c in sd["cohort"].unique()}
        for group in GROUPS:
            use = cohorts if group == "combined" else {group: cohorts[group]}
            point = summarise(_group_rows(sd, group))
            samples = {m: [] for m in HEADLINE_METRICS}
            for _ in range(draws):
                pick = [by_station[(c, s)] for c, stations in use.items() for s in rng.choice(stations, size=len(stations), replace=True)]
                m = mx.summarise(pd.concat(pick, ignore_index=True))
                for k in HEADLINE_METRICS:
                    samples[k].append(m[k])
            for k in HEADLINE_METRICS:
                arr = np.asarray(samples[k], dtype=float)
                rows.append(dict(method=method, group=group, metric=k, point=point[k],
                                 ci_low=float(np.nanpercentile(arr, lo)), ci_high=float(np.nanpercentile(arr, hi)),
                                 draws=draws))
    return pd.DataFrame(rows)


def coverage_table(m9_site_days: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """M9 confidence versus coverage: re-decide every headline day at each c in the grid.

    Per cohort and c: the share of days decided automatically, the days sent to review,
    the errors among auto-decided days (auto FP = corrected non-RPF day, auto FN = kept
    RPF day), and the energy and site-day metrics of the automatic decisions.
    """
    rows = []
    t = m9_site_days[m9_site_days["headline"] & m9_site_days["prob_day"].notna()].copy()
    for cohort in ("alpha", "beta"):
        g = t[t["cohort"] == cohort]
        p = g["prob_day"].to_numpy(float)
        has_window = (g["pred_start"] >= 0).to_numpy()
        rpf = g["rpf"].to_numpy().astype(bool)
        for c in settings["metrics"]["coverage_grid"]:
            c = float(c)
            correct = (p >= c) & has_window
            keep = p <= 1 - c
            uncertain = ~correct & ~keep
            work = g.assign(outcome=np.select([correct, keep], [AUTO_CORRECT, AUTO_KEEP], UNCERTAIN),
                            proposed_mwh=np.where(correct, g["candidate_mwh"], 0.0),
                            correct_mwh=np.where(correct, g["candidate_correct_mwh"], 0.0))
            m = mx.summarise(work)
            auto = work[~uncertain]
            rows.append(dict(
                cohort=cohort, c=c, n_days=len(g), auto_decided_share=float((~uncertain).mean()),
                review_days=int(uncertain.sum()), review_share=float(uncertain.mean()),
                auto_fp=int((correct & ~rpf).sum()), auto_fn=int((keep & rpf).sum()),
                auto_days=len(auto), auto_precision=m["day_precision"], auto_recall=float((correct & rpf).sum() / rpf.sum()) if rpf.sum() else np.nan,
                energy_iou=m["energy_iou"], energy_precision=m["energy_precision"],
                sure_day_recall=m["sure_day_recall"], day_f1=m["day_f1"],
            ))
    return pd.DataFrame(rows)


def gate_decision(pooled: pd.DataFrame, settings: Settings) -> dict:
    """The M9 release gate (decision D1) and the default-method recommendation."""
    gate = settings["gate"]
    threshold = float(gate["energy_precision_min"])
    out = {"energy_precision_min": threshold, "gated_cohort": gate["gated_cohort"],
           "alpha_energy_precision_ceiling": float(gate["alpha_energy_precision_ceiling"]), "methods": {}}
    for method in [m for m in METHODS if m in set(pooled["method"])]:
        row = pooled[pooled["method"] == method].set_index("group")
        beta_prec = float(row.loc["beta", "energy_precision"])
        alpha_prec = float(row.loc["alpha", "energy_precision"])
        out["methods"][method] = {
            "beta_energy_precision": beta_prec, "beta_gate_pass": bool(beta_prec >= threshold),
            "alpha_energy_precision": alpha_prec, "alpha_gate_pass": bool(alpha_prec >= threshold),
            "alpha_vs_ceiling": alpha_prec - out["alpha_energy_precision_ceiling"],
            "combined_energy_iou": float(row.loc["combined", "energy_iou"]),
        }
    best = max(out["methods"], key=lambda m: out["methods"][m]["combined_energy_iou"])
    m9 = out["methods"].get("m9", {"beta_gate_pass": False})
    out["strongest_combined_energy_iou"] = best
    out["complete"] = set(out["methods"]) == set(METHODS)
    out["m9_default"] = bool(out["complete"] and m9["beta_gate_pass"] and best == "m9")
    if not out["complete"]:
        out["decision"] = f"Incomplete run (methods present: {sorted(out['methods'])}); no release decision."
    elif out["m9_default"]:
        out["decision"] = "M9 is the default method"
    else:
        out["decision"] = f"M9 is not the default; the best validated method is {best.upper()}"
    return out


def sensitivity_table(site_days: pd.DataFrame) -> pd.DataFrame:
    """Beta 'unsure' and Beta-all results per method: reported, never used for selection."""
    rows = []
    beta = site_days[site_days["cohort"] == "beta"]
    for method in methods_present(site_days):
        sd = beta[beta["method"] == method]
        rows.append(dict(method=method, group="beta_unsure", **summarise(sd[~sd["headline"]])))
        rows.append(dict(method=method, group="beta_all", **summarise(sd)))
    return pd.DataFrame(rows)


def calibration_reliability(m9_site_days: pd.DataFrame) -> pd.DataFrame:
    """Expected calibration error, Brier score and calibration-in-the-large per cohort."""
    rows = []
    t = m9_site_days[m9_site_days["headline"] & m9_site_days["prob_day"].notna()]
    for cohort, g in t.groupby("cohort", sort=True):
        rows.append(dict(cohort=cohort, n_days=len(g), **mx.calibration_reliability(g["prob_day"].to_numpy(float), g["rpf"].to_numpy(float))))
    return pd.DataFrame(rows)

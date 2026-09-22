"""One function per stage: what each notebook runs.

Inputs:  the settings (``config.load``) and the files of earlier stages.
Outputs: the files of each stage under ``paths.output_dir`` and one manifest per stage
         under ``manifests/`` (repository-relative paths, hashes, code and package version).
Key steps: every stage reads only committed inputs or the files of earlier stages,
         writes its own files, and records both in its manifest. Stages are separate
         functions so a notebook can run one at a time and a failure is local.

    data_folds(settings)                         01_data_folds   population and the fold manifest
    baselines(settings, fold_id=None, force=False)   02_baselines   M8 bundles per fold, then M7 and M8 predictions
    m9(settings)                                 03_m9           scores, per-fold calibration, decisions
    metrics(settings)                            04_metrics      reference terms, site-day table, metrics,
                                                                 operating points
    gamma(settings)                              05_gamma        the forecasting case study
    artefacts(settings)                          paper/          every figure and table of the registries

A scripted rerun is one line from ``publication/2_journal_article``:
``python -c "from paper import config, stages; stages.metrics(config.load())"``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from . import figures, operating_points, reference, results, tables
from . import gamma as gamma_study
from . import m9 as m9_method
from . import metrics as metric_tables
from .baselines import m7, m8
from .config import Settings
from .data import KEY, METHODS, load_population
from .folds import build_folds, check_manifest, fold_manifest
from .manifest import write_stage_manifest


def _log(message: str) -> None:
    print(f"[paper] {message}", flush=True)


# ----------------------------------------------------------------------------- 01 data and folds

def data_folds(settings: Settings) -> dict[str, Any]:
    """The common site-day population per cohort and the fold manifest."""
    started = time.time()
    indexes, _ = load_population(settings)
    folds = build_folds(indexes, settings)
    manifest = fold_manifest(folds, indexes)
    check_manifest(manifest)
    folder = settings.out("data_folds")
    outputs = [folder / "fold_manifest.csv"]
    manifest.to_csv(outputs[0], index=False)
    population = []
    for cohort, index in indexes.items():
        path = folder / f"site_days_{cohort}.parquet"
        index.to_parquet(path, index=False)
        outputs.append(path)
        complete_headline = index["complete"] & index["headline"]
        population.append(dict(cohort=cohort, site_days=len(index), complete=int(index["complete"].sum()),
                               dropped_incomplete=int((~index["complete"]).sum()),
                               headline_complete=int(complete_headline.sum()),
                               rpf_headline_complete=int(index.loc[complete_headline, "rpf"].sum())))
    population = pd.DataFrame(population)
    population.to_csv(folder / "population.csv", index=False)
    outputs.append(folder / "population.csv")
    inputs = [settings.dataset(c) for c in indexes]
    write_stage_manifest(settings, "01_data_folds", inputs, outputs, started,
                         {"n_folds": len(folds), "population": population.to_dict("records")})
    _log(f"{len(folds)} folds; population:\n{population.to_string(index=False)}")
    return {"folds": folds, "manifest": manifest, "population": population}


# ----------------------------------------------------------------------------- 02 baselines

def baselines(settings: Settings, fold_id: str | None = None, force: bool = False) -> dict[str, Any]:
    """Train one M8 bundle per distinct training set, then predict M7 and M8 on every held-out station.

    Args:
        settings: the evaluation settings.
        fold_id: train this fold only and stop (no predictions, no manifest); for the
            long stage in pieces.
        force: retrain a fold whose bundle already exists.

    Returns:
        ``training`` (the summary table) and, on a full run, the M7 and M8 interval tables.
    """
    started = time.time()
    _, intervals = load_population(settings)
    folds = [f for f in results.folds(settings) if fold_id is None or f.fold_id == fold_id]
    if not folds:
        raise ValueError(f"No fold named {fold_id}.")
    records = []
    trained: dict[str, dict[str, Any]] = {}
    for k, fold in enumerate(folds, 1):
        t0 = time.time()
        record = m8.train_fold(fold, intervals, settings, force=force, trained=trained)
        records.append(record)
        if record.get("skipped"):
            state = "skipped (bundle exists)"
        elif record.get("shared_with"):
            state = f"shares the bundle of {record['shared_with']}"
        else:
            state = f"trained in {time.time() - t0:.0f} s"
        _log(f"[{k}/{len(folds)}] {fold.fold_id}: {state}")
    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "validation_metrics"} for r in records])
    summary["training_stations"] = summary["training_stations"].map(";".join)
    if fold_id is not None:
        return {"training": summary}

    folder = settings.out("baselines")
    summary_path = folder / "training_summary.csv"
    summary.to_csv(summary_path, index=False)
    outputs = [summary_path] + [m8.bundle_manifest_path(settings, f) for f in folds]
    predictions: dict[str, pd.DataFrame] = {}
    for method, module in (("m7", m7), ("m8", m8)):
        parts = []
        for cohort in settings["population"]["cohorts"]:
            parts.append(module.predict_cohort(intervals[cohort], folds, settings))
            _log(f"{method} {cohort}: {len(parts[-1]):,} interval rows")
        predictions[method] = pd.concat(parts, ignore_index=True)
        path = results.interval_path(settings, method)
        predictions[method].to_parquet(path, index=False)
        outputs.append(path)
    inputs = [settings.dataset(c) for c in settings["population"]["cohorts"]] + [results.fold_manifest_path(settings)]
    write_stage_manifest(settings, "02_baselines", inputs, outputs, started,
                         {"n_fits": int((~summary["skipped"] & summary["shared_with"].isna()).sum())})
    return {"training": summary, **predictions}


# ----------------------------------------------------------------------------- 03 M9

def m9(settings: Settings) -> dict[str, Any]:
    """Score every cohort, calibrate per fold and decide; write the M9 site-day and interval tables."""
    started = time.time()
    indexes, intervals = load_population(settings)
    folds = results.folds(settings)
    folder = settings.out("m9")
    cohorts = list(settings["population"]["cohorts"])
    outputs: list[Path] = []
    extra: dict[str, Any] = {}
    # Score every cohort first (label-free, one floor per cohort): a fold's calibration
    # stations may lie in another cohort, so the fit draws from all scores.
    scores: dict[str, pd.DataFrame] = {}
    for cohort in cohorts:
        scores[cohort] = m9_method.score_cohort(intervals[cohort], indexes[cohort], settings)
        path = folder / f"scores_{cohort}.parquet"
        scores[cohort].to_parquet(path, index=False)
        extra[f"sigma_floor_{cohort}"] = float(scores[cohort]["sigma_floor"].iloc[0])
        outputs.append(path)
        _log(f"m9 {cohort}: {len(scores[cohort]):,} site-days scored, phi = {extra[f'sigma_floor_{cohort}']:.3e} MW")
    calibration_pool = pd.concat(scores.values(), ignore_index=True)
    site_day_parts, fit_parts, interval_parts = [], [], []
    for cohort in cohorts:
        site_days, fits = m9_method.predict_cohort(scores[cohort], folds, settings, calibration_pool)
        site_day_parts.append(site_days)
        fit_parts.append(fits)
        interval_parts.append(m9_method.interval_table(site_days, intervals[cohort]))
    site_days = pd.concat(site_day_parts, ignore_index=True)
    fits = pd.concat(fit_parts, ignore_index=True)
    table = pd.concat(interval_parts, ignore_index=True)
    site_days.to_parquet(folder / "site_days_m9.parquet", index=False)
    fits.to_csv(folder / "calibration_fits.csv", index=False)
    table.to_parquet(results.interval_path(settings, "m9"), index=False)
    outputs += [folder / "site_days_m9.parquet", folder / "calibration_fits.csv", results.interval_path(settings, "m9")]
    inputs = [settings.dataset(c) for c in cohorts] + [results.fold_manifest_path(settings)]
    write_stage_manifest(settings, "03_m9", inputs, outputs, started, extra)
    _log(f"m9: {len(site_days):,} site-day decisions, {len(table):,} interval rows")
    return {"scores": scores, "site_days": site_days, "fits": fits, "intervals": table}


# ------------------------------------------------------------ 04 reference terms, metrics, operating points

def metrics(settings: Settings) -> dict[str, Any]:
    """Reference-side terms and the site-day table for all methods, the metric tables, the operating-point study."""
    started = time.time()
    indexes, intervals = load_population(settings)
    truth = pd.concat(intervals.values(), ignore_index=True)[KEY + ["slot", "y", "s", "truth"]]
    m9_site_days = results.m9_site_days(settings)
    inputs = [results.interval_path(settings, m) for m in METHODS]
    inputs += [settings.out("m9") / "site_days_m9.parquet", results.fold_manifest_path(settings)]
    parts = []
    for method in METHODS:
        table = results.intervals(settings, method)
        joined = table.merge(truth, on=KEY + ["slot"], how="left", validate="one_to_one")
        decided = reference.attach_outcomes(joined, m9_site_days if method == "m9" else None)
        for cohort in settings["population"]["cohorts"]:
            parts.append(reference.siteday_table(decided[decided["cohort"] == cohort], indexes[cohort]))
        _log(f"{method}: outcomes and reference terms for {sum(len(p) for p in parts[-len(indexes):]):,} site-days")
    site_days = pd.concat(parts, ignore_index=True)
    keys = site_days.groupby("method")[KEY].apply(lambda d: set(map(tuple, d.to_numpy())))
    if any(k != keys.iloc[0] for k in keys):
        raise ValueError("Methods do not share identical (cohort, station, date) keys.")
    folder = settings.out("metrics")
    site_days.to_parquet(results.site_days_path(settings), index=False)
    # A CSV of the decisions without the energy detail, for readers without parquet tools.
    decisions_csv = folder / "site_day_decisions.csv"
    site_days[["method", "cohort", "station", "fold_id", "date", "confidence", "rpf", "outcome", "prob_day",
               "pred_start", "pred_end", "true_start", "true_end", "proposed_mwh", "required_mwh", "correct_mwh",
               "raw_min_mw", "applied_min_mw", "min_change_mw"]].to_csv(decisions_csv, index=False)
    outputs = [results.site_days_path(settings), decisions_csv]

    pooled = metric_tables.pooled_table(site_days)
    stations = metric_tables.station_table(site_days)
    macro = metric_tables.macro_table(site_days)
    boot = metric_tables.bootstrap_stations(site_days, settings)
    m9_days = site_days[site_days["method"] == "m9"]
    coverage = metric_tables.coverage_table(m9_days, settings)
    gate = metric_tables.gate_decision(pooled, settings)
    sensitivity = metric_tables.sensitivity_table(site_days)
    reliability = metric_tables.calibration_reliability(m9_days)
    written = {"pooled.csv": pooled, "stations.csv": stations, "macro.csv": macro, "bootstrap.csv": boot,
               "coverage.csv": coverage, "sensitivity.csv": sensitivity, "calibration_reliability.csv": reliability}
    for name, frame in written.items():
        frame.to_csv(folder / name, index=False)
        outputs.append(folder / name)
    (folder / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    outputs.append(folder / "gate.json")
    show = pooled[["method", "group", "n_stations", "n_days", "n_rpf", "energy_iou", "energy_precision", "day_f1",
                   "day_precision", "sure_day_recall", "rate_uncertain"]]
    _log(f"pooled headline:\n{show.round(4).to_string(index=False)}\n{gate['decision']}")

    study = operating_points.run_study(m9_days, results.folds(settings), settings,
                                       results.operating_points_dir(settings))
    outputs += study["outputs"]
    targets = study["targets"]
    targets = targets[targets["applies"] == "both"][["cohort", "target", "rule", "c_correct_mean", "energy_precision",
                                                     "day_precision", "day_recall", "energy_iou", "fp_per_station_year",
                                                     "review_days_per_station_year"]]
    _log("operating points (both precisions):\n" + targets.round(3).to_string(index=False))
    write_stage_manifest(settings, "04_metrics", inputs, outputs, started,
                         {"n_site_days_per_method": int(len(keys.iloc[0])), "gate": gate})
    return {"site_days": site_days, "pooled": pooled, "stations": stations, "macro": macro, "bootstrap": boot,
            "coverage": coverage, "gate": gate, "sensitivity": sensitivity, "reliability": reliability,
            "operating_points": study}


# ----------------------------------------------------------------------------- 05 Gamma

def gamma(settings: Settings) -> dict[str, Any]:
    """The Gamma forecasting case study against the held-out M9 decisions."""
    started = time.time()
    m9_days = results.site_days(settings).query("method == 'm9'")
    result = gamma_study.run_study(gamma_study.load_gamma(settings), m9_days, settings, settings.out("gamma"))
    write_stage_manifest(settings, "05_gamma", [settings.dataset("gamma"), results.site_days_path(settings)],
                         result["outputs"], started, {"audit": result["audit"].to_dict("records")[0]})
    _log(f"Gamma impact:\n{result['impact'].round(3).to_string(index=False)}")
    return result


# ----------------------------------------------------------------------------- paper artefacts

def artefacts(settings: Settings) -> list[Path]:
    """Every figure and table of the registries, under ``paper/figures`` and ``paper/tables``."""
    started = time.time()
    outputs = figures.write_all(settings) + tables.write_all(settings)
    metric_dir, m9_dir, gamma_dir = settings.out("metrics"), settings.out("m9"), settings.out("gamma")
    inputs = [metric_dir / n for n in ("site_days.parquet", "pooled.csv", "stations.csv", "macro.csv", "bootstrap.csv",
                                       "coverage.csv", "sensitivity.csv", "gate.json")]
    inputs += [m9_dir / "calibration_fits.csv", results.operating_points_dir(settings) / "targets.csv",
               results.operating_points_dir(settings) / "frontier.csv", gamma_dir / "gamma_series.parquet",
               gamma_dir / "gamma_forecast_metrics.csv", gamma_dir / "gamma_forecast_impact.csv"]
    write_stage_manifest(settings, "paper", inputs, outputs, started,
                         {"figures": list(figures.FIGURES), "tables": list(tables.TABLES)})
    _log(f"{len(outputs)} figure and table files written")
    return outputs

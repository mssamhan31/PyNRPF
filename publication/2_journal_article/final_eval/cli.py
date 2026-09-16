"""Stage entry points: what each numbered notebook runs, callable from the command line too.

Inputs:  the settings (config/final_evaluation.yaml) and the outputs of earlier stages.
Outputs: the files of each stage under outputs/01_final_evaluation/ and one manifest
         per stage under manifests/.
Key steps: every stage reads only committed inputs or the files of earlier stages,
         writes its own files, and records both in its manifest. Stages are separate
         functions so a notebook can run one at a time and a failure is local.

Usage (from publication/2_journal_article/):
    python -m final_eval folds
    python -m final_eval train-m8 [--fold beta_beta_A] [--force]      # heavy; the author runs it
    python -m final_eval predict-m7 | predict-m8 | predict-m9
    python -m final_eval outcomes | metrics | report | gamma | operating-points
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from . import config as cfgmod
from . import figures, gamma, impact, m7, m8, m9, metrics, operating_points, outcomes, tables
from .common import METHODS
from .data import KEY, load_population
from .folds import Fold, build_folds, check_manifest, fold_manifest
from .manifest import write_stage_manifest

STAGES = ("folds", "train-m8", "predict-m7", "predict-m8", "predict-m9", "outcomes", "metrics", "report", "gamma", "operating-points")


# ----------------------------------------------------------------------------- shared readers

def read_index(settings: cfgmod.Settings, cohort: str) -> pd.DataFrame:
    return pd.read_parquet(settings.out("folds") / f"site_days_{cohort}.parquet")


def read_folds(settings: cfgmod.Settings) -> list[Fold]:
    """Rebuild the Fold objects from the committed fold manifest (the single source of truth)."""
    manifest = pd.read_csv(settings.out("folds") / "fold_manifest.csv")
    folds = []
    for row in manifest.itertuples(index=False):
        stations = row.m8_training_stations.split(";")
        cohorts = {s: ("alpha" if s.startswith("alpha") else "beta") for s in stations}
        folds.append(Fold(fold_id=row.fold_id, cohort=row.cohort, held_out=row.held_out,
                          m8_training=tuple((cohorts[s], s) for s in stations),
                          m9_calibration=tuple(row.m9_calibration_stations.split(";"))))
    return folds


def read_intervals(settings: cfgmod.Settings, method: str) -> pd.DataFrame:
    return pd.read_parquet(settings.out(method) / f"intervals_{method}.parquet")


def read_site_days(settings: cfgmod.Settings) -> pd.DataFrame:
    return pd.read_parquet(settings.out("site_days") / "site_days.parquet")


def _log(message: str) -> None:
    print(f"[final_eval] {message}", flush=True)


def available_methods(settings: cfgmod.Settings) -> list[str]:
    """Methods whose interval prediction table exists. A partial set is allowed while
    debugging; the frozen release requires all three and the report says so loudly."""
    present = [m for m in METHODS if (settings.out(m) / f"intervals_{m}.parquet").exists()]
    missing = [m for m in METHODS if m not in present]
    if missing:
        _log(f"WARNING: no predictions yet for {missing}; continuing with {present}. Not a release.")
    return present


# ----------------------------------------------------------------------------- stages

def stage_folds(settings: cfgmod.Settings) -> dict[str, Any]:
    """Common site-day population per cohort and the 18-fold manifest."""
    started = time.time()
    indexes, _ = load_population(settings)
    folds = build_folds(indexes, settings)
    manifest = fold_manifest(folds, indexes)
    check_manifest(manifest)
    folder = settings.out("folds")
    outputs = [folder / "fold_manifest.csv"]
    manifest.to_csv(outputs[0], index=False)
    population = []
    for cohort, index in indexes.items():
        path = folder / f"site_days_{cohort}.parquet"
        index.to_parquet(path, index=False)
        outputs.append(path)
        population.append(dict(cohort=cohort, site_days=len(index), complete=int(index["complete"].sum()),
                               dropped_incomplete=int((~index["complete"]).sum()),
                               headline_complete=int((index["complete"] & index["headline"]).sum()),
                               rpf_headline_complete=int(index.loc[index["complete"] & index["headline"], "rpf"].sum())))
    population = pd.DataFrame(population)
    population.to_csv(folder / "population.csv", index=False)
    outputs.append(folder / "population.csv")
    inputs = [settings.dataset(c) for c in indexes]
    write_stage_manifest(settings, "01_folds", inputs, outputs, started, {"n_folds": len(folds), "population": population.to_dict("records")})
    _log(f"{len(folds)} folds; population:\n{population.to_string(index=False)}")
    return {"folds": folds, "manifest": manifest, "population": population}


def stage_train_m8(settings: cfgmod.Settings, fold_id: str | None = None, force: bool = False) -> pd.DataFrame:
    """Train one M8 bundle per fold (or one fold). Heavy; resumable."""
    started = time.time()
    _, intervals = load_population(settings)
    folds = [f for f in read_folds(settings) if fold_id is None or f.fold_id == fold_id]
    if not folds:
        raise ValueError(f"No fold named {fold_id}.")
    records = []
    for k, fold in enumerate(folds, 1):
        t0 = time.time()
        record = m8.train_fold(fold, intervals, settings, force=force)
        records.append(record)
        state = "skipped (bundle exists)" if record.get("skipped") else f"trained in {time.time() - t0:.0f} s"
        _log(f"[{k}/{len(folds)}] {fold.fold_id}: {state}")
    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "validation_metrics"} for r in records])
    summary["training_stations"] = summary["training_stations"].map(";".join)
    path = settings.out("bundles") / "training_summary.csv"
    if fold_id is None:
        summary.to_csv(path, index=False)
        write_stage_manifest(settings, "02_train_m8", [settings.out("folds") / "fold_manifest.csv"],
                             [path] + [m8.bundle_manifest_path(settings, f) for f in folds], started)
    return summary


def stage_predict(settings: cfgmod.Settings, method: str) -> pd.DataFrame:
    """Held-out predictions of one method for both cohorts, in the common interval schema."""
    started = time.time()
    indexes, intervals = load_population(settings)
    folds = read_folds(settings)
    folder = settings.out(method)
    outputs, parts = [], []
    extra: dict[str, Any] = {}
    for cohort in settings["population"]["cohorts"]:
        if method == "m7":
            parts.append(m7.predict_cohort(intervals[cohort], folds, settings))
        elif method == "m8":
            parts.append(m8.predict_cohort(intervals[cohort], folds, settings))
        else:
            scores = m9.score_cohort(intervals[cohort], indexes[cohort], settings)
            scores_path = folder / f"scores_{cohort}.parquet"
            scores.to_parquet(scores_path, index=False)
            site_days, fits = m9.predict_cohort(scores, folds, settings)
            parts.append(m9.interval_table(site_days, intervals[cohort]))
            extra.setdefault("site_days", []).append(site_days)
            extra.setdefault("fits", []).append(fits)
            extra[f"sigma_floor_{cohort}"] = float(scores["sigma_floor"].iloc[0])
            outputs.append(scores_path)
        _log(f"{method} {cohort}: {len(parts[-1]):,} interval rows")
    table = pd.concat(parts, ignore_index=True)
    path = folder / f"intervals_{method}.parquet"
    table.to_parquet(path, index=False)
    outputs.append(path)
    if method == "m9":
        site_days = pd.concat(extra.pop("site_days"), ignore_index=True)
        fits = pd.concat(extra.pop("fits"), ignore_index=True)
        site_days.to_parquet(folder / "site_days_m9.parquet", index=False)
        fits.to_csv(folder / "calibration_fits.csv", index=False)
        outputs += [folder / "site_days_m9.parquet", folder / "calibration_fits.csv"]
    inputs = [settings.dataset(c) for c in settings["population"]["cohorts"]] + [settings.out("folds") / "fold_manifest.csv"]
    if method == "m8":
        inputs += [m8.bundle_manifest_path(settings, f) for f in folds]
    write_stage_manifest(settings, f"{cfgmod.STAGE_DIRS[method]}_predict", inputs, outputs, started, extra)
    return table


def stage_outcomes(settings: cfgmod.Settings) -> pd.DataFrame:
    """Outcomes, applied corrections, energy terms and minimum-demand impact for all methods."""
    started = time.time()
    indexes, intervals = load_population(settings)
    truth = pd.concat(intervals.values(), ignore_index=True)[KEY + ["slot", "y", "s", "truth"]]
    m9_site_days = pd.read_parquet(settings.out("m9") / "site_days_m9.parquet")
    parts, inputs = [], []
    for method in available_methods(settings):
        table = read_intervals(settings, method)
        inputs.append(settings.out(method) / f"intervals_{method}.parquet")
        joined = table.merge(truth, on=KEY + ["slot"], how="left", validate="one_to_one")
        decided = outcomes.attach_outcomes(joined, m9_site_days if method == "m9" else None)
        for cohort in settings["population"]["cohorts"]:
            parts.append(impact.siteday_table(decided[decided["cohort"] == cohort], indexes[cohort]))
        _log(f"{method}: outcomes and impact for {sum(len(p) for p in parts[-len(indexes):]):,} site-days")
    site_days = pd.concat(parts, ignore_index=True)
    keys = site_days.groupby("method")[KEY].apply(lambda d: set(map(tuple, d.to_numpy())))
    if any(k != keys.iloc[0] for k in keys):
        raise ValueError("Methods do not share identical (cohort, station, date) keys.")
    folder = settings.out("site_days")
    path = folder / "site_days.parquet"
    site_days.to_parquet(path, index=False)
    # A CSV of the decisions without the energy detail, for readers without parquet tools.
    csv = folder / "site_day_decisions.csv"
    site_days[["method", "cohort", "station", "fold_id", "date", "confidence", "rpf", "outcome", "prob_day",
               "pred_start", "pred_end", "true_start", "true_end", "proposed_mwh", "required_mwh", "correct_mwh",
               "raw_min_mw", "applied_min_mw", "min_change_mw"]].to_csv(csv, index=False)
    write_stage_manifest(settings, "06_outcomes", inputs, [path, csv], started,
                         {"n_site_days_per_method": int(len(keys.iloc[0])), "methods": list(keys.index)})
    return site_days


def stage_metrics(settings: cfgmod.Settings) -> dict[str, Any]:
    """Pooled and per-station metrics, bootstrap, coverage, gate, sensitivity, reliability."""
    started = time.time()
    site_days = read_site_days(settings)
    folder = settings.out("metrics")
    pooled = metrics.pooled_table(site_days)
    stations = metrics.station_table(site_days)
    boot = metrics.bootstrap_stations(site_days, settings)
    m9_days = site_days[site_days["method"] == "m9"]
    coverage = metrics.coverage_table(m9_days, settings)
    gate = metrics.gate_decision(pooled, settings)
    sensitivity = metrics.sensitivity_table(site_days)
    reliability = metrics.calibration_reliability(m9_days)
    written = {"pooled.csv": pooled, "stations.csv": stations, "bootstrap.csv": boot, "coverage.csv": coverage,
               "sensitivity.csv": sensitivity, "calibration_reliability.csv": reliability}
    outputs = []
    for name, frame in written.items():
        frame.to_csv(folder / name, index=False)
        outputs.append(folder / name)
    (folder / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    outputs.append(folder / "gate.json")
    write_stage_manifest(settings, "07_metrics", [settings.out("site_days") / "site_days.parquet"], outputs, started, {"gate": gate})
    show = pooled[["method", "group", "n_stations", "n_days", "n_rpf", "energy_iou", "energy_precision", "day_f1", "day_precision", "sure_day_recall", "rate_uncertain"]]
    _log(f"pooled headline:\n{show.round(4).to_string(index=False)}\n{gate['decision']}")
    return {"pooled": pooled, "stations": stations, "bootstrap": boot, "coverage": coverage, "gate": gate,
            "sensitivity": sensitivity, "reliability": reliability}


def stage_report(settings: cfgmod.Settings) -> list[Path]:
    """Paper tables and figures from the frozen metric tables and the site-day table."""
    started = time.time()
    mfolder = settings.out("metrics")
    pooled = pd.read_csv(mfolder / "pooled.csv")
    stations = pd.read_csv(mfolder / "stations.csv")
    boot = pd.read_csv(mfolder / "bootstrap.csv")
    coverage = pd.read_csv(mfolder / "coverage.csv")
    sensitivity = pd.read_csv(mfolder / "sensitivity.csv")
    gate = json.loads((mfolder / "gate.json").read_text(encoding="utf-8"))
    fits = pd.read_csv(settings.out("m9") / "calibration_fits.csv")
    site_days = read_site_days(settings)
    tfolder, ffolder = settings.out("tables"), settings.out("figures")
    outputs: list[Path] = []
    outputs += tables.write_table(tables.headline_table(pooled, boot), tfolder, "table01_headline")
    outputs += tables.write_table(tables.ausgrid_table(pooled, gate), tfolder, "table02_ausgrid")
    outputs += tables.write_table(tables.comparison_table(pooled), tfolder, "table03_method_comparison")
    outputs += tables.write_table(tables.station_wide(stations, "energy_iou"), tfolder, "table04_station_energy_iou")
    outputs += tables.write_table(tables.station_wide(stations, "energy_precision"), tfolder, "table05_station_energy_precision")
    outputs += tables.write_table(tables.station_wide(stations, "day_f1"), tfolder, "table06_station_day_f1")
    outputs += tables.write_table(coverage, tfolder, "table07_m9_confidence_coverage")
    outputs += tables.write_table(tables.comparison_table(sensitivity.assign(group=sensitivity["group"])).assign(group=sensitivity["group"]), tfolder, "table08_beta_unsure_sensitivity")
    outputs += tables.write_table(tables.fits_table(fits), tfolder, "table09_m9_calibration_fits", digits=4)
    outputs.append(figures.plot_headline(pooled, ffolder / "fig01_headline_metrics.png"))
    outputs.append(figures.plot_stations(stations, "energy_iou", "Reference Energy IoU", ffolder / "fig02_station_energy_iou.png"))
    outputs.append(figures.plot_stations(stations, "energy_precision", "Reference energy precision", ffolder / "fig03_station_energy_precision.png"))
    outputs.append(figures.plot_review_burden(coverage, "beta", ffolder / "fig04_m9_review_burden_beta.png"))
    outputs.append(figures.plot_review_burden(coverage, "alpha", ffolder / "fig05_m9_review_burden_alpha.png"))
    outputs.append(figures.plot_coverage_scores(coverage, "beta", ffolder / "fig06_m9_coverage_scores_beta.png"))
    outputs.append(figures.plot_calibration(site_days[site_days["method"] == "m9"], ffolder / "fig07_m9_calibration.png"))
    # Sample panels need the interval rows of the selected Beta days for every method.
    _, intervals = load_population(settings)
    beta_truth = intervals["beta"][KEY + ["slot", "y", "s", "truth"]]
    index = []
    for method in metrics.methods_present(site_days):
        selected = figures.select_samples(site_days, method, settings)
        table = read_intervals(settings, method)
        table = table[table["cohort"] == "beta"].merge(selected[KEY], on=KEY, how="inner")
        joined = table.merge(beta_truth, on=KEY + ["slot"], how="left", validate="one_to_one")
        outputs += figures.plot_samples(selected, joined, method, settings, ffolder)
        index.append(selected.assign(method=method))
    index = pd.concat(index, ignore_index=True)[["method", "kind", "cohort", "station", "date", "outcome", "prob_day", "pred_start", "pred_end",
                                                  "true_start", "true_end", "required_mwh", "candidate_mwh", "proposed_mwh", "correct_mwh"]]
    index.to_csv(ffolder / "samples_index.csv", index=False)
    outputs.append(ffolder / "samples_index.csv")
    inputs = [mfolder / n for n in ("pooled.csv", "stations.csv", "bootstrap.csv", "coverage.csv", "sensitivity.csv", "gate.json")]
    write_stage_manifest(settings, "08_report", inputs, outputs, started)
    _log(f"{len(outputs)} tables and figures written")
    return outputs


def stage_gamma(settings: cfgmod.Settings) -> dict[str, Any]:
    """The Gamma downstream forecasting study against the held-out M9 decisions."""
    started = time.time()
    site_days = read_site_days(settings)
    m9_days = site_days[site_days["method"] == "m9"]
    result = gamma.run_study(gamma.load_gamma(settings), m9_days, settings, settings.out("gamma"))
    write_stage_manifest(settings, "09_gamma", [settings.dataset("gamma"), settings.out("site_days") / "site_days.parquet"],
                         result["outputs"], started, {"audit": result["audit"].to_dict("records")[0]})
    _log(f"Gamma impact:\n{result['impact'].round(3).to_string(index=False)}")
    return result


def stage_operating_points(settings: cfgmod.Settings) -> dict[str, Any]:
    """The operating-point study: precision targets and the two-control policy, M9 only."""
    started = time.time()
    site_days = read_site_days(settings)
    m9_days = site_days[site_days["method"] == "m9"]
    result = operating_points.run_study(m9_days, read_folds(settings), settings, settings.out("operating_points"))
    write_stage_manifest(settings, "10_operating_points",
                         [settings.out("site_days") / "site_days.parquet", settings.out("folds") / "fold_manifest.csv"],
                         result["outputs"], started, {"paper_figures": list(settings["operating_points"]["paper_figures"])})
    show = result["targets"]
    show = show[(show["applies"] == "both")][["cohort", "target", "rule", "c_correct_mean", "energy_precision", "day_precision", "day_recall", "energy_iou", "fp_per_station_year", "review_days_per_station_year"]]
    _log("targets (both precisions):\n" + show.round(3).to_string(index=False))
    return result


# ----------------------------------------------------------------------------- command line

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m final_eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--fold", default=None, help="train-m8 only: one fold id, e.g. beta_beta_A")
    parser.add_argument("--force", action="store_true", help="train-m8 only: retrain even if a bundle exists")
    parser.add_argument("--config", default=None, help="alternative configuration file")
    args = parser.parse_args(argv)
    settings = cfgmod.load(args.config)
    if args.stage == "folds":
        stage_folds(settings)
    elif args.stage == "train-m8":
        stage_train_m8(settings, args.fold, args.force)
    elif args.stage.startswith("predict-"):
        stage_predict(settings, args.stage.split("-")[1])
    elif args.stage == "outcomes":
        stage_outcomes(settings)
    elif args.stage == "metrics":
        stage_metrics(settings)
    elif args.stage == "report":
        stage_report(settings)
    elif args.stage == "gamma":
        stage_gamma(settings)
    elif args.stage == "operating-points":
        stage_operating_points(settings)

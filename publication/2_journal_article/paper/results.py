"""Readers for the files a finished stage wrote under the output directory.

Inputs:  the settings (which name the output directory) and a stage's file names.
Outputs: the frames, records and paths the later stages, the figures and the tables read.
Key steps: one function per file family, so no other module builds a results path by
         hand and the layout of ``results/`` is written down once, here:

    01_data_folds/   fold_manifest.csv, population.csv, site_days_<cohort>.parquet
    02_baselines/    <fold_id>.json, training_summary.csv, intervals_m7.parquet, intervals_m8.parquet, bundles/
    03_m9/           scores_<cohort>.parquet, calibration_fits.csv, site_days_m9.parquet, intervals_m9.parquet
    04_metrics/      site_days.parquet, site_day_decisions.csv, the metric tables, gate.json, operating_points/
    05_gamma/        the Gamma study tables, series and figures
    paper/           figures/ and tables/, one file pair per registry entry
    manifests/       one JSON per stage
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import Settings
from .folds import Fold, folds_from_manifest


def index(settings: Settings, cohort: str) -> pd.DataFrame:
    """The site-day index of a cohort (``data.siteday_index``) as the data stage wrote it."""
    return pd.read_parquet(settings.out("data_folds") / f"site_days_{cohort}.parquet")


def fold_manifest_path(settings: Settings) -> Path:
    return settings.out("data_folds") / "fold_manifest.csv"


def folds(settings: Settings) -> list[Fold]:
    """The Fold objects rebuilt from the written fold manifest, the single source of truth."""
    return folds_from_manifest(pd.read_csv(fold_manifest_path(settings)))


def interval_path(settings: Settings, method: str) -> Path:
    """Where a method's interval prediction table lives (M7 and M8 with the baselines, M9 in its own stage)."""
    stage = "m9" if method == "m9" else "baselines"
    return settings.out(stage) / f"intervals_{method}.parquet"


def intervals(settings: Settings, method: str) -> pd.DataFrame:
    return pd.read_parquet(interval_path(settings, method))


def scores(settings: Settings, cohort: str) -> pd.DataFrame:
    return pd.read_parquet(settings.out("m9") / f"scores_{cohort}.parquet")


def m9_site_days(settings: Settings) -> pd.DataFrame:
    return pd.read_parquet(settings.out("m9") / "site_days_m9.parquet")


def calibration_fits(settings: Settings) -> pd.DataFrame:
    return pd.read_csv(settings.out("m9") / "calibration_fits.csv")


def site_days_path(settings: Settings) -> Path:
    return settings.out("metrics") / "site_days.parquet"


def site_days(settings: Settings) -> pd.DataFrame:
    """The site-day decision-and-impact table of all methods (``reference.siteday_table``)."""
    return pd.read_parquet(site_days_path(settings))


def metric(settings: Settings, name: str) -> pd.DataFrame:
    """A metric table by file name (``pooled``, ``stations``, ``macro``, ``bootstrap``, ``coverage``, ...)."""
    return pd.read_csv(settings.out("metrics") / f"{name}.csv")


def gate(settings: Settings) -> dict:
    return json.loads((settings.out("metrics") / "gate.json").read_text(encoding="utf-8"))


def operating_points_dir(settings: Settings) -> Path:
    path = settings.out("metrics") / "operating_points"
    path.mkdir(parents=True, exist_ok=True)
    return path


def operating_point(settings: Settings, name: str) -> pd.DataFrame:
    """An operating-point study table by file name (``frontier``, ``targets``, ``selections``, ...)."""
    return pd.read_csv(operating_points_dir(settings) / f"{name}.csv")


def gamma_table(settings: Settings, name: str) -> pd.DataFrame:
    return pd.read_csv(settings.out("gamma") / f"{name}.csv")


def gamma_series(settings: Settings) -> pd.DataFrame:
    return pd.read_parquet(settings.out("gamma") / "gamma_series.parquet")


def paper_dir(settings: Settings, kind: str) -> Path:
    """``paper/figures`` or ``paper/tables`` under the output directory, created on first use."""
    path = settings.out("paper") / kind
    path.mkdir(parents=True, exist_ok=True)
    return path

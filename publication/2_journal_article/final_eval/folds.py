"""The eighteen leave-one-station-out folds and the manifest that proves the exclusions.

Inputs:  the site-day index of each cohort.
Outputs: a list of Fold objects and a fold manifest table (one row per fold) with
         the held-out station, the M8 training stations, the M9 calibration stations,
         record counts and a hash of the training keys.
Key steps: every station of every cohort is held out exactly once; M8 trains on all
         other stations of both cohorts (Beta 'sure' days only); M9 calibrates on the
         other stations of the same cohort, exactly as frozen in m9_dev.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd

from .config import Settings


@dataclass(frozen=True)
class Fold:
    """One outer fold: who is held out and who may be fitted on."""

    fold_id: str
    cohort: str
    held_out: str
    m8_training: tuple[tuple[str, str], ...] = field(default_factory=tuple)   # (cohort, station)
    m9_calibration: tuple[str, ...] = field(default_factory=tuple)             # stations, same cohort

    def training_stations(self) -> set[str]:
        return {station for _, station in self.m8_training}


def build_folds(indexes: dict[str, pd.DataFrame], settings: Settings) -> list[Fold]:
    """One fold per station across the configured cohorts, in cohort then station order."""
    stations = [(cohort, station) for cohort in settings["population"]["cohorts"]
                for station in sorted(indexes[cohort]["station"].unique())]
    folds = []
    for cohort, held_out in stations:
        others = tuple((c, s) for c, s in stations if s != held_out)
        same = tuple(s for c, s in stations if c == cohort and s != held_out)
        folds.append(Fold(fold_id=f"{cohort}_{held_out}", cohort=cohort, held_out=held_out,
                          m8_training=others, m9_calibration=same))
    return folds


def training_days(indexes: dict[str, pd.DataFrame], fold: Fold) -> pd.DataFrame:
    """Site-days M8 may fit on in this fold: complete, headline confidence, other stations."""
    parts = []
    for cohort, station in fold.m8_training:
        idx = indexes[cohort]
        parts.append(idx[(idx["station"] == station) & idx["complete"] & idx["headline"]])
    return pd.concat(parts, ignore_index=True)


def assert_no_leakage(fold: Fold, frame: pd.DataFrame, station_col: str) -> None:
    """Raise if the held-out station appears in a frame about to be fitted on."""
    present = set(frame[station_col].astype(str).unique())
    if fold.held_out in present:
        raise ValueError(f"Fold {fold.fold_id}: held-out station {fold.held_out} is present in a training frame.")
    unexpected = present - fold.training_stations()
    if unexpected:
        raise ValueError(f"Fold {fold.fold_id}: stations outside the training set: {sorted(unexpected)}")


def _keys_hash(days: pd.DataFrame) -> str:
    text = "\n".join(f"{c}|{s}|{d}" for c, s, d in days[["cohort", "station", "date"]].itertuples(index=False))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fold_manifest(folds: list[Fold], indexes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per fold with counts and a hash of the exact training site-day keys."""
    rows = []
    for fold in folds:
        held = indexes[fold.cohort]
        held = held[(held["station"] == fold.held_out) & held["complete"]]
        train = training_days(indexes, fold)
        cal = indexes[fold.cohort]
        cal = cal[cal["station"].isin(fold.m9_calibration) & cal["complete"] & cal["headline"]]
        rows.append(dict(
            fold_id=fold.fold_id, cohort=fold.cohort, held_out=fold.held_out,
            n_heldout_days=len(held), n_heldout_headline_days=int(held["headline"].sum()),
            n_heldout_rpf_days=int(held.loc[held["headline"], "rpf"].sum()),
            m8_training_stations=";".join(s for _, s in fold.m8_training),
            n_m8_training_days=len(train), n_m8_training_rpf_days=int(train["rpf"].sum()),
            m8_training_keys_sha256=_keys_hash(train),
            m9_calibration_stations=";".join(fold.m9_calibration),
            n_m9_calibration_days=len(cal), n_m9_calibration_rpf_days=int(cal["rpf"].sum()),
        ))
    return pd.DataFrame(rows)


def check_manifest(manifest: pd.DataFrame) -> None:
    """Every station held out exactly once, and never inside its own training set."""
    if manifest["held_out"].duplicated().any():
        raise ValueError("A station is held out in more than one fold.")
    for row in manifest.itertuples(index=False):
        if row.held_out in row.m8_training_stations.split(";") or row.held_out in row.m9_calibration_stations.split(";"):
            raise ValueError(f"Fold {row.fold_id} trains on its own held-out station.")

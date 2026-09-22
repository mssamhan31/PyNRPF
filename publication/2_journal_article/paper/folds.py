"""The eighteen leave-one-station-out folds and the manifest that proves the exclusions.

Inputs:  the site-day index of each cohort and the ``folds`` block of the settings.
Outputs: a list of Fold objects and a fold manifest table (one row per fold) with
         the held-out station, the M8 training stations, the M9 calibration stations,
         record counts, a hash of the training keys and the training signature that
         identifies which folds share one M8 bundle.
Key steps: every station of every cohort is held out exactly once. Who may be fitted
         on is a configured scope. ``all_other_stations_both_cohorts`` (M8) trains on
         every other station of both cohorts, Beta 'sure' days only;
         ``other_stations_same_cohort`` (M9) calibrates on the other stations of the
         same cohort; ``beta_only`` (either method, the reference run) fits on the
         other Beta stations for a Beta fold and on all Beta stations for an Alpha
         fold, so Alpha becomes a pure transfer cohort and its ten folds share one fit.
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
    m9_calibration: tuple[str, ...] = field(default_factory=tuple)             # stations, any cohort

    def training_stations(self) -> set[str]:
        return {station for _, station in self.m8_training}


# Who a fold may be fitted on. The first value of each pair is the default when the
# settings carry no ``folds`` block; the reference run names ``beta_only`` for both.
M8_SCOPES = ("all_other_stations_both_cohorts", "beta_only")
M9_SCOPES = ("other_stations_same_cohort", "beta_only")
FITTING_COHORT = "beta"   # the cohort that ``beta_only`` names


def fold_scopes(settings: Settings) -> tuple[str, str]:
    """The configured (M8 training scope, M9 calibration scope), validated.

    Args:
        settings: the evaluation settings; ``folds.m8_training_scope`` and
            ``folds.m9_calibration_scope`` are optional and default to the first
            value of ``M8_SCOPES`` and ``M9_SCOPES``.

    Returns:
        A pair of scope names, each one of ``M8_SCOPES`` or ``M9_SCOPES``.
    """
    block = settings.raw.get("folds", {}) or {}
    m8_scope = block.get("m8_training_scope", M8_SCOPES[0])
    m9_scope = block.get("m9_calibration_scope", M9_SCOPES[0])
    if m8_scope not in M8_SCOPES:
        raise ValueError(f"folds.m8_training_scope must be one of {M8_SCOPES}, not {m8_scope!r}.")
    if m9_scope not in M9_SCOPES:
        raise ValueError(f"folds.m9_calibration_scope must be one of {M9_SCOPES}, not {m9_scope!r}.")
    return m8_scope, m9_scope


def _fitting_stations(scope: str, cohort: str, held_out: str,
                      stations: list[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """(cohort, station) pairs a fold may fit on under one scope; never the held-out station."""
    if scope == "all_other_stations_both_cohorts":
        return tuple((c, s) for c, s in stations if s != held_out)
    if scope == "other_stations_same_cohort":
        return tuple((c, s) for c, s in stations if c == cohort and s != held_out)
    if scope == "beta_only":
        return tuple((c, s) for c, s in stations if c == FITTING_COHORT and s != held_out)
    raise ValueError(f"Unknown fitting scope {scope!r}.")


def build_folds(indexes: dict[str, pd.DataFrame], settings: Settings) -> list[Fold]:
    """One fold per station across the configured cohorts, in cohort then station order.

    Args:
        indexes: site-day index per cohort (``data.siteday_index``).
        settings: the evaluation settings; the ``folds`` block chooses the fitting scopes.

    Returns:
        The folds, each naming its held-out station, its M8 training stations and its
        M9 calibration stations according to the configured scopes.
    """
    m8_scope, m9_scope = fold_scopes(settings)
    stations = [(cohort, station) for cohort in settings["population"]["cohorts"]
                for station in sorted(indexes[cohort]["station"].unique())]
    folds = []
    for cohort, held_out in stations:
        m8_training = _fitting_stations(m8_scope, cohort, held_out, stations)
        m9_calibration = tuple(s for _, s in _fitting_stations(m9_scope, cohort, held_out, stations))
        folds.append(Fold(fold_id=f"{cohort}_{held_out}", cohort=cohort, held_out=held_out,
                          m8_training=m8_training, m9_calibration=m9_calibration))
    return folds


def training_signature(fold: Fold) -> str:
    """Identity of a fold's M8 training set: SHA-256 of its sorted (cohort, station) pairs.

    Two folds with the same signature train on the same rows, so one bundle serves
    both; under ``beta_only`` the ten Alpha folds share a signature.

    Args:
        fold: the fold.

    Returns:
        A 64-character hexadecimal digest.
    """
    text = "\n".join(f"{c}|{s}" for c, s in sorted(fold.m8_training))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    """One row per fold with counts, a hash of the exact training site-day keys and the
    training signature (folds with equal signatures share one M8 bundle)."""
    rows = []
    for fold in folds:
        held = indexes[fold.cohort]
        held = held[(held["station"] == fold.held_out) & held["complete"]]
        train = training_days(indexes, fold)
        # Calibration stations may lie in another cohort (beta_only), so pool every index.
        cal = pd.concat(indexes.values(), ignore_index=True)
        cal = cal[cal["station"].isin(fold.m9_calibration) & cal["complete"] & cal["headline"]]
        rows.append(dict(
            fold_id=fold.fold_id, cohort=fold.cohort, held_out=fold.held_out,
            n_heldout_days=len(held), n_heldout_headline_days=int(held["headline"].sum()),
            n_heldout_rpf_days=int(held.loc[held["headline"], "rpf"].sum()),
            m8_training_stations=";".join(s for _, s in fold.m8_training),
            n_m8_training_days=len(train), n_m8_training_rpf_days=int(train["rpf"].sum()),
            m8_training_keys_sha256=_keys_hash(train),
            m8_training_signature=training_signature(fold),
            m9_calibration_stations=";".join(fold.m9_calibration),
            n_m9_calibration_days=len(cal), n_m9_calibration_rpf_days=int(cal["rpf"].sum()),
        ))
    return pd.DataFrame(rows)


def check_manifest(manifest: pd.DataFrame) -> None:
    """Every station held out exactly once, and never inside its own training set."""
    if manifest["held_out"].duplicated().any():
        raise ValueError("A station is held out in more than one fold.")
    for row in manifest.itertuples(index=False):
        fitted_on = row.m8_training_stations.split(";") + row.m9_calibration_stations.split(";")
        if row.held_out in fitted_on:
            raise ValueError(f"Fold {row.fold_id} trains on its own held-out station.")


def folds_from_manifest(manifest: pd.DataFrame) -> list[Fold]:
    """Rebuild the Fold objects from a fold manifest (the single source of truth once written)."""
    folds = []
    for row in manifest.itertuples(index=False):
        stations = row.m8_training_stations.split(";")
        cohorts = {s: ("alpha" if s.startswith("alpha") else "beta") for s in stations}
        folds.append(Fold(fold_id=row.fold_id, cohort=row.cohort, held_out=row.held_out,
                          m8_training=tuple((cohorts[s], s) for s in stations),
                          m9_calibration=tuple(row.m9_calibration_stations.split(";"))))
    return folds

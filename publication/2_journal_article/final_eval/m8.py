"""M8, the two-stage XGBoost classifier: one bundle per training set, then held-out prediction.

Inputs:  the complete-day interval frames of both cohorts, the folds, the settings.
Outputs: <output_dir>/02_bundles/<fold_id>/... (bundle.pkl, gitignored) for every fold
         that fits, and 02_bundles/<fold_id>.json (kept) for every fold; the M8 interval
         prediction table.
Key steps: assemble the training frame from the fold's training stations (Beta 'sure'
         days only), assert the held-out station is absent, train through
         ``pynrpf.api.train_m8_xgb`` with the frozen thresholds and the configured fit
         and validation windows; predict the held-out station through
         ``pynrpf.api.run_inference`` with the fold's bundle.

Training is the only heavy stage of the evaluation. It is resumable: a fold whose
bundle manifest exists is skipped unless ``force`` is set. Folds with the same training
signature (``folds.training_signature``) share one bundle: the first such fold fits, the
others get a manifest pointing at its bundle with ``shared_with`` naming the donor. Under
the Phase 3 scope every fold has its own signature (18 fits); under ``beta_only`` the ten
Alpha folds share one (9 fits).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .common import INTERVAL_COLUMNS, finish_interval_table
from .config import Settings, ensure_pynrpf_importable
from .data import model_input
from .folds import Fold, assert_no_leakage, training_signature
from .m7 import inference_config, run_package


def bundle_manifest_path(settings: Settings, fold: Fold) -> Path:
    """Path of the fold's kept bundle manifest under ``02_bundles/``."""
    return settings.out("bundles") / f"{fold.fold_id}.json"


def _read_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_shared_bundle(settings: Settings, fold: Fold, signature: str) -> dict[str, Any] | None:
    """The manifest of another fold in this run that already fitted this training set.

    Args:
        settings: the evaluation settings (the run is its output directory).
        fold: the fold about to be trained; its own manifest is ignored.
        signature: ``training_signature(fold)``.

    Returns:
        The donor's bundle manifest when one exists with the same signature and a
        bundle on disk; None otherwise. Manifests without a signature (Phase 3 layout)
        never match.
    """
    for path in sorted(settings.out("bundles").glob("*.json")):
        if path.name == f"{fold.fold_id}.json":
            continue
        record = _read_record(path)
        if record.get("training_signature") != signature or record.get("shared_with"):
            continue
        if (settings.root / record["bundle"]).exists():
            return record
    return None


def shared_record(fold: Fold, donor: dict[str, Any]) -> dict[str, Any]:
    """The bundle manifest of a fold that reuses a donor fold's bundle.

    The training rows, validation metrics and bundle path are the donor's, because the
    bundle is the same object; ``shared_with`` names the donor and ``elapsed_s`` is zero.
    """
    record = dict(donor)
    record.update(fold_id=fold.fold_id, held_out=fold.held_out, shared_with=donor["fold_id"],
                  elapsed_s=0.0, skipped=False)
    return record


def training_frame(fold: Fold, intervals: dict[str, pd.DataFrame], settings: Settings) -> pd.DataFrame:
    """Labelled package input for the fold's training stations: complete, headline-confidence days."""
    headline = set(settings["population"]["headline_confidence"])
    parts = []
    for cohort, station in fold.m8_training:
        df = intervals[cohort]
        parts.append(df[(df["station"] == station) & df["confidence"].isin(headline)])
    train = pd.concat(parts, ignore_index=True)
    frame = model_input(train, settings, with_labels=True)
    assert_no_leakage(fold, frame, settings.columns["site"])
    return frame


def training_config(settings: Settings, bundle_dir: Path) -> dict[str, Any]:
    """Inference plus training blocks in the layout ``train_m8_xgb`` accepts."""
    cols = settings.columns
    m8 = settings["m8"]
    return {
        "pynrpf_inference": inference_config(settings, "m8_xgb"),
        "pynrpf_training": {
            "model_id": "m8_xgb",
            "labels": {"day": cols["label_day"], "interval": cols["label_interval"]},
            "split": dict(m8["split"]),
            "thresholds": dict(m8["thresholds"]),
            "random_seed": int(m8["random_seed"]),
            "output": {"base_uri": str(bundle_dir)},
        },
    }


def train_fold(fold: Fold, intervals: dict[str, pd.DataFrame], settings: Settings, force: bool = False,
               trained: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Train the fold's bundle unless one already serves it; return the bundle manifest.

    Args:
        fold: the fold to train.
        intervals: complete-day interval frame per cohort (``data.load_population``).
        settings: the evaluation settings.
        force: retrain even if the fold's manifest and bundle exist. Bundles fitted
            earlier in the same invocation are still shared.
        trained: bundle manifests fitted so far in this invocation, keyed by training
            signature; updated in place when this fold fits. A fold whose signature is
            present shares that bundle instead of fitting again.

    Returns:
        The fold's bundle manifest: training stations, row and RPF-day counts, the
        repository-relative bundle path, in-bundle validation metrics, ``elapsed_s``
        (seconds), ``skipped`` (an existing manifest was reused) and ``shared_with``
        (the donor fold id, or None when this fold fitted its own bundle).
    """
    manifest_path = bundle_manifest_path(settings, fold)
    if manifest_path.exists() and not force:
        record = _read_record(manifest_path)
        if (settings.root / record["bundle"]).exists():
            record["skipped"] = True
            return record
    signature = training_signature(fold)
    trained = trained if trained is not None else {}
    donor = trained.get(signature) or (None if force else find_shared_bundle(settings, fold, signature))
    if donor is not None:
        record = shared_record(fold, donor)
        manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return record
    ensure_pynrpf_importable()
    from pynrpf.api import train_m8_xgb

    started = time.time()
    frame = training_frame(fold, intervals, settings)
    bundle_dir = settings.out("bundles") / fold.fold_id
    result = train_m8_xgb(frame, training_config(settings, bundle_dir))
    record = {
        "fold_id": fold.fold_id, "held_out": fold.held_out,
        "training_stations": [s for _, s in fold.m8_training],
        "training_signature": signature,
        "shared_with": None,
        "n_training_rows": int(len(frame)),
        "n_training_rpf_days": int(frame.groupby([settings.columns["site"], frame[settings.columns["timestamp"]].dt.date])[settings.columns["label_day"]].any().sum()),
        "bundle": settings.relative(result["resolved_artifact_path"] if "resolved_artifact_path" in result else result["artifact_uri"]),
        "validation_metrics": result["validation_metrics"],
        "elapsed_s": round(time.time() - started, 1),
        "skipped": False,
    }
    manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    trained[signature] = record
    return record


def bundle_uri(settings: Settings, fold: Fold) -> str:
    """Absolute bundle path for a fold, from its kept manifest; raises if training has not run.

    A shared bundle resolves through the fold's own manifest, which carries the donor's path.
    """
    path = bundle_manifest_path(settings, fold)
    if not path.exists():
        raise FileNotFoundError(f"No M8 bundle for fold {fold.fold_id}: run the train-m8 stage (notebook 02) first.")
    record = _read_record(path)
    bundle = settings.root / record["bundle"]
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle listed for fold {fold.fold_id} is missing: {record['bundle']}")
    return str(bundle)


def predict_station(intervals: pd.DataFrame, fold: Fold, settings: Settings) -> pd.DataFrame:
    """M8 interval predictions for one held-out station from its fold's bundle."""
    station = intervals[intervals["station"] == fold.held_out]
    merged = run_package(station, settings, inference_config(settings, "m8_xgb", bundle_uri(settings, fold)))
    table = station[["cohort", "station", "date", "slot", "ts"]].copy()
    table["pred_interval"] = merged["pynrpf_interval_flag"].fillna(False).astype(bool).to_numpy()
    table["pred_day"] = merged["pynrpf_day_flag"].fillna(False).astype(bool).to_numpy()
    table["prob_day"] = merged["m8_prob_day"].to_numpy(float)
    table["prob_interval"] = merged["m8_prob_ts"].to_numpy(float)
    return finish_interval_table(table, "m8", fold.fold_id)


def predict_cohort(intervals: pd.DataFrame, folds: list[Fold], settings: Settings) -> pd.DataFrame:
    """M8 interval predictions for every held-out station of one cohort."""
    cohort = intervals["cohort"].iloc[0]
    parts = [predict_station(intervals, f, settings) for f in folds if f.cohort == cohort]
    return pd.concat(parts, ignore_index=True)[INTERVAL_COLUMNS]

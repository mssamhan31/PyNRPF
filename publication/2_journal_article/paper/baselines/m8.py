"""M8, the two-stage XGBoost classifier: one bundle per training set, then held-out prediction.

Stage one (``xgb1_day``) classifies a site-day from its shape; stage two
(``xgb2_timestamp``) classifies each daytime interval of the days stage one flagged.
Both are fitted on the fold's training stations and applied once to its held-out
station. A bundle is a local pickle holding the two fitted models, their feature
columns and thresholds, and the in-bundle validation metrics.

Inputs:  the complete-day interval frames of both cohorts, the folds, the settings
         (``m8``: split windows, thresholds, random seed, classifier parameters).
Outputs: ``<output_dir>/02_baselines/bundles/<fold_id>/bundle.pkl`` (gitignored) for
         every fold that fits, ``02_baselines/<fold_id>.json`` (kept) for every fold,
         and the M8 interval prediction table in the common schema.
Key steps: assemble the training frame from the fold's training stations (headline
         confidence days only), assert the held-out station is absent, fit both stages
         with the configured fit and validation windows, predict the held-out station.

Training is the only heavy stage of the evaluation. It is resumable: a fold whose
bundle manifest and bundle exist is skipped unless ``force`` is set. Folds with the
same training signature (``folds.training_signature``) share one bundle: the first such
fold fits, the others get a manifest pointing at its bundle with ``shared_with`` naming
the donor. Under ``beta_only`` the ten Alpha folds share one bundle, so nine fits serve
eighteen folds.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from ..config import Settings
from ..data import INTERVAL_COLUMNS, finish_interval_table, model_input
from ..folds import Fold, assert_no_leakage, training_signature
from . import check_frame
from .m8_features import build_xgb1_features, build_xgb2_features

# ----------------------------------------------------------------------------- the classifier

def make_classifier(params: dict[str, Any]) -> xgb.XGBClassifier:
    """An ``XGBClassifier`` from a stage's parameter block (``eta`` is the learning rate)."""
    return xgb.XGBClassifier(
        objective=params.get("objective", "binary:logistic"),
        eval_metric=params.get("eval_metric", "aucpr"),
        tree_method=params.get("tree_method", "hist"),
        learning_rate=params.get("eta", 0.1),
        n_estimators=int(params.get("n_estimators", 500)),
        max_depth=int(params.get("max_depth", 6)),
        min_child_weight=int(params.get("min_child_weight", 3)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        scale_pos_weight=float(params.get("scale_pos_weight", 5)),
        random_state=int(params.get("seed", 9)),
        missing=np.nan,
    )


def _align_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """The feature matrix in training order; a column absent from this batch (a station one-hot) is 0."""
    aligned = df.copy()
    for col in [c for c in feature_columns if c not in aligned.columns]:
        aligned[col] = 0.0
    return aligned[feature_columns]


def _date_window_mask(values: pd.Series, start_date: str, end_date: str) -> pd.Series:
    dates = pd.to_datetime(values).dt.normalize()
    return (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))


def _binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    """Confusion counts, precision, recall and F1 of ``prob >= threshold`` against binary labels."""
    pred = (prob >= threshold).astype(np.uint8)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"rows": int(len(y_true)), "positives": int(y_true.sum()), "threshold": float(threshold),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _stage_params(settings: Settings, stage: str) -> dict[str, Any]:
    """A stage's parameter block with the configured threshold and seed applied."""
    params = dict(settings["m8"]["m8_xgb"][stage])
    params["threshold"] = float(settings["m8"]["thresholds"][stage])
    params["seed"] = int(settings["m8"]["random_seed"])
    return params


# ----------------------------------------------------------------------------- fit and predict

def fit_bundle(frame: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    """Fit both stages on a labelled frame and return the bundle.

    Args:
        frame: interval rows in the configuration's column names with the day and
            interval label columns (``data.model_input(..., with_labels=True)``).
        settings: the evaluation settings; ``m8.split`` gives the fit window
            (train_start .. train_end) and the validation window, ``m8.thresholds``
            the decision thresholds, ``m8.random_seed`` the seed of both stages.

    Returns:
        The bundle: per stage the fitted model, its feature columns and threshold;
        the validation metrics of both stages; the split and the seed.
    """
    check_frame(frame, settings.columns)
    cols = settings.columns
    m8_cfg = settings["m8"]["m8_xgb"]
    split = settings["m8"]["split"]
    train_start, train_end = str(split["train_start"]), str(split["train_end"])
    validation_start, validation_end = str(split["validation_start"]), str(split["validation_end"])
    xgb1_params, xgb2_params = _stage_params(settings, "xgb1_day"), _stage_params(settings, "xgb2_timestamp")
    holiday_range = (train_start, validation_end)

    # Stage one: day labels encoded as the feature builder expects (negative = wrong sign).
    day_work = frame.copy()
    day_work["_gt_day"] = np.where(day_work[cols["label_day"]].fillna(False).astype(bool), -1.0, 1.0)
    day_df, feat_cols1, label_col1 = build_xgb1_features(day_work, m8_cfg, cols, "_gt_day", holiday_range)
    day_dates = pd.to_datetime(day_df["date"])
    is_train_day = _date_window_mask(day_dates, train_start, train_end)
    is_validation_day = _date_window_mask(day_dates, validation_start, validation_end)
    if not is_train_day.any() or not is_validation_day.any():
        raise ValueError("The fit or the validation window holds no site-day.")
    day_train, day_val = day_df.loc[is_train_day], day_df.loc[is_validation_day]
    clf1 = make_classifier(xgb1_params)
    clf1.fit(day_train[feat_cols1].to_numpy(dtype=np.float32), day_train[label_col1].to_numpy(dtype=np.uint8))
    prob1_val = clf1.predict_proba(day_val[feat_cols1].to_numpy(dtype=np.float32))[:, 1]
    metrics_day = _binary_metrics(day_val[label_col1].to_numpy(dtype=np.uint8), prob1_val, xgb1_params["threshold"])

    # Stage two: interval labels, on the days whose day label is positive.
    interval_work = frame.copy()
    wrong_sign = interval_work[cols["label_interval"]].fillna(False).astype(bool)
    interval_work["_gt_interval"] = np.where(wrong_sign, -1.0, 1.0)
    positive_keys = day_df.loc[day_df[label_col1] == 1, [cols["site"], "date"]].copy()
    ts_df, feat_cols2, label_col2 = build_xgb2_features(interval_work, m8_cfg, cols, "_gt_interval", day_df,
                                                        positive_keys)
    if ts_df.empty:
        raise ValueError("No interval training rows for stage two: check the labels and daytime coverage.")
    ts_dates = pd.to_datetime(ts_df["date"])
    is_train_ts = _date_window_mask(ts_dates, train_start, train_end)
    is_validation_ts = _date_window_mask(ts_dates, validation_start, validation_end)
    if not is_train_ts.any() or not is_validation_ts.any():
        raise ValueError("The fit or the validation window holds no interval row.")
    ts_train, ts_val = ts_df.loc[is_train_ts], ts_df.loc[is_validation_ts]
    clf2 = make_classifier(xgb2_params)
    clf2.fit(ts_train[feat_cols2].to_numpy(dtype=np.float32), ts_train[label_col2].to_numpy(dtype=np.uint8))
    prob2_val = clf2.predict_proba(ts_val[feat_cols2].to_numpy(dtype=np.float32))[:, 1]
    metrics_interval = _binary_metrics(ts_val[label_col2].to_numpy(dtype=np.uint8), prob2_val, xgb2_params["threshold"])

    return {
        "xgb1_day": {"model": clf1, "feature_columns": feat_cols1, "threshold": xgb1_params["threshold"]},
        "xgb2_timestamp": {"model": clf2, "feature_columns": feat_cols2, "threshold": xgb2_params["threshold"]},
        "validation_metrics": {"xgb1_day": metrics_day, "xgb2_timestamp": metrics_interval},
        "split": {"train_start": train_start, "train_end": train_end,
                  "validation_start": validation_start, "validation_end": validation_end},
        "random_seed": int(settings["m8"]["random_seed"]),
    }


def predict(frame: pd.DataFrame, bundle: dict[str, Any], settings: Settings) -> pd.DataFrame:
    """Score an interval frame with a bundle.

    Args:
        frame: interval rows in the configuration's column names.
        bundle: a bundle from ``fit_bundle`` (or loaded with ``load_bundle``).
        settings: the evaluation settings (``m8.m8_xgb`` feature parameters).

    Returns:
        One row per input row, in input order: ``m8_prob_day`` and ``m8_rpf_day``
        (stage one), ``m8_prob_ts`` and ``m8_rpf_flag`` (stage two; NaN and False on
        the intervals of days stage one did not flag).
    """
    check_frame(frame, settings.columns)
    cols = settings.columns
    site_col, ts_col = cols["site"], cols["timestamp"]
    m8_cfg = settings["m8"]["m8_xgb"]
    score_all_days = bool(m8_cfg.get("xgb2_timestamp", {}).get("score_all_days_for_review", False))
    clf1, feat_cols1, thr1 = (bundle["xgb1_day"][k] for k in ("model", "feature_columns", "threshold"))
    clf2, feat_cols2, thr2 = (bundle["xgb2_timestamp"][k] for k in ("model", "feature_columns", "threshold"))

    work = frame.copy()
    work["_gt_dummy"] = 1.0                      # no labels at prediction time
    ts = pd.to_datetime(work[ts_col])
    holiday_range = (ts.min().date().isoformat(), ts.max().date().isoformat())
    day_df, _, _ = build_xgb1_features(work, m8_cfg, cols, "_gt_dummy", holiday_range)
    prob_day = clf1.predict_proba(_align_features(day_df, feat_cols1).to_numpy(dtype=np.float32))[:, 1]
    day_df["m8_prob_day"] = prob_day
    day_df["m8_rpf_day"] = (prob_day >= thr1).astype(bool)

    candidate_days = day_df if score_all_days else day_df.loc[day_df["m8_rpf_day"]]
    candidate_keys = candidate_days[[site_col, "date"]]
    ts_results = pd.DataFrame(columns=[site_col, ts_col, "m8_prob_ts", "m8_rpf_flag"])
    if not candidate_keys.empty:
        ts_df, _, _ = build_xgb2_features(work, m8_cfg, cols, "_gt_dummy", day_df, candidate_keys.copy())
        if not ts_df.empty:
            prob_ts = clf2.predict_proba(_align_features(ts_df, feat_cols2).to_numpy(dtype=np.float32))[:, 1]
            ts_df["m8_prob_ts"] = prob_ts
            ts_df["m8_rpf_flag"] = (prob_ts >= thr2).astype(bool)
            ts_results = ts_df[[site_col, ts_col, "m8_prob_ts", "m8_rpf_flag"]].copy()

    result = frame[[site_col, ts_col]].copy()
    result["date"] = pd.to_datetime(result[ts_col]).dt.date
    day_map = day_df.set_index([site_col, "date"])[["m8_rpf_day", "m8_prob_day"]]
    idx = result.set_index([site_col, "date"]).index
    result["m8_prob_day"] = idx.map(day_map["m8_prob_day"]).values
    day_flag = pd.Series(idx.map(day_map["m8_rpf_day"]).values)
    result["m8_rpf_day"] = day_flag.map(lambda v: bool(v) if pd.notna(v) else False).to_numpy()
    if ts_results.empty:
        result["m8_prob_ts"] = np.nan
        result["m8_rpf_flag"] = False
    else:
        result = result.merge(ts_results, on=[site_col, ts_col], how="left")   # a left merge keeps the input order
        result["m8_rpf_flag"] = result["m8_rpf_flag"].map(lambda v: bool(v) if pd.notna(v) else False).astype(bool)
    if len(result) != len(frame):
        raise ValueError("M8 output does not align one-to-one with the input intervals.")
    return result[["m8_prob_day", "m8_rpf_day", "m8_prob_ts", "m8_rpf_flag"]].reset_index(drop=True)


# ----------------------------------------------------------------------------- bundles on disk

def bundle_path(settings: Settings, fold: Fold) -> Path:
    """Where the fold's own bundle lives (gitignored): ``02_baselines/bundles/<fold_id>/bundle.pkl``."""
    return settings.out("baselines") / "bundles" / fold.fold_id / "bundle.pkl"


def bundle_manifest_path(settings: Settings, fold: Fold) -> Path:
    """Path of the fold's kept bundle manifest, ``02_baselines/<fold_id>.json``."""
    return settings.out("baselines") / f"{fold.fold_id}.json"


def save_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(bundle))


def load_bundle(path: Path) -> dict[str, Any]:
    bundle = pickle.loads(Path(path).read_bytes())
    if not isinstance(bundle, dict):
        raise TypeError(f"{path} does not hold a bundle.")
    return bundle


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
        bundle on disk; None otherwise. A manifest that itself shares is never a donor.
    """
    for path in sorted(settings.out("baselines").glob("*.json")):
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
    """Labelled input for the fold's training stations: complete, headline-confidence days."""
    headline = set(settings["population"]["headline_confidence"])
    parts = []
    for cohort, station in fold.m8_training:
        df = intervals[cohort]
        parts.append(df[(df["station"] == station) & df["confidence"].isin(headline)])
    train = pd.concat(parts, ignore_index=True)
    frame = model_input(train, settings, with_labels=True)
    assert_no_leakage(fold, frame, settings.columns["site"])
    return frame


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

    started = time.time()
    frame = training_frame(fold, intervals, settings)
    bundle = fit_bundle(frame, settings)
    path = bundle_path(settings, fold)
    save_bundle(bundle, path)
    cols = settings.columns
    day_key = [cols["site"], frame[cols["timestamp"]].dt.date]
    record = {
        "fold_id": fold.fold_id, "held_out": fold.held_out,
        "training_stations": [s for _, s in fold.m8_training],
        "training_signature": signature,
        "shared_with": None,
        "n_training_rows": int(len(frame)),
        "n_training_rpf_days": int(frame.groupby(day_key)[cols["label_day"]].any().sum()),
        "bundle": settings.relative(path),
        "validation_metrics": bundle["validation_metrics"],
        "elapsed_s": round(time.time() - started, 1),
        "skipped": False,
    }
    manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    trained[signature] = record
    return record


def fold_bundle(settings: Settings, fold: Fold) -> dict[str, Any]:
    """The bundle serving a fold, from its kept manifest; raises if training has not run.

    A shared bundle resolves through the fold's own manifest, which carries the donor's path.
    """
    path = bundle_manifest_path(settings, fold)
    if not path.exists():
        raise FileNotFoundError(f"No M8 bundle for fold {fold.fold_id}: run the baselines stage (notebook 02) first.")
    record = _read_record(path)
    bundle = settings.root / record["bundle"]
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle listed for fold {fold.fold_id} is missing: {record['bundle']}")
    return load_bundle(bundle)


# ----------------------------------------------------------------------------- held-out prediction

def predict_station(intervals: pd.DataFrame, fold: Fold, settings: Settings) -> pd.DataFrame:
    """M8 interval predictions for one held-out station from its fold's bundle."""
    station = intervals[intervals["station"] == fold.held_out]
    scored = predict(model_input(station, settings), fold_bundle(settings, fold), settings)
    table = station[["cohort", "station", "date", "slot", "ts"]].copy()
    table["pred_interval"] = scored["m8_rpf_flag"].to_numpy(bool)
    table["pred_day"] = scored["m8_rpf_day"].to_numpy(bool)
    table["prob_day"] = scored["m8_prob_day"].to_numpy(float)
    table["prob_interval"] = scored["m8_prob_ts"].to_numpy(float)
    return finish_interval_table(table, "m8", fold.fold_id)


def predict_cohort(intervals: pd.DataFrame, folds: list[Fold], settings: Settings) -> pd.DataFrame:
    """M8 interval predictions for every held-out station of one cohort."""
    cohort = intervals["cohort"].iloc[0]
    parts = [predict_station(intervals, f, settings) for f in folds if f.cohort == cohort]
    return pd.concat(parts, ignore_index=True)[INTERVAL_COLUMNS]

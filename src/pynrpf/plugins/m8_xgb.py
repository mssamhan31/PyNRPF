from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from .._legacy.features import build_xgb1_features, build_xgb2_features
from ..artifacts import load_artifact_bundle
from .base import BaseModelPlugin


def _make_clf(params: Dict[str, Any]) -> xgb.XGBClassifier:
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
    aligned = df.copy()
    missing = [c for c in feature_columns if c not in aligned.columns]
    for col in missing:
        aligned[col] = np.nan
    return aligned[feature_columns]


def _feature_cfg(df: pd.DataFrame, ts_col: str, m8_cfg: dict[str, Any]) -> dict[str, Any]:
    ts = pd.to_datetime(df[ts_col])
    min_date = ts.min().date().isoformat()
    max_date = ts.max().date().isoformat()
    return {
        "m8_xgb": m8_cfg,
        "split": {
            "train_start": min_date,
            "test_end": max_date,
        },
    }


def _bundle_section(
    bundle: dict[str, Any],
    key: str,
    fallback_threshold: float,
) -> tuple[Any, list[str], float]:
    sec = bundle.get(key)
    if not isinstance(sec, dict):
        raise ValueError(f"Artifact bundle is missing '{key}' section.")

    model = sec.get("model")
    if model is None:
        raise ValueError(f"Artifact bundle section '{key}' does not contain a model.")

    feat_cols = sec.get("feature_columns", sec.get("feat_cols"))
    if not isinstance(feat_cols, list) or not feat_cols:
        raise ValueError(f"Artifact bundle section '{key}' does not contain feature_columns.")

    threshold = float(sec.get("threshold", fallback_threshold))
    return model, feat_cols, threshold


def _date_window_mask(values: pd.Series, start_date: str, end_date: str) -> pd.Series:
    dates = pd.to_datetime(values).dt.normalize()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return (dates >= start) & (dates <= end)


def _binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (prob >= threshold).astype(np.uint8)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "rows": int(len(y_true)),
        "positives": int(y_true.sum()),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


class M8XGBPlugin(BaseModelPlugin):
    name = "m8_xgb"

    def _load_bundle(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        artifact_cfg = cfg.get("artifacts", {})
        uri = artifact_cfg.get("m8_pretrained_bundle_uri")
        if not uri:
            raise ValueError(
                "m8_pretrained_bundle_uri is required for m8_xgb inference."
            )
        return load_artifact_bundle(uri)

    def run_inference(
        self,
        df: pd.DataFrame,
        cfg: Dict[str, Any],
        columns: Dict[str, str],
    ) -> pd.DataFrame:
        site_col = columns["site"]
        ts_col = columns["timestamp"]
        net_col = columns["net_load"]
        solar_col = columns["solar"]

        model_cfg = cfg.get("model", {})
        m8_cfg = model_cfg.get("m8_xgb", {})
        xgb1_cfg = m8_cfg.get("xgb1_day", {})
        xgb2_cfg = m8_cfg.get("xgb2_timestamp", {})

        bundle = self._load_bundle(cfg)
        clf1, feat_cols1, thr1 = _bundle_section(
            bundle, "xgb1_day", float(xgb1_cfg.get("threshold", 0.5))
        )
        clf2, feat_cols2, thr2 = _bundle_section(
            bundle, "xgb2_timestamp", float(xgb2_cfg.get("threshold", 0.5))
        )

        work = df.copy()
        work["_pynrpf_gt_dummy"] = 1.0
        feature_cfg = _feature_cfg(work, ts_col, m8_cfg)

        day_df, _, _ = build_xgb1_features(
            work,
            feature_cfg,
            site_col,
            ts_col,
            net_col,
            solar_col,
            "_pynrpf_gt_dummy",
        )
        X_day = _align_features(day_df, feat_cols1).to_numpy(dtype=np.float32)
        prob_day = clf1.predict_proba(X_day)[:, 1]
        day_df["m8_prob_day"] = prob_day
        day_df["m8_rpf_day"] = (prob_day >= thr1).astype(bool)

        pos_keys = day_df.loc[day_df["m8_rpf_day"], [site_col, "date"]].copy()
        ts_results: pd.DataFrame
        if pos_keys.empty:
            ts_results = pd.DataFrame(columns=[site_col, ts_col, "m8_prob_ts", "m8_rpf_flag"])
        else:
            ts_df, _, _ = build_xgb2_features(
                work,
                feature_cfg,
                day_df,
                pos_keys,
                site_col,
                ts_col,
                net_col,
                solar_col,
                "_pynrpf_gt_dummy",
            )
            if ts_df.empty:
                ts_results = pd.DataFrame(
                    columns=[site_col, ts_col, "m8_prob_ts", "m8_rpf_flag"]
                )
            else:
                X_ts = _align_features(ts_df, feat_cols2).to_numpy(dtype=np.float32)
                prob_ts = clf2.predict_proba(X_ts)[:, 1]
                ts_df["m8_prob_ts"] = prob_ts
                ts_df["m8_rpf_flag"] = (prob_ts >= thr2).astype(bool)
                ts_results = ts_df[[site_col, ts_col, "m8_prob_ts", "m8_rpf_flag"]].copy()

        result = df.copy()
        if "date" not in result.columns:
            result["date"] = pd.to_datetime(result[ts_col]).dt.date

        day_map = day_df.set_index([site_col, "date"])[["m8_rpf_day", "m8_prob_day"]]
        idx = result.set_index([site_col, "date"]).index
        result["m8_rpf_day"] = idx.map(day_map["m8_rpf_day"]).values
        result["m8_prob_day"] = idx.map(day_map["m8_prob_day"]).values
        result["m8_rpf_day"] = pd.Series(result["m8_rpf_day"]).fillna(False).astype(bool)

        if ts_results.empty:
            result["m8_prob_ts"] = np.nan
            result["m8_rpf_flag"] = False
        else:
            result = result.merge(ts_results, on=[site_col, ts_col], how="left")
            result["m8_rpf_flag"] = result["m8_rpf_flag"].fillna(False).astype(bool)

        result["net_load_MW_m8"] = np.where(
            result["m8_rpf_flag"], -result[net_col].values, result[net_col].values
        )
        if result[net_col].isna().any():
            result.loc[result[net_col].isna(), "net_load_MW_m8"] = np.nan

        result["pynrpf_interval_flag"] = result["m8_rpf_flag"]
        result["pynrpf_day_flag"] = result["m8_rpf_day"]
        result["pynrpf_confidence"] = result["m8_prob_ts"]
        result["pynrpf_corrected_net_load"] = result["net_load_MW_m8"]
        return result

    def train(
        self,
        df: pd.DataFrame,
        cfg: Dict[str, Any],
        columns: Dict[str, str],
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        labels = labels or {}
        day_label_col = labels.get("day")
        interval_label_col = labels.get("interval")
        if not day_label_col or not interval_label_col:
            raise ValueError(
                "m8_xgb training requires labels={'day': <col>, 'interval': <col>}."
            )
        if day_label_col not in df.columns or interval_label_col not in df.columns:
            raise KeyError(
                "Training label columns not found. "
                f"day={day_label_col}, interval={interval_label_col}"
            )

        training_cfg = cfg.get("training", {})
        if not isinstance(training_cfg, dict):
            raise TypeError("Training settings must be a mapping at cfg['training'].")
        split_cfg = training_cfg.get("split", {})
        if not isinstance(split_cfg, dict):
            raise TypeError(
                "Training split settings must be a mapping at cfg['training']['split']."
            )
        train_start = str(split_cfg.get("train_start", "")).strip()
        train_end = str(split_cfg.get("train_end", "")).strip()
        validation_start = str(split_cfg.get("validation_start", "")).strip()
        validation_end = str(split_cfg.get("validation_end", "")).strip()
        if not (train_start and train_end and validation_start and validation_end):
            raise ValueError(
                "Training split dates are required: train_start, train_end, "
                "validation_start, validation_end."
            )

        site_col = columns["site"]
        ts_col = columns["timestamp"]
        net_col = columns["net_load"]
        solar_col = columns["solar"]

        model_cfg = cfg.get("model", {})
        m8_cfg = model_cfg.get("m8_xgb", {})
        xgb1_cfg = m8_cfg.get("xgb1_day", {})
        xgb2_cfg = m8_cfg.get("xgb2_timestamp", {})
        feature_cfg = {
            "m8_xgb": m8_cfg,
            "split": {
                "train_start": train_start,
                "test_end": validation_end,
            },
        }
        thr1 = float(xgb1_cfg.get("threshold", 0.5))
        thr2 = float(xgb2_cfg.get("threshold", 0.5))

        # Stage 1 training labels (day).
        day_work = df.copy()
        day_work["_pynrpf_gt_day"] = np.where(
            day_work[day_label_col].fillna(False).astype(bool), -1.0, 1.0
        )
        day_df, feat_cols1, label_col1 = build_xgb1_features(
            day_work,
            feature_cfg,
            site_col,
            ts_col,
            net_col,
            solar_col,
            "_pynrpf_gt_day",
        )
        day_dates = pd.to_datetime(day_df["date"])
        is_train_day = _date_window_mask(day_dates, train_start, train_end)
        is_validation_day = _date_window_mask(day_dates, validation_start, validation_end)
        if not is_train_day.any():
            raise ValueError("No day-level rows found in training split window for m8_xgb.")
        if not is_validation_day.any():
            raise ValueError("No day-level rows found in validation split window for m8_xgb.")

        day_train = day_df.loc[is_train_day]
        day_val = day_df.loc[is_validation_day]
        X1 = day_train[feat_cols1].to_numpy(dtype=np.float32)
        y1 = day_train[label_col1].to_numpy(dtype=np.uint8)
        clf1 = _make_clf(xgb1_cfg)
        clf1.fit(X1, y1)
        X1_val = day_val[feat_cols1].to_numpy(dtype=np.float32)
        y1_val = day_val[label_col1].to_numpy(dtype=np.uint8)
        prob1_val = clf1.predict_proba(X1_val)[:, 1]
        val_metrics_day = _binary_metrics(y1_val, prob1_val, thr1)

        # Stage 2 training labels (interval).
        interval_work = df.copy()
        interval_work["_pynrpf_gt_interval"] = np.where(
            interval_work[interval_label_col].fillna(False).astype(bool), -1.0, 1.0
        )
        true_positive_keys = day_df.loc[day_df[label_col1] == 1, [site_col, "date"]].copy()
        ts_df, feat_cols2, label_col2 = build_xgb2_features(
            interval_work,
            feature_cfg,
            day_df,
            true_positive_keys,
            site_col,
            ts_col,
            net_col,
            solar_col,
            "_pynrpf_gt_interval",
        )
        if ts_df.empty:
            raise ValueError(
                "No interval training rows available for m8_xgb stage-2 model. "
                "Check labels and daytime coverage."
            )
        ts_dates = pd.to_datetime(ts_df["date"])
        is_train_ts = _date_window_mask(ts_dates, train_start, train_end)
        is_validation_ts = _date_window_mask(ts_dates, validation_start, validation_end)
        if not is_train_ts.any():
            raise ValueError("No interval rows found in training split window for m8_xgb.")
        if not is_validation_ts.any():
            raise ValueError("No interval rows found in validation split window for m8_xgb.")

        ts_train = ts_df.loc[is_train_ts]
        ts_val = ts_df.loc[is_validation_ts]
        X2 = ts_train[feat_cols2].to_numpy(dtype=np.float32)
        y2 = ts_train[label_col2].to_numpy(dtype=np.uint8)
        clf2 = _make_clf(xgb2_cfg)
        clf2.fit(X2, y2)
        X2_val = ts_val[feat_cols2].to_numpy(dtype=np.float32)
        y2_val = ts_val[label_col2].to_numpy(dtype=np.uint8)
        prob2_val = clf2.predict_proba(X2_val)[:, 1]
        val_metrics_interval = _binary_metrics(y2_val, prob2_val, thr2)

        return {
            "bundle_schema": "pynrpf.m8_xgb.bundle.v2",
            "model_name": self.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "xgb1_day": {
                "model": clf1,
                "feature_columns": feat_cols1,
                "threshold": thr1,
            },
            "xgb2_timestamp": {
                "model": clf2,
                "feature_columns": feat_cols2,
                "threshold": thr2,
            },
            "training_metadata": {
                "feature_pipeline": str(
                    training_cfg.get("features", {}).get("pipeline", "legacy_v1")
                ),
                "labels": {
                    "day": day_label_col,
                    "interval": interval_label_col,
                },
                "split": {
                    "train_start": train_start,
                    "train_end": train_end,
                    "validation_start": validation_start,
                    "validation_end": validation_end,
                },
                "random_seed": int(training_cfg.get("random_seed", xgb1_cfg.get("seed", 9))),
                "validation_metrics": {
                    "xgb1_day": val_metrics_day,
                    "xgb2_timestamp": val_metrics_interval,
                },
            },
        }

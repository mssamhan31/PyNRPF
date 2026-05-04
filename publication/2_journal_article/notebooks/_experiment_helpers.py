from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def find_article_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (
            candidate.name == "2_journal_article"
            and (candidate / "dataset" / "processed").exists()
        ):
            return candidate
        nested = candidate / "publication" / "2_journal_article"
        if (nested / "dataset" / "processed").exists():
            return nested.resolve()
    raise RuntimeError(f"Could not locate publication/2_journal_article from {start}")


def load_config(article_root: Path) -> dict[str, Any]:
    path = article_root / "config" / "experiment_config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg["_config_path"] = str(path)
    return cfg


def article_path(article_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else article_root / path


def experiment_output_dir(article_root: Path, cfg: dict[str, Any], experiment_id: str) -> Path:
    output_base = article_path(article_root, cfg["paths"]["output_base_dir"])
    folder = cfg["outputs"]["experiments"][experiment_id]
    return output_base / folder


VALID_METHODS = {"m8_xgb", "m7_dtr"}


def enabled_methods(cfg: dict[str, Any]) -> list[str]:
    methods = list(cfg.get("methods", {}).get("enabled", []))
    if not methods:
        raise ValueError("Configure at least one method under methods.enabled")
    invalid = sorted(set(methods) - VALID_METHODS)
    if invalid:
        raise ValueError(f"Unknown methods in methods.enabled: {invalid}")
    return methods


def method_enabled(cfg: dict[str, Any], method: str) -> bool:
    return method in enabled_methods(cfg)


def _resume_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("resume", {}).get("skip_completed", False))


def _overwrite_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("execution", {}).get("overwrite_outputs", False))


def _task_file_stem(fold_id: str, method: str) -> str:
    return f"{fold_id}__{method}"


def task_paths(
    out_dir: Path,
    cfg: dict[str, Any],
    fold_id: str,
    method: str,
) -> dict[str, Path]:
    stem = _task_file_stem(fold_id, method)
    return {
        "prediction": out_dir / cfg["outputs"]["predictions_dir"] / f"{stem}.csv",
        "metrics": out_dir / cfg["outputs"].get("metrics_dir", "metrics") / f"{stem}.csv",
        "status": out_dir / cfg["outputs"].get("status_dir", "status") / f"{stem}.yaml",
    }


def ensure_output_dirs(out_dir: Path, cfg: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / cfg["outputs"]["predictions_dir"]).mkdir(parents=True, exist_ok=True)
    (out_dir / cfg["outputs"].get("metrics_dir", "metrics")).mkdir(parents=True, exist_ok=True)
    (out_dir / cfg["outputs"].get("status_dir", "status")).mkdir(parents=True, exist_ok=True)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)
    return path


def _atomic_write_yaml(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    tmp_path.replace(path)
    return path


def task_is_complete(
    out_dir: Path,
    cfg: dict[str, Any],
    fold_id: str,
    method: str,
) -> bool:
    paths = task_paths(out_dir, cfg, fold_id, method)
    if not all(path.exists() for path in paths.values()):
        return False
    try:
        with paths["status"].open("r", encoding="utf-8") as handle:
            status = yaml.safe_load(handle) or {}
    except yaml.YAMLError:
        return False
    return status.get("status") == "complete"


def should_skip_task(
    out_dir: Path,
    cfg: dict[str, Any],
    fold_id: str,
    method: str,
) -> bool:
    return _resume_enabled(cfg) and not _overwrite_enabled(cfg) and task_is_complete(
        out_dir, cfg, fold_id, method
    )


def _begin_task(out_dir: Path, cfg: dict[str, Any], fold_id: str, method: str) -> None:
    status_path = task_paths(out_dir, cfg, fold_id, method)["status"]
    if status_path.exists():
        status_path.unlink()


def _planned_task(
    fold_id: str,
    method: str,
    dataset: str,
    partition: str,
) -> dict[str, str]:
    return {
        "fold_id": fold_id,
        "method": method,
        "dataset": dataset,
        "partition": partition,
    }


def _m8_has_incomplete_tasks(
    out_dir: Path,
    cfg: dict[str, Any],
    task_specs: list[dict[str, str]],
) -> bool:
    if not method_enabled(cfg, "m8_xgb"):
        return False
    return any(
        spec["method"] == "m8_xgb"
        and not should_skip_task(out_dir, cfg, spec["fold_id"], spec["method"])
        for spec in task_specs
    )


def _clear_expected_task_outputs(
    out_dir: Path,
    cfg: dict[str, Any],
    task_specs: list[dict[str, str]],
) -> None:
    if not _overwrite_enabled(cfg):
        return
    for spec in task_specs:
        for path in task_paths(out_dir, cfg, spec["fold_id"], spec["method"]).values():
            if path.exists():
                path.unlink()


def _run_control_details(cfg: dict[str, Any]) -> dict[str, Any]:
    methods = enabled_methods(cfg)
    ev = cfg.get("evaluation", {})
    return {
        "active_methods": methods,
        "m8_enabled": "m8_xgb" in methods,
        "m7_enabled": "m7_dtr" in methods,
        "resume_skip_completed": _resume_enabled(cfg),
        "overwrite_outputs": _overwrite_enabled(cfg),
        "interval_prediction_day_scope": ev.get(
            "interval_prediction_day_scope", "predicted_positive_days"
        ),
        "interval_metric_scope": ev.get("interval_metric_scope", "tp_days_only"),
        "interval_metric_level_name": ev.get(
            "interval_metric_level_name", "interval_tp_days_only"
        ),
    }


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    normalized = series.astype("string").fillna("").str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f", "", "nan", "none"}
    unknown = sorted(set(normalized.unique()) - truthy - falsy)
    if unknown:
        raise ValueError(f"Unknown boolean label values: {unknown}")
    return normalized.isin(truthy)


def load_dataset(article_root: Path, cfg: dict[str, Any], dataset_key: str) -> pd.DataFrame:
    if dataset_key == "actual":
        path = article_path(article_root, cfg["paths"]["actual_dataset_csv"])
    elif dataset_key == "synthetic":
        path = article_path(article_root, cfg["paths"]["synthetic_dataset_csv"])
    else:
        raise ValueError(f"Unknown dataset_key: {dataset_key}")

    cols = cfg["columns"]
    expected = [
        cols["site"],
        cols["date"],
        cols["timestamp"],
        cols["net_load"],
        cols["solar"],
        cols["label_interval"],
        cols["label_day"],
    ]
    df = pd.read_csv(path)
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing expected columns: {missing}")

    df = df[expected].copy()
    df[cols["timestamp"]] = pd.to_datetime(df[cols["timestamp"]], errors="raise")
    if getattr(df[cols["timestamp"]].dt, "tz", None) is not None:
        # Match the conference workflow: strip timezone without converting.
        df[cols["timestamp"]] = df[cols["timestamp"]].dt.tz_localize(None)
    df[cols["date"]] = df[cols["timestamp"]].dt.date
    df[cols["label_interval"]] = _coerce_bool(df[cols["label_interval"]])
    df[cols["label_day"]] = df.groupby([cols["site"], cols["date"]])[
        cols["label_interval"]
    ].transform("any")
    df = df.sort_values([cols["site"], cols["timestamp"]]).reset_index(drop=True)
    validate_dataset(df, cfg, dataset_key)
    return df


def validate_dataset(df: pd.DataFrame, cfg: dict[str, Any], dataset_name: str) -> None:
    cols = cfg["columns"]
    duplicate_count = int(df.duplicated([cols["site"], cols["timestamp"]], keep=False).sum())
    if duplicate_count:
        raise ValueError(
            f"{dataset_name} has duplicate ({cols['site']}, {cols['timestamp']}) rows: "
            f"{duplicate_count}"
        )
    if df[cols["timestamp"]].isna().any():
        raise ValueError(f"{dataset_name} has null timestamps")
    expected_date = df[cols["timestamp"]].dt.date
    if not (df[cols["date"]] == expected_date).all():
        bad = int((df[cols["date"]] != expected_date).sum())
        raise ValueError(f"{dataset_name} has date/timestamp mismatches: {bad}")
    recomputed = df.groupby([cols["site"], cols["date"]])[cols["label_interval"]].transform(
        "any"
    )
    if not (df[cols["label_day"]] == recomputed).all():
        bad = int((df[cols["label_day"]] != recomputed).sum())
        raise ValueError(f"{dataset_name} has stale label_day rows: {bad}")


def dataset_summary(df: pd.DataFrame, cfg: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    cols = cfg["columns"]
    site_day = df[[cols["site"], cols["date"], cols["label_day"]]].drop_duplicates()
    return {
        "dataset": dataset_name,
        "n_rows": int(len(df)),
        "n_stations": int(df[cols["site"]].nunique()),
        "min_timestamp": str(df[cols["timestamp"]].min()),
        "max_timestamp": str(df[cols["timestamp"]].max()),
        "n_dates": int(df[cols["date"]].nunique()),
        "null_timestamp": int(df[cols["timestamp"]].isna().sum()),
        "null_net_load_MW": int(df[cols["net_load"]].isna().sum()),
        "null_solar_MW": int(df[cols["solar"]].isna().sum()),
        "positive_label_interval": int(df[cols["label_interval"]].sum()),
        "positive_label_day_site_days": int(site_day[cols["label_day"]].sum()),
    }


def time_masks(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    date_col = cfg["columns"]["date"]
    dates = pd.to_datetime(df[date_col])
    split = cfg["split"]
    train = (dates >= pd.Timestamp(split["train_start"])) & (
        dates <= pd.Timestamp(split["train_end"])
    )
    test = (dates >= pd.Timestamp(split["test_start"])) & (
        dates <= pd.Timestamp(split["test_end"])
    )
    return train, test


def station_folds(df: pd.DataFrame, cfg: dict[str, Any], dataset_key: str) -> list[str]:
    cols = cfg["columns"]
    stations = sorted(df[cols["site"]].dropna().unique())
    excluded = set()
    if dataset_key == "synthetic" and cfg["station_split"].get("exclude_zero_positive_folds", True):
        excluded.update(cfg["station_split"].get("synthetic_excluded_stations", []))
    return [station for station in stations if station not in excluded]


def prepare_smoke_output(
    article_root: Path,
    cfg: dict[str, Any],
    experiment_id: str,
    notebook_name: str,
    details: dict[str, Any],
) -> Path | None:
    if not cfg.get("execution", {}).get("smoke_create_output_dirs", True):
        return None
    out_dir = experiment_output_dir(article_root, cfg, experiment_id)
    ensure_output_dirs(out_dir, cfg)
    manifest_path = out_dir / cfg["outputs"]["manifest_yaml"]
    manifest = {
        "status": "smoke_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "notebook": notebook_name,
        "config_path": cfg.get("_config_path"),
        "experiment_id": experiment_id,
        "run_control": _run_control_details(cfg),
        "output_structure": {
            "predictions_dir": str(out_dir / cfg["outputs"]["predictions_dir"]),
            "metrics_dir": str(out_dir / cfg["outputs"].get("metrics_dir", "metrics")),
            "status_dir": str(out_dir / cfg["outputs"].get("status_dir", "status")),
        },
        "details": details,
    }
    _atomic_write_yaml(manifest, manifest_path)
    return manifest_path


def _ensure_pynrpf_imports(article_root: Path) -> dict[str, Any]:
    repo_root = article_root.parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from pynrpf._legacy.features import build_xgb1_features, build_xgb2_features
    from pynrpf._legacy.m7_threshold import run_m7
    from pynrpf.plugins.m8_xgb import _align_features, _make_clf

    return {
        "build_xgb1_features": build_xgb1_features,
        "build_xgb2_features": build_xgb2_features,
        "run_m7": run_m7,
        "_align_features": _align_features,
        "_make_clf": _make_clf,
    }


def _feature_cfg(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    ts_col = cfg["columns"]["timestamp"]
    ts = pd.to_datetime(df[ts_col])
    return {
        "m8_xgb": cfg["m8_xgb"],
        "split": {
            "train_start": ts.min().date().isoformat(),
            "test_end": ts.max().date().isoformat(),
        },
    }


def _binary_classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    tn = int((~pred & ~true).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": int(true.sum()),
        "n_rows": int(len(true)),
    }


def train_m8_model(train_df: pd.DataFrame, cfg: dict[str, Any], article_root: Path) -> dict[str, Any]:
    tools = _ensure_pynrpf_imports(article_root)
    cols = cfg["columns"]
    m8_cfg = cfg["m8_xgb"]
    work = train_df.copy()
    work["_pynrpf_gt"] = np.where(work[cols["label_interval"]], -1.0, 1.0)
    feature_cfg = _feature_cfg(work, cfg)

    day_df, feat_cols1, label_col1 = tools["build_xgb1_features"](
        work,
        feature_cfg,
        cols["site"],
        cols["timestamp"],
        cols["net_load"],
        cols["solar"],
        "_pynrpf_gt",
    )
    y1 = day_df[label_col1].to_numpy(dtype=np.uint8)
    if len(np.unique(y1)) < 2:
        raise ValueError("M8 stage-1 training needs both positive and negative day labels.")
    clf1 = tools["_make_clf"](m8_cfg["xgb1_day"])
    clf1.fit(day_df[feat_cols1].to_numpy(dtype=np.float32), y1)

    true_positive_keys = day_df.loc[day_df[label_col1] == 1, [cols["site"], "date"]].copy()
    ts_df, feat_cols2, label_col2 = tools["build_xgb2_features"](
        work,
        feature_cfg,
        day_df,
        true_positive_keys,
        cols["site"],
        cols["timestamp"],
        cols["net_load"],
        cols["solar"],
        "_pynrpf_gt",
    )
    y2 = ts_df[label_col2].to_numpy(dtype=np.uint8)
    if len(np.unique(y2)) < 2:
        raise ValueError("M8 stage-2 training needs both positive and negative interval labels.")
    clf2 = tools["_make_clf"](m8_cfg["xgb2_timestamp"])
    clf2.fit(ts_df[feat_cols2].to_numpy(dtype=np.float32), y2)

    return {
        "xgb1_day": {
            "model": clf1,
            "feature_columns": feat_cols1,
            "threshold": float(m8_cfg["xgb1_day"]["threshold"]),
        },
        "xgb2_timestamp": {
            "model": clf2,
            "feature_columns": feat_cols2,
            "threshold": float(m8_cfg["xgb2_timestamp"]["threshold"]),
        },
        "training_rows": int(len(train_df)),
        "training_site_days": int(day_df.shape[0]),
        "training_interval_rows": int(ts_df.shape[0]),
    }


def predict_m8(
    model: dict[str, Any],
    eval_df: pd.DataFrame,
    cfg: dict[str, Any],
    article_root: Path,
) -> pd.DataFrame:
    tools = _ensure_pynrpf_imports(article_root)
    cols = cfg["columns"]
    work = eval_df.copy()
    work["_pynrpf_gt_dummy"] = 1.0
    feature_cfg = _feature_cfg(work, cfg)

    day_df, _, _ = tools["build_xgb1_features"](
        work,
        feature_cfg,
        cols["site"],
        cols["timestamp"],
        cols["net_load"],
        cols["solar"],
        "_pynrpf_gt_dummy",
    )
    sec1 = model["xgb1_day"]
    X_day = tools["_align_features"](day_df, sec1["feature_columns"]).to_numpy(dtype=np.float32)
    day_df["prob_day"] = sec1["model"].predict_proba(X_day)[:, 1]
    day_df["pred_day"] = day_df["prob_day"] >= sec1["threshold"]

    candidate_keys = day_df.loc[day_df["pred_day"], [cols["site"], "date"]].copy()

    if candidate_keys.empty:
        ts_results = pd.DataFrame(
            columns=[cols["site"], cols["timestamp"], "prob_interval", "pred_interval"]
        )
    else:
        ts_df, _, _ = tools["build_xgb2_features"](
            work,
            feature_cfg,
            day_df,
            candidate_keys,
            cols["site"],
            cols["timestamp"],
            cols["net_load"],
            cols["solar"],
            "_pynrpf_gt_dummy",
        )
        if ts_df.empty:
            ts_results = pd.DataFrame(
                columns=[cols["site"], cols["timestamp"], "prob_interval", "pred_interval"]
            )
        else:
            sec2 = model["xgb2_timestamp"]
            X_ts = tools["_align_features"](ts_df, sec2["feature_columns"]).to_numpy(
                dtype=np.float32
            )
            ts_df["prob_interval"] = sec2["model"].predict_proba(X_ts)[:, 1]
            ts_df["pred_interval"] = ts_df["prob_interval"] >= sec2["threshold"]
            ts_results = ts_df[
                [cols["site"], cols["timestamp"], "prob_interval", "pred_interval"]
            ].copy()

    result = eval_df.copy()
    day_map = day_df.set_index([cols["site"], "date"])[["prob_day", "pred_day"]]
    idx = result.set_index([cols["site"], cols["date"]]).index
    result["prob_day"] = idx.map(day_map["prob_day"]).values
    result["pred_day"] = idx.map(day_map["pred_day"]).values
    result["pred_day"] = result["pred_day"].fillna(False).astype(bool)

    result = result.merge(ts_results, on=[cols["site"], cols["timestamp"]], how="left")
    result["pred_interval"] = (
        result["pred_interval"].fillna(False).infer_objects(copy=False).astype(bool)
    )
    result.loc[~result["pred_day"], "pred_interval"] = False
    result["corrected_net_load_MW"] = np.where(
        result["pred_interval"], -result[cols["net_load"]], result[cols["net_load"]]
    )
    result.loc[result[cols["net_load"]].isna(), "corrected_net_load_MW"] = np.nan
    return result


def predict_m7(eval_df: pd.DataFrame, cfg: dict[str, Any], article_root: Path) -> pd.DataFrame:
    tools = _ensure_pynrpf_imports(article_root)
    cols = cfg["columns"]
    out = tools["run_m7"](
        eval_df.copy(),
        {"m7_threshold": cfg["m7_threshold"]},
        cols["site"],
        cols["timestamp"],
        cols["net_load"],
        cols["solar"],
    )
    out["pred_day"] = out["m7_rpf_day"].fillna(False).astype(bool)
    out["pred_interval"] = out["m7_rpf_flag"].fillna(False).astype(bool) & out["pred_day"]
    out["prob_day"] = np.nan
    out["prob_interval"] = np.nan
    out["corrected_net_load_MW"] = np.where(
        out["pred_interval"], -out[cols["net_load"]], out[cols["net_load"]]
    )
    out.loc[out[cols["net_load"]].isna(), "corrected_net_load_MW"] = np.nan
    return out


def evaluate_prediction_rows(
    pred_df: pd.DataFrame,
    cfg: dict[str, Any],
    experiment_id: str,
    fold_id: str,
    dataset: str,
    method: str,
    partition: str,
) -> list[dict[str, Any]]:
    cols = cfg["columns"]
    work = pred_df.copy()
    neg_days = work.groupby([cols["site"], cols["date"]])[cols["net_load"]].min().reset_index()
    neg_days["_has_negative_raw_mw"] = neg_days[cols["net_load"]] < 0
    keep_days = neg_days.loc[~neg_days["_has_negative_raw_mw"], [cols["site"], cols["date"]]]

    day_df = (
        work.groupby([cols["site"], cols["date"]])
        .agg(
            label_day=(cols["label_day"], "first"),
            pred_day=("pred_day", "first"),
            net_min=(cols["net_load"], "min"),
        )
        .reset_index()
    )
    day_df = day_df.loc[~(day_df["net_min"] < 0)].copy()
    day_metrics = _binary_classification_metrics(day_df["label_day"], day_df["pred_day"])

    tp_days = day_df.loc[
        day_df[cols["label_day"]].astype(bool) & day_df["pred_day"].astype(bool),
        [cols["site"], cols["date"]],
    ].copy()

    interval_df = work.merge(keep_days, on=[cols["site"], cols["date"]], how="inner")
    interval_df = interval_df.merge(tp_days, on=[cols["site"], cols["date"]], how="inner")
    hours = interval_df[cols["timestamp"]].dt.hour
    ev = cfg["evaluation"]
    interval_df = interval_df.loc[
        (hours >= int(ev["daytime_start_hour"])) & (hours < int(ev["daytime_end_hour"]))
    ].copy()
    interval_metrics = _binary_classification_metrics(
        interval_df[cols["label_interval"]],
        interval_df["pred_interval"],
    )

    rows = []
    interval_level_name = ev.get("interval_metric_level_name", "interval_tp_days_only")
    for level, metrics in [("day", day_metrics), (interval_level_name, interval_metrics)]:
        rows.append(
            {
                "experiment_id": experiment_id,
                "fold_id": fold_id,
                "dataset": dataset,
                "method": method,
                "partition": partition,
                "metric_level": level,
                **metrics,
            }
        )
    return rows


def audit_prediction_frame(
    pred_df: pd.DataFrame,
    cfg: dict[str, Any],
    experiment_id: str,
    fold_id: str,
    dataset: str,
    method: str,
) -> pd.DataFrame:
    cols = cfg["columns"]
    out = pred_df[
        [
            cols["site"],
            cols["date"],
            cols["timestamp"],
            cols["net_load"],
            cols["solar"],
            cols["label_day"],
            cols["label_interval"],
            "pred_day",
            "pred_interval",
            "prob_day",
            "prob_interval",
            "corrected_net_load_MW",
        ]
    ].copy()
    out.insert(0, "method", method)
    out.insert(0, "dataset", dataset)
    out.insert(0, "fold_id", fold_id)
    out.insert(0, "experiment_id", experiment_id)
    return out


def _write_task_outputs(
    pred_df: pd.DataFrame,
    out_dir: Path,
    cfg: dict[str, Any],
    experiment_id: str,
    fold_id: str,
    dataset: str,
    method: str,
    partition: str,
) -> dict[str, Path]:
    paths = task_paths(out_dir, cfg, fold_id, method)
    audit = audit_prediction_frame(pred_df, cfg, experiment_id, fold_id, dataset, method)
    metric_rows = evaluate_prediction_rows(
        pred_df, cfg, experiment_id, fold_id, dataset, method, partition
    )
    metrics_df = pd.DataFrame(metric_rows)

    _atomic_write_csv(audit, paths["prediction"])
    _atomic_write_csv(metrics_df, paths["metrics"])

    status = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "fold_id": fold_id,
        "dataset": dataset,
        "method": method,
        "partition": partition,
        "prediction_rows": int(len(audit)),
        "metrics_rows": int(len(metrics_df)),
        "positive_label_interval_rows": int(audit[cfg["columns"]["label_interval"]].sum()),
        "prediction_csv": str(paths["prediction"]),
        "metrics_csv": str(paths["metrics"]),
    }
    _atomic_write_yaml(status, paths["status"])
    return paths


def _completed_task_statuses(out_dir: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    status_dir = out_dir / cfg["outputs"].get("status_dir", "status")
    statuses: list[dict[str, Any]] = []
    if not status_dir.exists():
        return statuses
    for path in sorted(status_dir.glob("*.yaml")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                status = yaml.safe_load(handle) or {}
        except yaml.YAMLError:
            continue
        if status.get("status") != "complete":
            continue
        fold_id = str(status.get("fold_id", ""))
        method = str(status.get("method", ""))
        if not fold_id or not method:
            continue
        if not task_is_complete(out_dir, cfg, fold_id, method):
            continue
        statuses.append(status)
    return statuses


def _completed_metric_rows(out_dir: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for status in _completed_task_statuses(out_dir, cfg):
        metrics_path = task_paths(out_dir, cfg, status["fold_id"], status["method"])["metrics"]
        if metrics_path.exists():
            rows.extend(pd.read_csv(metrics_path).to_dict("records"))
    return rows


def _completed_prediction_paths(out_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for status in _completed_task_statuses(out_dir, cfg):
        prediction_path = task_paths(out_dir, cfg, status["fold_id"], status["method"])[
            "prediction"
        ]
        if prediction_path.exists():
            paths.append(prediction_path)
    return paths


def _read_prediction_csv(path: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    cols = cfg["columns"]
    df = pd.read_csv(path)
    df[cols["timestamp"]] = pd.to_datetime(df[cols["timestamp"]], errors="raise")
    df[cols["date"]] = df[cols["timestamp"]].dt.date
    for col in [cols["label_day"], cols["label_interval"], "pred_day", "pred_interval"]:
        df[col] = _coerce_bool(df[col])
    return df


def _station_pooled_metric_rows(
    out_dir: Path,
    cfg: dict[str, Any],
    experiment_id: str,
    dataset: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    statuses = _completed_task_statuses(out_dir, cfg)
    for method in enabled_methods(cfg):
        prediction_paths = [
            task_paths(out_dir, cfg, status["fold_id"], method)["prediction"]
            for status in statuses
            if status.get("method") == method and str(status.get("fold_id", "")).startswith("station_")
        ]
        if not prediction_paths:
            continue
        frames = [_read_prediction_csv(path, cfg) for path in prediction_paths]
        pooled = pd.concat(frames, ignore_index=True)
        rows.extend(
            evaluate_prediction_rows(
                pooled,
                cfg,
                experiment_id,
                "pooled",
                dataset,
                method,
                "held_out_station",
            )
        )
    return rows


def _write_experiment_outputs(
    out_dir: Path,
    cfg: dict[str, Any],
    experiment_id: str,
    notebook_name: str,
    details: dict[str, Any],
) -> None:
    ensure_output_dirs(out_dir, cfg)
    fold_rows = _completed_metric_rows(out_dir, cfg)
    summary_rows = list(fold_rows)
    if details.get("include_station_pooled_metrics"):
        summary_rows.extend(
            _station_pooled_metric_rows(out_dir, cfg, experiment_id, details["pooled_dataset"])
        )
    prediction_paths = _completed_prediction_paths(out_dir, cfg)
    completed_statuses = _completed_task_statuses(out_dir, cfg)
    completed_task_keys = {
        (status["fold_id"], status["method"])
        for status in completed_statuses
        if "fold_id" in status and "method" in status
    }
    expected_tasks = details.get("expected_tasks", [])
    all_expected_complete = bool(expected_tasks) and all(
        (task["fold_id"], task["method"]) in completed_task_keys for task in expected_tasks
    )
    run_status = "complete" if all_expected_complete else "partial"

    metrics_path = out_dir / cfg["outputs"]["metrics_summary_csv"]
    folds_path = out_dir / cfg["outputs"]["fold_metrics_csv"]
    manifest_path = out_dir / cfg["outputs"]["manifest_yaml"]
    _atomic_write_csv(pd.DataFrame(summary_rows), metrics_path)
    _atomic_write_csv(pd.DataFrame(fold_rows), folds_path)
    manifest = {
        "status": run_status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "notebook": notebook_name,
        "config_path": cfg.get("_config_path"),
        "experiment_id": experiment_id,
        "run_control": _run_control_details(cfg),
        "completed_task_count": len(completed_task_keys),
        "expected_task_count": len(expected_tasks),
        "completed_tasks": [
            {"fold_id": fold_id, "method": method}
            for fold_id, method in sorted(completed_task_keys)
        ],
        "metrics_summary_csv": str(metrics_path),
        "fold_metrics_csv": str(folds_path),
        "prediction_files": [str(path) for path in prediction_paths],
        "metrics_dir": str(out_dir / cfg["outputs"].get("metrics_dir", "metrics")),
        "status_dir": str(out_dir / cfg["outputs"].get("status_dir", "status")),
        "details": details,
    }
    _atomic_write_yaml(manifest, manifest_path)


def run_time_split_experiment(
    article_root: Path,
    cfg: dict[str, Any],
    dataset_key: str,
    experiment_id: str,
    notebook_name: str,
) -> None:
    df = load_dataset(article_root, cfg, dataset_key)
    train_mask, test_mask = time_masks(df, cfg)
    methods = enabled_methods(cfg)
    expected_tasks = [
        _planned_task(f"time_{partition}", method, dataset_key, partition)
        for partition in ["train", "test"]
        for method in methods
    ]
    details = {
        **_run_control_details(cfg),
        "dataset_summary": dataset_summary(df, cfg, dataset_key),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "expected_tasks": expected_tasks,
    }
    print(details)
    if not cfg["execution"]["run_full_experiment"]:
        manifest = prepare_smoke_output(article_root, cfg, experiment_id, notebook_name, details)
        print(f"Smoke mode: full training skipped. Manifest: {manifest}")
        return

    out_dir = experiment_output_dir(article_root, cfg, experiment_id)
    ensure_output_dirs(out_dir, cfg)
    _clear_expected_task_outputs(out_dir, cfg, expected_tasks)
    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()
    m8_model = None
    if _m8_has_incomplete_tasks(out_dir, cfg, expected_tasks):
        print("Training M8 once for incomplete time-split M8 task(s).")
        m8_model = train_m8_model(train_df, cfg, article_root)
    else:
        print("M8 training skipped: M8 is disabled or all M8 time-split tasks are complete.")

    for partition, part_df in [("train", train_df), ("test", test_df)]:
        fold_id = f"time_{partition}"
        for method in methods:
            if should_skip_task(out_dir, cfg, fold_id, method):
                print(f"Skipping completed task: {fold_id} / {method}")
                continue
            _begin_task(out_dir, cfg, fold_id, method)
            pred = (
                predict_m8(m8_model, part_df, cfg, article_root)
                if method == "m8_xgb"
                else predict_m7(part_df, cfg, article_root)
            )
            _write_task_outputs(
                pred,
                out_dir,
                cfg,
                experiment_id,
                fold_id,
                dataset_key,
                method,
                partition,
            )
            print(f"Completed task: {fold_id} / {method}")

    _write_experiment_outputs(out_dir, cfg, experiment_id, notebook_name, details)


def run_station_split_experiment(
    article_root: Path,
    cfg: dict[str, Any],
    dataset_key: str,
    experiment_id: str,
    notebook_name: str,
) -> None:
    df = load_dataset(article_root, cfg, dataset_key)
    folds = station_folds(df, cfg, dataset_key)
    cols = cfg["columns"]
    methods = enabled_methods(cfg)
    expected_tasks = [
        _planned_task(f"station_{station}", method, dataset_key, "held_out_station")
        for station in folds
        for method in methods
    ]
    details = {
        **_run_control_details(cfg),
        "dataset_summary": dataset_summary(df, cfg, dataset_key),
        "folds": folds,
        "excluded_synthetic_stations": cfg["station_split"].get("synthetic_excluded_stations", [])
        if dataset_key == "synthetic"
        else [],
        "expected_tasks": expected_tasks,
        "include_station_pooled_metrics": True,
        "pooled_dataset": dataset_key,
    }
    print(details)
    if not cfg["execution"]["run_full_experiment"]:
        manifest = prepare_smoke_output(article_root, cfg, experiment_id, notebook_name, details)
        print(f"Smoke mode: full training skipped. Manifest: {manifest}")
        return

    out_dir = experiment_output_dir(article_root, cfg, experiment_id)
    ensure_output_dirs(out_dir, cfg)
    _clear_expected_task_outputs(out_dir, cfg, expected_tasks)

    for station in folds:
        train_df = df.loc[df[cols["site"]] != station].copy()
        test_df = df.loc[df[cols["site"]] == station].copy()
        fold_id = f"station_{station}"
        fold_tasks = [
            _planned_task(fold_id, method, dataset_key, "held_out_station") for method in methods
        ]
        m8_model = None
        if _m8_has_incomplete_tasks(out_dir, cfg, fold_tasks):
            print(f"Training M8 for incomplete station fold: {fold_id}")
            m8_model = train_m8_model(train_df, cfg, article_root)
        else:
            print(f"M8 training skipped for {fold_id}: disabled or already complete.")

        for method in methods:
            if should_skip_task(out_dir, cfg, fold_id, method):
                print(f"Skipping completed task: {fold_id} / {method}")
                continue
            _begin_task(out_dir, cfg, fold_id, method)
            pred = (
                predict_m8(m8_model, test_df, cfg, article_root)
                if method == "m8_xgb"
                else predict_m7(test_df, cfg, article_root)
            )
            _write_task_outputs(
                pred,
                out_dir,
                cfg,
                experiment_id,
                fold_id,
                dataset_key,
                method,
                "held_out_station",
            )
            print(f"Completed task: {fold_id} / {method}")

    _write_experiment_outputs(out_dir, cfg, experiment_id, notebook_name, details)


def run_transfer_experiment(
    article_root: Path,
    cfg: dict[str, Any],
    experiment_id: str,
    notebook_name: str,
) -> None:
    synthetic = load_dataset(article_root, cfg, "synthetic")
    actual = load_dataset(article_root, cfg, "actual")
    syn_train_mask, _ = time_masks(synthetic, cfg)
    _, actual_test_mask = time_masks(actual, cfg)
    methods = enabled_methods(cfg)
    fold_id = "transfer_actual_test"
    expected_tasks = [
        _planned_task(fold_id, method, "actual", "test")
        for method in methods
    ]
    details = {
        **_run_control_details(cfg),
        "synthetic_summary": dataset_summary(synthetic, cfg, "synthetic"),
        "actual_summary": dataset_summary(actual, cfg, "actual"),
        "synthetic_train_rows": int(syn_train_mask.sum()),
        "actual_test_rows": int(actual_test_mask.sum()),
        "expected_tasks": expected_tasks,
    }
    print(details)
    if not cfg["execution"]["run_full_experiment"]:
        manifest = prepare_smoke_output(article_root, cfg, experiment_id, notebook_name, details)
        print(f"Smoke mode: full training skipped. Manifest: {manifest}")
        return

    out_dir = experiment_output_dir(article_root, cfg, experiment_id)
    ensure_output_dirs(out_dir, cfg)
    _clear_expected_task_outputs(out_dir, cfg, expected_tasks)
    synthetic_train = synthetic.loc[syn_train_mask].copy()
    actual_test = actual.loc[actual_test_mask].copy()
    m8_model = None
    if _m8_has_incomplete_tasks(out_dir, cfg, expected_tasks):
        print("Training M8 once for incomplete synthetic-to-real transfer M8 task(s).")
        m8_model = train_m8_model(synthetic_train, cfg, article_root)
    else:
        print("M8 training skipped: M8 is disabled or the transfer M8 task is complete.")

    for method in methods:
        if should_skip_task(out_dir, cfg, fold_id, method):
            print(f"Skipping completed task: {fold_id} / {method}")
            continue
        _begin_task(out_dir, cfg, fold_id, method)
        pred = (
            predict_m8(m8_model, actual_test, cfg, article_root)
            if method == "m8_xgb"
            else predict_m7(actual_test, cfg, article_root)
        )
        _write_task_outputs(
            pred,
            out_dir,
            cfg,
            experiment_id,
            fold_id,
            "actual",
            method,
            "test",
        )
        print(f"Completed task: {fold_id} / {method}")

    _write_experiment_outputs(out_dir, cfg, experiment_id, notebook_name, details)

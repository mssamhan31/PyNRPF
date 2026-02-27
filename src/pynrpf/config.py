from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Union

import yaml

ConfigInput = Union[Mapping[str, Any], str, Path]


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": "0.2.0",
    "columns": {
        "site": "substation_id",
        "timestamp": "timestamp",
        "net_load": "net_load_MW",
        "solar": "solar_MW",
    },
    "runtime": {
        "interval_minutes": 15,
        "strict_validation": True,
    },
    "model": {
        "selected_model": "m7_dtr",
        "primary_model": "m7_dtr",
        "enabled_models": ["m7_dtr"],
        "m7_threshold": {
            "solar_peak_tiebreak_time": "12:30",
            "peak_window_minutes": 150,
            "min_threshold": 0.05,
            "min_threshold_both": 0.25,
        },
        "m8_xgb": {
            "noon_hour_start": 6,
            "noon_hour_end": 18,
            "holiday_country": "AU",
            "holiday_subdivision": "NSW",
            "xgb1_day": {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "tree_method": "hist",
                "eta": 0.1,
                "n_estimators": 500,
                "max_depth": 6,
                "min_child_weight": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 5,
                "seed": 9,
                "threshold": 0.585985,
            },
            "xgb2_timestamp": {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "tree_method": "hist",
                "eta": 0.1,
                "n_estimators": 500,
                "max_depth": 6,
                "min_child_weight": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 5,
                "seed": 9,
                "threshold": 0.892234,
            },
        },
    },
    "artifacts": {
        "m8_pretrained_bundle_uri": None,
    },
    "monitoring": {
        "enable_data_quality": True,
        "enable_confidence_summary": True,
        "enable_drift_summary": True,
        "reference_stats": {},
    },
}


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise TypeError("Config payload must be a mapping.")
    return payload


def _is_new_schema(cfg: dict[str, Any]) -> bool:
    return all(key in cfg for key in ("columns", "runtime", "model"))


def _looks_legacy(cfg: dict[str, Any]) -> bool:
    legacy_keys = {"run", "paths", "data", "validation", "split", "m7_threshold", "m8_xgb"}
    return any(key in cfg for key in legacy_keys)


def _adapt_legacy(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    cols = data_cfg.get("columns", {}) if isinstance(data_cfg.get("columns"), dict) else {}
    validation_cfg = (
        cfg.get("validation", {}) if isinstance(cfg.get("validation"), dict) else {}
    )
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}

    selected_model = model_cfg.get("selected_model", "m7_dtr")
    primary_model = model_cfg.get("primary_model", selected_model)
    enabled_models = model_cfg.get("enabled_models", [selected_model])
    if not isinstance(enabled_models, list) or not enabled_models:
        enabled_models = [selected_model]

    adapted = {
        "schema_version": "0.2.0",
        "columns": {
            "site": cols.get("site", "substation_id"),
            "timestamp": cols.get("ts", "timestamp"),
            "net_load": cols.get("net_load", "net_load_MW"),
            "solar": cols.get("solar", "solar_MW"),
        },
        "runtime": {
            "interval_minutes": int(data_cfg.get("interval_minutes", 15)),
            "strict_validation": bool(validation_cfg.get("enforce_interval_alignment", True)),
        },
        "model": {
            "selected_model": selected_model,
            "primary_model": primary_model,
            "enabled_models": enabled_models,
            "m7_threshold": deepcopy(cfg.get("m7_threshold", {})),
            "m8_xgb": deepcopy(cfg.get("m8_xgb", {})),
            "training_labels": deepcopy(model_cfg.get("training_labels", {})),
        },
        "artifacts": (
            deepcopy(cfg.get("artifacts", {}))
            if isinstance(cfg.get("artifacts"), dict)
            else {}
        ),
        "monitoring": (
            deepcopy(cfg.get("monitoring", {}))
            if isinstance(cfg.get("monitoring"), dict)
            else {}
        ),
    }
    return adapted


def load_config(config: ConfigInput) -> dict[str, Any]:
    if isinstance(config, Mapping):
        raw = dict(config)
    elif isinstance(config, (str, Path)):
        raw = _load_yaml(Path(config))
    else:
        raise TypeError("Config must be a mapping, file path string, or pathlib.Path.")

    if _looks_legacy(raw) and not _is_new_schema(raw):
        raw = _adapt_legacy(raw)

    cfg = _deep_merge(DEFAULT_CONFIG, raw)

    runtime_cfg = cfg.get("runtime", {})
    interval_minutes = int(runtime_cfg.get("interval_minutes", 15))
    if interval_minutes != 15:
        raise ValueError(
            "v0.2.0 supports only 15-minute interval data. "
            f"Got interval_minutes={interval_minutes}."
        )
    cfg["runtime"]["interval_minutes"] = interval_minutes

    selected_model = str(cfg.get("model", {}).get("selected_model", "m7_dtr")).strip()
    if not selected_model:
        raise ValueError("model.selected_model must be set.")
    cfg["model"]["selected_model"] = selected_model

    primary_model = str(cfg.get("model", {}).get("primary_model", "")).strip()
    if not primary_model:
        primary_model = selected_model
    cfg["model"]["primary_model"] = primary_model

    enabled_models = cfg.get("model", {}).get("enabled_models", [selected_model])
    if not isinstance(enabled_models, list) or not enabled_models:
        enabled_models = [selected_model]
    if selected_model not in enabled_models:
        enabled_models = [selected_model] + [m for m in enabled_models if m != selected_model]
    cfg["model"]["enabled_models"] = enabled_models

    return cfg

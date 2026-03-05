from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import ConfigInput

DEFAULT_TRAINING_CONFIG: dict[str, Any] = {
    "schema_version": "0.3.0",
    "model_id": "m8_xgb",
    "labels": {
        "day": "label_day",
        "interval": "label_interval",
    },
    "split": {
        "train_start": "2021-11-01",
        "train_end": "2023-09-30",
        "validation_start": "2023-10-01",
        "validation_end": "2024-09-30",
    },
    "features": {
        "pipeline": "legacy_v1",
    },
    "thresholds": {
        "xgb1_day": 0.585985,
        "xgb2_timestamp": 0.892234,
    },
    "random_seed": 9,
    "output": {
        "base_uri": "outputs/artifacts",
        "versioning": "timestamp_dir",
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Training config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise TypeError("Training config payload must be a mapping.")
    return payload


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _extract_training_block(raw: dict[str, Any]) -> dict[str, Any]:
    payload = raw.get("pynrpf_training")
    if isinstance(payload, dict):
        return deepcopy(payload)
    return raw


def _require_non_empty_str(container: dict[str, Any], key: str) -> str:
    value = str(container.get(key, "")).strip()
    if not value:
        raise ValueError(f"Training config key '{key}' must be a non-empty string.")
    return value


def _parse_date(key: str, value: Any) -> date:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Training split key '{key}' must be provided.")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"Training split key '{key}' must be YYYY-MM-DD. Got '{text}'."
        ) from exc


def _validate_threshold(name: str, value: Any) -> float:
    numeric = float(value)
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"Training threshold '{name}' must be between 0 and 1.")
    return numeric


def load_training_config(config: ConfigInput) -> dict[str, Any]:
    if isinstance(config, Mapping):
        raw = _extract_training_block(dict(config))
    elif isinstance(config, (str, Path)):
        raw = _extract_training_block(_load_yaml(Path(config)))
    else:
        raise TypeError("Training config must be a mapping, file path string, or pathlib.Path.")

    cfg = _deep_merge(DEFAULT_TRAINING_CONFIG, raw)

    model_id = _require_non_empty_str(cfg, "model_id")
    if model_id != "m8_xgb":
        raise ValueError(f"pynrpf_training.model_id must be 'm8_xgb'. Got '{model_id}'.")
    cfg["model_id"] = model_id

    labels = cfg.get("labels", {})
    if not isinstance(labels, dict):
        raise TypeError("pynrpf_training.labels must be a mapping.")
    day_label = _require_non_empty_str(labels, "day")
    interval_label = _require_non_empty_str(labels, "interval")
    cfg["labels"] = {"day": day_label, "interval": interval_label}

    features_cfg = cfg.get("features", {})
    if not isinstance(features_cfg, dict):
        raise TypeError("pynrpf_training.features must be a mapping.")
    pipeline = _require_non_empty_str(features_cfg, "pipeline")
    if pipeline != "legacy_v1":
        raise ValueError(
            "pynrpf_training.features.pipeline must be 'legacy_v1' for this release."
        )
    cfg["features"] = {"pipeline": pipeline}

    split_cfg = cfg.get("split", {})
    if not isinstance(split_cfg, dict):
        raise TypeError("pynrpf_training.split must be a mapping.")
    train_start = _parse_date("train_start", split_cfg.get("train_start"))
    train_end = _parse_date("train_end", split_cfg.get("train_end"))
    validation_start = _parse_date("validation_start", split_cfg.get("validation_start"))
    validation_end = _parse_date("validation_end", split_cfg.get("validation_end"))
    if not (train_start <= train_end < validation_start <= validation_end):
        raise ValueError(
            "Training split must satisfy: train_start <= train_end < "
            "validation_start <= validation_end."
        )
    cfg["split"] = {
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "validation_start": validation_start.isoformat(),
        "validation_end": validation_end.isoformat(),
    }

    thresholds = cfg.get("thresholds", {})
    if not isinstance(thresholds, dict):
        raise TypeError("pynrpf_training.thresholds must be a mapping.")
    cfg["thresholds"] = {
        "xgb1_day": _validate_threshold("xgb1_day", thresholds.get("xgb1_day")),
        "xgb2_timestamp": _validate_threshold("xgb2_timestamp", thresholds.get("xgb2_timestamp")),
    }

    random_seed = int(cfg.get("random_seed", 9))
    cfg["random_seed"] = random_seed

    output_cfg = cfg.get("output", {})
    if not isinstance(output_cfg, dict):
        raise TypeError("pynrpf_training.output must be a mapping.")
    base_uri = _require_non_empty_str(output_cfg, "base_uri")
    versioning = _require_non_empty_str(output_cfg, "versioning")
    if versioning != "timestamp_dir":
        raise ValueError("pynrpf_training.output.versioning must be 'timestamp_dir'.")
    cfg["output"] = {"base_uri": base_uri, "versioning": versioning}

    return cfg

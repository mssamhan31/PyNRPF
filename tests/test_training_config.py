from __future__ import annotations

from pathlib import Path

import yaml

from pynrpf.training_config import load_training_config


def _training_payload(base_uri: str = "outputs/artifacts") -> dict:
    return {
        "model_id": "m8_xgb",
        "labels": {"day": "label_day", "interval": "label_interval"},
        "split": {
            "train_start": "2024-01-01",
            "train_end": "2024-01-31",
            "validation_start": "2024-02-01",
            "validation_end": "2024-02-29",
        },
        "features": {"pipeline": "legacy_v1"},
        "thresholds": {"xgb1_day": 0.55, "xgb2_timestamp": 0.66},
        "random_seed": 7,
        "output": {"base_uri": base_uri, "versioning": "timestamp_dir"},
    }


def test_load_training_config_from_pipeline_mapping() -> None:
    cfg = {
        "pipeline_schema_version": "1.0.0",
        "pynrpf_training": _training_payload(),
    }
    training_cfg = load_training_config(cfg)
    assert training_cfg["model_id"] == "m8_xgb"
    assert training_cfg["labels"]["day"] == "label_day"
    assert training_cfg["thresholds"]["xgb2_timestamp"] == 0.66


def test_load_training_config_from_yaml_path(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        yaml.safe_dump({"pynrpf_training": _training_payload("dbfs:/FileStore/models")}),
        encoding="utf-8",
    )
    training_cfg = load_training_config(path)
    assert training_cfg["output"]["base_uri"] == "dbfs:/FileStore/models"


def test_load_training_config_rejects_invalid_split_order() -> None:
    payload = _training_payload()
    payload["split"]["validation_start"] = "2024-01-15"
    try:
        load_training_config({"pynrpf_training": payload})
    except ValueError as exc:
        assert "train_start <= train_end < validation_start <= validation_end" in str(exc)
        return
    raise AssertionError("Expected ValueError for invalid training split ordering.")


def test_load_training_config_rejects_invalid_threshold() -> None:
    payload = _training_payload()
    payload["thresholds"]["xgb1_day"] = 1.5
    try:
        load_training_config({"pynrpf_training": payload})
    except ValueError as exc:
        assert "xgb1_day" in str(exc)
        return
    raise AssertionError("Expected ValueError for threshold outside [0, 1].")

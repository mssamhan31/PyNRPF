from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pynrpf.api import run_inference, train_m8_xgb


def _sample_training_df() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for i, day in enumerate(pd.date_range("2024-01-01", periods=4, freq="D")):
        ts = pd.date_range(day, periods=96, freq="15min")
        x = np.linspace(0, 2 * np.pi, len(ts), endpoint=False)
        net = 4.0 + 2.0 * np.sin(x) + 0.3 * np.cos(3 * x) + i * 0.05
        solar = np.maximum(0, np.sin((ts.hour.values - 6) / 12 * np.pi))

        is_positive_day = i in (0, 2)  # one positive day in train window and one in validation
        is_positive_interval = is_positive_day & (ts.hour.values >= 10) & (ts.hour.values <= 13)

        frames.append(
            pd.DataFrame(
                {
                    "substation_id": "A",
                    "timestamp": ts,
                    "net_load_MW": net,
                    "solar_MW": solar,
                    "label_day": bool(is_positive_day),
                    "label_interval": is_positive_interval.astype(bool),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _pipeline_cfg(base_uri: Path) -> dict:
    return {
        "pipeline_schema_version": "1.0.0",
        "tables": {"inputs": {"ignored": "value"}},
        "pynrpf_inference": {
            "columns": {
                "site": "substation_id",
                "timestamp": "timestamp",
                "net_load": "net_load_MW",
                "solar": "solar_MW",
            },
            "runtime": {"interval_minutes": 15, "strict_validation": True},
            "model": {"selected_model": "m8_xgb"},
        },
        "pynrpf_training": {
            "model_id": "m8_xgb",
            "labels": {
                "day": "label_day",
                "interval": "label_interval",
            },
            "split": {
                "train_start": "2024-01-01",
                "train_end": "2024-01-02",
                "validation_start": "2024-01-03",
                "validation_end": "2024-01-04",
            },
            "features": {"pipeline": "legacy_v1"},
            "thresholds": {"xgb1_day": 0.55, "xgb2_timestamp": 0.6},
            "random_seed": 11,
            "output": {
                "base_uri": str(base_uri),
                "versioning": "timestamp_dir",
            },
        },
    }


def test_train_m8_xgb_writes_artifact_and_inference_loads_bundle(tmp_path: Path) -> None:
    df = _sample_training_df()
    cfg = _pipeline_cfg(tmp_path / "artifacts")

    train_out = train_m8_xgb(df, cfg)
    assert train_out["model"] == "m8_xgb"
    assert train_out["bundle_schema"] == "pynrpf.m8_xgb.bundle.v2"
    assert train_out["artifact_uri"]
    assert "xgb1_day" in train_out["validation_metrics"]
    assert "xgb2_timestamp" in train_out["validation_metrics"]

    inference_cfg = {
        "pynrpf_inference": {
            "columns": cfg["pynrpf_inference"]["columns"],
            "runtime": cfg["pynrpf_inference"]["runtime"],
            "model": cfg["pynrpf_inference"]["model"] | {"selected_model": "m8_xgb"},
            "artifacts": {"m8_pretrained_bundle_uri": train_out["artifact_uri"]},
        }
    }
    infer_df = df[["substation_id", "timestamp", "net_load_MW", "solar_MW"]].copy()
    infer_out = run_inference(infer_df, inference_cfg)
    result_df = infer_out["data"]

    assert "pynrpf_interval_flag" in result_df.columns
    assert "pynrpf_day_flag" in result_df.columns
    assert "pynrpf_corrected_net_load" in result_df.columns


def test_train_m8_xgb_requires_day_and_interval_labels(tmp_path: Path) -> None:
    df = _sample_training_df().drop(columns=["label_interval"])
    cfg = _pipeline_cfg(tmp_path / "artifacts")

    try:
        train_m8_xgb(df, cfg)
    except KeyError as exc:
        assert "label_interval" in str(exc)
        return
    raise AssertionError("Expected KeyError when required training label columns are missing.")

from __future__ import annotations

import numpy as np
import pandas as pd

from pynrpf.api import run_inference


def _sample_df() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=96, freq="15min")
    x = np.linspace(0, 2 * np.pi, len(ts))
    net = 4.0 + 2.0 * np.sin(x) + 0.5 * np.sin(4 * x)
    solar = np.maximum(0, np.sin((ts.hour.values - 6) / 12 * np.pi))

    return pd.DataFrame(
        {
            "substation_id": "A",
            "timestamp": ts,
            "net_load_MW": net,
            "solar_MW": solar,
        }
    )


def test_run_inference_m7_pandas_dataframe() -> None:
    df = _sample_df()
    cfg = {
        "columns": {
            "site": "substation_id",
            "timestamp": "timestamp",
            "net_load": "net_load_MW",
            "solar": "solar_MW",
        },
        "runtime": {"interval_minutes": 15, "strict_validation": True},
        "model": {
            "selected_model": "m7_dtr",
            "m7_threshold": {
                "solar_peak_tiebreak_time": "12:30",
                "peak_window_minutes": 150,
                "min_threshold": 0.05,
                "min_threshold_both": 0.25,
            },
        },
    }

    out = run_inference(df, cfg)
    result_df = out["data"]
    summary = out["summary"]

    assert out["model"] == "m7_dtr"
    assert "pynrpf_interval_flag" in result_df.columns
    assert "pynrpf_day_flag" in result_df.columns
    assert "pynrpf_corrected_net_load" in result_df.columns
    assert "rows_processed" in summary
    assert summary["rows_processed"] == 96


def test_run_inference_m8_requires_artifact_uri() -> None:
    df = _sample_df()
    cfg = {
        "columns": {
            "site": "substation_id",
            "timestamp": "timestamp",
            "net_load": "net_load_MW",
            "solar": "solar_MW",
        },
        "runtime": {"interval_minutes": 15, "strict_validation": True},
        "model": {"selected_model": "m8_xgb"},
        "artifacts": {"m8_pretrained_bundle_uri": None},
    }

    try:
        run_inference(df, cfg)
    except ValueError as exc:
        assert "m8_pretrained_bundle_uri" in str(exc)
        return
    raise AssertionError("Expected ValueError when m8_pretrained_bundle_uri is not provided.")

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pynrpf.api import run_inference


def _m7_cfg() -> dict:
    return {
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


def _solar_profile(ts: pd.DatetimeIndex) -> np.ndarray:
    hour_fraction = ts.hour.values + ts.minute.values / 60.0
    return np.maximum(0.0, 1.0 - np.abs(hour_fraction - 12.5) / 6.5)


def _build_full_day(overrides: dict[str, float], day: str = "2024-01-01") -> pd.DataFrame:
    ts = pd.date_range(day, periods=96, freq="15min")
    net = np.full(len(ts), 8.0)
    labels = ts.strftime("%H:%M")
    for hhmm, value in overrides.items():
        net[labels == hhmm] = value

    return pd.DataFrame(
        {
            "substation_id": "A",
            "timestamp": ts,
            "net_load_MW": net,
            "solar_MW": _solar_profile(ts),
        }
    )


def _build_insufficient_midday_day() -> pd.DataFrame:
    ts = pd.to_datetime(["2024-01-01 11:45", "2024-01-01 12:00"])
    return pd.DataFrame(
        {
            "substation_id": "A",
            "timestamp": ts,
            "net_load_MW": [6.0, 10.0],
            "solar_MW": [0.9, 1.0],
        }
    )


def test_m7_strict_positive_day_flags_day_and_intervals() -> None:
    df = _build_full_day(
        {
            "10:00": 0.0,
            "11:45": 6.0,
            "12:00": 10.0,
            "12:15": 6.0,
            "14:00": 0.3,
        }
    )

    out = run_inference(df, _m7_cfg())
    result_df = out["data"]
    flagged = result_df["pynrpf_interval_flag"]

    assert result_df["m7_rpf_day"].all()
    assert result_df["pynrpf_day_flag"].all()
    assert flagged.any()
    assert out["summary"]["predicted_positive_days"] == 1
    assert out["summary"]["rows_corrected"] == int(flagged.sum())
    assert (
        result_df.loc[flagged, "pynrpf_corrected_net_load"]
        == -result_df.loc[flagged, "net_load_MW"]
    ).all()


def test_m7_relaxed_interval_correction_can_flag_strict_negative_day() -> None:
    df = _build_full_day(
        {
            "10:00": 3.0,
            "11:45": 6.0,
            "12:00": 10.0,
            "12:15": 6.0,
            "14:00": 3.5,
        }
    )

    out = run_inference(df, _m7_cfg())
    result_df = out["data"]
    flagged = result_df["pynrpf_interval_flag"]

    assert not result_df["m7_rpf_day"].any()
    assert not result_df["pynrpf_day_flag"].any()
    assert flagged.any()
    assert out["summary"]["predicted_positive_days"] == 0
    assert out["summary"]["rows_corrected"] == int(flagged.sum()) > 0
    assert (
        result_df.loc[flagged, "pynrpf_corrected_net_load"]
        == -result_df.loc[flagged, "net_load_MW"]
    ).all()


def test_m7_without_relaxed_pair_leaves_day_and_intervals_unflagged() -> None:
    out = run_inference(_build_full_day({}), _m7_cfg())
    result_df = out["data"]

    assert not result_df["m7_rpf_day"].any()
    assert not result_df["m7_rpf_flag"].any()
    assert not result_df["pynrpf_interval_flag"].any()
    assert out["summary"]["predicted_positive_days"] == 0
    assert out["summary"]["rows_corrected"] == 0


def test_m7_relaxed_interval_requires_daytime_minima() -> None:
    df = _build_full_day(
        {
            "05:00": 1.0,
            "11:45": 6.0,
            "12:00": 10.0,
            "12:15": 6.0,
            "14:00": 3.0,
        }
    )

    out = run_inference(df, _m7_cfg())
    result_df = out["data"]

    assert not result_df["m7_rpf_day"].any()
    assert not result_df["m7_rpf_flag"].any()
    assert not result_df["pynrpf_interval_flag"].any()


@pytest.mark.parametrize(
    ("df", "label"),
    [
        (
            _build_full_day(
                {
                    "10:00": 0.0,
                    "11:45": 6.0,
                    "12:00": np.nan,
                    "12:15": 6.0,
                    "14:00": 0.3,
                }
            ),
            "missing",
        ),
        (
            _build_full_day(
                {
                    "09:00": -0.1,
                    "10:00": 0.0,
                    "11:45": 6.0,
                    "12:00": 10.0,
                    "12:15": 6.0,
                    "14:00": 0.3,
                }
            ),
            "negative",
        ),
        (_build_insufficient_midday_day(), "insufficient_midday"),
    ],
)
def test_m7_relaxed_interval_preserves_early_gates(df: pd.DataFrame, label: str) -> None:
    out = run_inference(df, _m7_cfg())
    result_df = out["data"]

    assert not result_df["m7_rpf_day"].any(), label
    assert not result_df["m7_rpf_flag"].any(), label
    assert not result_df["pynrpf_interval_flag"].any(), label
    assert out["summary"]["predicted_positive_days"] == 0, label
    assert out["summary"]["rows_corrected"] == 0, label

from __future__ import annotations

import numpy as np
import pandas as pd

from pynrpf.api import run_inference


class _FixedProbClf:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        prob = np.full(len(X), self.probability, dtype=np.float32)
        return np.column_stack([1.0 - prob, prob])


def _sample_interval_df() -> pd.DataFrame:
    ts = pd.to_datetime(
        [
            "2024-01-01 10:00:00",
            "2024-01-01 10:15:00",
            "2024-01-02 10:00:00",
            "2024-01-02 10:15:00",
        ]
    )
    return pd.DataFrame(
        {
            "substation_id": "A",
            "timestamp": ts,
            "net_load_MW": [4.0, 5.0, 6.0, 7.0],
            "solar_MW": [1.0, 1.0, 1.0, 1.0],
        }
    )


def _base_cfg(score_all_days_for_review: bool = False) -> dict:
    return {
        "columns": {
            "site": "substation_id",
            "timestamp": "timestamp",
            "net_load": "net_load_MW",
            "solar": "solar_MW",
        },
        "runtime": {"interval_minutes": 15, "strict_validation": True},
        "model": {
            "selected_model": "m8_xgb",
            "m8_xgb": {
                "xgb1_day": {"threshold": 0.5},
                "xgb2_timestamp": {
                    "threshold": 0.5,
                    "score_all_days_for_review": score_all_days_for_review,
                },
            },
        },
    }


def _patch_m8_inference(monkeypatch) -> None:
    bundle = {
        "xgb1_day": {
            "model": _FixedProbClf(0.1),
            "feature_columns": ["f_day"],
            "threshold": 0.5,
        },
        "xgb2_timestamp": {
            "model": _FixedProbClf(0.9),
            "feature_columns": ["f_ts"],
            "threshold": 0.5,
        },
    }

    def fake_xgb1_features(
        df: pd.DataFrame,
        cfg: dict,
        col_site: str,
        col_ts: str,
        col_net: str,
        col_solar: str,
        col_gt: str,
    ):
        day_df = (
            df[[col_site, col_ts]]
            .assign(date=pd.to_datetime(df[col_ts]).dt.date, f_day=1.0)
            [[col_site, "date", "f_day"]]
            .drop_duplicates([col_site, "date"])
            .reset_index(drop=True)
        )
        return day_df, ["f_day"], "ignored_day_label"

    def fake_xgb2_features(
        df: pd.DataFrame,
        cfg: dict,
        day_features_df: pd.DataFrame,
        candidate_keys: pd.DataFrame,
        col_site: str,
        col_ts: str,
        col_net: str,
        col_solar: str,
        col_gt: str,
    ):
        ts_df = (
            df[[col_site, col_ts]]
            .assign(date=pd.to_datetime(df[col_ts]).dt.date, f_ts=1.0)
            .merge(candidate_keys[[col_site, "date"]], on=[col_site, "date"], how="inner")
            [[col_site, "date", col_ts, "f_ts"]]
            .reset_index(drop=True)
        )
        return ts_df, ["f_ts"], "ignored_interval_label"

    monkeypatch.setattr(
        "pynrpf.plugins.m8_xgb.M8XGBPlugin._load_bundle",
        lambda self, cfg: bundle,
    )
    monkeypatch.setattr("pynrpf.plugins.m8_xgb.build_xgb1_features", fake_xgb1_features)
    monkeypatch.setattr("pynrpf.plugins.m8_xgb.build_xgb2_features", fake_xgb2_features)


def test_m8_stage2_default_scoring_keeps_day_gate(monkeypatch) -> None:
    _patch_m8_inference(monkeypatch)

    out = run_inference(_sample_interval_df(), _base_cfg(score_all_days_for_review=False))
    result_df = out["data"]

    assert not result_df["pynrpf_day_flag"].any()
    assert not result_df["pynrpf_interval_flag"].any()
    assert result_df["m8_prob_ts"].isna().all()
    assert (result_df["pynrpf_corrected_net_load"] == result_df["net_load_MW"]).all()


def test_m8_stage2_review_mode_can_score_day_negative_rows(monkeypatch) -> None:
    _patch_m8_inference(monkeypatch)

    out = run_inference(_sample_interval_df(), _base_cfg(score_all_days_for_review=True))
    result_df = out["data"]

    assert not result_df["pynrpf_day_flag"].any()
    assert result_df["pynrpf_interval_flag"].all()
    assert result_df["m8_prob_ts"].notna().all()
    assert (result_df["pynrpf_corrected_net_load"] == -result_df["net_load_MW"]).all()

"""M7, the deterministic threshold rule, scored on every held-out station through the package.

Inputs:  the complete-day interval frame of a cohort and the folds.
Outputs: the interval prediction table for M7 (one row per quarter-hour) in the
         common schema shared by all methods (see ``common.py``).
Key steps: build the package input for one station; call ``pynrpf.api.run_inference``
         with the M7 configuration from the settings; merge the flags back by
         (site, timestamp). Nothing is fitted; the thresholds are configuration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import INTERVAL_COLUMNS, finish_interval_table
from .config import Settings, ensure_pynrpf_importable
from .data import model_input
from .folds import Fold


def inference_config(settings: Settings, model: str, bundle_uri: str | None = None) -> dict[str, Any]:
    """The pynrpf inference configuration for M7 or M8 from the evaluation settings."""
    cols = settings.columns
    cfg: dict[str, Any] = {
        "columns": {k: cols[k] for k in ("site", "timestamp", "net_load", "solar")},
        "runtime": {"interval_minutes": 15, "strict_validation": True},
        "model": {
            "selected_model": model,
            "m7_threshold": dict(settings["m7"]["m7_threshold"]),
            "m8_xgb": dict(settings["m8"]["m8_xgb"]),
        },
        "artifacts": {"m8_pretrained_bundle_uri": bundle_uri},
    }
    return cfg


def run_package(intervals: pd.DataFrame, settings: Settings, cfg: dict[str, Any]) -> pd.DataFrame:
    """Run pynrpf inference on an interval frame and return its flags aligned to the input rows."""
    ensure_pynrpf_importable()
    from pynrpf.api import run_inference

    cols = settings.columns
    frame = model_input(intervals, settings)
    out = run_inference(frame, cfg)["data"]
    keep = [cols["site"], cols["timestamp"], "pynrpf_interval_flag", "pynrpf_day_flag", "pynrpf_confidence"]
    extra = [c for c in ("m8_prob_day", "m8_prob_ts") if c in out.columns]
    out = out[keep + extra].rename(columns={cols["site"]: "station", cols["timestamp"]: "ts_naive"})
    out["ts_naive"] = pd.to_datetime(out["ts_naive"])
    key = pd.DataFrame({"station": intervals["station"].to_numpy(), "ts_naive": intervals["ts"].dt.tz_localize(None).to_numpy()})
    merged = key.merge(out, on=["station", "ts_naive"], how="left", validate="one_to_one")
    if len(merged) != len(intervals):
        raise ValueError("Package output does not align one-to-one with the input intervals.")
    return merged


def predict_station(intervals: pd.DataFrame, fold: Fold, settings: Settings) -> pd.DataFrame:
    """M7 interval predictions for one held-out station."""
    station = intervals[intervals["station"] == fold.held_out]
    merged = run_package(station, settings, inference_config(settings, "m7_dtr"))
    table = station[["cohort", "station", "date", "slot", "ts"]].copy()
    table["pred_interval"] = merged["pynrpf_interval_flag"].fillna(False).astype(bool).to_numpy()
    table["pred_day"] = merged["pynrpf_day_flag"].fillna(False).astype(bool).to_numpy()
    table["prob_day"] = np.nan
    table["prob_interval"] = merged["pynrpf_confidence"].to_numpy(float)
    return finish_interval_table(table, "m7", fold.fold_id)


def predict_cohort(intervals: pd.DataFrame, folds: list[Fold], settings: Settings) -> pd.DataFrame:
    """M7 interval predictions for every held-out station of one cohort."""
    cohort = intervals["cohort"].iloc[0]
    parts = [predict_station(intervals, f, settings) for f in folds if f.cohort == cohort]
    return pd.concat(parts, ignore_index=True)[INTERVAL_COLUMNS]

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .._legacy.m7_threshold import run_m7
from .base import BaseModelPlugin


class M7DTRPlugin(BaseModelPlugin):
    name = "m7_dtr"

    def run_inference(
        self,
        df: pd.DataFrame,
        cfg: Dict[str, Any],
        columns: Dict[str, str],
    ) -> pd.DataFrame:
        model_cfg = cfg.get("model", {})
        m7_cfg = model_cfg.get("m7_threshold", {})
        run_cfg = {"m7_threshold": m7_cfg}

        site_col = columns["site"]
        ts_col = columns["timestamp"]
        net_col = columns["net_load"]
        solar_col = columns["solar"]

        result = run_m7(
            df=df,
            cfg=run_cfg,
            col_site=site_col,
            col_ts=ts_col,
            col_net=net_col,
            col_solar=solar_col,
        )

        result["pynrpf_interval_flag"] = result["m7_rpf_flag"].fillna(False).astype(bool)
        result["pynrpf_day_flag"] = result["m7_rpf_day"].fillna(False).astype(bool)
        result["pynrpf_corrected_net_load"] = result["net_load_MW_m7"]
        result["pynrpf_confidence"] = np.where(result["pynrpf_interval_flag"], 1.0, 0.0)
        return result

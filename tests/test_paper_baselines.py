"""The re-homed M7 rule and M8 feature builders reproduce the conference codebase on a committed slice.

Protects: moving M7 and M8 from the old package into ``paper/baselines`` changed no
number. The expectations under ``tests/fixtures/baseline_*.parquet`` were generated
once with the old modules (``pynrpf._legacy.m7_threshold``, ``pynrpf._legacy.features``
at commit ab6e8bd) on four beta_D days: a wrong-sign day the rule skips for a missing
reading, a wrong-sign day it flags, Australia Day 2024 (the holiday feature) and a
clean day it wrongly flags. The M8 fit and prediction path is covered end to end by
the reference run under ``results/`` (``test_paper_results``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ARTICLE = Path(__file__).resolve().parents[1] / "publication" / "2_journal_article"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ARTICLE))

from paper import config  # noqa: E402
from paper.baselines import m7  # noqa: E402
from paper.baselines.m8_features import build_xgb1_features, build_xgb2_features  # noqa: E402

COLUMNS = {"site": "substation_id", "timestamp": "timestamp", "net_load": "net_load_MW", "solar": "solar_MW"}


@pytest.fixture(scope="module")
def days() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "baseline_days.parquet")


@pytest.fixture(scope="module")
def settings() -> config.Settings:
    return config.load(check_hashes=False)


def test_m7_rule_matches_the_conference_output(days, settings):
    expected = pd.read_parquet(FIXTURES / "baseline_m7_expected.parquet")
    got = m7.apply_rule(days, settings["m7"]["m7_threshold"], COLUMNS)
    assert (got["timestamp"].to_numpy() == expected["timestamp"].to_numpy()).all()
    assert np.array_equal(got["m7_rpf_flag"].to_numpy(), expected["m7_rpf_flag"].to_numpy())
    assert np.array_equal(got["m7_rpf_day"].to_numpy(), expected["m7_rpf_day"].to_numpy())
    # The slice exercises both paths: one flagged day, one skipped for a missing reading.
    assert got["m7_rpf_day"].any() and not got["m7_rpf_day"].all()


def _assert_frames_equal(got: pd.DataFrame, expected: pd.DataFrame) -> None:
    assert list(got.columns) == list(expected.columns)
    assert len(got) == len(expected)
    for column in expected.columns:
        g, e = got[column].to_numpy(), expected[column].to_numpy()
        if np.issubdtype(e.dtype, np.number):
            assert np.array_equal(g.astype(float), e.astype(float), equal_nan=True), column
        else:
            assert (g.astype(str) == e.astype(str)).all(), column


def test_m8_feature_builders_match_the_conference_output(days, settings):
    m8_cfg = settings["m8"]["m8_xgb"]
    work = days.copy()
    work["_gt_day"] = np.where(work["label_day"], -1.0, 1.0)
    work["_gt_interval"] = np.where(work["label_interval"], -1.0, 1.0)
    dates = sorted(work["timestamp"].dt.strftime("%Y-%m-%d").unique())
    day_df, feat1, label1 = build_xgb1_features(work, m8_cfg, COLUMNS, "_gt_day", (dates[0], dates[-1]))
    ts_df, feat2, label2 = build_xgb2_features(work, m8_cfg, COLUMNS, "_gt_interval", day_df,
                                               day_df[["substation_id", "date"]])
    day_df["date"] = day_df["date"].astype(str)
    ts_df["date"] = ts_df["date"].astype(str)
    expected_day = pd.read_parquet(FIXTURES / "baseline_xgb1_expected.parquet")
    expected_ts = pd.read_parquet(FIXTURES / "baseline_xgb2_expected.parquet")
    _assert_frames_equal(day_df, expected_day)
    _assert_frames_equal(ts_df, expected_ts)
    assert feat1 == [c for c in expected_day.columns if c not in ("substation_id", "date", label1)]
    assert feat2 == [c for c in expected_ts.columns if c not in ("substation_id", "date", "timestamp", label2)]
    assert int(expected_day.loc[expected_day["date"] == "2024-01-26", "is_holiday"].iloc[0]) == 1

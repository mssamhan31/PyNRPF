"""Hand-worked fixture for the macro (per-station mean) table of the evaluation package.

Protects: the macro table averages the station-level metrics with one vote per station,
so it differs from the pooled table exactly as the hand calculation says; a station
without an applied correction or without an RPF day is left out of the means that are
undefined for it but keeps its frozen Energy IoU of zero; counts are summed, not averaged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ARTICLE = Path(__file__).resolve().parents[1] / "publication" / "2_journal_article"
sys.path.insert(0, str(ARTICLE))

from final_eval import metrics  # noqa: E402
from final_eval.common import AUTO_CORRECT, AUTO_KEEP  # noqa: E402


def _day(
    cohort: str,
    station: str,
    date: str,
    rpf: int,
    outcome: str,
    proposed: float,
    required: float,
    correct: float,
) -> dict:
    """One site-day row with the columns the pooled and station summaries read (MWh)."""
    return dict(
        method="m9",
        cohort=cohort,
        station=station,
        date=date,
        confidence="controlled" if cohort == "alpha" else "sure",
        headline=True,
        rpf=rpf,
        outcome=outcome,
        prob_day=0.9 if outcome == AUTO_CORRECT else 0.1,
        proposed_mwh=proposed,
        required_mwh=required,
        correct_mwh=correct,
        false_mwh=proposed - correct,
        slot_tp=int(correct),
        slot_fp=int(proposed - correct),
        slot_fn=int(required - correct),
        window_iou=np.nan,
        start_error=np.nan,
        end_error=np.nan,
        min_change_mw=0.0,
        reference_min_change_mw=0.0,
    )


@pytest.fixture
def site_days() -> pd.DataFrame:
    # beta_A: two RPF days, both corrected perfectly -> IoU 1, precision 1.
    # beta_B: one RPF day corrected with half the energy right (IoU 5/15, precision 0.5)
    #         and one clean day kept -> day precision 1, day recall 1.
    # alpha_A: one clean day kept -> nothing applied, no RPF day: IoU 0, precisions undefined.
    rows = [
        _day("beta", "beta_A", "2024-01-01", 1, AUTO_CORRECT, 10.0, 10.0, 10.0),
        _day("beta", "beta_A", "2024-01-02", 1, AUTO_CORRECT, 10.0, 10.0, 10.0),
        _day("beta", "beta_B", "2024-01-01", 1, AUTO_CORRECT, 10.0, 10.0, 5.0),
        _day("beta", "beta_B", "2024-01-02", 0, AUTO_KEEP, 0.0, 0.0, 0.0),
        _day("alpha", "alpha_A", "2024-01-01", 0, AUTO_KEEP, 0.0, 0.0, 0.0),
    ]
    return pd.DataFrame(rows)


def test_macro_table_is_the_unweighted_station_mean(site_days):
    macro = metrics.macro_table(site_days).set_index("group")
    pooled = metrics.pooled_table(site_days).set_index("group")
    beta = macro.loc["beta"]
    # Pooled Beta: proposed 30, required 30, correct 25 -> IoU 25/35, precision 25/30.
    assert pooled.loc["beta", "energy_iou"] == pytest.approx(25 / 35)
    assert pooled.loc["beta", "energy_precision"] == pytest.approx(25 / 30)
    # Macro Beta: mean of (1, 1/3) and of (1, 0.5).
    assert beta["energy_iou"] == pytest.approx((1 + 1 / 3) / 2)
    assert beta["energy_precision"] == pytest.approx(0.75)
    assert beta["day_precision"] == pytest.approx(1.0)
    assert beta["day_recall"] == pytest.approx(1.0)
    assert beta["day_f1"] == pytest.approx(1.0)
    assert beta["n_stations"] == 2 and beta["n_days"] == 4 and beta["n_rpf"] == 3


def test_macro_table_drops_undefined_stations_but_keeps_zero_energy_iou(site_days):
    macro = metrics.macro_table(site_days).set_index("group")
    combined = macro.loc["combined"]
    alpha = macro.loc["alpha"]
    # alpha_A has no RPF day and no correction: IoU 0 by the frozen definition, the rest undefined.
    assert alpha["energy_iou"] == 0.0
    assert np.isnan(alpha["energy_precision"]) and np.isnan(alpha["day_recall"])
    assert alpha["n_stations"] == 1 and alpha["n_stations_with_rpf"] == 0
    assert alpha["n_stations_with_correction"] == 0
    # Combined: IoU averages all three stations; precision and recall only the two defined.
    assert combined["energy_iou"] == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert combined["energy_precision"] == pytest.approx(0.75)
    assert combined["day_recall"] == pytest.approx(1.0)
    assert combined["n_stations"] == 3 and combined["n_stations_with_rpf"] == 2
    assert combined["n_stations_with_correction"] == 2
    assert list(macro.index) == ["combined", "alpha", "beta"]

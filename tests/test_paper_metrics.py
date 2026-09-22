"""Hand-worked fixtures for the headline definitions, the macro table and the gate wording.

Protects: the headline numbers pool energies before dividing; the macro table averages
the station-level metrics with one vote per station, so it differs from the pooled
table exactly as the hand calculation says; a station without an applied correction or
without an RPF day is left out of the means that are undefined for it but keeps its
Energy IoU of zero; counts are summed, not averaged; the gate names M9 the default only
when every method is present and M9 passes on the gated cohort with the best Energy IoU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pynrpf.m9 import AUTO_CORRECT, AUTO_KEEP

ARTICLE = Path(__file__).resolve().parents[1] / "publication" / "2_journal_article"
sys.path.insert(0, str(ARTICLE))

from paper import metrics  # noqa: E402
from paper.config import Settings  # noqa: E402


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


def test_headline_pools_energy_before_dividing(site_days):
    beta = site_days[site_days["cohort"] == "beta"]
    h = metrics.headline(beta)
    assert h["n_days"] == 4 and h["n_rpf"] == 3
    assert h["energy_iou"] == pytest.approx(25 / 35)
    assert h["energy_precision"] == pytest.approx(25 / 30)
    assert h["day_precision"] == 1.0 and h["day_recall"] == 1.0 and h["day_f1"] == 1.0
    assert h["sure_day_recall"] == 1.0 and h["rate_uncertain"] == 0.0
    assert h["rate_auto_correct"] == pytest.approx(0.75)
    # Nothing proposed: IoU is 0 by definition and precision undefined.
    iou, precision = metrics.pooled_energy(site_days.iloc[4:5], np.array([False]))
    assert iou == 0.0 and np.isnan(precision)


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
    # alpha_A has no RPF day and no correction: IoU 0 by definition, the rest undefined.
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


def _pooled(rows: dict[str, tuple[float, float, float]]) -> pd.DataFrame:
    """A pooled table from (combined Energy IoU, Alpha precision, Beta precision) per method."""
    out = []
    for method, (iou, alpha, beta) in rows.items():
        out += [dict(method=method, group="combined", energy_iou=iou, energy_precision=np.nan),
                dict(method=method, group="alpha", energy_iou=np.nan, energy_precision=alpha),
                dict(method=method, group="beta", energy_iou=np.nan, energy_precision=beta)]
    return pd.DataFrame(out)


def test_gate_wording(site_days):
    settings = Settings(raw={"columns": {}, "paths": {"output_dir": "results"},
                             "gate": {"energy_precision_min": 0.9, "gated_cohort": "beta",
                                      "alpha_energy_precision_ceiling": 0.904}})
    incomplete = metrics.gate_decision(metrics.pooled_table(site_days), settings)
    assert not incomplete["complete"] and incomplete["decision"].startswith("Incomplete run")
    default = metrics.gate_decision(_pooled({"m7": (0.5, 0.8, 0.8), "m8": (0.7, 0.95, 0.85), "m9": (0.8, 0.9, 0.95)}),
                                    settings)
    assert default["m9_default"] and default["decision"] == "M9 is the default method"
    assert default["methods"]["m9"]["beta_gate_pass"] and not default["methods"]["m8"]["beta_gate_pass"]
    pooled = _pooled({"m7": (0.5, 0.8, 0.8), "m8": (0.7, 0.95, 0.95), "m9": (0.8, 0.9, 0.85)})
    fails_gate = metrics.gate_decision(pooled, settings)
    assert not fails_gate["m9_default"]
    assert fails_gate["decision"] == "M9 is not the default; the best validated method is M9"
    assert fails_gate["strongest_combined_energy_iou"] == "m9"

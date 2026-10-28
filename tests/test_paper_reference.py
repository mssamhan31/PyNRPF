"""Hand-worked fixtures for the reference-side terms: energy, minimum-demand impact and the outcome policy.

Protects: correction energy with partial overlap, an interior gap and non-contiguous
flags (M7 / M8 style); the site-day minimum-demand change; the three-way policy at
the band edges; binary outcomes for M7 / M8; the identical-key guarantee on a
synthetic cohort run through outcomes and the site-day table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pynrpf.m9 import AUTO_CORRECT, AUTO_KEEP, UNCERTAIN, decide

ARTICLE = Path(__file__).resolve().parents[1] / "publication" / "2_journal_article"
sys.path.insert(0, str(ARTICLE))

from paper import reference  # noqa: E402


def _day(values: dict[int, float]) -> np.ndarray:
    y = np.full(96, 2.0)
    for k, v in values.items():
        y[k] = v
    return y


def test_energy_terms_partial_overlap_gap_and_noncontiguous():
    # y = 2 MW everywhere except slot 40 (missing) and slot 41 (4 MW).
    # Slot energy = 2 * y * 0.25 = 1 MWh per 2 MW slot.
    y = _day({40: np.nan, 41: 4.0})
    truth = np.zeros(96, bool)
    truth[36:44] = True  # 8 labelled slots; slot 40 missing -> 6 * 1 + 2 = 8 MWh required
    proposed = np.zeros(96, bool)
    proposed[38:42] = True  # 38, 39 (1 MWh each), 40 (missing, 0), 41 (2 MWh) -> 4 MWh correct
    proposed[50] = True  # a stray non-contiguous flag outside the label: 1 MWh false
    p, r, c = reference.energy_terms(y, truth, proposed)
    assert p == pytest.approx(5.0)
    assert r == pytest.approx(8.0)
    assert c == pytest.approx(4.0)
    # IoU = c / (p + r - c) = 4 / 9; precision = 4 / 5
    assert c / (p + r - c) == pytest.approx(4 / 9)
    assert c / p == pytest.approx(0.8)


def test_minimum_demand_change_hand_computed():
    y = _day({30: 0.5, 31: 0.4, 32: 0.6})  # a shallow dip that a sign error hides
    flipped = np.zeros(96, bool)
    flipped[30:33] = True
    raw, applied, change = reference.minimum_demand(y, flipped)
    assert raw == pytest.approx(0.4)
    assert applied == pytest.approx(-0.6)
    assert change == pytest.approx(-1.0)
    raw2, applied2, change2 = reference.minimum_demand(y, np.zeros(96, bool))
    assert (raw2, applied2, change2) == pytest.approx((0.4, 0.4, 0.0))
    assert np.isnan(reference.minimum_demand(np.full(96, np.nan), flipped)[0])


def test_three_way_policy_at_band_edges():
    c = 0.7
    assert decide(0.7, c) == AUTO_CORRECT
    assert decide(0.6999, c) == UNCERTAIN
    assert decide(0.5, c) == UNCERTAIN
    assert decide(0.3, c) == AUTO_KEEP
    assert decide(0.3001, c) == UNCERTAIN
    assert decide(float("nan"), c) == UNCERTAIN


def _intervals(
    method: str,
    station: str,
    date: str,
    flags: dict[int, bool],
    y: np.ndarray,
    truth: np.ndarray,
    p=np.nan,
) -> pd.DataFrame:
    slot = np.arange(96)
    return pd.DataFrame(
        {
            "cohort": "beta",
            "station": station,
            "fold_id": f"beta_{station}",
            "method": method,
            "date": date,
            "slot": slot,
            "ts": pd.date_range(f"{date} 00:00", periods=96, freq="15min", tz="UTC"),
            "pred_interval": [flags.get(k, False) for k in slot],
            "pred_day": any(flags.values()),
            "prob_day": p,
            "prob_interval": np.nan,
            "y": y,
            "s": 0.0,
            "truth": truth,
        }
    )


def test_binary_outcomes_and_m9_outcomes_drive_applied_energy():
    y = _day({})
    truth = np.zeros(96, bool)
    truth[40:44] = True
    flags = {k: True for k in range(40, 44)}
    m7 = pd.concat(
        [
            _intervals("m7", "beta_A", "2024-01-01", flags, y, truth),
            _intervals("m7", "beta_A", "2024-01-02", {}, y, ~truth & False),
        ],
        ignore_index=True,
    )
    decided = reference.attach_outcomes(m7)
    by_day = decided.groupby("date")["outcome"].first()
    assert by_day["2024-01-01"] == AUTO_CORRECT and by_day["2024-01-02"] == AUTO_KEEP
    assert decided.loc[decided["applied"], "applied_y"].eq(-2.0).all()
    # M9: same window but the day is UNCERTAIN -> candidate energy kept, nothing applied.
    m9 = _intervals("m9", "beta_A", "2024-01-01", flags, y, truth, p=0.6)
    site = pd.DataFrame(
        {"cohort": ["beta"], "station": ["beta_A"], "date": ["2024-01-01"], "outcome": [UNCERTAIN]}
    )
    decided9 = reference.attach_outcomes(m9, site)
    assert not decided9["applied"].any()
    index = pd.DataFrame(
        {
            "cohort": ["beta"],
            "station": ["beta_A"],
            "date": ["2024-01-01"],
            "confidence": ["sure"],
            "headline": [True],
            "rpf": [1],
            "true_start": [40],
            "true_end": [43],
            "true_slots": [4],
        }
    )
    row = reference.siteday_table(decided9, index).iloc[0]
    assert row["candidate_mwh"] == pytest.approx(4.0)
    assert row["proposed_mwh"] == 0.0 and row["correct_mwh"] == 0.0
    assert row["required_mwh"] == pytest.approx(4.0)
    assert row["outcome"] == UNCERTAIN and row["min_change_mw"] == 0.0


def test_methods_share_identical_keys_after_pipeline():
    y = _day({})
    truth = np.zeros(96, bool)
    frames = []
    for method in ("m7", "m8"):
        for date in ("2024-01-01", "2024-01-02"):
            frames.append(
                reference.attach_outcomes(_intervals(method, "beta_A", date, {}, y, truth))
            )
    index = pd.DataFrame(
        {
            "cohort": "beta",
            "station": "beta_A",
            "date": ["2024-01-01", "2024-01-02"],
            "confidence": "sure",
            "headline": True,
            "rpf": 0,
            "true_start": -1,
            "true_end": -1,
            "true_slots": 0,
        }
    )
    tables = [
        reference.siteday_table(pd.concat(frames[i : i + 2], ignore_index=True), index) for i in (0, 2)
    ]
    keys = [set(map(tuple, t[["cohort", "station", "date"]].to_numpy())) for t in tables]
    assert keys[0] == keys[1] and len(keys[0]) == 2

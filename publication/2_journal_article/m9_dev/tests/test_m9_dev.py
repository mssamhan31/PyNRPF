"""Synthetic fixtures for the M9 reference scorer and metrics. No real site-day appears here.

Each test plants a known situation on a smooth synthetic day and checks the scorer's
answer by hand-derivable expectation: the null wins on a clean day, a planted sign
error is found at its exact edges, edge cases and tie rules behave as specified,
missing inputs abstain, raw data is preserved, and the energy metrics match a
hand-worked two-day example with a gap.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import m9_metrics as mx  # noqa: E402
import m9_scorer as ms  # noqa: E402

FLOOR = 0.01


def synthetic_day(demand_level: float = 4.0, solar_peak: float = 6.0, ramp: float = 0.0, noise: float = 0.0, seed: int = 0):
    """Smooth demand plus a midday solar bell; returns (y_true, s) with y_true = demand - solar."""
    t = np.arange(ms.SLOTS)
    rng = np.random.default_rng(seed)
    demand = demand_level + ramp * (t - 48) / 48 + noise * rng.standard_normal(ms.SLOTS)
    solar = np.clip(solar_peak * np.exp(-0.5 * ((t - 52) / 10.0) ** 2), 0, None)
    return demand - solar, solar


def plant_error(y_true: np.ndarray, start: int, end: int) -> np.ndarray:
    """Recorded net load if the sign was lost inside [start, end]: magnitude only."""
    y = y_true.copy()
    y[start : end + 1] = np.abs(y_true[start : end + 1])
    return y


def scored(y, s, **kw):
    return ms.score_siteday(y, s, FLOOR, **kw)


# --------------------------------------------------------------------------- decisions

def test_clean_day_null_wins():
    y, s = synthetic_day(demand_level=8.0, solar_peak=5.0)  # demand always above solar
    r = scored(y, s)
    assert r["input_ok"] and r["best_start"] < 0 and r["r_best"] <= 0.0


def test_planted_window_found():
    # The true edges sit where net load crosses zero, so the outermost slots carry
    # almost no correction energy and per-slot normalisation may drop them. The
    # physically meaningful requirement is high energy overlap, not exact slots.
    y_true, s = synthetic_day(demand_level=4.0, solar_peak=6.0)
    neg = np.flatnonzero(y_true < 0)
    a, b = int(neg[0]), int(neg[-1])
    y = plant_error(y_true, a, b)
    r = scored(y, s)
    assert r["best_score"] > 0
    assert a <= r["best_start"] <= a + 1 and b - 1 <= r["best_end"] <= b
    truth = y_true < 0
    p, req, c = mx.energy_terms(y, truth, r["best_start"], r["best_end"])
    assert c / (p + req - c) > 0.98 and c / p > 0.999


def test_vectorised_matches_reference():
    for seed in range(3):
        y_true, s = synthetic_day(demand_level=4.0, solar_peak=6.0, ramp=1.0, noise=0.2, seed=seed)
        y = plant_error(y_true, 44, 58)
        u0 = ms.reconstruct_uncorrected(y, s)
        for variant in ("sq", "abs", "tv"):
            g_ref, l_ref = ms.bridge_gain_matrix_reference(u0, y, variant)
            g_fast, l_fast = ms.bridge_gain_matrix(u0, y, variant)
            finite = np.isfinite(g_ref)
            assert np.array_equal(finite, np.isfinite(g_fast))
            assert np.allclose(g_ref[finite], g_fast[finite], rtol=1e-9, atol=1e-9)
            assert np.array_equal(l_ref, l_fast)


def test_llr_statistic_finds_planted_window_and_null_on_clean_day():
    y_true, s = synthetic_day(demand_level=4.0, solar_peak=6.0, noise=0.05, seed=1)
    neg = np.flatnonzero(y_true < 0)
    a, b = int(neg[0]), int(neg[-1])
    r = scored(plant_error(y_true, a, b), s, stat="llr")
    assert r["best_score"] > 0 and a <= r["best_start"] <= a + 1 and b - 1 <= r["best_end"] <= b
    y_clean, s_clean = synthetic_day(demand_level=8.0, solar_peak=5.0, noise=0.05, seed=2)
    assert scored(y_clean, s_clean, stat="llr")["best_start"] < 0


def test_one_slot_and_full_window():
    y_true, s = synthetic_day(demand_level=4.0, solar_peak=6.0)
    for a, b in [(50, 50), (ms.SCAN_START, ms.SCAN_END - 1)]:
        y = y_true.copy()
        y[a : b + 1] = np.abs(y_true[a : b + 1]) + 0.5  # force a visible flip everywhere in the window
        r = scored(y, s)
        assert r["best_start"] >= ms.SCAN_START and r["best_end"] <= ms.SCAN_END - 1


def test_windows_at_scan_edges_are_representable():
    sc = np.full((ms.N_WINDOWS, ms.N_WINDOWS), -np.inf)
    sc[0, 0] = 1.0
    assert ms.best_window(sc).start == ms.SCAN_START
    sc[:] = -np.inf
    sc[-1, -1] = 1.0
    assert ms.best_window(sc).end == ms.SCAN_END - 1


def test_tie_rules_shorter_then_earlier_then_null():
    sc = np.full((ms.N_WINDOWS, ms.N_WINDOWS), -np.inf)
    sc[2, 5] = 3.0   # length 4, start 26
    sc[3, 5] = 3.0   # length 3, start 27  <- shorter wins
    sc[4, 6] = 3.0   # length 3, start 28  <- later, loses to the earlier length-3
    assert ms.best_window(sc) == ms.Candidate(27, 29, 3.0)
    sc[:] = -np.inf
    sc[1, 1] = 0.0   # exactly zero ties the null; null must win
    best, ru = ms.rank(sc)
    assert best.is_null and ru.start == 25


def test_runner_up_does_not_overlap_best():
    y_true, s = synthetic_day(demand_level=4.0, solar_peak=6.0)
    neg = np.flatnonzero(y_true < 0)
    r = scored(plant_error(y_true, int(neg[0]), int(neg[-1])), s)
    assert r["runner_end"] < r["best_start"] or r["runner_start"] > r["best_end"] or r["runner_start"] < 0


# --------------------------------------------------------------------------- inputs

@pytest.mark.parametrize("slot", [ms.SCAN_START - 1, ms.SCAN_START + 10, ms.SCAN_END])
def test_missing_net_load_abstains_day_under_default_rule(slot):
    y, s = synthetic_day()
    y[slot] = np.nan
    assert not scored(y, s)["input_ok"]


def test_mask_windows_rule_scores_around_a_missing_slot():
    # A planted error at 44-58 and a missing reading at slot 30, well before it:
    # the day must still be scored and the chosen window must not touch slot 30.
    y_true, s = synthetic_day(demand_level=4.0, solar_peak=6.0)
    neg = np.flatnonzero(y_true < 0)
    y = plant_error(y_true, int(neg[0]), int(neg[-1]))
    y[30] = np.nan
    r = scored(y, s, missing="mask_windows")
    assert r["input_ok"] and r["best_score"] > 0
    assert r["best_start"] - 1 > 30 or r["best_end"] + 1 < 30
    adm = ms.admissible_windows(y, s)
    i, j = 30 - ms.SCAN_START, 31 - ms.SCAN_START
    assert not adm[i, j] and not adm[j - 1, j] and adm[j + 1, j + 3]


def test_nearest_anchor_reaches_a_window_beside_a_missing_block():
    # Slots 40-43 missing and the error starting at 44: under adjacent anchors the
    # true window cannot start before 45; under nearest anchors it bridges from 39.
    y_true, s = synthetic_day(demand_level=4.0, solar_peak=6.0)
    y = plant_error(y_true, 44, 58)
    y[40:44] = np.nan
    adm = ms.admissible_windows(y, s)
    assert adm[44 - ms.SCAN_START, 58 - ms.SCAN_START]
    r = scored(y, s, missing="mask_windows")
    assert r["input_ok"] and r["best_score"] > 0 and r["best_start"] <= 45 and r["best_end"] >= 57


def test_mask_windows_rule_abstains_when_nothing_is_scorable():
    y, s = synthetic_day()
    y[ms.SCAN_START - 1 : ms.SCAN_END + 1] = np.nan
    assert not scored(y, s, missing="mask_windows")["input_ok"]


def test_missing_solar_abstains_but_missing_overnight_does_not():
    y, s = synthetic_day()
    s2 = s.copy()
    s2[40] = np.nan
    assert not scored(y, s2)["input_ok"]
    y2 = y.copy()
    y2[3] = np.nan  # overnight slot: not read by the window scorer
    assert scored(y2, s)["input_ok"]


def test_zero_overnight_scale_uses_floor():
    u0 = np.full(ms.SLOTS, 5.0)  # perfectly flat: median step is 0
    assert ms.overnight_scale(u0, FLOOR) == FLOOR


# --------------------------------------------------------------------------- calibration and action

def test_decide_boundaries():
    c = 0.7
    assert ms.decide(0.7, c) == ms.AUTO_CORRECT and ms.decide(0.6999, c) == ms.UNCERTAIN
    assert ms.decide(0.3, c) == ms.AUTO_KEEP and ms.decide(0.3001, c) == ms.UNCERTAIN
    assert ms.decide(0.5, c) == ms.UNCERTAIN


def test_calibrator_monotone_and_threshold_roundtrip():
    r = np.concatenate([np.linspace(-500, -1, 200), np.linspace(1, 500, 200)])
    label = (r > 0).astype(int)
    cal = ms.Calibrator().fit(r, label)
    assert cal.beta > 0
    thr = cal.raw_threshold(0.7)
    assert abs(cal.probability(thr) - 0.7) < 1e-6


def test_raw_series_preserved_under_null():
    y, s = synthetic_day()
    out = ms.corrected_series(y, ms.NULL)
    assert np.array_equal(out, y) and out is not y


def test_corrected_series_flips_only_inside_window():
    y, _ = synthetic_day()
    out = ms.corrected_series(y, ms.Candidate(40, 45, 1.0))
    assert np.array_equal(out[40:46], -y[40:46]) and np.array_equal(out[:40], y[:40]) and np.array_equal(out[46:], y[46:])


# --------------------------------------------------------------------------- metrics, hand-worked

def test_energy_terms_hand_worked_with_gap():
    # Two labelled runs 40-41 and 44-45 (gap at 42-43), net load 2 MW everywhere.
    # Slot energy = 2*2*0.25 = 1 MWh. Proposed window 40-45 covers 6 slots.
    y = np.full(ms.SLOTS, 2.0)
    truth = np.zeros(ms.SLOTS, dtype=bool)
    truth[[40, 41, 44, 45]] = True
    p, r, c = mx.energy_terms(y, truth, 40, 45)
    assert (p, r, c) == (6.0, 4.0, 4.0)
    assert c / (p + r - c) == pytest.approx(4 / 6)   # Energy IoU
    assert c / p == pytest.approx(4 / 6)             # energy precision
    p0, r0, c0 = mx.energy_terms(y, truth, -1, -1)   # no window proposed
    assert (p0, r0, c0) == (0.0, 4.0, 0.0)


def test_missing_reading_carries_no_energy():
    y = np.full(ms.SLOTS, 2.0)
    y[41] = np.nan
    truth = np.zeros(ms.SLOTS, dtype=bool)
    truth[40:42] = True
    assert mx.energy_terms(y, truth, 40, 41) == (1.0, 1.0, 1.0)

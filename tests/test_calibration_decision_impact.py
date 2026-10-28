"""Steps 8 and 9, and the impact numbers, on planted inputs."""

import numpy as np
import pytest

from pynrpf import RELEASE_CALIBRATION, Calibration, fit_calibration
from pynrpf.impact import minimum_demand, proposed_energy_mwh, slot_energy
from pynrpf.m9.calibration import logistic, signed_log
from pynrpf.m9.decision import AUTO_CORRECT, AUTO_KEEP, UNCERTAIN, corrected_series, decide
from pynrpf.m9.winner import NO_CORRECTION, Candidate


def test_release_calibration_matches_the_reference_fit():
    assert RELEASE_CALIBRATION.intercept == pytest.approx(-4.590095, abs=1e-6)
    assert RELEASE_CALIBRATION.slope == pytest.approx(2.162080, abs=1e-6)
    # the thresholds the deck quotes: p = 0.7 near r = 11.4, p = 0.3 near r = 4.6
    assert RELEASE_CALIBRATION.evidence_at(0.7) == pytest.approx(11.37, abs=0.05)
    assert RELEASE_CALIBRATION.evidence_at(0.3) == pytest.approx(4.65, abs=0.05)


def test_signed_log_and_probability_are_monotone():
    r = np.array([-100.0, -10.0, 0.0, 10.0, 100.0, 1000.0])
    z = signed_log(r)
    assert np.all(np.diff(z) > 0) and z[2] == 0.0
    assert z[3] == pytest.approx(np.log(11.0))
    p = RELEASE_CALIBRATION.probability(r)
    assert np.all(np.diff(p) > 0) and 0.0 < p[0] < p[-1] < 1.0
    assert logistic(-1e6) == 0.0


def test_fit_with_slope_frozen_recovers_a_planted_intercept():
    rng = np.random.default_rng(9)
    truth = Calibration(-3.0, 2.0)
    evidence = rng.uniform(-5.0, 60.0, 4000)
    labels = rng.uniform(size=4000) < truth.probability(evidence)
    fitted = fit_calibration(evidence, labels, slope=2.0, provenance="planted")
    assert fitted.slope == 2.0
    assert fitted.intercept == pytest.approx(-3.0, abs=0.15)
    assert fitted.provenance == "planted"


def test_fit_with_slope_free_recovers_both_numbers():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(11)
    truth = Calibration(-3.0, 2.0)
    evidence = rng.uniform(-5.0, 60.0, 6000)
    labels = rng.uniform(size=6000) < truth.probability(evidence)
    fitted = fit_calibration(evidence, labels)
    assert fitted.intercept == pytest.approx(-3.0, abs=0.25)
    assert fitted.slope == pytest.approx(2.0, abs=0.2)


def test_fit_needs_both_labels():
    with pytest.raises(ValueError):
        fit_calibration(np.array([1.0, 2.0]), np.array([1, 1]), slope=2.0)


def test_decision_bands_and_guards():
    assert decide(0.70) == AUTO_CORRECT and decide(0.30) == AUTO_KEEP
    assert decide(0.69) == UNCERTAIN and decide(0.31) == UNCERTAIN
    assert decide(float("nan")) == UNCERTAIN
    assert decide(0.95, c=0.9) == AUTO_CORRECT and decide(0.85, c=0.9) == UNCERTAIN
    with pytest.raises(ValueError):
        decide(0.9, c=0.5)


def test_corrected_series_flips_only_the_window():
    y = np.arange(96, dtype=float)
    out = corrected_series(y, Candidate(40, 42, 1.0))
    assert (out[40:43] == -y[40:43]).all() and (out[:40] == y[:40]).all() and (out[43:] == y[43:]).all()
    assert (corrected_series(y, NO_CORRECTION) == y).all()
    assert y[40] == 40.0                                              # the input is untouched


def test_impact_on_a_hand_worked_day():
    y = np.full(96, 2.0)
    y[40:44] = 3.0                                                    # four slots of 3 MW: 2 * 3 * 0.25 MWh each
    y[10] = np.nan                                                    # a missing reading carries no energy
    window = Candidate(40, 43, 1.0)
    assert proposed_energy_mwh(y, window) == pytest.approx(4 * 1.5)
    assert proposed_energy_mwh(y, NO_CORRECTION) == 0.0
    assert slot_energy(np.array([np.nan]))[0] == 0.0
    recorded, corrected, change = minimum_demand(y, window)
    assert recorded == 2.0 and corrected == -3.0 and change == -5.0
    assert all(np.isnan(v) for v in minimum_demand(np.full(96, np.nan), window))

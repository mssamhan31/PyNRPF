"""The package reproduces the reference run on committed slices of two stations, to the last digit."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pynrpf
from pynrpf import Calibration

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.parametrize("name", ["beta_slice", "alpha_slice"])
def test_scores_probabilities_and_outcomes_match_the_reference(name):
    slice_ = pd.read_parquet(FIXTURES / f"{name}.parquet")
    pair = Calibration(float(slice_["cal_intercept"].iloc[0]), float(slice_["cal_slope"].iloc[0]))
    result = pynrpf.run(slice_[["substation_id", "timestamp", "net_load_MW", "solar_MW"]],
                        calibration=pair, phi=float(slice_["phi"].iloc[0]))
    expected = slice_.drop_duplicates("date").set_index("date")
    got = result.site_days.set_index("date").loc[expected.index]
    assert (got["input_ok"].to_numpy() == expected["input_ok"].to_numpy()).all()
    assert (got["n_admissible"].to_numpy() == expected["n_admissible"].to_numpy()).all()
    assert (got["window_start"].to_numpy() == expected["best_start"].to_numpy()).all()
    assert (got["window_end"].to_numpy() == expected["best_end"].to_numpy()).all()
    assert (got["runner_start"].to_numpy() == expected["runner_start"].to_numpy()).all()
    scored = expected["input_ok"].to_numpy()
    assert np.array_equal(got["evidence"].to_numpy()[scored], expected["r_best"].to_numpy()[scored])
    assert np.array_equal(got["p"].to_numpy()[scored], expected["prob_day"].to_numpy()[scored])
    assert (got["outcome"].to_numpy() == expected["outcome"].to_numpy()).all()
    assert np.array_equal(got["proposed_mwh"].to_numpy(), expected["candidate_mwh"].to_numpy())


def test_beta_slice_covers_the_awkward_cases():
    slice_ = pd.read_parquet(FIXTURES / "beta_slice.parquet").drop_duplicates("date")
    assert (slice_["best_start"] < 0).sum() > 50                      # days where no correction wins
    assert (~slice_["input_ok"]).sum() > 0                            # days with no admissible window
    readings = pd.read_parquet(FIXTURES / "beta_slice.parquet")
    assert readings["net_load_MW"].isna().sum() > 0                   # days with missing readings

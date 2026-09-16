"""Operating-point study on synthetic probabilities: selection rules and leakage guard.

Protects: rule L picks the smallest threshold whose expected precision meets the target
on a hand-built set; the operating threshold is the larger of the energy and day
thresholds; the per-fold selection never reads the held-out station's labels; the
symmetric two-control policy at c = 0.7 equals the locked one-control policy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ARTICLE = Path(__file__).resolve().parents[1] / "publication" / "2_journal_article"
sys.path.insert(0, str(ARTICLE))

from final_eval import operating_points as op  # noqa: E402
from final_eval.config import Settings  # noqa: E402
from final_eval.folds import Fold  # noqa: E402

SETTINGS = Settings(
    raw={
        "columns": {},
        "paths": {"output_dir": "outputs/01_final_evaluation", "m9_dev_dir": "m9_dev"},
        "m9": {"c": 0.7},
        "gate": {"energy_precision_min": 0.9},
        "metrics": {"bootstrap": {"draws": 20, "seed": 1, "percentiles": [2.5, 97.5]}},
        "operating_points": {
            "targets": [0.9, 0.95],
            "c_correct_grid": {"start": 0.5, "stop": 0.995, "step": 0.005},
            "heatmap_c_correct": [0.7, 0.9],
            "heatmap_c_keep": [0.1, 0.3],
            "recommended_rule": "L",
            "per_station_points": [{"label": "locked", "rule": "locked", "target": None}],
        },
    }
)


def _days(station: str, probs: list[float], rpf: list[int], mwh: float = 1.0) -> pd.DataFrame:
    n = len(probs)
    return pd.DataFrame(
        {
            "method": "m9",
            "cohort": "beta",
            "station": station,
            "fold_id": f"beta_{station}",
            "date": [f"2024-01-{k + 1:02d}" for k in range(n)],
            "headline": True,
            "prob_day": probs,
            "rpf": rpf,
            "pred_start": 40,
            "candidate_mwh": mwh,
            "candidate_correct_mwh": [mwh if r else 0.0 for r in rpf],
            "required_mwh": [mwh if r else 0.0 for r in rpf],
        }
    )


def test_rule_l_picks_smallest_threshold_meeting_expected_precision():
    cal = op._prepare(_days("beta_A", [0.99, 0.9, 0.8, 0.6], [1, 1, 1, 0]))
    thresholds = np.array([0.6, 0.8, 0.9, 0.99])
    curve = op.precision_curve(cal, thresholds, "L")
    # expected precision above t: 0.6 -> 0.8225, 0.8 -> 0.8967, 0.9 -> 0.945, 0.99 -> 0.99
    assert op.choose_threshold(curve, 0.90, "day_precision") == pytest.approx(0.9)
    assert op.choose_threshold(curve, 0.80, "day_precision") == pytest.approx(0.6)
    assert op.choose_threshold(curve, 0.995, "day_precision") == pytest.approx(
        0.99
    )  # unattainable: top of grid


def test_operating_threshold_is_the_larger_of_energy_and_day():
    t = op._prepare(
        pd.concat(
            [
                _days("beta_A", [0.99, 0.9, 0.8, 0.6], [1, 1, 0, 0], mwh=1.0),
                _days("beta_B", [0.95, 0.85, 0.7, 0.55], [1, 1, 1, 0], mwh=5.0),
                _days("beta_C", [0.97, 0.9, 0.75, 0.5], [1, 0, 1, 0], mwh=2.0),
            ]
        )
    )
    fold = Fold("beta_beta_C", "beta", "beta_C", m9_calibration=("beta_A", "beta_B"))
    thresholds = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    both = op.select_fold_threshold(t, fold, 0.9, 0.9, "E", thresholds)
    energy = op.select_fold_threshold(t, fold, 0.9, None, "E", thresholds)
    day = op.select_fold_threshold(t, fold, None, 0.9, "E", thresholds)
    assert both["c_correct"] == pytest.approx(max(energy["c_correct"], day["c_correct"]))
    assert both["binding"] in ("energy", "day", "both")


def test_selection_never_reads_held_out_labels():
    base = pd.concat(
        [
            _days("beta_A", [0.99, 0.9, 0.8, 0.6], [1, 1, 0, 0]),
            _days("beta_B", [0.95, 0.85, 0.7, 0.55], [1, 1, 1, 0]),
            _days("beta_C", [0.97, 0.9, 0.75, 0.5], [1, 0, 1, 0]),
        ]
    )
    flipped = base.copy()
    held = flipped["station"] == "beta_C"
    flipped.loc[held, "rpf"] = 1 - flipped.loc[held, "rpf"]
    flipped.loc[held, "candidate_correct_mwh"] = (
        flipped.loc[held, "candidate_mwh"] * flipped.loc[held, "rpf"]
    )
    fold = Fold("beta_beta_C", "beta", "beta_C", m9_calibration=("beta_A", "beta_B"))
    thresholds = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    for rule in ("L", "E"):
        a = op.select_fold_threshold(op._prepare(base), fold, 0.9, 0.9, rule, thresholds)
        b = op.select_fold_threshold(op._prepare(flipped), fold, 0.9, 0.9, rule, thresholds)
        assert a["c_correct"] == b["c_correct"]
    leaky = Fold("beta_beta_C", "beta", "beta_C", m9_calibration=("beta_A", "beta_C"))
    with pytest.raises(ValueError, match="held-out"):
        op.select_fold_threshold(op._prepare(base), leaky, 0.9, 0.9, "L", thresholds)


def test_symmetric_policy_at_locked_c_matches_one_control_policy():
    t = op._prepare(
        _days("beta_A", [0.95, 0.75, 0.69, 0.5, 0.31, 0.29, 0.1], [1, 1, 1, 0, 1, 0, 0])
    )
    point = op.evaluate_point(t, 0.7, 0.3)
    p = t["prob_day"].to_numpy()
    assert point["corrected_days"] == int((p >= 0.7).sum()) == 2
    assert point["review_days"] == int(((p > 0.3) & (p < 0.7)).sum()) == 3
    assert point["auto_fn"] == 0  # the RPF day at 0.31 is reviewed, not kept
    assert point["day_precision"] == pytest.approx(1.0)
    assert point["day_recall"] == pytest.approx(0.5)

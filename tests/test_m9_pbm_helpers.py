from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARTICLE_ROOT = Path(__file__).resolve().parents[1] / "publication" / "2_journal_article"
NOTEBOOK_DIR = ARTICLE_ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOK_DIR))

import _gamma_forecast as gamma_forecast  # noqa: E402
import _m9_pbm_data as m9_data  # noqa: E402
import _m9_pbm_features as m9_features  # noqa: E402
import _m9_pbm_validation as m9_validation  # noqa: E402


def test_m9_config_and_paths_are_portable() -> None:
    config = m9_data.load_experiment_config(ARTICLE_ROOT)
    paths = m9_data.resolve_paths(ARTICLE_ROOT, config)

    assert paths.article == ARTICLE_ROOT
    assert config["m9_pbm"]["example"] == {
        "substation_id": "alpha_F",
        "date": "2024-02-17",
    }
    assert config["m9_pbm"]["features"]["compact_names"] == (
        m9_features.COMPACT_FEATURE_COLUMNS
    )


def test_demand_reconstructions_flip_only_inside_candidate() -> None:
    net = np.array([1.0, 2.0, 3.0, 4.0])
    solar = np.array([10.0, 10.0, 10.0, 10.0])

    uncorrected, corrected = m9_features.reconstruct_demand(net, solar, 1, 2)

    assert np.array_equal(uncorrected, [11.0, 12.0, 13.0, 14.0])
    assert np.array_equal(corrected, [11.0, 8.0, 7.0, 14.0])


def test_candidate_geometry_respects_duration_daytime_and_solar_peak() -> None:
    spec = m9_features.CandidateSpec()
    solar = np.zeros(spec.slots_per_day)
    solar[48] = 5.0

    candidates = m9_features.candidate_windows(solar, spec)

    assert candidates["duration_slots"].min() == 2
    assert candidates["duration_slots"].max() == 32
    assert candidates["left_slot"].min() >= spec.scan_start_slot
    assert candidates["right_slot"].max() <= spec.scan_end_slot
    midpoint = (candidates["left_slot"] + candidates["right_slot"]) / 2
    assert (midpoint.sub(48).abs() <= spec.solar_peak_radius_slots).all()


def test_bridge_uses_outside_anchors_and_median_absolute_error() -> None:
    spec = m9_features.CandidateSpec(slots_per_day=8)
    anchor_curve = np.array([0.0, 1.0, 2.0, 30.0, 40.0, 5.0, 6.0, 7.0])
    candidate_curve = anchor_curve.copy()
    candidate_curve[3:5] = [3.0, 4.0]

    line, anchors = m9_features.bridge_line(anchor_curve, 3, 4, spec)
    error = m9_features.bridge_error(candidate_curve, anchor_curve, 3, 4, spec)

    assert anchors == (2, 5)
    assert np.allclose(line, [3.0, 4.0])
    assert np.isclose(error, 0.0)


def test_candidate_features_are_finite_and_duration_score_saturates() -> None:
    spec = m9_features.CandidateSpec()
    slots = np.arange(spec.slots_per_day)
    solar = np.maximum(0.0, 6 * np.sin(np.pi * (slots - 24) / 48))
    net = 3 + 0.2 * np.cos(2 * np.pi * slots / 96)
    net[40:56] += 2 * np.sin(np.pi * np.arange(16) / 15)

    candidates, audit = m9_features.compute_candidate_features(
        net,
        solar,
        substation_solar_scale=5.0,
        spec=spec,
    )

    duration_scores = candidates.set_index("duration_slots")["F4_duration_plausibility"]
    assert np.isclose(duration_scores.loc[2].iloc[0], 0.5 / 1.5)
    assert np.isclose(duration_scores.loc[6].iloc[0], 1.0)
    assert np.isclose(duration_scores.loc[32].iloc[0], 1.0)
    assert np.isfinite(candidates.filter(regex=r"^F[1-7]_").to_numpy()).all()
    assert candidates["F5_n_height_ratio"].between(0, 1).all()
    assert candidates["F6_solar_strength_ratio"].between(0, 1).all()
    assert candidates["F7_solar_peak_alignment"].between(0, 1).all()
    assert audit["candidate_count"] == len(candidates)


def test_best_candidate_tie_break_is_earlier_then_shorter() -> None:
    candidates = pd.DataFrame(
        {
            "dataset": "beta",
            "substation_id": "beta_A",
            "date": "2024-01-01",
            "candidate_id": [0, 1, 2],
            "left_slot": [40, 39, 39],
            "right_slot": [45, 46, 44],
            "duration_slots": [6, 8, 6],
            "F1_bridge_improvement": [1.0, 1.0, 1.0],
        }
    )

    selected = m9_features.select_best_candidates(
        candidates, {"F1_bridge_improvement": 1.0}
    )

    assert selected.iloc[0]["candidate_id"] == 2


def test_window_and_energy_iou_cover_tp_fp_fn_and_empty_cases() -> None:
    assert np.isclose(m9_features.window_iou({1, 2}, {2, 3}), 1 / 3)
    assert m9_features.window_iou({1, 2}, set()) == 0.0
    assert m9_features.window_iou(set(), {1, 2}) == 0.0
    assert np.isnan(m9_features.window_iou(set(), set()))

    metrics = m9_features.correction_energy_components(
        np.array([1.0, 2.0, 3.0]),
        np.array([True, True, False]),
        np.array([False, True, True]),
    )
    assert np.isclose(metrics["overlap_correction_MWh"], 1.0)
    assert np.isclose(metrics["union_correction_MWh"], 3.0)
    assert np.isclose(metrics["energy_iou"], 1 / 3)


def test_threshold_selection_uses_macro_substation_tie_breaks() -> None:
    training = pd.DataFrame(
        {
            "dataset": ["beta"] * 6,
            "substation_id": ["beta_A"] * 3 + ["beta_B"] * 3,
            "true_day": [False, True, True, False, False, True],
            "score": [0.1, 0.7, 0.9, 0.2, 0.4, 0.8],
        }
    )

    selected = m9_validation.select_threshold(training)

    assert np.isclose(selected.threshold, 0.7)
    assert np.isclose(selected.metrics["macro_f1"], 1.0)


def test_loso_never_includes_heldout_substation() -> None:
    frame = pd.DataFrame(
        {
            "substation_id": ["beta_A", "beta_A", "beta_B", "beta_B"],
            "true_day": [False, True, False, True],
        }
    )
    for heldout, train_index, test_index in m9_validation.beta_loso_folds(frame):
        training = frame.loc[train_index]
        testing = frame.loc[test_index]
        m9_validation.assert_heldout_absent(training, heldout)
        assert testing["substation_id"].eq(heldout).all()


def test_weight_generators_meet_positive_simplex_contract() -> None:
    grid = m9_validation.simplex_grid()
    random = m9_validation.random_simplex_weights(1000, seed=9)

    assert len(grid) == 171
    assert np.allclose(grid.sum(axis=1), 1.0)
    assert np.allclose(random.sum(axis=1), 1.0)
    assert grid.min().min() >= 0.05
    assert random.min().min() >= 0.05
    assert random.equals(m9_validation.random_simplex_weights(1000, seed=9))


def test_all_feature_subsets_and_partition_maxima_are_self_consistent() -> None:
    definitions = m9_validation.all_nonempty_feature_subsets(
        m9_features.FEATURE_COLUMNS
    )
    weights = m9_validation.subset_equal_weight_matrix(
        definitions, m9_features.FEATURE_COLUMNS
    )
    assert len(definitions) == 511
    assert definitions["subset_mask"].nunique() == 511
    assert set(definitions["feature_count"]) == set(range(1, 10))
    assert np.allclose(weights.sum(axis=1), 1.0)

    candidates = pd.DataFrame(
        {
            "dataset": "beta",
            "substation_id": "beta_A",
            "date": ["2024-01-01"] * 2 + ["2024-01-02"] * 2,
            "candidate_id": [0, 1, 0, 1],
            "F1": [0.1, 0.8, 0.4, 0.2],
            "F2": [0.9, 0.1, 0.3, 0.7],
        }
    )
    keys, maxima = m9_features.maximum_subset_scores(
        candidates,
        feature_columns=["F1", "F2"],
        weight_matrix=np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        batch_days=1,
    )
    assert keys["date"].tolist() == ["2024-01-01", "2024-01-02"]
    assert np.allclose(maxima, [[0.8, 0.9, 0.5], [0.4, 0.7, 0.45]])


def test_confidence_coverage_counts_auto_accept_and_manual_review() -> None:
    predictions = pd.DataFrame(
        {
            "substation_id": ["beta_A"] * 4,
            "date": pd.date_range("2024-01-01", periods=4).strftime("%Y-%m-%d"),
            "true_day": [True, False, True, False],
            "predicted_day": [True, False, False, True],
            "confidence_margin": [0.9, 0.8, 0.2, 0.1],
        }
    )

    coverage = m9_validation.confidence_coverage_metrics(predictions, [50, 100])

    first = coverage.iloc[0]
    assert first["auto_accepted_days"] == 2
    assert first["manual_review_days"] == 2
    assert first["auto_errors"] == 0
    assert coverage.iloc[1]["auto_errors"] == 2


def test_gamma_correction_does_not_use_heldout_labels_for_m9_prediction() -> None:
    timestamps = pd.date_range("2024-01-01", periods=96, freq="15min", tz="UTC")
    gamma = pd.DataFrame(
        {
            "substation_id": "beta_B",
            "date": "2024-01-01",
            "timestamp": timestamps,
            "net_load_MW": np.arange(96, dtype=float),
            "label_interval": False,
        }
    )
    candidates = pd.DataFrame(
        {
            "dataset": "beta",
            "substation_id": "beta_B",
            "date": "2024-01-01",
            "candidate_id": [0],
            "left_slot": [40],
            "right_slot": [45],
            "duration_slots": [6],
            "F1_bridge_improvement": [1.0],
            "F3_slope_continuity_improvement": [0.5],
            "F4_duration_plausibility": [1.0],
        }
    )
    model = {
        "heldout_substation": "beta_B",
        "threshold": 0.5,
        "weights": {
            "F1_bridge_improvement": 1 / 3,
            "F3_slope_continuity_improvement": 1 / 3,
            "F4_duration_plausibility": 1 / 3,
        },
    }

    first, _ = gamma_forecast.apply_m9_pbm_correction(gamma, candidates, model)
    gamma.loc[gamma.index[10:20], "label_interval"] = True
    second, _ = gamma_forecast.apply_m9_pbm_correction(gamma, candidates, model)

    assert first["predicted_interval"].equals(second["predicted_interval"])
    assert first["m9_pbm_corrected_MW"].equals(second["m9_pbm_corrected_MW"])
    assert not first["manually_corrected_MW"].equals(second["manually_corrected_MW"])


def test_gamma_forecast_features_stop_at_exact_seven_day_origin() -> None:
    timestamps = pd.date_range("2024-01-01", periods=45 * 96, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "raw_uncorrected_MW": np.arange(len(timestamps), dtype=float),
        }
    )
    frame["manually_corrected_MW"] = frame["raw_uncorrected_MW"]
    target = pd.Timestamp("2024-02-10 12:00:00", tz="UTC")
    baseline = gamma_forecast.build_forecast_examples(
        frame,
        "raw_uncorrected_MW",
        target_start="2024-02-10",
        target_end="2024-02-10",
    )
    row = baseline.loc[baseline["target_timestamp"].eq(target)].iloc[0]
    assert row["target_timestamp"] - row["origin_timestamp"] == pd.Timedelta(days=7)

    changed = frame.copy()
    future = changed["timestamp"].gt(row["origin_timestamp"])
    changed.loc[future, "raw_uncorrected_MW"] += 1_000_000
    rebuilt = gamma_forecast.build_forecast_examples(
        changed,
        "raw_uncorrected_MW",
        target_start="2024-02-10",
        target_end="2024-02-10",
    )
    changed_row = rebuilt.loc[rebuilt["target_timestamp"].eq(target)].iloc[0]
    feature_columns = gamma_forecast.forecast_feature_columns()
    assert np.allclose(
        row[feature_columns].to_numpy(dtype=float),
        changed_row[feature_columns].to_numpy(dtype=float),
        equal_nan=True,
    )

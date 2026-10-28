"""Fold construction and leakage guards of the paper's evaluation code, on synthetic indexes.

Protects: every station held out exactly once; no fold trains or calibrates on its
held-out station; Beta 'unsure' days never enter a training set; the M8 training path
refuses a frame containing the held-out station; manifests carry no absolute path; the
``beta_only`` scope fits every Alpha fold on all Beta stations and every Beta fold on the
other Beta stations, so the ten Alpha folds share one training key and one M8 bundle.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

ARTICLE = Path(__file__).resolve().parents[1] / "publication" / "2_journal_article"
sys.path.insert(0, str(ARTICLE))

from paper import folds as fl  # noqa: E402
from paper import manifest as mf  # noqa: E402
from paper.baselines import m8  # noqa: E402
from paper.config import Settings  # noqa: E402

SETTINGS = Settings(
    raw={
        "columns": {
            "site": "substation_id",
            "date": "date",
            "timestamp": "timestamp",
            "net_load": "net_load_MW",
            "solar": "solar_MW",
            "label_interval": "label_interval",
            "label_day": "label_day",
            "confidence": "confidence",
        },
        "population": {
            "slots_per_day": 96,
            "cohorts": ["alpha", "beta"],
            "alpha_confidence": "controlled",
            "headline_confidence": ["controlled", "sure"],
        },
        "paths": {"output_dir": "results"},
    }
)


def _index(cohort: str, stations: list[str], confidences: list[str]) -> pd.DataFrame:
    rows = []
    for station in stations:
        for k, conf in enumerate(confidences):
            rows.append(
                dict(
                    cohort=cohort,
                    station=station,
                    date=f"2024-01-{k + 1:02d}",
                    n_slots=96,
                    n_finite=96,
                    confidence=conf,
                    rpf=int(k % 2),
                    true_start=-1,
                    true_end=-1,
                    true_slots=0,
                    complete=True,
                    headline=conf in ("controlled", "sure"),
                )
            )
    return pd.DataFrame(rows)


@pytest.fixture
def indexes() -> dict[str, pd.DataFrame]:
    return {
        "alpha": _index("alpha", ["alpha_A", "alpha_B"], ["controlled"] * 3),
        "beta": _index("beta", ["beta_A", "beta_B"], ["sure", "unsure", "sure"]),
    }


def test_every_station_held_out_exactly_once(indexes):
    folds = fl.build_folds(indexes, SETTINGS)
    manifest = fl.fold_manifest(folds, indexes)
    fl.check_manifest(manifest)
    assert sorted(manifest["held_out"]) == ["alpha_A", "alpha_B", "beta_A", "beta_B"]
    assert len(folds) == 4
    rebuilt = fl.folds_from_manifest(manifest)
    assert [f.fold_id for f in rebuilt] == [f.fold_id for f in folds]
    assert all(r.m8_training == f.m8_training for r, f in zip(rebuilt, folds, strict=True))


def test_training_sets_exclude_held_out_and_unsure(indexes):
    folds = fl.build_folds(indexes, SETTINGS)
    for fold in folds:
        assert fold.held_out not in fold.training_stations()
        assert fold.held_out not in fold.m9_calibration
        train = fl.training_days(indexes, fold)
        assert fold.held_out not in set(train["station"])
        assert not (train["confidence"] == "unsure").any()
        # Under the default scopes M9 calibrates within cohort only; M8 sees the other cohort too.
        assert set(fold.m9_calibration) <= set(indexes[fold.cohort]["station"])
        assert len(fold.m8_training) == 3


def test_leakage_guard_refuses_held_out_station(indexes):
    fold = fl.build_folds(indexes, SETTINGS)[0]
    clean = pd.DataFrame({"substation_id": ["alpha_B", "beta_A"]})
    fl.assert_no_leakage(fold, clean, "substation_id")
    leaky = pd.DataFrame({"substation_id": ["alpha_B", fold.held_out]})
    with pytest.raises(ValueError, match="held-out station"):
        fl.assert_no_leakage(fold, leaky, "substation_id")


def test_manifest_rejects_absolute_paths(tmp_path):
    mf.assert_repository_relative(
        {"inputs": [{"path": "dataset/final/dataset_alpha.parquet"}], "nested": ["outputs/x.csv"]}
    )
    with pytest.raises(ValueError, match="absolute path"):
        mf.assert_repository_relative({"inputs": [{"path": str(tmp_path / "x.parquet")}]})
    with pytest.raises(ValueError, match="absolute path"):
        mf.assert_repository_relative(["/home/someone/file.csv"])


# ----------------------------------------------------------------------------- beta_only scope


def _settings(**folds_block) -> Settings:
    """A copy of SETTINGS with an optional ``folds`` block (tests omit it; the YAML has one)."""
    raw = deepcopy(SETTINGS.raw)
    if folds_block:
        raw["folds"] = folds_block
    return Settings(raw=raw)


@pytest.fixture
def full_indexes() -> dict[str, pd.DataFrame]:
    """The real station layout: ten Alpha and eight Beta stations, three days each."""
    alpha = [f"alpha_{c}" for c in "ABCDEFGHIJ"]
    beta = [f"beta_{c}" for c in "ABCDEFGH"]
    return {
        "alpha": _index("alpha", alpha, ["controlled"] * 3),
        "beta": _index("beta", beta, ["sure", "unsure", "sure"]),
    }


def test_default_scopes_and_unknown_scope_is_refused(indexes):
    defaults = ("all_other_stations_both_cohorts", "other_stations_same_cohort")
    assert fl.fold_scopes(SETTINGS) == defaults
    with pytest.raises(ValueError, match="m8_training_scope"):
        fl.build_folds(indexes, _settings(m8_training_scope="everything"))
    with pytest.raises(ValueError, match="m9_calibration_scope"):
        fl.build_folds(indexes, _settings(m9_calibration_scope="everything"))


def test_beta_only_scope_fits_on_beta_stations_only(full_indexes):
    settings = _settings(m8_training_scope="beta_only", m9_calibration_scope="beta_only")
    folds = fl.build_folds(full_indexes, settings)
    manifest = fl.fold_manifest(folds, full_indexes)
    fl.check_manifest(manifest)
    beta = set(full_indexes["beta"]["station"])
    assert len(folds) == 18
    for fold in folds:
        assert fold.held_out not in fold.training_stations()
        assert fold.held_out not in fold.m9_calibration
        assert fold.training_stations() <= beta and set(fold.m9_calibration) <= beta
        if fold.cohort == "alpha":
            # An Alpha fold is pure transfer: fitted on all eight Beta stations.
            assert fold.training_stations() == beta and set(fold.m9_calibration) == beta
        else:
            assert fold.training_stations() == beta - {fold.held_out}
            assert len(fold.m8_training) == 7 and len(fold.m9_calibration) == 7
        train = fl.training_days(full_indexes, fold)
        assert not (train["confidence"] == "unsure").any()
    alpha_rows = manifest[manifest["cohort"] == "alpha"]
    beta_rows = manifest[manifest["cohort"] == "beta"]
    # Ten Alpha folds share one training key and one signature; every Beta fold has its own.
    assert len(alpha_rows) == 10 and alpha_rows["m8_training_keys_sha256"].nunique() == 1
    assert alpha_rows["m8_training_signature"].nunique() == 1
    assert beta_rows["m8_training_signature"].nunique() == 8
    assert manifest["m8_training_signature"].nunique() == 9


def test_default_scopes_give_every_fold_its_own_signature(full_indexes):
    manifest = fl.fold_manifest(fl.build_folds(full_indexes, SETTINGS), full_indexes)
    assert manifest["m8_training_signature"].nunique() == 18


def test_shared_bundle_is_reused_for_an_equal_training_signature(tmp_path, full_indexes):
    beta_only = _settings(m8_training_scope="beta_only", m9_calibration_scope="beta_only")
    settings = Settings(raw=deepcopy(beta_only.raw), root=tmp_path)
    folds = fl.build_folds(full_indexes, settings)
    alpha_a, alpha_b = folds[0], folds[1]
    beta_a = next(f for f in folds if f.cohort == "beta")
    bundle = m8.bundle_path(settings, alpha_a)
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"a bundle stands here")
    donor = {
        "fold_id": alpha_a.fold_id,
        "held_out": alpha_a.held_out,
        "training_stations": sorted(alpha_a.training_stations()),
        "training_signature": fl.training_signature(alpha_a),
        "shared_with": None,
        "n_training_rows": 1,
        "n_training_rpf_days": 0,
        "bundle": settings.relative(bundle),
        "validation_metrics": {},
        "elapsed_s": 1.0,
        "skipped": False,
    }
    m8.bundle_manifest_path(settings, alpha_a).write_text(json.dumps(donor), encoding="utf-8")
    # The second Alpha fold never trains: intervals are not even read.
    record = m8.train_fold(alpha_b, intervals={}, settings=settings)
    assert record["shared_with"] == alpha_a.fold_id
    assert record["bundle"] == donor["bundle"] and record["fold_id"] == alpha_b.fold_id
    assert settings.root / record["bundle"] == bundle
    written = json.loads(m8.bundle_manifest_path(settings, alpha_b).read_text())
    assert written["shared_with"] == alpha_a.fold_id
    # A Beta fold has a different training set and finds no donor.
    assert m8.find_shared_bundle(settings, beta_a, fl.training_signature(beta_a)) is None
    # A shared manifest is never itself a donor, so a chain cannot form.
    alpha_c = folds[2]
    donor_again = m8.find_shared_bundle(settings, alpha_c, fl.training_signature(alpha_c))
    assert donor_again["fold_id"] == alpha_a.fold_id

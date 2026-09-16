"""Fold construction and leakage guards of the Phase 3 evaluation package, on synthetic indexes.

Protects: every station held out exactly once; no fold trains or calibrates on its
held-out station; Beta 'unsure' days never enter a training set; the M8 training path
refuses a frame containing the held-out station; manifests carry no absolute path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ARTICLE = Path(__file__).resolve().parents[1] / "publication" / "2_journal_article"
sys.path.insert(0, str(ARTICLE))

from final_eval import folds as fl  # noqa: E402
from final_eval import manifest as mf  # noqa: E402
from final_eval.config import Settings  # noqa: E402

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
        "paths": {"output_dir": "outputs/01_final_evaluation", "m9_dev_dir": "m9_dev"},
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


def test_training_sets_exclude_held_out_and_unsure(indexes):
    folds = fl.build_folds(indexes, SETTINGS)
    for fold in folds:
        assert fold.held_out not in fold.training_stations()
        assert fold.held_out not in fold.m9_calibration
        assert all(c == fold.cohort for c in [fold.cohort])
        train = fl.training_days(indexes, fold)
        assert fold.held_out not in set(train["station"])
        assert not (train["confidence"] == "unsure").any()
        # M9 calibrates within cohort only; M8 sees the other cohort too.
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

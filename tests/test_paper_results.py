"""Checks on the committed reference run under publication/2_journal_article/results/.

Protects: every stage manifest matches the files on disk (paths, SHA-256); the fold
manifest holds every station exactly once; the site-day table holds only held-out
predictions; the committed M9 pooled Beta row reproduces from the committed site-day
table; no manifest carries an absolute path.

Each test skips when the reference run is not present, so a checkout without
``results/`` still passes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ARTICLE = Path(__file__).resolve().parents[1] / "publication" / "2_journal_article"
RESULTS = ARTICLE / "results"
sys.path.insert(0, str(ARTICLE))

from paper import config, metrics  # noqa: E402
from paper import folds as fl  # noqa: E402
from paper import manifest as mf  # noqa: E402

HEADLINE_KEYS = ("n_days", "n_rpf", "energy_iou", "energy_precision", "day_f1", "day_precision", "sure_day_recall",
                 "rate_uncertain")


def _need(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"reference run file not present: {path.relative_to(ARTICLE)}")
    return path


@pytest.fixture(scope="module")
def settings() -> config.Settings:
    return config.load(check_hashes=False)


def test_manifests_match_the_files_on_disk(settings):
    folder = _need(RESULTS / "manifests")
    files = sorted(folder.glob("*.json"))
    assert files, "no manifests written"
    problems = {path.name: mf.check_manifest(settings, path) for path in files}
    assert all(not p for p in problems.values()), {k: v for k, v in problems.items() if v}


def test_manifests_are_repository_relative():
    folder = _need(RESULTS / "manifests")
    for path in sorted(folder.glob("*.json")):
        mf.assert_repository_relative(json.loads(path.read_text(encoding="utf-8")))


def test_fold_manifest_is_valid():
    manifest = pd.read_csv(_need(RESULTS / "01_data_folds" / "fold_manifest.csv"))
    fl.check_manifest(manifest)
    assert len(manifest) == 18
    assert manifest["n_m8_training_days"].gt(0).all()
    assert manifest["m8_training_signature"].nunique() == 9   # ten Alpha folds share one M8 bundle


def test_site_day_table_holds_only_held_out_predictions():
    site_days = pd.read_parquet(_need(RESULTS / "04_metrics" / "site_days.parquet"))
    manifest = pd.read_csv(_need(RESULTS / "01_data_folds" / "fold_manifest.csv"))
    held = dict(zip(manifest["fold_id"], manifest["held_out"], strict=True))
    assert (site_days["fold_id"].map(held) == site_days["station"]).all()
    counts = site_days.groupby("method").size()
    assert counts.nunique() == 1, "methods differ in site-day counts"


def test_m9_pooled_beta_row_reproduces_from_the_site_day_table():
    site_days = pd.read_parquet(_need(RESULTS / "04_metrics" / "site_days.parquet"))
    committed = pd.read_csv(_need(RESULTS / "04_metrics" / "pooled.csv"), float_precision="round_trip")
    recomputed = metrics.pooled_table(site_days[site_days["method"] == "m9"])
    new = recomputed[recomputed["group"] == "beta"].iloc[0]
    old = committed[(committed["method"] == "m9") & (committed["group"] == "beta")].iloc[0]
    for key in HEADLINE_KEYS:
        assert float(new[key]) == pytest.approx(float(old[key]), abs=1e-12), key

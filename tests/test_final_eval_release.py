"""Checks on the committed Phase 3 release files (no computation, no real data read).

Protects: the M9 port reproduces the frozen m9_dev run to four decimals; the fold
manifest of the release holds every station exactly once; the pooled table holds
only held-out predictions; every manifest is repository-relative; the command line
entry point answers --help.

Each test skips when the release outputs are not present, so a fresh checkout that
has not run the evaluation still passes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ARTICLE = Path(__file__).resolve().parents[1] / "publication" / "2_journal_article"
OUT = ARTICLE / "outputs" / "01_final_evaluation"
sys.path.insert(0, str(ARTICLE))

from final_eval import folds as fl  # noqa: E402
from final_eval import manifest as mf  # noqa: E402


def _need(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"release file not present: {path.relative_to(ARTICLE)}")
    return path


def test_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "final_eval", "--help"], cwd=ARTICLE, capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "train-m8" in result.stdout


def test_m9_port_reproduces_frozen_run():
    pooled = pd.read_csv(_need(OUT / "07_metrics" / "pooled.csv"))
    frozen = pd.read_csv(
        _need(ARTICLE / "m9_dev" / "runs" / "phase5_final_rev2" / "summary_pooled.csv")
    )
    for cohort in ("alpha", "beta"):
        new = pooled[(pooled["method"] == "m9") & (pooled["group"] == cohort)].iloc[0]
        old = frozen[(frozen["cohort"] == cohort) & (frozen["group"] == "headline")].iloc[0]
        for key in (
            "n_days",
            "n_rpf",
            "energy_iou",
            "energy_precision",
            "day_f1",
            "day_precision",
            "sure_day_recall",
        ):
            assert float(new[key]) == pytest.approx(float(old[key]), abs=5e-5), f"{cohort} {key}"


def test_release_fold_manifest_is_valid():
    manifest = pd.read_csv(_need(OUT / "01_folds" / "fold_manifest.csv"))
    fl.check_manifest(manifest)
    assert len(manifest) == 18
    assert manifest["n_m8_training_days"].gt(0).all()


def test_pooled_table_contains_only_held_out_predictions():
    site_days = pd.read_parquet(_need(OUT / "06_site_days" / "site_days.parquet"))
    manifest = pd.read_csv(_need(OUT / "01_folds" / "fold_manifest.csv"))
    held = dict(zip(manifest["fold_id"], manifest["held_out"], strict=True))
    assert (site_days["fold_id"].map(held) == site_days["station"]).all()
    counts = site_days.groupby("method").size()
    assert counts.nunique() == 1, "methods differ in site-day counts"


def test_release_manifests_are_repository_relative():
    folder = _need(OUT / "manifests")
    files = sorted(folder.glob("*.json"))
    assert files, "no manifests written"
    for path in files:
        mf.assert_repository_relative(json.loads(path.read_text(encoding="utf-8")))

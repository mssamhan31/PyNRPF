"""Load and validate the one configuration file, and resolve paths under the article root.

Inputs:  config/final_evaluation.yaml and the three frozen datasets it names.
Outputs: a Settings object exposing the raw mapping, absolute paths, output-stage
         directories and repository-relative path strings for manifests.
Key steps: read the YAML; verify each dataset's SHA-256 against the recorded value
         (a hash mismatch is an error, because every result must trace to the frozen
         data); expose helpers so no other module builds paths by hand.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import ARTICLE_ROOT

DEFAULT_CONFIG = ARTICLE_ROOT / "config" / "final_evaluation.yaml"
STAGE_DIRS = {
    "folds": "01_folds",
    "bundles": "02_bundles",
    "m7": "03_m7",
    "m8": "04_m8",
    "m9": "05_m9",
    "site_days": "06_site_days",
    "metrics": "07_metrics",
    "figures": "08_figures",
    "tables": "08_tables",
    "gamma": "09_gamma",
    "operating_points": "10_operating_points",
    "manifests": "manifests",
}


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file, streamed so large parquet files are not held in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Settings:
    """The resolved configuration: the raw mapping plus path helpers."""

    raw: dict[str, Any]
    root: Path = ARTICLE_ROOT
    config_path: Path = DEFAULT_CONFIG
    columns: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.columns = dict(self.raw["columns"])

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def dataset(self, cohort: str) -> Path:
        """Absolute path of a frozen dataset by cohort name (alpha, beta, gamma)."""
        return self.root / self.raw["paths"][f"{cohort}_dataset"]

    def output_root(self) -> Path:
        return self.root / self.raw["paths"]["output_dir"]

    def out(self, stage: str) -> Path:
        """Output directory of a stage, created on first use."""
        path = self.output_root() / STAGE_DIRS[stage]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def relative(self, path: Path | str) -> str:
        """Repository-relative POSIX string for manifests; never an absolute path."""
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    def m9_dev(self) -> Path:
        return self.root / self.raw["paths"]["m9_dev_dir"]


def verify_hashes(settings: Settings) -> dict[str, str]:
    """Check every dataset hash recorded in the configuration; raise on the first mismatch."""
    observed: dict[str, str] = {}
    for name, expected in settings.raw["input_hashes"].items():
        cohort = name.replace("dataset_", "").replace(".parquet", "")
        path = settings.dataset(cohort)
        observed[name] = sha256_file(path)
        if observed[name] != expected:
            raise ValueError(
                f"{name} does not match the frozen hash in {settings.config_path.name}: "
                f"expected {expected[:12]}…, found {observed[name][:12]}…"
            )
    return observed


def load(config_path: Path | str | None = None, check_hashes: bool = True) -> Settings:
    """Read the configuration and, by default, verify the frozen dataset hashes.

    Args:
        config_path: YAML file; defaults to config/final_evaluation.yaml.
        check_hashes: set False only in tests that never touch the real datasets.

    Returns:
        Settings.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    settings = Settings(raw=raw, config_path=path)
    if check_hashes:
        verify_hashes(settings)
    return settings


def import_m9_dev(settings: Settings):
    """Make the frozen M9 modules importable and return (m9_scorer, m9_metrics)."""
    folder = str(settings.m9_dev())
    if folder not in sys.path:
        sys.path.insert(0, folder)
    import m9_metrics  # noqa: E402
    import m9_scorer  # noqa: E402

    return m9_scorer, m9_metrics


def ensure_pynrpf_importable() -> None:
    """Prefer the repository's src/ package over any installed copy, as the notebooks do."""
    src = ARTICLE_ROOT.parents[1] / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

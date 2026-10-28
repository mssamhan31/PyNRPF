"""Load and validate the one configuration file, and resolve paths under the article root.

Inputs:  config/evaluation.yaml and the three datasets it names.
Outputs: a Settings object exposing the raw mapping, absolute paths, the output
         directory of every stage and repository-relative path strings for manifests.
Key steps: read the YAML; verify each dataset's SHA-256 against the recorded value
         (a mismatch is an error, because every result must trace to the released
         data); expose helpers so no other module builds paths by hand.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import ARTICLE_ROOT

DEFAULT_CONFIG = ARTICLE_ROOT / "config" / "evaluation.yaml"

# One folder per stage under the output directory, in the order the notebooks run.
STAGE_DIRS = {
    "data_folds": "01_data_folds",
    "baselines": "02_baselines",
    "m9": "03_m9",
    "metrics": "04_metrics",
    "gamma": "05_gamma",
    "paper": "paper",
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
        """Absolute path of a dataset by cohort name (alpha, beta, gamma)."""
        return self.root / self.raw["paths"][f"{cohort}_dataset"]

    def output_root(self) -> Path:
        """The output directory named by ``paths.output_dir``, under the article root."""
        return self.root / self.raw["paths"]["output_dir"]

    def out(self, stage: str) -> Path:
        """Output directory of a stage (a key of ``STAGE_DIRS``), created on first use."""
        path = self.output_root() / STAGE_DIRS[stage]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def relative(self, path: Path | str) -> str:
        """Repository-relative POSIX string for manifests; never an absolute path."""
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()


def verify_hashes(settings: Settings) -> dict[str, str]:
    """Check every dataset hash recorded in the configuration; raise on the first mismatch."""
    observed: dict[str, str] = {}
    for name, expected in settings.raw["input_hashes"].items():
        cohort = name.replace("dataset_", "").replace(".parquet", "")
        path = settings.dataset(cohort)
        observed[name] = sha256_file(path)
        if observed[name] != expected:
            raise ValueError(
                f"{name} does not match the hash recorded in {settings.config_path.name}: "
                f"expected {expected[:12]}…, found {observed[name][:12]}…"
            )
    return observed


def load(config_path: Path | str | None = None, check_hashes: bool = True) -> Settings:
    """Read the configuration and, by default, verify the dataset hashes.

    Args:
        config_path: YAML file; defaults to config/evaluation.yaml.
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

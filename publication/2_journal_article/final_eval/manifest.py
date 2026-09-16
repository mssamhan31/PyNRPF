"""Stage manifests: what a stage read, what it wrote, which code and configuration ran.

Inputs:  the settings, a stage name, the input and output paths of the stage.
Outputs: outputs/01_final_evaluation/manifests/<stage>.json with repository-relative
         paths only, SHA-256 and byte size per file, a hash of the evaluation code,
         the configuration hash and the elapsed time.
Key steps: hash files; convert every path with Settings.relative; refuse to write a
         manifest that still carries an absolute path (machine names leaked once).
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import ARTICLE_ROOT
from .config import Settings, sha256_file

ABSOLUTE = re.compile(r"^([A-Za-z]:[\\/]|/|\\\\)")


def code_hash(settings: Settings) -> str:
    """One digest over the evaluation package and the frozen M9 modules it imports."""
    files = sorted((ARTICLE_ROOT / "final_eval").glob("*.py"))
    files += [settings.m9_dev() / "m9_scorer.py", settings.m9_dev() / "m9_metrics.py"]
    import hashlib

    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_record(settings: Settings, path: Path) -> dict[str, Any]:
    path = Path(path)
    return {"path": settings.relative(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def assert_repository_relative(payload: Any) -> None:
    """Walk a manifest payload and raise on any string that looks like an absolute path."""
    if isinstance(payload, dict):
        for value in payload.values():
            assert_repository_relative(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            assert_repository_relative(value)
    elif isinstance(payload, str) and ABSOLUTE.match(payload):
        raise ValueError(f"Manifest carries an absolute path: {payload!r}")


def write_stage_manifest(settings: Settings, stage: str, inputs: list[Path], outputs: list[Path],
                         started: float, extra: dict[str, Any] | None = None) -> Path:
    """Write the manifest for one stage and return its path."""
    payload = {
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": file_record(settings, settings.config_path),
        "code_sha256": code_hash(settings),
        "inputs": [file_record(settings, p) for p in inputs],
        "outputs": [file_record(settings, p) for p in outputs],
        "elapsed_s": round(time.time() - started, 1),
        **(extra or {}),
    }
    assert_repository_relative(payload)
    path = settings.out("manifests") / f"{stage}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path

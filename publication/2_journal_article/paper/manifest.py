"""Stage manifests: what a stage read, what it wrote, which code and configuration ran.

Inputs:  the settings, a stage name, the input and output paths of the stage.
Outputs: <output_dir>/manifests/<stage>.json with repository-relative paths only,
         SHA-256 and byte size per file, a hash of the paper code, the installed
         ``pynrpf`` version, the configuration hash and the elapsed time.
Key steps: hash files; convert every path with Settings.relative; refuse to write a
         manifest that still carries an absolute path (machine names leaked once).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pynrpf

from . import ARTICLE_ROOT
from .config import Settings, sha256_file

ABSOLUTE = re.compile(r"^([A-Za-z]:[\\/]|/|\\\\)")
PAPER_DIR = ARTICLE_ROOT / "paper"


def code_hash() -> str:
    """One digest over every module of ``paper/`` and the installed ``pynrpf`` version."""
    digest = hashlib.sha256()
    for path in sorted(PAPER_DIR.rglob("*.py")):
        digest.update(path.relative_to(PAPER_DIR).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(f"pynrpf=={pynrpf.__version__}".encode("utf-8"))
    return digest.hexdigest()


def file_record(settings: Settings, path: Path) -> dict[str, Any]:
    """Repository-relative path, SHA-256 and size in bytes of one file."""
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
    """Write the manifest for one stage and return its path.

    Args:
        settings: the evaluation settings.
        stage: the manifest's name, the stage folder name (``01_data_folds``, ...).
        inputs: files the stage read.
        outputs: files the stage wrote.
        started: ``time.time()`` when the stage began, for ``elapsed_s``.
        extra: any further JSON-serialisable fields (counts, floors, decisions).
    """
    payload = {
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": file_record(settings, settings.config_path),
        "code_sha256": code_hash(),
        "pynrpf_version": pynrpf.__version__,
        "inputs": [file_record(settings, p) for p in inputs],
        "outputs": [file_record(settings, p) for p in outputs],
        "elapsed_s": round(time.time() - started, 1),
        **(extra or {}),
    }
    assert_repository_relative(payload)
    path = settings.out("manifests") / f"{stage}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def check_manifest(settings: Settings, path: Path) -> list[str]:
    """Compare a written manifest with the files on disk; return the mismatches (empty when all agree).

    Every input and output must exist with the recorded SHA-256 and size.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = []
    for record in payload["inputs"] + payload["outputs"]:
        file = settings.root / record["path"]
        if not file.exists():
            problems.append(f"missing: {record['path']}")
        elif sha256_file(file) != record["sha256"]:
            problems.append(f"changed: {record['path']}")
    return problems

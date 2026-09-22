"""Validated lightweight data sources for fast journal figure rerendering."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

FIGURE_SOURCE_SCHEMA_VERSION = 1
RENDER_ONLY_ENV = "PYNRPF_RENDER_ONLY"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"", "0", "false", "no", "off"}


def render_only_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether notebook execution should use cached figure inputs only."""

    value = (environ or os.environ).get(RENDER_ONLY_ENV, "").strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{RENDER_ONLY_ENV} must be one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}; "
        f"received {value!r}."
    )


def figure_source_path(root: Path, slug: str, name: str) -> Path:
    """Return the canonical Parquet path for one notebook figure source."""

    return root / slug / f"{name}.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(".json")


def write_figure_source(
    frame: pd.DataFrame,
    path: Path,
    *,
    required_columns: Sequence[str] = (),
) -> Path:
    """Write one atomic Parquet source plus a versioned integrity sidecar."""

    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Figure source is missing required columns: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    payload: dict[str, Any] = {
        "schema_version": FIGURE_SOURCE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "file": path.name,
        "sha256": _sha256(path),
        "rows": int(len(frame)),
        "columns": frame.columns.tolist(),
        "required_columns": list(required_columns),
    }
    metadata_path = _metadata_path(path)
    metadata_temporary = metadata_path.with_suffix(".json.tmp")
    metadata_temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata_temporary.replace(metadata_path)
    return path


def load_figure_source(
    path: Path,
    *,
    required_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Load one source after validating its sidecar, hash, and columns."""

    metadata_path = _metadata_path(path)
    if not path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Figure source cache is incomplete for {path}. Run the notebook once "
            f"without {RENDER_ONLY_ENV}=1."
        )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FIGURE_SOURCE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported figure source schema for {path}.")
    if payload.get("sha256") != _sha256(path):
        raise ValueError(f"Figure source hash mismatch for {path}.")
    frame = pd.read_parquet(path)
    expected_columns = list(payload.get("columns", []))
    if frame.columns.tolist() != expected_columns or len(frame) != payload.get("rows"):
        raise ValueError(f"Figure source shape or columns do not match metadata for {path}.")
    all_required = set(required_columns) | set(payload.get("required_columns", []))
    missing = all_required - set(frame.columns)
    if missing:
        raise ValueError(f"Figure source is missing required columns: {sorted(missing)}")
    return frame
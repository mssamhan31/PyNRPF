"""Filesystem and configuration helpers retained from the conference codebase.

Inputs:  repository-relative paths, YAML configuration files and parquet datasets.
Outputs: loaded configuration mappings, dataframes, and resolved paths.
Key steps: locate the repository root, read YAML and parquet, and provide the
         nested-key accessors the legacy modules were written against.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml


def repo_root(from_path: Optional[Path] = None) -> Path:
    """Return the package root two levels above the given file."""
    p = (from_path or Path(__file__)).resolve()
    return p.parent.parent


def load_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML file into a dict, returning {} when it is empty.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Read a nested config value by dotted key, or return the default."""
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def req(cfg: Dict[str, Any], dotted: str) -> Any:
    """Read a nested config value by dotted key, requiring it to be present.

    Raises:
        KeyError: If the key is absent or None.
    """
    v = get(cfg, dotted, None)
    if v is None:
        raise KeyError(f"Missing required config key: '{dotted}'")
    return v


def ensure_dir(p: Path) -> None:
    """Create a directory and any missing parents, ignoring one that exists."""
    p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_mb: int = 1) -> str:
    """Return the SHA-256 hex digest of a file, read in chunk_mb megabyte chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_mb * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sha256_txt(sha_path: Path) -> Tuple[str, Optional[str]]:
    """Parse a checksum sidecar of the form ``<hex>  <filename>``.

    Returns:
        Tuple of the hex digest and the recorded filename, which may be None.

    Raises:
        ValueError: If the file is empty.
    """
    # expected format: "<hex>  <filename>"
    txt = sha_path.read_text(encoding="utf-8").strip()
    if not txt:
        raise ValueError(f"Empty sha256 file: {sha_path}")
    parts = txt.split()
    digest = parts[0].strip()
    fname = parts[1].strip() if len(parts) >= 2 else None
    return digest, fname


def verify_sha256_best_effort(parquet_path: Path, sha_path: Path) -> Dict[str, Any]:
    """Check a dataset against its checksum sidecar without raising on mismatch.

    Args:
        parquet_path: Dataset to hash.
        sha_path: Checksum sidecar to compare against.

    Returns:
        Dict carrying the two paths and a ``status`` of ``skipped``, ``ok`` or
        ``mismatch``, so a caller can report the outcome rather than abort.
    """
    out: Dict[str, Any] = {
        "parquet": str(parquet_path),
        "sha_file": str(sha_path),
        "status": "skipped",
        "expected": None,
        "actual": None,
        "filename_in_sha": None,
        "note": None,
    }

    if not parquet_path.exists():
        out["status"] = "failed"
        out["note"] = "parquet missing"
        return out

    if not sha_path.exists():
        out["status"] = "skipped"
        out["note"] = "sha256.txt missing"
        return out

    try:
        expected, fname = parse_sha256_txt(sha_path)
        actual = sha256_file(parquet_path)
        out["expected"] = expected
        out["actual"] = actual
        out["filename_in_sha"] = fname
        out["status"] = "ok" if expected.lower() == actual.lower() else "failed"
        if out["status"] == "failed":
            out["note"] = "checksum mismatch"
        return out
    except Exception as e:
        out["status"] = "skipped"
        out["note"] = f"sha256 parse/compute error: {type(e).__name__}: {e}"
        return out


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a payload as indented JSON, creating parent directories as needed."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    """Write *payload* as a YAML file (uses safe_dump with default_flow_style=False)."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)


def load_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file into a pandas DataFrame and print basic info."""
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} cols from {path.name}")
    return df

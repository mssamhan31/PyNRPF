from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml


def repo_root(from_path: Optional[Path] = None) -> Path:
    p = (from_path or Path(__file__)).resolve()
    return p.parent.parent


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def req(cfg: Dict[str, Any], dotted: str) -> Any:
    v = get(cfg, dotted, None)
    if v is None:
        raise KeyError(f"Missing required config key: '{dotted}'")
    return v


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_mb: int = 1) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_mb * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sha256_txt(sha_path: Path) -> Tuple[str, Optional[str]]:
    # expected format: "<hex>  <filename>"
    txt = sha_path.read_text(encoding="utf-8").strip()
    if not txt:
        raise ValueError(f"Empty sha256 file: {sha_path}")
    parts = txt.split()
    digest = parts[0].strip()
    fname = parts[1].strip() if len(parts) >= 2 else None
    return digest, fname


def verify_sha256_best_effort(parquet_path: Path, sha_path: Path) -> Dict[str, Any]:
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

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Union
from urllib.parse import urlparse
from urllib.request import urlopen

Location = Union[str, Path]


def _is_http_url(location: str) -> bool:
    scheme = urlparse(location).scheme.lower()
    return scheme in {"http", "https"}


def _to_local_path(location: Location) -> Path:
    loc = str(location)
    if loc.startswith("dbfs:/Volumes/"):
        # Unity Catalog Volumes alias routed through dbfs:/ URI.
        return Path("/Volumes") / loc.replace("dbfs:/Volumes/", "", 1).lstrip("/")
    if loc.startswith("dbfs:/"):
        # Databricks DBFS FUSE mount path.
        return Path("/dbfs") / loc.replace("dbfs:/", "", 1).lstrip("/")

    parsed = urlparse(loc)
    if len(parsed.scheme) == 1 and len(loc) >= 3 and loc[1] == ":":
        # Windows drive letter path (e.g. C:\path\file.pkl).
        return Path(loc)
    if parsed.scheme == "file":
        return Path(parsed.path)
    if parsed.scheme and parsed.scheme not in {"", "file"}:
        raise ValueError(f"Unsupported local path scheme: {parsed.scheme}")
    return Path(loc)


def _read_bytes(location: Location) -> bytes:
    loc = str(location)
    if _is_http_url(loc):
        with urlopen(loc) as resp:  # nosec B310 - intentional user-configured URI fetch
            return resp.read()

    path = _to_local_path(location)
    if not path.exists():
        raise FileNotFoundError(f"Artifact bundle not found: {path}")
    return path.read_bytes()


def _write_bytes(location: Location, payload: bytes) -> str:
    loc = str(location)
    if _is_http_url(loc):
        raise ValueError("Writing to HTTP/HTTPS URI is not supported for artifacts.")

    path = _to_local_path(location)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path)


def _join_location(base: Location, *parts: str) -> str:
    clean_parts = [p.strip("/\\") for p in parts if p]
    loc = str(base)

    if loc.startswith("dbfs:/"):
        suffix = "/".join(clean_parts)
        if not suffix:
            return loc.rstrip("/")
        return f"{loc.rstrip('/')}/{suffix}"

    if loc.startswith("file://"):
        path = _to_local_path(loc).joinpath(*clean_parts)
        return path.as_uri()

    return str(Path(loc).joinpath(*clean_parts))


def load_artifact_bundle(location: Location) -> Dict[str, Any]:
    payload = _read_bytes(location)
    bundle = pickle.loads(payload)
    if not isinstance(bundle, dict):
        raise TypeError("Artifact bundle payload must deserialize to a dictionary.")
    return bundle


def save_artifact_bundle(bundle: Dict[str, Any], location: Location) -> str:
    loc = str(location)
    if _is_http_url(loc):
        raise ValueError("Saving to HTTP/HTTPS URI is not supported. Use a local or DBFS path.")

    return _write_bytes(location, pickle.dumps(bundle))


def save_versioned_artifact_bundle(
    bundle: Dict[str, Any],
    base_location: Location,
    model_name: str = "m8_xgb",
    manifest: Dict[str, Any] | None = None,
    timestamp_utc: str | None = None,
) -> Dict[str, str]:
    ts = timestamp_utc or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir_uri = _join_location(base_location, model_name, ts)
    artifact_uri = _join_location(artifact_dir_uri, "bundle.pkl")
    manifest_uri = _join_location(artifact_dir_uri, "manifest.json")

    resolved_artifact_path = save_artifact_bundle(bundle, artifact_uri)
    manifest_payload = manifest or {
        "bundle_schema": bundle.get("bundle_schema"),
        "model_name": bundle.get("model_name"),
        "created_at_utc": bundle.get("created_at_utc"),
    }
    resolved_manifest_path = _write_bytes(
        manifest_uri,
        json.dumps(manifest_payload, indent=2, sort_keys=True).encode("utf-8"),
    )

    return {
        "artifact_dir_uri": artifact_dir_uri,
        "artifact_uri": artifact_uri,
        "manifest_uri": manifest_uri,
        "resolved_artifact_path": resolved_artifact_path,
        "resolved_manifest_path": resolved_manifest_path,
    }

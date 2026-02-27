from __future__ import annotations

import pickle
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

    path = _to_local_path(location)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(bundle))
    return str(path)

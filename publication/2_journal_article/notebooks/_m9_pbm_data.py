"""Data, path, cache, and manifest helpers for the journal m9_pbm workflow."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

BASE_COLUMNS = [
    "substation_id",
    "date",
    "timestamp",
    "net_load_MW",
    "solar_MW",
    "label_interval",
    "label_day",
]
CONFIDENCE_VALUES = {"sure", "unsure"}


@dataclass(frozen=True)
class M9Paths:
    """Resolved article input and output paths."""

    article: Path
    config: Path
    notebooks: Path
    final_data: Path
    outputs: Path
    intermediate: Path
    metrics: Path
    tables: Path
    figures: Path
    manifests: Path


@dataclass(frozen=True)
class DayArrays:
    """One row per substation-day plus aligned 96-slot input arrays."""

    keys: pd.DataFrame
    net_load: np.ndarray
    solar: np.ndarray


def find_article_root(start: Path | None = None) -> Path:
    """Find ``publication/2_journal_article`` from a notebook or repository path."""

    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if candidate.name == "2_journal_article" and (candidate / "dataset").is_dir():
            return candidate
        nested = candidate / "publication" / "2_journal_article"
        if (nested / "dataset").is_dir():
            return nested.resolve()
    raise FileNotFoundError(f"Could not find publication/2_journal_article from {start}")


def load_experiment_config(article_root: Path | None = None) -> dict[str, Any]:
    """Load and minimally validate the journal experiment configuration."""

    article = (article_root or find_article_root()).resolve()
    path = article / "config" / "experiment_config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("schema_version") != "journal_v2":
        raise ValueError("Expected experiment_config.yaml schema_version journal_v2.")
    if "m9_pbm" not in config:
        raise ValueError("experiment_config.yaml is missing the m9_pbm section.")
    config["_config_path"] = str(path)
    return config


def resolve_paths(
    article_root: Path | None = None,
    config: dict[str, Any] | None = None,
) -> M9Paths:
    """Resolve all m9 input/output paths without relying on a user-specific path."""

    article = (article_root or find_article_root()).resolve()
    cfg = config or load_experiment_config(article)
    output_root = Path(cfg["paths"]["output_base_dir"])
    if not output_root.is_absolute():
        output_root = article / output_root
    output_cfg = cfg["outputs"]
    return M9Paths(
        article=article,
        config=Path(cfg["_config_path"]),
        notebooks=article / "notebooks",
        final_data=article / "dataset" / "final",
        outputs=output_root,
        intermediate=output_root / output_cfg["intermediate_dir"],
        metrics=output_root / output_cfg["metrics_dir"],
        tables=output_root / output_cfg["tables_dir"],
        figures=output_root / output_cfg["figures_dir"],
        manifests=output_root / output_cfg["manifests_dir"],
    )


def output_dirs(paths: M9Paths, slug: str) -> dict[str, Path]:
    """Create and return the category-first output directories for one notebook."""

    result = {
        "intermediate": paths.intermediate / slug,
        "metrics": paths.metrics / slug,
        "tables": paths.tables / slug,
        "figures": paths.figures / slug,
    }
    for directory in [*result.values(), paths.manifests]:
        directory.mkdir(parents=True, exist_ok=True)
    return result


def _coerce_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).astype(int).astype(bool)
    text = values.astype("string").fillna("").str.strip().str.lower()
    truthy = {"true", "t", "1", "yes", "y"}
    falsy = {"false", "f", "0", "no", "n", "", "nan", "none"}
    unknown = sorted(set(text.unique()) - truthy - falsy)
    if unknown:
        raise ValueError(f"Unknown boolean values: {unknown}")
    return text.isin(truthy)


def load_dataset(
    dataset: str,
    *,
    article_root: Path | None = None,
    config: dict[str, Any] | None = None,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load a final Alpha, Beta, or Gamma dataset with a consistent schema."""

    key = dataset.strip().lower()
    if key not in {"alpha", "beta", "gamma"}:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    article = (article_root or find_article_root()).resolve()
    cfg = config or load_experiment_config(article)
    relative_path = cfg["paths"][f"{key}_dataset_path"]
    path = Path(relative_path)
    if not path.is_absolute():
        path = article / path
    requested = list(columns) if columns is not None else None
    frame = pd.read_parquet(path, columns=requested)

    if "substation_id" in frame:
        frame["substation_id"] = frame["substation_id"].astype(str)
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    if "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    for label in ["label_interval", "label_day"]:
        if label in frame:
            frame[label] = _coerce_bool(frame[label])
    if "confidence" in frame:
        frame["confidence"] = (
            frame["confidence"].astype("string").fillna("missing").str.strip().str.lower()
        )
        unknown = set(frame["confidence"].unique()) - CONFIDENCE_VALUES
        if unknown:
            raise ValueError(f"Unexpected {key} confidence values: {sorted(unknown)}")
    return frame.sort_values(
        [column for column in ["substation_id", "timestamp"] if column in frame]
    ).reset_index(drop=True)


def to_day_arrays(frame: pd.DataFrame, slots_per_day: int = 96) -> DayArrays:
    """Convert interval data to aligned day rows and fixed-width net/solar arrays."""

    required = set(BASE_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Day-array input is missing columns: {sorted(missing)}")

    data = frame.copy()
    data["slot"] = data["timestamp"].dt.hour * 4 + data["timestamp"].dt.minute // 15
    data = data.loc[data["slot"].between(0, slots_per_day - 1)]
    duplicate_count = int(data.duplicated(["substation_id", "date", "slot"]).sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate substation-date-slot rows.")

    keys: list[dict[str, Any]] = []
    net_rows: list[np.ndarray] = []
    solar_rows: list[np.ndarray] = []
    for (substation, date), group in data.groupby(["substation_id", "date"], sort=True):
        net = np.full(slots_per_day, np.nan, dtype=np.float64)
        solar = np.full(slots_per_day, np.nan, dtype=np.float64)
        slots = group["slot"].to_numpy(dtype=int)
        net[slots] = pd.to_numeric(group["net_load_MW"], errors="coerce")
        solar[slots] = pd.to_numeric(group["solar_MW"], errors="coerce")
        row: dict[str, Any] = {
            "substation_id": substation,
            "date": date,
            "true_day": bool(group["label_day"].max()),
            "true_interval_count": int(group["label_interval"].sum()),
            "n_rows": int(len(group)),
            "n_missing_net": int(np.isnan(net).sum()),
            "n_missing_solar": int(np.isnan(solar).sum()),
        }
        if "confidence" in group:
            confidence = group["confidence"].dropna().unique().tolist()
            if len(confidence) != 1:
                raise ValueError(f"Confidence varies within {substation} {date}: {confidence}")
            row["confidence"] = confidence[0]
        else:
            row["confidence"] = "not_applicable"
        keys.append(row)
        net_rows.append(net)
        solar_rows.append(solar)

    return DayArrays(
        keys=pd.DataFrame(keys),
        net_load=np.vstack(net_rows),
        solar=np.vstack(solar_rows),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    """Hash configuration content while ignoring the resolved config path."""

    serialisable = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = json.dumps(serialisable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_input_hashes(paths: M9Paths, config: dict[str, Any]) -> pd.DataFrame:
    """Validate final datasets against hashes recorded in configuration."""

    rows = []
    for filename, expected in config.get("input_hashes", {}).items():
        path = paths.final_data / filename
        actual = sha256_file(path)
        rows.append(
            {
                "filename": filename,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": actual == expected,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) and not result["matches"].all():
        failed = result.loc[~result["matches"], "filename"].tolist()
        raise ValueError(f"Final dataset hashes differ from configuration: {failed}")
    return result


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    return path


def write_table_formats(frame: pd.DataFrame, stem: Path) -> list[Path]:
    """Write one compact publication table as CSV, Markdown, and LaTeX."""

    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(suffix) for suffix in [".csv", ".md", ".tex"]]
    frame.to_csv(paths[0], index=False)
    paths[1].write_text(frame.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    paths[2].write_text(
        frame.to_latex(index=False, float_format=lambda value: f"{value:.4f}"),
        encoding="utf-8",
    )
    return paths


def artifact_inventory(
    artifacts: Mapping[str, Path],
    *,
    relative_to: Path | None = None,
) -> pd.DataFrame:
    """Build an explicit existence and size audit for expected upstream outputs."""

    rows = []
    for name, path in artifacts.items():
        resolved = path.resolve()
        display_path = (
            str(resolved.relative_to(relative_to.resolve()))
            if relative_to is not None and resolved.is_relative_to(relative_to.resolve())
            else str(resolved)
        )
        rows.append(
            {
                "artifact": name,
                "path": display_path.replace("\\", "/"),
                "exists": resolved.exists(),
                "bytes": resolved.stat().st_size if resolved.exists() else 0,
            }
        )
    return pd.DataFrame(rows)


def consolidate_parquet_files(source_paths: Iterable[Path], output_path: Path) -> Path:
    """Stream compatible Parquet partitions into one file without a large concat."""

    import pyarrow.parquet as pq

    sources = list(source_paths)
    if not sources:
        raise ValueError("At least one Parquet partition is required.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    writer = None
    try:
        for source in sources:
            table = pq.read_table(source)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                    use_dictionary=["dataset", "substation_id", "date"],
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output_path)
    return output_path


def _git_commit(article_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=article_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def manifest_payload(
    *,
    paths: M9Paths,
    config: dict[str, Any],
    started_at: float,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    row_counts: dict[str, int],
    status: str = "publication_ready",
) -> dict[str, Any]:
    """Build the common reproducibility payload used by every notebook manifest."""

    input_paths = list(inputs)
    output_paths = [path for path in outputs if path.exists()]
    return {
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started_at,
        "config_path": str(paths.config.relative_to(paths.article)),
        "config_sha256": config_hash(config),
        "git_commit": _git_commit(paths.article),
        "random_seed": config["execution"]["random_seed"],
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "row_counts": row_counts,
        "inputs": [
            {
                "path": str(path.relative_to(paths.article)),
                "sha256": sha256_file(path),
            }
            for path in input_paths
        ],
        "outputs": [
            {
                "path": str(path.relative_to(paths.article)),
                "sha256": sha256_file(path),
            }
            for path in output_paths
        ],
    }


def write_manifest(paths: M9Paths, name: str, payload: dict[str, Any]) -> Path:
    paths.manifests.mkdir(parents=True, exist_ok=True)
    path = paths.manifests / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path

"""Minimal Databricks Connect hello check for local VS Code execution."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = (_repo_root() / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_connect_env(cfg: dict[str, Any]) -> None:
    local = cfg.get("local_connect") or {}
    workspace_host = str(
        local.get("workspace_host", cfg.get("workspace_host", ""))
    ).strip()
    profile = str(local.get("profile", "dbx")).strip()
    cli_path = Path(str(local.get("cli_path", ".tools/databricks-cli/databricks.exe")))

    if not cli_path.is_absolute():
        cli_path = (_repo_root() / cli_path).resolve()

    if workspace_host:
        os.environ.setdefault("DATABRICKS_HOST", workspace_host)
    os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", profile)
    os.environ.setdefault("DATABRICKS_CLI_PATH", str(cli_path))

    # Respect explicit shell overrides from scripts/run_local_connect.ps1.
    if ("DATABRICKS_CLUSTER_ID" in os.environ or
            "DATABRICKS_SERVERLESS_COMPUTE_ID" in os.environ):
        return

    compute_mode = str(local.get("compute_mode", "cluster")).strip().lower()
    if compute_mode == "cluster":
        cluster_id = str(local.get("cluster_id", "")).strip()
        if not cluster_id:
            raise ValueError("local_connect.cluster_id is required when compute_mode='cluster'")
        os.environ["DATABRICKS_CLUSTER_ID"] = cluster_id
        os.environ.pop("DATABRICKS_SERVERLESS_COMPUTE_ID", None)
        return

    serverless_id = str(local.get("serverless_compute_id", "auto")).strip() or "auto"
    os.environ["DATABRICKS_SERVERLESS_COMPUTE_ID"] = serverless_id
    os.environ.pop("DATABRICKS_CLUSTER_ID", None)


def run(config_path: str) -> None:
    try:
        from databricks.connect import DatabricksSession
    except ImportError as exc:
        raise ModuleNotFoundError(
            "databricks-connect is not installed. Run scripts/setup_local_connect.ps1 first."
        ) from exc

    cfg = load_config(config_path)
    apply_connect_env(cfg)

    app_name = str(cfg.get("app_name", "pynrpf_hello"))
    greeting = str(cfg.get("greeting", "Hello from Databricks Connect"))

    spark = DatabricksSession.builder.getOrCreate()

    print(f"[{app_name}] {greeting}")
    print(f"Profile: {os.environ.get('DATABRICKS_CONFIG_PROFILE', '<unset>')}")
    if os.environ.get("DATABRICKS_CLUSTER_ID"):
        print(f"Compute: cluster {os.environ['DATABRICKS_CLUSTER_ID']}")
    else:
        print(f"Compute: serverless {os.environ.get('DATABRICKS_SERVERLESS_COMPUTE_ID', 'auto')}")

    spark.sql("select current_user() as user, current_timestamp() as ts").show(truncate=False)
    spark.range(5).show(truncate=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/databricks_connect.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

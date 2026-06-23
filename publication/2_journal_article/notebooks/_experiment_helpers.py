from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


EXPECTED_COLUMNS = [
    "substation_id",
    "date",
    "timestamp",
    "net_load_MW",
    "solar_MW",
    "label_interval",
    "label_day",
]


@dataclass(frozen=True)
class ArticlePaths:
    root: Path
    config_path: Path
    final: Path
    outputs: Path
    intermediate: Path
    metrics: Path
    tables: Path
    figures: Path
    manifests: Path


def find_article_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (
            candidate.name == "2_journal_article"
            and (candidate / "dataset").exists()
        ):
            return candidate
        nested = candidate / "publication" / "2_journal_article"
        if (nested / "dataset").exists():
            return nested.resolve()
    raise RuntimeError(f"Could not locate publication/2_journal_article from {start}")


def article_path(article_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else article_root / path


def load_config(article_root: Path) -> dict[str, Any]:
    path = article_root / "config" / "experiment_config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if cfg.get("schema_version") != "journal_v2":
        raise ValueError("Expected journal_v2 config schema.")
    cfg["_config_path"] = str(path)
    return cfg


def article_paths(article_root: Path, cfg: dict[str, Any]) -> ArticlePaths:
    output_root = article_path(article_root, cfg["paths"]["output_base_dir"])
    outputs = cfg["outputs"]
    return ArticlePaths(
        root=article_root,
        config_path=Path(cfg["_config_path"]),
        final=article_root / "dataset" / "final",
        outputs=output_root,
        intermediate=output_root / outputs["intermediate_dir"],
        metrics=output_root / outputs["metrics_dir"],
        tables=output_root / outputs["tables_dir"],
        figures=output_root / outputs["figures_dir"],
        manifests=output_root / outputs["manifests_dir"],
    )


def ensure_output_dirs(paths: ArticlePaths) -> None:
    for path in [
        paths.outputs,
        paths.final,
        paths.intermediate,
        paths.metrics,
        paths.tables,
        paths.figures,
        paths.manifests,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def notebook_output_dirs(paths: ArticlePaths, slug: str) -> dict[str, Path]:
    dirs = {
        "intermediate": paths.intermediate / slug,
        "metrics": paths.metrics / slug,
        "tables": paths.tables / slug,
        "figures": paths.figures / slug,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    paths.manifests.mkdir(parents=True, exist_ok=True)
    return dirs


def write_manifest(paths: ArticlePaths, name: str, payload: dict[str, Any]) -> Path:
    ensure_output_dirs(paths)
    out = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(paths.config_path),
        **payload,
    }
    path = paths.manifests / name
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)
    return path


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256(paths_to_hash: Iterable[Path], output_path: Path) -> Path:
    rows = [f"{sha256_file(path)}  {path.name}" for path in paths_to_hash if path.exists()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f"{output_path.name}.tmp")
    tmp.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    tmp.replace(output_path)
    return output_path


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    normalized = series.astype("string").fillna("").str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f", "", "nan", "none"}
    unknown = sorted(set(normalized.unique()) - truthy - falsy)
    if unknown:
        raise ValueError(f"Unknown boolean label values: {unknown}")
    return normalized.isin(truthy)


def _parse_wall_clock(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    stripped = text.str.replace(r"(Z|[+-]\d{2}:\d{2})$", "", regex=True)
    return pd.to_datetime(stripped, errors="raise")


def validate_schema(df: pd.DataFrame, dataset_name: str) -> None:
    actual = list(df.columns)
    if actual != EXPECTED_COLUMNS:
        raise ValueError(f"{dataset_name} expected {EXPECTED_COLUMNS}, found {actual}.")


def read_dataset_file(path: Path, dataset_name: str) -> pd.DataFrame:
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"{dataset_name} must be a Parquet final dataset, got {path}.")
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} dataset not found: {path}")
    return pd.read_parquet(path)


def raw_dataset_for_write(path: Path, dataset_name: str) -> pd.DataFrame:
    df = read_dataset_file(path, dataset_name)
    validate_schema(df, dataset_name)
    return df[EXPECTED_COLUMNS].copy()


def prepare_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    validate_schema(df, dataset_name)
    work = df.copy()
    work["_timestamp_dt"] = _parse_wall_clock(work["timestamp"])
    work["date"] = work["date"].astype(str)
    parsed_dates = work["_timestamp_dt"].dt.strftime("%Y-%m-%d")
    mismatch = work["date"] != parsed_dates
    if mismatch.any():
        raise ValueError(f"{dataset_name} has {int(mismatch.sum())} date mismatches.")
    work["label_interval"] = _coerce_bool(work["label_interval"])
    work["label_day"] = work.groupby(["substation_id", "date"])["label_interval"].transform(
        "any"
    )
    duplicate_count = int(work.duplicated(["substation_id", "timestamp"]).sum())
    if duplicate_count:
        raise ValueError(f"{dataset_name} has duplicate site/timestamp keys: {duplicate_count}")
    work["reference_net_load_MW"] = reference_net_load(work)
    work["hour"] = work["_timestamp_dt"].dt.hour
    work["month"] = work["_timestamp_dt"].dt.month
    work["weekday"] = work["_timestamp_dt"].dt.dayofweek
    work["is_weekend"] = work["weekday"] >= 5
    work["season"] = work["month"].map(month_to_season)
    return work.sort_values(["substation_id", "_timestamp_dt"]).reset_index(drop=True)


def load_dataset(article_root: Path, cfg: dict[str, Any], dataset_key: str) -> pd.DataFrame:
    if dataset_key == "alpha":
        path = article_path(article_root, cfg["paths"]["alpha_dataset_path"])
        alpha = prepare_dataset(read_dataset_file(path, "Alpha"), "Alpha")
        validate_final_dataset(alpha, "Alpha", cfg)
        return alpha
    if dataset_key == "beta":
        path = article_path(article_root, cfg["paths"]["beta_dataset_path"])
        beta = prepare_dataset(read_dataset_file(path, "Beta"), "Beta")
        validate_final_dataset(beta, "Beta", cfg)
        return beta
    if dataset_key == "gamma":
        path = article_path(article_root, cfg["paths"]["gamma_dataset_path"])
        gamma = prepare_dataset(read_dataset_file(path, "Gamma"), "Gamma")
        validate_final_dataset(gamma, "Gamma", cfg)
        return gamma
    raise ValueError(f"Unknown dataset_key: {dataset_key}")


def filter_date_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].copy().reset_index(drop=True)


def validate_final_dataset(df: pd.DataFrame, dataset_name: str, cfg: dict[str, Any]) -> None:
    validate_schema(df[EXPECTED_COLUMNS], dataset_name)
    windows = cfg["windows"]
    if dataset_name == "Alpha":
        if df["date"].min() > windows["train_start"] or df["date"].max() < windows["test_end"]:
            raise ValueError("Alpha final dataset does not cover configured train/test windows.")
    if dataset_name == "Beta":
        if df["date"].min() != windows["beta_start"] or df["date"].max() != windows["beta_end"]:
            raise ValueError(
                f"Beta final dataset must span {windows['beta_start']} to {windows['beta_end']}."
            )
        if len(df) != 280_800:
            raise ValueError(f"Beta final dataset expected 280800 rows, found {len(df)}.")
        site_days = df[["substation_id", "date"]].drop_duplicates().shape[0]
        if site_days != 2_928:
            raise ValueError(f"Beta final dataset expected 2928 site-days, found {site_days}.")
    if dataset_name == "Gamma":
        if df["substation_id"].nunique() != 1:
            raise ValueError("Gamma final dataset must contain exactly one site.")
        if df["date"].min() != windows["beta_start"] or df["date"].max() != windows["beta_end"]:
            raise ValueError("Gamma final dataset must use the same date range as Beta.")
        if len(df) != 35_136:
            raise ValueError(f"Gamma final dataset expected 35136 rows, found {len(df)}.")


def final_dataset_summary(
    df: pd.DataFrame,
    dataset_name: str,
    path: Path,
    article_root: Path,
) -> dict[str, Any]:
    summary = dataset_summary(df, dataset_name)
    summary["source_stage"] = "final"
    try:
        summary["source_file"] = path.relative_to(article_root).as_posix()
    except ValueError:
        summary["source_file"] = str(path)
    return summary


def build_final_datasets(
    article_root: Path,
    cfg: dict[str, Any],
    gamma_site_override: str | None = None,
) -> dict[str, Any]:
    paths = article_paths(article_root, cfg)
    paths.final.mkdir(parents=True, exist_ok=True)

    alpha_source = article_path(article_root, cfg["source_paths"]["alpha_processed_path"])
    beta_source = article_path(article_root, cfg["source_paths"]["beta_processed_path"])
    alpha_path = article_path(article_root, cfg["paths"]["alpha_dataset_path"])
    beta_path = article_path(article_root, cfg["paths"]["beta_dataset_path"])
    gamma_path = article_path(article_root, cfg["paths"]["gamma_dataset_path"])

    alpha = prepare_dataset(raw_dataset_for_write(alpha_source, "Alpha processed"), "Alpha")
    beta_full = prepare_dataset(raw_dataset_for_write(beta_source, "Beta processed"), "Beta")
    beta = filter_date_window(beta_full, cfg["windows"]["beta_start"], cfg["windows"]["beta_end"])

    gamma_rank = gamma_site_rankings(beta, cfg)
    gamma_site = select_gamma_site(beta, cfg, gamma_site_override)
    gamma = beta.loc[beta["substation_id"] == gamma_site].copy().reset_index(drop=True)

    validate_final_dataset(alpha, "Alpha", cfg)
    validate_final_dataset(beta, "Beta", cfg)
    validate_final_dataset(gamma, "Gamma", cfg)

    write_parquet(alpha[EXPECTED_COLUMNS], alpha_path)
    write_parquet(beta[EXPECTED_COLUMNS], beta_path)
    write_parquet(gamma[EXPECTED_COLUMNS], gamma_path)

    final_rows = [
        final_dataset_summary(alpha, "Alpha", alpha_path, article_root),
        final_dataset_summary(beta, "Beta", beta_path, article_root),
        final_dataset_summary(gamma, "Gamma", gamma_path, article_root),
    ]
    final_summary = pd.DataFrame(final_rows)
    write_csv(final_summary, paths.final / "dataset_final_summary.csv")
    write_csv(gamma_rank, paths.final / "gamma_selection_summary.csv")
    write_sha256(
        [
            alpha_path,
            beta_path,
            gamma_path,
            paths.final / "dataset_final_summary.csv",
            paths.final / "gamma_selection_summary.csv",
        ],
        paths.final / "sha256.txt",
    )

    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "gamma_site": gamma_site,
        "gamma_rankings": gamma_rank,
        "final_summary": final_summary,
    }


def reference_net_load(df: pd.DataFrame) -> pd.Series:
    flags = _coerce_bool(df["label_interval"])
    values = np.where(flags, -df["net_load_MW"], df["net_load_MW"])
    result = pd.Series(values, index=df.index, name="reference_net_load_MW")
    result.loc[df["net_load_MW"].isna()] = np.nan
    return result


def month_to_season(month: int) -> str:
    if month in {12, 1, 2}:
        return "summer"
    if month in {3, 4, 5}:
        return "autumn"
    if month in {6, 7, 8}:
        return "winter"
    return "spring"


def dataset_summary(df: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
    site_day = df[["substation_id", "date", "label_day"]].drop_duplicates()
    return {
        "dataset": dataset_name,
        "n_rows": int(len(df)),
        "n_sites": int(df["substation_id"].nunique()),
        "min_timestamp": str(df["_timestamp_dt"].min()),
        "max_timestamp": str(df["_timestamp_dt"].max()),
        "n_dates": int(df["date"].nunique()),
        "null_net_load_MW": int(df["net_load_MW"].isna().sum()),
        "null_solar_MW": int(df["solar_MW"].isna().sum()),
        "positive_label_interval": int(df["label_interval"].sum()),
        "positive_label_day_site_days": int(site_day["label_day"].sum()),
    }


def site_rpf_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    site_day = df.groupby(["substation_id", "date"])["label_interval"].any()
    day_counts = site_day.groupby("substation_id").sum().rename("rpf_days")
    total_days = site_day.groupby("substation_id").size().rename("total_days")
    interval_counts = df.groupby("substation_id")["label_interval"].sum().rename(
        "rpf_intervals"
    )
    rows = pd.concat([day_counts, total_days, interval_counts], axis=1).fillna(0)
    rows["dataset"] = dataset_name
    rows["rpf_day_pct"] = rows["rpf_days"] / rows["total_days"] * 100.0
    return rows.reset_index()[
        ["dataset", "substation_id", "total_days", "rpf_days", "rpf_day_pct", "rpf_intervals"]
    ]


def alpha_loso_sites(alpha: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    top_n = int(cfg["correction"]["alpha_top_n_loso_sites"])
    rankings = site_rpf_summary(alpha, "Alpha").sort_values(
        ["rpf_days", "rpf_intervals", "substation_id"],
        ascending=[False, False, True],
    )
    return rankings.head(top_n)["substation_id"].tolist()


def gamma_site_rankings(beta: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    rows = []
    for site, group in beta.groupby("substation_id"):
        err = group["net_load_MW"] - group["reference_net_load_MW"]
        site_days = group.groupby("date")["label_interval"].any()
        flagged = group.loc[group["label_interval"]]
        min_reference = float(group["reference_net_load_MW"].min())
        rows.append(
            {
                "substation_id": site,
                "rpf_days": int(site_days.sum()),
                "rpf_intervals": int(group["label_interval"].sum()),
                "data_error_rmse_MW": rmse(group["reference_net_load_MW"], group["net_load_MW"]),
                "data_error_mae_MW": mae(group["reference_net_load_MW"], group["net_load_MW"]),
                "max_raw_flagged_MW": float(flagged["net_load_MW"].max())
                if not flagged.empty
                else 0.0,
                "mean_raw_flagged_MW": float(flagged["net_load_MW"].mean())
                if not flagged.empty
                else 0.0,
                "min_reference_net_load_MW": min_reference,
                "abs_min_reference_net_load_MW": abs(min_reference),
                "raw_reference_error_MW_sum": float(err.abs().sum()),
            }
        )
    ranking_metric = (
        (cfg or {}).get("gamma", {}).get("ranking_metric", "data_error_rmse_MW")
    )
    ranking = pd.DataFrame(rows)
    if ranking_metric not in ranking.columns:
        raise ValueError(f"Unknown Gamma ranking metric: {ranking_metric}")
    ranking = ranking.sort_values(
        [ranking_metric, "rpf_days", "rpf_intervals", "substation_id"],
        ascending=[False, False, False, True],
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking.reset_index(drop=True)


def select_gamma_site(
    beta: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
    gamma_site_override: str | None = None,
) -> str:
    available_sites = set(beta["substation_id"].astype(str).unique())
    selected = gamma_site_override
    if selected is None and cfg is not None:
        gamma_cfg = cfg.get("gamma", {})
        if gamma_cfg.get("selection_mode") == "manual":
            selected = gamma_cfg.get("manual_site")
    if selected:
        selected = str(selected)
        if selected not in available_sites:
            raise ValueError(f"Gamma site {selected!r} is not present in final Beta.")
        return selected
    return str(gamma_site_rankings(beta, cfg).iloc[0]["substation_id"])


def extract_rpf_events(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    events: list[dict[str, Any]] = []
    for (site, date), group in df.groupby(["substation_id", "date"], sort=True):
        group = group.sort_values("_timestamp_dt").reset_index(drop=True)
        active = False
        start_idx = 0
        for idx, flag in enumerate(group["label_interval"].to_numpy(dtype=bool)):
            if flag and not active:
                active = True
                start_idx = idx
            is_last = idx == len(group) - 1
            if active and ((not flag) or is_last):
                end_idx = idx if flag and is_last else idx - 1
                event = group.iloc[start_idx : end_idx + 1]
                events.append(
                    {
                        "dataset": dataset_name,
                        "substation_id": site,
                        "date": date,
                        "start_timestamp": event["_timestamp_dt"].iloc[0],
                        "end_timestamp": event["_timestamp_dt"].iloc[-1],
                        "duration_minutes": int(len(event) * 15),
                        "n_intervals": int(len(event)),
                        "min_reference_net_load_MW": float(
                            event["reference_net_load_MW"].min()
                        ),
                        "max_raw_net_load_MW": float(event["net_load_MW"].max()),
                    }
                )
                active = False
    return pd.DataFrame(events)


def temporal_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    summaries = []
    for level, columns in {
        "month": ["month"],
        "season": ["season"],
        "hour": ["hour"],
        "weekday_weekend": ["is_weekend"],
        "month_hour": ["month", "hour"],
    }.items():
        grouped = (
            df.groupby(columns)
            .agg(
                total_intervals=("label_interval", "size"),
                rpf_intervals=("label_interval", "sum"),
            )
            .reset_index()
        )
        grouped["dataset"] = dataset_name
        grouped["level"] = level
        grouped["rpf_interval_pct"] = (
            grouped["rpf_intervals"] / grouped["total_intervals"] * 100.0
        )
        summaries.append(grouped)
    return pd.concat(summaries, ignore_index=True, sort=False)


def dataset_occurrence_summary(occurrence: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        occurrence.groupby("dataset", as_index=False)
        .agg(
            n_sites=("substation_id", "nunique"),
            total_site_days=("total_days", "sum"),
            rpf_site_days=("rpf_days", "sum"),
            rpf_intervals=("rpf_intervals", "sum"),
        )
        .sort_values("dataset")
    )
    grouped["rpf_site_day_pct"] = (
        grouped["rpf_site_days"] / grouped["total_site_days"] * 100.0
    )
    return grouped[
        [
            "dataset",
            "n_sites",
            "total_site_days",
            "rpf_site_days",
            "rpf_site_day_pct",
            "rpf_intervals",
        ]
    ]


def event_dataset_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "n_events",
                "duration_minutes_mean",
                "duration_minutes_median",
                "duration_minutes_max",
                "min_reference_net_load_MW_min",
                "max_raw_net_load_MW_max",
            ]
        )
    return (
        events.groupby("dataset", as_index=False)
        .agg(
            n_events=("duration_minutes", "size"),
            duration_minutes_mean=("duration_minutes", "mean"),
            duration_minutes_median=("duration_minutes", "median"),
            duration_minutes_max=("duration_minutes", "max"),
            min_reference_net_load_MW_min=("min_reference_net_load_MW", "min"),
            max_raw_net_load_MW_max=("max_raw_net_load_MW", "max"),
        )
        .sort_values("dataset")
    )


def final_dataset_validation_summary(
    final_summary: pd.DataFrame,
    cfg: dict[str, Any],
    gamma_site: str,
) -> pd.DataFrame:
    beta = final_summary.loc[final_summary["dataset"] == "Beta"].iloc[0]
    gamma = final_summary.loc[final_summary["dataset"] == "Gamma"].iloc[0]
    rows = [
        {
            "check": "schema_version",
            "expected": "journal_v2",
            "actual": cfg.get("schema_version"),
            "status": cfg.get("schema_version") == "journal_v2",
        },
        {
            "check": "beta_date_range",
            "expected": f"{cfg['windows']['beta_start']} to {cfg['windows']['beta_end']}",
            "actual": f"{str(beta['min_timestamp'])[:10]} to {str(beta['max_timestamp'])[:10]}",
            "status": str(beta["min_timestamp"])[:10] == cfg["windows"]["beta_start"]
            and str(beta["max_timestamp"])[:10] == cfg["windows"]["beta_end"],
        },
        {
            "check": "beta_rows",
            "expected": 280_800,
            "actual": int(beta["n_rows"]),
            "status": int(beta["n_rows"]) == 280_800,
        },
        {
            "check": "gamma_one_site",
            "expected": "one selected Beta site",
            "actual": gamma_site,
            "status": int(gamma["n_sites"]) == 1,
        },
        {
            "check": "gamma_rows",
            "expected": 35_136,
            "actual": int(gamma["n_rows"]),
            "status": int(gamma["n_rows"]) == 35_136,
        },
    ]
    return pd.DataFrame(rows)


def output_dir_manifest_payload(dirs: dict[str, Path]) -> dict[str, str]:
    return {name: str(path) for name, path in dirs.items()}


def run_prepare_datasets(
    article_root: Path | None = None,
    gamma_site_override: str | None = None,
) -> dict[str, Any]:
    root = find_article_root(article_root)
    cfg = load_config(root)
    paths = article_paths(root, cfg)
    ensure_output_dirs(paths)
    dirs = notebook_output_dirs(paths, "00_prepare_datasets")

    result = build_final_datasets(root, cfg, gamma_site_override=gamma_site_override)
    alpha = result["alpha"]
    beta = result["beta"]
    gamma = result["gamma"]
    gamma_rank = result["gamma_rankings"]
    gamma_site = result["gamma_site"]
    final_summary = result["final_summary"]

    alpha_rank = site_rpf_summary(alpha, "Alpha").sort_values(
        ["rpf_days", "rpf_intervals", "substation_id"],
        ascending=[False, False, True],
    )
    alpha_rank.insert(0, "rank", np.arange(1, len(alpha_rank) + 1))
    alpha_sites = alpha_loso_sites(alpha, cfg)
    validation = final_dataset_validation_summary(final_summary, cfg, gamma_site)

    write_csv(final_summary, dirs["intermediate"] / "01_dataset_summary.csv")
    write_csv(alpha_rank, dirs["intermediate"] / "02_alpha_site_rankings.csv")
    write_csv(gamma_rank, dirs["intermediate"] / "03_beta_gamma_site_rankings.csv")
    write_csv(validation, dirs["intermediate"] / "04_final_dataset_validation.csv")
    write_manifest(
        paths,
        "00_prepare_datasets.json",
        {
            "notebook": "00_prepare_datasets.ipynb",
            "schema_version": cfg["schema_version"],
            "output_subfolders": output_dir_manifest_payload(dirs),
            "alpha_dataset_path": str(article_path(root, cfg["paths"]["alpha_dataset_path"])),
            "beta_dataset_path": str(article_path(root, cfg["paths"]["beta_dataset_path"])),
            "gamma_dataset_path": str(article_path(root, cfg["paths"]["gamma_dataset_path"])),
            "alpha_loso_sites": alpha_sites,
            "gamma_site": gamma_site,
            "gamma_selection_mode": cfg.get("gamma", {}).get("selection_mode", "auto"),
            "gamma_ranking_metric": cfg.get("gamma", {}).get(
                "ranking_metric", "data_error_rmse_MW"
            ),
            "gamma_override_used": gamma_site_override is not None,
            "beta_review_status": cfg["datasets"]["beta_review_status"],
            "row_counts": {
                "alpha": int(len(alpha)),
                "beta": int(len(beta)),
                "gamma": int(len(gamma)),
            },
        },
    )
    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "gamma_site": gamma_site,
        "alpha_sites": alpha_sites,
        "final_summary": final_summary,
        "gamma_rankings": gamma_rank,
        "validation": validation,
    }


def run_characterisation(article_root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = find_article_root(article_root)
    cfg = load_config(root)
    paths = article_paths(root, cfg)
    ensure_output_dirs(paths)
    dirs = notebook_output_dirs(paths, "01_characterisation")
    alpha = load_dataset(root, cfg, "alpha")
    beta = load_dataset(root, cfg, "beta")

    occurrence = pd.concat(
        [site_rpf_summary(alpha, "Alpha"), site_rpf_summary(beta, "Beta")],
        ignore_index=True,
    )
    temporal = pd.concat(
        [temporal_summary(alpha, "Alpha"), temporal_summary(beta, "Beta")],
        ignore_index=True,
        sort=False,
    )
    events = pd.concat(
        [extract_rpf_events(alpha, "Alpha"), extract_rpf_events(beta, "Beta")],
        ignore_index=True,
    )
    occurrence_dataset = dataset_occurrence_summary(occurrence)
    event_summary = event_dataset_summary(events)

    write_csv(occurrence_dataset, dirs["intermediate"] / "01_rpf_occurrence_by_dataset.csv")
    write_csv(occurrence, dirs["intermediate"] / "02_rpf_occurrence_by_site.csv")
    write_csv(temporal, dirs["intermediate"] / "03_rpf_temporal_summary.csv")
    write_csv(events, dirs["intermediate"] / "04_rpf_event_summary.csv")
    write_csv(
        occurrence_dataset,
        dirs["tables"] / "table01_rpf_occurrence_summary_alpha_beta.csv",
    )
    write_csv(event_summary, dirs["tables"] / "table02_rpf_event_summary_alpha_beta.csv")
    figure_paths = write_characterisation_figures(
        occurrence, temporal, events, dirs["figures"]
    )
    write_manifest(
        paths,
        "01_characterisation.json",
        {
            "notebook": "01_characterisation.ipynb",
            "schema_version": cfg["schema_version"],
            "output_subfolders": output_dir_manifest_payload(dirs),
            "datasets": ["Alpha", "Beta"],
            "n_events": int(len(events)),
            "tables": [
                "table01_rpf_occurrence_summary_alpha_beta.csv",
                "table02_rpf_event_summary_alpha_beta.csv",
            ],
            "figures": [path.name for path in figure_paths],
        },
    )
    return {
        "occurrence_dataset": occurrence_dataset,
        "occurrence": occurrence,
        "temporal": temporal,
        "events": events,
        "event_summary": event_summary,
    }


def _load_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_characterisation_figures(
    occurrence: pd.DataFrame,
    temporal: pd.DataFrame,
    events: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    plt = _load_matplotlib()
    figure_paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 4))
    pivot = occurrence.pivot(index="substation_id", columns="dataset", values="rpf_days").fillna(0)
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("RPF days")
    ax.set_xlabel("Site")
    ax.set_title("RPF day counts by site")
    fig.tight_layout()
    path = figures_dir / "fig01_site_rpf_day_counts_alpha_beta.png"
    fig.savefig(path, dpi=200)
    figure_paths.append(path)
    plt.close(fig)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True)
    for ax, dataset in zip(axes, ["Alpha", "Beta"]):
        month_hour = temporal[
            (temporal["dataset"] == dataset) & (temporal["level"] == "month_hour")
        ]
        heat = month_hour.pivot(index="month", columns="hour", values="rpf_interval_pct").fillna(0)
        image = ax.imshow(heat.to_numpy(), aspect="auto", origin="lower")
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels(heat.columns)
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(heat.index)
        ax.set_xlabel("Hour")
        ax.set_title(dataset)
    axes[0].set_ylabel("Month")
    fig.suptitle("RPF interval percentage by month and hour")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="% intervals")
    path = figures_dir / "fig02_month_hour_heatmap_alpha_beta.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    figure_paths.append(path)
    plt.close(fig)

    if not events.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        for dataset, group in events.groupby("dataset"):
            group["duration_minutes"].hist(ax=ax, bins=30, alpha=0.55, label=dataset)
        ax.set_xlabel("Event duration (minutes)")
        ax.set_ylabel("Count")
        ax.set_title("RPF event duration distribution")
        ax.legend()
        fig.tight_layout()
        path = figures_dir / "fig03_event_duration_distribution_alpha_beta.png"
        fig.savefig(path, dpi=200)
        figure_paths.append(path)
        plt.close(fig)
    return figure_paths


def binary_metrics(y_true: Iterable[Any], y_pred: Iterable[Any]) -> dict[str, Any]:
    true = np.asarray(list(y_true), dtype=bool)
    pred = np.asarray(list(y_pred), dtype=bool)
    tp = int((true & pred).sum())
    fp = int((~true & pred).sum())
    fn = int((true & ~pred).sum())
    tn = int((~true & ~pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "support": int(len(true)),
        "positive_support": int(true.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def daytime_mask(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    start = int(cfg["correction"]["interval_daytime_start_hour"])
    end = int(cfg["correction"]["interval_daytime_end_hour"])
    return df["hour"].between(start, end, inclusive="both")


def evaluate_prediction_frame(
    pred_df: pd.DataFrame,
    cfg: dict[str, Any],
    dataset: str,
    fold_id: str,
    method: str,
) -> pd.DataFrame:
    day_df = (
        pred_df.groupby(["substation_id", "date"])
        .agg(label_day=("label_interval", "any"), pred_day=("pred_interval", "any"))
        .reset_index()
    )
    rows = []
    for level, values in [
        ("day", binary_metrics(day_df["label_day"], day_df["pred_day"])),
        (
            "interval_daytime",
            binary_metrics(
                pred_df.loc[daytime_mask(pred_df, cfg), "label_interval"],
                pred_df.loc[daytime_mask(pred_df, cfg), "pred_interval"],
            ),
        ),
    ]:
        rows.append(
            {
                "dataset": dataset,
                "fold_id": fold_id,
                "method": method,
                "level": level,
                **values,
            }
        )
    return pd.DataFrame(rows)


def _repo_src_path(article_root: Path) -> Path:
    return article_root.parents[1] / "src"


def _ensure_package_import(article_root: Path) -> None:
    src_path = _repo_src_path(article_root)
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def inference_config(cfg: dict[str, Any], selected_model: str) -> dict[str, Any]:
    return {
        "columns": {
            "site": cfg["columns"]["site"],
            "timestamp": cfg["columns"]["timestamp"],
            "net_load": cfg["columns"]["net_load"],
            "solar": cfg["columns"]["solar"],
        },
        "runtime": {"interval_minutes": 15, "strict_validation": True},
        "model": {
            "selected_model": selected_model,
            "m7_threshold": cfg["correction"]["m7_threshold"],
            "m8_xgb": cfg["correction"]["m8_xgb"],
        },
    }


def training_config(cfg: dict[str, Any]) -> dict[str, Any]:
    windows = cfg["windows"]
    return {
        "pynrpf_training": {
            "model_id": "m8_xgb",
            "labels": {"day": "label_day", "interval": "label_interval"},
            "split": {
                "train_start": windows["train_start"],
                "train_end": windows["train_end"],
                "validation_start": windows["test_start"],
                "validation_end": windows["test_end"],
            },
            "thresholds": {
                "xgb1_day": cfg["correction"]["m8_xgb"]["xgb1_day"]["threshold"],
                "xgb2_timestamp": cfg["correction"]["m8_xgb"]["xgb2_timestamp"]["threshold"],
            },
            "random_seed": int(cfg["execution"]["random_seed"]),
            "output": {"base_uri": "unused_notebook_bundle"},
        }
    }


def _model_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[EXPECTED_COLUMNS].copy()


def train_m8_bundle(
    train_source_df: pd.DataFrame, cfg: dict[str, Any], article_root: Path
) -> dict[str, Any]:
    _ensure_package_import(article_root)
    from pynrpf.config import load_config as load_inference_config
    from pynrpf.registry import get_model
    from pynrpf.training_config import load_training_config
    from pynrpf.validation import validate_dataframe

    infer_cfg = inference_config(cfg, "m8_xgb")
    train_cfg = training_config(cfg)
    package_cfg = load_inference_config(infer_cfg)
    package_cfg["training"] = load_training_config({**infer_cfg, **train_cfg})
    cleaned, _ = validate_dataframe(_model_input_columns(train_source_df), package_cfg)
    return get_model("m8_xgb").train(
        cleaned,
        package_cfg,
        package_cfg["columns"],
        labels={"day": "label_day", "interval": "label_interval"},
    )


def predict_m8_bundle(
    eval_df: pd.DataFrame,
    bundle: dict[str, Any],
    cfg: dict[str, Any],
    article_root: Path,
) -> pd.DataFrame:
    _ensure_package_import(article_root)
    from pynrpf.plugins.m8_xgb import _align_features
    from pynrpf.plugins.m8_xgb import _bundle_section
    from pynrpf.plugins.m8_xgb import _feature_cfg
    from pynrpf.plugins.m8_xgb import build_xgb1_features, build_xgb2_features

    cols = cfg["columns"]
    m8_cfg = cfg["correction"]["m8_xgb"]
    work = _model_input_columns(eval_df).copy()
    work["timestamp"] = _parse_wall_clock(work["timestamp"])
    work["_pynrpf_gt_dummy"] = 1.0
    feature_cfg = _feature_cfg(work, cols["timestamp"], m8_cfg)
    xgb1_cfg = m8_cfg["xgb1_day"]
    xgb2_cfg = m8_cfg["xgb2_timestamp"]
    clf1, feat_cols1, thr1 = _bundle_section(bundle, "xgb1_day", xgb1_cfg["threshold"])
    clf2, feat_cols2, thr2 = _bundle_section(bundle, "xgb2_timestamp", xgb2_cfg["threshold"])

    day_df, _, _ = build_xgb1_features(
        work,
        feature_cfg,
        cols["site"],
        cols["timestamp"],
        cols["net_load"],
        cols["solar"],
        "_pynrpf_gt_dummy",
    )
    prob_day = clf1.predict_proba(_align_features(day_df, feat_cols1).to_numpy(np.float32))[:, 1]
    day_df["pred_day"] = prob_day >= thr1
    day_df["prob_day"] = prob_day
    candidates = day_df.loc[day_df["pred_day"], [cols["site"], "date"]]
    interval_cols = [cols["site"], cols["timestamp"], "prob_interval", "pred_interval"]

    if candidates.empty:
        ts_results = pd.DataFrame(columns=interval_cols)
    else:
        ts_df, _, _ = build_xgb2_features(
            work,
            feature_cfg,
            day_df,
            candidates,
            cols["site"],
            cols["timestamp"],
            cols["net_load"],
            cols["solar"],
            "_pynrpf_gt_dummy",
        )
        if ts_df.empty:
            ts_results = pd.DataFrame(columns=interval_cols)
        else:
            X_ts = _align_features(ts_df, feat_cols2).to_numpy(np.float32)
            prob_ts = clf2.predict_proba(X_ts)[:, 1]
            ts_df["prob_interval"] = prob_ts
            ts_df["pred_interval"] = prob_ts >= thr2
            ts_results = ts_df[[cols["site"], cols["timestamp"], "prob_interval", "pred_interval"]]

    result = eval_df.copy()
    result["pred_interval"] = False
    result["prob_interval"] = np.nan
    result_ts = result["_timestamp_dt"]
    result_key = pd.DataFrame({cols["site"]: result[cols["site"]], cols["timestamp"]: result_ts})
    if not ts_results.empty:
        merged = result_key.merge(ts_results, on=[cols["site"], cols["timestamp"]], how="left")
        result["pred_interval"] = merged["pred_interval"].fillna(False).astype(bool).to_numpy()
        result["prob_interval"] = merged["prob_interval"].to_numpy()
    result["corrected_net_load_MW"] = np.where(
        result["pred_interval"], -result["net_load_MW"], result["net_load_MW"]
    )
    return result


def predict_m7(eval_df: pd.DataFrame, cfg: dict[str, Any], article_root: Path) -> pd.DataFrame:
    _ensure_package_import(article_root)
    from pynrpf.api import run_inference

    out = run_inference(_model_input_columns(eval_df), inference_config(cfg, "m7_dtr"))["data"]
    result = eval_df.copy()
    result["pred_interval"] = out["pynrpf_interval_flag"].fillna(False).astype(bool).to_numpy()
    result["prob_interval"] = np.where(result["pred_interval"], 1.0, 0.0)
    result["corrected_net_load_MW"] = out["pynrpf_corrected_net_load"].to_numpy()
    return result


def correction_smoke_plan(
    alpha: pd.DataFrame, beta: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    for site in alpha_loso_sites(alpha, cfg):
        rows.append(
            {"experiment": "alpha_top3_loso", "fold_id": f"holdout_{site}", "status": "planned"}
        )
    rows.append(
        {"experiment": "beta_transfer", "fold_id": "alpha_train_to_beta", "status": "planned"}
    )
    rows.append({"experiment": "beta_transfer", "fold_id": "m7_dtr_beta", "status": "planned"})
    rows.append({"experiment": "beta_rows", "fold_id": "beta_filtered", "status": str(len(beta))})
    return pd.DataFrame(rows)


def correction_smoke_metrics(alpha: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    methods = cfg["correction"]["methods"]
    for holdout_site in alpha_loso_sites(alpha, cfg):
        for method in methods:
            for level in ["day", "interval_daytime"]:
                rows.append(
                    {
                        "dataset": "Alpha",
                        "fold_id": f"alpha_holdout_{holdout_site}",
                        "method": method,
                        "level": level,
                        "support": np.nan,
                        "positive_support": np.nan,
                        "tp": np.nan,
                        "fp": np.nan,
                        "fn": np.nan,
                        "tn": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "status": "planned_smoke_only",
                    }
                )
    for method in methods:
        for level in ["day", "interval_daytime"]:
            rows.append(
                {
                    "dataset": "Beta",
                    "fold_id": "beta_transfer",
                    "method": method,
                    "level": level,
                    "support": np.nan,
                    "positive_support": np.nan,
                    "tp": np.nan,
                    "fp": np.nan,
                    "fn": np.nan,
                    "tn": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "status": "planned_smoke_only",
                }
            )
    return pd.DataFrame(rows)


def correction_confusion_matrices(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = ["dataset", "fold_id", "method", "level", "tp", "fp", "fn", "tn"]
    return metrics[[col for col in cols if col in metrics.columns]].copy()


def write_correction_figures(metrics: pd.DataFrame, figures_dir: Path) -> list[Path]:
    usable = metrics.dropna(subset=["precision", "recall", "f1"], how="all").copy()
    if usable.empty:
        return []
    plt = _load_matplotlib()
    figure_paths: list[Path] = []

    matrix = correction_confusion_matrices(usable)
    matrix["label"] = (
        matrix["dataset"]
        + "\n"
        + matrix["fold_id"].astype(str)
        + "\n"
        + matrix["method"].astype(str)
        + "\n"
        + matrix["level"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(max(8, len(matrix) * 0.45), 4))
    bottom = np.zeros(len(matrix))
    for col in ["tp", "fp", "fn", "tn"]:
        values = pd.to_numeric(matrix[col], errors="coerce").fillna(0).to_numpy()
        ax.bar(matrix["label"], values, bottom=bottom, label=col.upper())
        bottom += values
    ax.set_ylabel("Count")
    ax.set_title("Correction confusion-matrix components")
    ax.tick_params(axis="x", rotation=75)
    ax.legend(ncol=4)
    fig.tight_layout()
    path = figures_dir / "fig01_correction_confusion_matrices.png"
    fig.savefig(path, dpi=200)
    figure_paths.append(path)
    plt.close(fig)

    score = usable.melt(
        id_vars=["dataset", "fold_id", "method", "level"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="value",
    )
    score["label"] = score["dataset"] + " / " + score["method"] + " / " + score["level"]
    pivot = (
        score.groupby(["label", "metric"], as_index=False)["value"]
        .mean()
        .pivot(index="label", columns="metric", values="value")
    )
    fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 0.45), 4))
    pivot[["precision", "recall", "f1"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Correction precision, recall, and F1")
    ax.tick_params(axis="x", rotation=65)
    fig.tight_layout()
    path = figures_dir / "fig02_correction_precision_recall_f1.png"
    fig.savefig(path, dpi=200)
    figure_paths.append(path)
    plt.close(fig)
    return figure_paths


def run_correction_validation(article_root: Path | None = None) -> dict[str, Any]:
    root = find_article_root(article_root)
    cfg = load_config(root)
    paths = article_paths(root, cfg)
    ensure_output_dirs(paths)
    dirs = notebook_output_dirs(paths, "02_correction_validation")
    alpha = load_dataset(root, cfg, "alpha")
    beta = load_dataset(root, cfg, "beta")

    if not bool(cfg["execution"]["run_full_correction_validation"]):
        plan = correction_smoke_plan(alpha, beta, cfg)
        metrics = correction_smoke_metrics(alpha, cfg)
        confusion = correction_confusion_matrices(metrics)
        write_csv(plan, dirs["intermediate"] / "01_correction_validation_plan.csv")
        write_csv(metrics, dirs["metrics"] / "01_correction_metrics.csv")
        write_csv(confusion, dirs["metrics"] / "02_correction_confusion_matrices.csv")
        write_csv(metrics, dirs["tables"] / "table01_correction_metrics_summary.csv")
        write_csv(
            metrics.loc[metrics["dataset"] == "Beta"].copy(),
            dirs["tables"] / "table02_beta_transfer_key_metrics.csv",
        )
        figure_paths = write_correction_figures(metrics, dirs["figures"])
        write_manifest(
            paths,
            "02_correction_validation.json",
            {
                "notebook": "02_correction_validation.ipynb",
                "schema_version": cfg["schema_version"],
                "status": "smoke_only",
                "output_subfolders": output_dir_manifest_payload(dirs),
                "row_counts": {"alpha": int(len(alpha)), "beta": int(len(beta))},
                "tables": [
                    "table01_correction_metrics_summary.csv",
                    "table02_beta_transfer_key_metrics.csv",
                ],
                "figures": [path.name for path in figure_paths],
            },
        )
        return {"status": "smoke_only", "plan": plan, "metrics": metrics}

    metric_frames = []
    pred_index = 2
    for holdout_site in alpha_loso_sites(alpha, cfg):
        train_source = alpha.loc[alpha["substation_id"] != holdout_site].copy()
        eval_df = filter_date_window(
            alpha.loc[alpha["substation_id"] == holdout_site].copy(),
            cfg["windows"]["test_start"],
            cfg["windows"]["test_end"],
        )
        bundle = train_m8_bundle(train_source, cfg, root)
        for method, pred in [
            ("m8_xgb", predict_m8_bundle(eval_df, bundle, cfg, root)),
            ("m7_dtr", predict_m7(eval_df, cfg, root)),
        ]:
            fold_id = f"alpha_holdout_{holdout_site}"
            pred_cols = EXPECTED_COLUMNS + ["pred_interval", "corrected_net_load_MW"]
            pred_path = (
                dirs["intermediate"]
                / f"{pred_index:02d}_correction_predictions_{fold_id}_{method}.csv"
            )
            write_csv(pred[pred_cols], pred_path)
            pred_index += 1
            metric_frames.append(evaluate_prediction_frame(pred, cfg, "Alpha", fold_id, method))

    alpha_train = alpha.copy()
    beta_bundle = train_m8_bundle(alpha_train, cfg, root)
    for method, pred in [
        ("m8_xgb", predict_m8_bundle(beta, beta_bundle, cfg, root)),
        ("m7_dtr", predict_m7(beta, cfg, root)),
    ]:
        fold_id = "beta_transfer"
        pred_cols = EXPECTED_COLUMNS + ["pred_interval", "corrected_net_load_MW"]
        pred_path = (
            dirs["intermediate"]
            / f"{pred_index:02d}_correction_predictions_{fold_id}_{method}.csv"
        )
        write_csv(pred[pred_cols], pred_path)
        pred_index += 1
        metric_frames.append(evaluate_prediction_frame(pred, cfg, "Beta", fold_id, method))

    metrics = pd.concat(metric_frames, ignore_index=True)
    plan = correction_smoke_plan(alpha, beta, cfg)
    confusion = correction_confusion_matrices(metrics)
    write_csv(plan, dirs["intermediate"] / "01_correction_validation_plan.csv")
    write_csv(metrics, dirs["metrics"] / "01_correction_metrics.csv")
    write_csv(confusion, dirs["metrics"] / "02_correction_confusion_matrices.csv")
    write_csv(metrics, dirs["tables"] / "table01_correction_metrics_summary.csv")
    write_csv(
        metrics.loc[metrics["dataset"] == "Beta"].copy(),
        dirs["tables"] / "table02_beta_transfer_key_metrics.csv",
    )
    figure_paths = write_correction_figures(metrics, dirs["figures"])
    write_manifest(
        paths,
        "02_correction_validation.json",
        {
            "notebook": "02_correction_validation.ipynb",
            "schema_version": cfg["schema_version"],
            "status": "complete",
            "output_subfolders": output_dir_manifest_payload(dirs),
            "row_counts": {"alpha": int(len(alpha)), "beta": int(len(beta))},
            "tables": [
                "table01_correction_metrics_summary.csv",
                "table02_beta_transfer_key_metrics.csv",
            ],
            "figures": [path.name for path in figure_paths],
        },
    )
    return {"status": "complete", "metrics": metrics}


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true = pd.Series(y_true, dtype="float64")
    pred = pd.Series(y_pred, dtype="float64")
    mask = true.notna() & pred.notna()
    if not mask.any():
        return math.nan
    return float(np.sqrt(np.mean((true[mask] - pred[mask]) ** 2)))


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true = pd.Series(y_true, dtype="float64")
    pred = pd.Series(y_pred, dtype="float64")
    mask = true.notna() & pred.notna()
    if not mask.any():
        return math.nan
    return float(np.mean(np.abs(true[mask] - pred[mask])))


def forecast_metric_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, model), group in df.groupby(["data_condition", "model"]):
        rows.append(
            {
                "data_condition": condition,
                "model": model,
                "n_targets": int(len(group)),
                "rmse_MW": rmse(group["y_reference"], group["y_pred"]),
                "mae_MW": mae(group["y_reference"], group["y_pred"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["data_condition", "model"]).reset_index(drop=True)


def add_forecast_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ts = pd.to_datetime(out["target_timestamp"])
    out["target_hour"] = ts.dt.hour
    out["target_minute"] = ts.dt.minute
    out["target_dayofweek"] = ts.dt.dayofweek
    out["target_month"] = ts.dt.month
    out["target_is_weekend"] = (out["target_dayofweek"] >= 5).astype(int)
    minutes = out["target_hour"] * 60 + out["target_minute"]
    out["target_time_sin"] = np.sin(2 * np.pi * minutes / 1440)
    out["target_time_cos"] = np.cos(2 * np.pi * minutes / 1440)
    out["target_dow_sin"] = np.sin(2 * np.pi * out["target_dayofweek"] / 7)
    out["target_dow_cos"] = np.cos(2 * np.pi * out["target_dayofweek"] / 7)
    return out


def build_forecast_examples(
    gamma: pd.DataFrame,
    series_column: str,
    cfg: dict[str, Any],
    target_start: str,
    target_end: str,
) -> pd.DataFrame:
    horizon = pd.Timedelta(days=int(cfg["forecast"]["horizon_days"]))
    lookback = pd.Timedelta(days=int(cfg["forecast"]["lookback_days"]))
    work = gamma.sort_values("_timestamp_dt").set_index("_timestamp_dt")
    targets = pd.date_range(target_start, f"{target_end} 23:45:00", freq="15min")
    rows = []
    for target in targets:
        origin = target - horizon
        window_start = origin - lookback
        if target not in work.index or origin not in work.index:
            continue
        window = work.loc[(work.index > window_start) & (work.index <= origin)]
        if window.empty:
            continue
        values = window[series_column].astype(float)
        target_row = work.loc[target]
        rows.append(
            {
                "target_timestamp": target,
                "origin_timestamp": origin,
                "data_condition": series_column,
                "origin_value": float(work.loc[origin, series_column]),
                "lookback_mean": float(values.mean()),
                "lookback_std": float(values.std(ddof=0)),
                "lookback_min": float(values.min()),
                "lookback_max": float(values.max()),
                "lookback_p05": float(values.quantile(0.05)),
                "lookback_p95": float(values.quantile(0.95)),
                "last_day_mean": float(values.tail(96).mean()),
                "last_day_min": float(values.tail(96).min()),
                "last_day_max": float(values.tail(96).max()),
                "y_condition": float(target_row[series_column]),
                "y_reference": float(target_row["reference_net_load_MW"]),
                "y_raw": float(target_row["net_load_MW"]),
            }
        )
    return add_forecast_calendar_features(pd.DataFrame(rows))


def forecast_feature_columns() -> list[str]:
    return [
        "origin_value",
        "lookback_mean",
        "lookback_std",
        "lookback_min",
        "lookback_max",
        "lookback_p05",
        "lookback_p95",
        "last_day_mean",
        "last_day_min",
        "last_day_max",
        "target_hour",
        "target_minute",
        "target_dayofweek",
        "target_month",
        "target_is_weekend",
        "target_time_sin",
        "target_time_cos",
        "target_dow_sin",
        "target_dow_cos",
    ]


def data_error_benchmark(gamma: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    start = cfg["windows"]["gamma_forecast_test_start"]
    end = cfg["windows"]["gamma_forecast_test_end"]
    examples = build_forecast_examples(gamma, "net_load_MW", cfg, start, end)
    out = examples[["target_timestamp", "y_reference", "y_raw"]].copy()
    out["data_condition"] = "raw_uncorrected"
    out["model"] = "data_error_only"
    out["y_pred"] = out["y_raw"]
    return out[["target_timestamp", "data_condition", "model", "y_reference", "y_pred"]]


def run_gamma_forecast_impact(article_root: Path | None = None) -> dict[str, Any]:
    root = find_article_root(article_root)
    cfg = load_config(root)
    paths = article_paths(root, cfg)
    ensure_output_dirs(paths)
    dirs = notebook_output_dirs(paths, "03_gamma_forecast_impact")
    gamma = load_dataset(root, cfg, "gamma")
    gamma_site = str(gamma["substation_id"].iloc[0])
    gamma["raw_uncorrected_MW"] = gamma["net_load_MW"]
    gamma["reference_corrected_MW"] = gamma["reference_net_load_MW"]
    gamma["m8_xgb_corrected_MW"] = np.nan

    benchmark = data_error_benchmark(gamma, cfg)
    write_csv(
        gamma[
            EXPECTED_COLUMNS
            + [
                "raw_uncorrected_MW",
                "m8_xgb_corrected_MW",
                "reference_corrected_MW",
            ]
        ],
        dirs["intermediate"] / "01_gamma_series.csv",
    )
    write_csv(benchmark, dirs["metrics"] / "01_gamma_data_error_benchmark.csv")
    figure_paths = [write_gamma_series_figure(gamma, dirs["figures"], gamma_site)]

    if not bool(cfg["execution"]["run_full_forecast"]):
        metrics = forecast_metric_rows(benchmark)
        write_csv(metrics, dirs["metrics"] / "02_gamma_forecast_metrics.csv")
        write_csv(metrics, dirs["tables"] / "table01_forecast_impact.csv")
        write_csv(
            metrics.loc[metrics["model"] == "data_error_only"].copy(),
            dirs["tables"] / "table02_gamma_data_error_benchmark.csv",
        )
        rmse_path = write_forecast_metric_figure(metrics, dirs["figures"])
        residual_path = write_forecast_residual_figure(benchmark, dirs["figures"])
        figure_paths.extend([path for path in [rmse_path, residual_path] if path is not None])
        write_manifest(
            paths,
            "03_gamma_forecast_impact.json",
            {
                "notebook": "03_gamma_forecast_impact.ipynb",
                "schema_version": cfg["schema_version"],
                "status": "smoke_only",
                "gamma_site": gamma_site,
                "output_subfolders": output_dir_manifest_payload(dirs),
                "row_counts": {"gamma": int(len(gamma))},
                "tables": [
                    "table01_forecast_impact.csv",
                    "table02_gamma_data_error_benchmark.csv",
                ],
                "figures": [path.name for path in figure_paths if path is not None],
            },
        )
        return {"status": "smoke_only", "gamma_site": gamma_site, "metrics": metrics}

    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor

    alpha = load_dataset(root, cfg, "alpha")
    m8_bundle = train_m8_bundle(alpha, cfg, root)
    gamma_pred = predict_m8_bundle(gamma, m8_bundle, cfg, root)
    gamma["m8_xgb_corrected_MW"] = gamma_pred["corrected_net_load_MW"].to_numpy()
    write_csv(
        gamma[
            EXPECTED_COLUMNS
            + [
                "raw_uncorrected_MW",
                "m8_xgb_corrected_MW",
                "reference_corrected_MW",
            ]
        ],
        dirs["intermediate"] / "01_gamma_series.csv",
    )
    figure_paths = [write_gamma_series_figure(gamma, dirs["figures"], gamma_site)]

    train_end = pd.Timestamp(cfg["windows"]["gamma_forecast_test_start"]) - pd.Timedelta(minutes=15)
    train_start = gamma["_timestamp_dt"].min().strftime("%Y-%m-%d")
    test_start = cfg["windows"]["gamma_forecast_test_start"]
    test_end = cfg["windows"]["gamma_forecast_test_end"]
    condition_map = {
        "raw_uncorrected": "raw_uncorrected_MW",
        "m8_xgb_corrected": "m8_xgb_corrected_MW",
        "reference_corrected": "reference_corrected_MW",
    }
    forecast_frames = [benchmark]
    example_index = 2
    for condition, column in condition_map.items():
        train_examples = build_forecast_examples(
            gamma, column, cfg, train_start, train_end.strftime("%Y-%m-%d")
        )
        test_examples = build_forecast_examples(gamma, column, cfg, test_start, test_end)
        train_examples = train_examples.loc[train_examples["target_timestamp"] <= train_end]
        train_path = (
            dirs["intermediate"]
            / f"{example_index:02d}_gamma_forecast_examples_train_{condition}.csv"
        )
        test_path = (
            dirs["intermediate"]
            / f"{example_index + 1:02d}_gamma_forecast_examples_test_{condition}.csv"
        )
        write_csv(train_examples, train_path)
        write_csv(test_examples, test_path)
        example_index += 2

        seasonal = test_examples[["target_timestamp", "y_reference", "origin_value"]].copy()
        seasonal["data_condition"] = condition
        seasonal["model"] = "seasonal_naive"
        seasonal["y_pred"] = seasonal["origin_value"]
        forecast_cols = ["target_timestamp", "data_condition", "model", "y_reference", "y_pred"]
        forecast_frames.append(seasonal[forecast_cols])

        X_train = train_examples[forecast_feature_columns()].to_numpy(dtype=float)
        y_train = train_examples["y_condition"].to_numpy(dtype=float)
        X_test = test_examples[forecast_feature_columns()].to_numpy(dtype=float)
        for model_name, model in [
            ("linear_regression", LinearRegression()),
            ("xgboost", XGBRegressor(**cfg["forecast"]["xgboost"])),
        ]:
            model.fit(X_train, y_train)
            pred = test_examples[["target_timestamp", "y_reference"]].copy()
            pred["data_condition"] = condition
            pred["model"] = model_name
            pred["y_pred"] = model.predict(X_test)
            forecast_frames.append(pred)

    forecasts = pd.concat(forecast_frames, ignore_index=True)
    metrics = forecast_metric_rows(forecasts)
    write_csv(forecasts, dirs["intermediate"] / f"{example_index:02d}_gamma_forecasts.csv")
    write_csv(benchmark, dirs["metrics"] / "01_gamma_data_error_benchmark.csv")
    write_csv(metrics, dirs["metrics"] / "02_gamma_forecast_metrics.csv")
    write_csv(metrics, dirs["tables"] / "table01_forecast_impact.csv")
    write_csv(
        metrics.loc[metrics["model"] == "data_error_only"].copy(),
        dirs["tables"] / "table02_gamma_data_error_benchmark.csv",
    )
    rmse_path = write_forecast_metric_figure(metrics, dirs["figures"])
    residual_path = write_forecast_residual_figure(forecasts, dirs["figures"])
    figure_paths.extend([path for path in [rmse_path, residual_path] if path is not None])
    write_manifest(
        paths,
        "03_gamma_forecast_impact.json",
        {
            "notebook": "03_gamma_forecast_impact.ipynb",
            "schema_version": cfg["schema_version"],
            "status": "complete",
            "gamma_site": gamma_site,
            "output_subfolders": output_dir_manifest_payload(dirs),
            "row_counts": {"alpha": int(len(alpha)), "gamma": int(len(gamma))},
            "tables": [
                "table01_forecast_impact.csv",
                "table02_gamma_data_error_benchmark.csv",
            ],
            "figures": [path.name for path in figure_paths if path is not None],
        },
    )
    return {"status": "complete", "gamma_site": gamma_site, "metrics": metrics}


def write_gamma_series_figure(gamma: pd.DataFrame, figures_dir: Path, gamma_site: str) -> Path:
    plt = _load_matplotlib()
    plot_df = gamma.loc[
        (gamma["date"] >= "2024-09-01") & (gamma["date"] <= "2024-09-07")
    ].copy()
    if plot_df.empty:
        plot_df = gamma.tail(7 * 96).copy()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(plot_df["_timestamp_dt"], plot_df["net_load_MW"], label="Raw")
    if "m8_xgb_corrected_MW" in plot_df.columns and plot_df["m8_xgb_corrected_MW"].notna().any():
        ax.plot(plot_df["_timestamp_dt"], plot_df["m8_xgb_corrected_MW"], label="m8_xgb corrected")
    ax.plot(plot_df["_timestamp_dt"], plot_df["reference_net_load_MW"], label="Reference")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title(f"Gamma site {gamma_site}: raw vs reference net load")
    ax.set_ylabel("MW")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "fig01_gamma_series_raw_corrected_reference.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_forecast_metric_figure(metrics: pd.DataFrame, figures_dir: Path) -> Path | None:
    if metrics.empty:
        return None
    plt = _load_matplotlib()
    labels = metrics["data_condition"] + " / " + metrics["model"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, metrics["rmse_MW"])
    ax.set_ylabel("RMSE (MW)")
    ax.set_title("Gamma forecast impact")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = figures_dir / "fig02_gamma_forecast_rmse.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_forecast_residual_figure(forecasts: pd.DataFrame, figures_dir: Path) -> Path | None:
    if forecasts.empty or {"y_reference", "y_pred"}.difference(forecasts.columns):
        return None
    plot_df = forecasts.copy()
    plot_df["residual_MW"] = plot_df["y_pred"] - plot_df["y_reference"]
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = plot_df["data_condition"] + " / " + plot_df["model"]
    order = labels.drop_duplicates().tolist()
    data = [plot_df.loc[labels == label, "residual_MW"].dropna().to_numpy() for label in order]
    if not data:
        plt.close(fig)
        return None
    ax.boxplot(data, labels=order, showfliers=False)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_ylabel("Forecast residual (MW)")
    ax.set_title("Gamma forecast residual comparison")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = figures_dir / "fig03_gamma_forecast_residuals.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_table_formats(df: pd.DataFrame, csv_path: Path) -> dict[str, Path]:
    paths = {"csv": write_csv(df, csv_path)}
    md_path = csv_path.with_suffix(".md")
    tex_path = csv_path.with_suffix(".tex")
    try:
        markdown = df.to_markdown(index=False)
    except ImportError:
        markdown = df.to_csv(index=False)
    md_path.write_text(markdown + "\n", encoding="utf-8")
    tex_path.write_text(df.to_latex(index=False), encoding="utf-8")
    paths["md"] = md_path
    paths["tex"] = tex_path
    return paths


def inventory_existing(paths_to_check: dict[str, Path]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "artifact": name,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
            for name, path in paths_to_check.items()
        ]
    )


def run_publication_tables(article_root: Path | None = None) -> dict[str, Path]:
    root = find_article_root(article_root)
    cfg = load_config(root)
    paths = article_paths(root, cfg)
    ensure_output_dirs(paths)
    dirs = notebook_output_dirs(paths, "04_publication_tables_figures")
    outputs: dict[str, Path] = {}

    upstream_tables = {
        "dataset_summary": paths.final / "dataset_final_summary.csv",
        "characterisation_summary": paths.tables
        / "01_characterisation"
        / "table01_rpf_occurrence_summary_alpha_beta.csv",
        "correction_metrics": paths.tables
        / "02_correction_validation"
        / "table01_correction_metrics_summary.csv",
        "forecast_impact": paths.tables
        / "03_gamma_forecast_impact"
        / "table01_forecast_impact.csv",
    }
    table_targets = {
        "dataset_summary": "table01_dataset_summary.csv",
        "characterisation_summary": "table02_characterisation_summary.csv",
        "correction_metrics": "table03_correction_metrics.csv",
        "forecast_impact": "table04_forecast_impact.csv",
    }
    for name, src in upstream_tables.items():
        if src.exists():
            target = dirs["tables"] / table_targets[name]
            written = write_table_formats(pd.read_csv(src), target)
            outputs[table_targets[name]] = written["csv"]

    expected_figures = {
        "fig01_site_rpf_day_counts_alpha_beta": paths.figures
        / "01_characterisation"
        / "fig01_site_rpf_day_counts_alpha_beta.png",
        "fig02_month_hour_heatmap_alpha_beta": paths.figures
        / "01_characterisation"
        / "fig02_month_hour_heatmap_alpha_beta.png",
        "fig03_event_duration_distribution_alpha_beta": paths.figures
        / "01_characterisation"
        / "fig03_event_duration_distribution_alpha_beta.png",
        "fig01_correction_confusion_matrices": paths.figures
        / "02_correction_validation"
        / "fig01_correction_confusion_matrices.png",
        "fig02_correction_precision_recall_f1": paths.figures
        / "02_correction_validation"
        / "fig02_correction_precision_recall_f1.png",
        "fig01_gamma_series_raw_corrected_reference": paths.figures
        / "03_gamma_forecast_impact"
        / "fig01_gamma_series_raw_corrected_reference.png",
        "fig02_gamma_forecast_rmse": paths.figures
        / "03_gamma_forecast_impact"
        / "fig02_gamma_forecast_rmse.png",
        "fig03_gamma_forecast_residuals": paths.figures
        / "03_gamma_forecast_impact"
        / "fig03_gamma_forecast_residuals.png",
    }
    table_inventory = inventory_existing(
        {target: dirs["tables"] / target for target in table_targets.values()}
    )
    figure_inventory = inventory_existing(expected_figures)
    missing = pd.concat(
        [
            inventory_existing(upstream_tables).assign(kind="upstream_table"),
            figure_inventory.assign(kind="figure"),
        ],
        ignore_index=True,
    )
    missing = missing.loc[~missing["exists"]].reset_index(drop=True)
    write_csv(table_inventory, dirs["intermediate"] / "01_table_inventory.csv")
    write_csv(figure_inventory, dirs["intermediate"] / "02_figure_inventory.csv")
    write_csv(missing, dirs["intermediate"] / "03_missing_upstream_outputs.csv")
    write_manifest(
        paths,
        "04_publication_tables_figures.json",
        {
            "notebook": "04_publication_tables_figures.ipynb",
            "schema_version": cfg["schema_version"],
            "output_subfolders": output_dir_manifest_payload(dirs),
            "tables": sorted(outputs),
            "missing_upstream_outputs": int(len(missing)),
        },
    )
    return outputs

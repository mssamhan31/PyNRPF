from __future__ import annotations

import hashlib
import json
import math
import calendar
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

JOURNAL_COLORS = {
    "orange": "#eb932c",
    "dark_blue": "#22303d",
    "grey": "#2F4D67",
    "light_grey": "#5C7D99",
    "light_white": "#ebe3e3",
}
JOURNAL_BAR_COLORS = [
    JOURNAL_COLORS["dark_blue"],
    JOURNAL_COLORS["orange"],
    JOURNAL_COLORS["grey"],
    JOURNAL_COLORS["light_grey"],
]
JOURNAL_LINE_COLORS = {
    "raw": JOURNAL_COLORS["orange"],
    "m8": JOURNAL_COLORS["light_grey"],
    "reference": JOURNAL_COLORS["dark_blue"],
}
FORECAST_DISPLAY_LABELS = {
    "raw_uncorrected": "Uncorrected data",
    "m8_xgb_corrected": "m8_xgb-corrected data",
    "reference_corrected": "Manually corrected data",
    "perfect_model_baseline": "Perfect-model baseline",
    "seasonal_naive": "Seasonal naive",
    "linear_regression": "Linear regression",
    "xgboost": "XGBoost",
}


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


def remove_file_if_exists(path: Path) -> None:
    if not path.exists():
        return
    try:
        path.unlink()
    except PermissionError:
        # A stale output may be open in Excel/VS Code on Windows. Do not let
        # cleanup block regeneration of the current manifest and table set.
        return


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


def rename_final_site_ids(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    prefix_map = {
        "Alpha": ("syn_", "alpha_"),
        "Beta": ("act_", "beta_"),
    }
    if dataset_name not in prefix_map:
        return df.copy()

    old_prefix, new_prefix = prefix_map[dataset_name]
    work = df.copy()
    site_text = work["substation_id"].astype(str)
    if not site_text.str.startswith(old_prefix).all():
        bad = sorted(site_text.loc[~site_text.str.startswith(old_prefix)].unique())
        raise ValueError(
            f"{dataset_name} processed dataset has unexpected site IDs for paper rename: {bad}"
        )
    work["substation_id"] = site_text.str.replace(
        rf"^{old_prefix}", new_prefix, regex=True
    )
    return work


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

    alpha_raw = rename_final_site_ids(raw_dataset_for_write(alpha_source, "Alpha processed"), "Alpha")
    beta_raw = rename_final_site_ids(raw_dataset_for_write(beta_source, "Beta processed"), "Beta")
    alpha = prepare_dataset(alpha_raw, "Alpha")
    beta_full = prepare_dataset(beta_raw, "Beta")
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
                duration_minutes = int(len(event) * 15)
                events.append(
                    {
                        "dataset": dataset_name,
                        "substation_id": site,
                        "date": date,
                        "start_timestamp": event["_timestamp_dt"].iloc[0],
                        "end_timestamp": event["_timestamp_dt"].iloc[-1],
                        "duration_minutes": duration_minutes,
                        "duration_hours": duration_minutes / 60.0,
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


def rpf_day_of_month_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    site_days = (
        df.groupby(["substation_id", "date"], as_index=False)
        .agg(label_day=("label_interval", "any"))
        .assign(
            month=lambda x: pd.to_datetime(x["date"]).dt.month,
            day=lambda x: pd.to_datetime(x["date"]).dt.day,
        )
    )
    grouped = (
        site_days.groupby(["month", "day"], as_index=False)
        .agg(total_site_days=("label_day", "size"), rpf_site_days=("label_day", "sum"))
        .assign(dataset=dataset_name)
    )
    grouped["rpf_site_day_pct"] = (
        grouped["rpf_site_days"] / grouped["total_site_days"] * 100.0
    )
    grid = pd.MultiIndex.from_product(
        [range(1, 13), range(1, 32)], names=["month", "day"]
    ).to_frame(index=False)
    out = grid.merge(grouped, on=["month", "day"], how="left")
    out["dataset"] = out["dataset"].fillna(dataset_name)
    out["valid_calendar_day"] = out.apply(
        lambda row: int(row["day"]) <= calendar.monthrange(2024, int(row["month"]))[1],
        axis=1,
    )
    for col in ["total_site_days", "rpf_site_days"]:
        out.loc[out["valid_calendar_day"] & out[col].isna(), col] = 0
    out.loc[~out["valid_calendar_day"], ["total_site_days", "rpf_site_days"]] = np.nan
    out.loc[~out["valid_calendar_day"], "rpf_site_day_pct"] = np.nan
    return out[
        [
            "dataset",
            "month",
            "day",
            "valid_calendar_day",
            "total_site_days",
            "rpf_site_days",
            "rpf_site_day_pct",
        ]
    ]


def rpf_event_count_by_day(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "n_contiguous_events",
                "n_rpf_site_days",
                "share_pct",
                "plot_category",
            ]
        )
    event_counts = (
        events.groupby(["dataset", "substation_id", "date"], as_index=False)
        .size()
        .rename(columns={"size": "n_contiguous_events"})
    )
    distribution = (
        event_counts.groupby(["dataset", "n_contiguous_events"], as_index=False)
        .size()
        .rename(columns={"size": "n_rpf_site_days"})
    )
    totals = distribution.groupby("dataset")["n_rpf_site_days"].transform("sum")
    distribution["share_pct"] = distribution["n_rpf_site_days"] / totals * 100.0
    distribution["plot_category"] = np.where(
        distribution["n_contiguous_events"] >= 5,
        "5+",
        distribution["n_contiguous_events"].astype(int).astype(str),
    )
    return distribution[
        [
            "dataset",
            "n_contiguous_events",
            "n_rpf_site_days",
            "share_pct",
            "plot_category",
        ]
    ].sort_values(["dataset", "n_contiguous_events"]).reset_index(drop=True)


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
                "duration_hours_mean",
                "duration_hours_median",
                "duration_hours_max",
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
            duration_hours_mean=("duration_hours", "mean"),
            duration_hours_median=("duration_hours", "median"),
            duration_hours_max=("duration_hours", "max"),
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
    day_of_month = pd.concat(
        [
            rpf_day_of_month_summary(alpha, "Alpha"),
            rpf_day_of_month_summary(beta, "Beta"),
        ],
        ignore_index=True,
    )
    event_count_distribution = rpf_event_count_by_day(events)
    occurrence_dataset = dataset_occurrence_summary(occurrence)
    event_summary = event_dataset_summary(events)

    write_csv(occurrence_dataset, dirs["intermediate"] / "01_rpf_occurrence_by_dataset.csv")
    write_csv(occurrence, dirs["intermediate"] / "02_rpf_occurrence_by_site.csv")
    write_csv(temporal, dirs["intermediate"] / "03_rpf_temporal_summary.csv")
    write_csv(events, dirs["intermediate"] / "04_rpf_event_summary.csv")
    write_csv(day_of_month, dirs["intermediate"] / "05_rpf_day_of_month_summary.csv")
    write_csv(
        event_count_distribution,
        dirs["intermediate"] / "06_rpf_event_count_by_day_distribution.csv",
    )
    write_csv(
        occurrence_dataset,
        dirs["tables"] / "table01_rpf_occurrence_summary_alpha_beta.csv",
    )
    write_csv(event_summary, dirs["tables"] / "table02_rpf_event_summary_alpha_beta.csv")
    figure_paths = write_characterisation_figures(
        occurrence,
        temporal,
        events,
        day_of_month,
        event_count_distribution,
        dirs["figures"],
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
        "day_of_month": day_of_month,
        "event_count_distribution": event_count_distribution,
        "event_summary": event_summary,
        "figure_paths": figure_paths,
    }


def _load_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "axes.edgecolor": JOURNAL_COLORS["dark_blue"],
            "axes.labelcolor": JOURNAL_COLORS["dark_blue"],
            "axes.titlecolor": JOURNAL_COLORS["dark_blue"],
            "xtick.color": JOURNAL_COLORS["dark_blue"],
            "ytick.color": JOURNAL_COLORS["dark_blue"],
            "text.color": JOURNAL_COLORS["dark_blue"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
        }
    )
    import matplotlib.pyplot as plt

    return plt


def journal_colormap(name: str = "journal_heat") -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        name,
        [
            JOURNAL_COLORS["light_white"],
            JOURNAL_COLORS["light_grey"],
            JOURNAL_COLORS["orange"],
            JOURNAL_COLORS["dark_blue"],
        ],
    )


def style_axis_grid(ax: Any, axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=JOURNAL_COLORS["light_white"], linewidth=0.8, alpha=0.7)


def write_characterisation_figures(
    occurrence: pd.DataFrame,
    temporal: pd.DataFrame,
    events: pd.DataFrame,
    day_of_month: pd.DataFrame,
    event_count_distribution: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    plt = _load_matplotlib()
    figure_paths: list[Path] = []

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8.6, 4.1), sharey=True)
    for ax, dataset, colour in zip(
        axes,
        ["Alpha", "Beta"],
        [JOURNAL_COLORS["dark_blue"], JOURNAL_COLORS["orange"]],
    ):
        plot_df = (
            occurrence.loc[occurrence["dataset"] == dataset]
            .sort_values(["rpf_day_pct", "substation_id"], ascending=[False, True])
            .reset_index(drop=True)
        )
        x = np.arange(len(plot_df))
        bars = ax.bar(x, plot_df["rpf_day_pct"], width=0.86, color=colour)
        ax.bar_label(
            bars,
            labels=[f"{value:.0f}" for value in plot_df["rpf_day_pct"]],
            padding=2,
            fontsize=9,
            color=JOURNAL_COLORS["dark_blue"],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["substation_id"], rotation=45, ha="right", fontsize=10)
        ax.set_xlabel("Site", fontsize=13)
        ax.set_title(dataset, fontsize=15)
        ax.set_ylim(0, 105)
        ax.tick_params(axis="y", labelsize=11)
        ax.margins(x=0.01)
        style_axis_grid(ax)
    axes[0].set_ylabel("RPF days (%)", fontsize=13)
    fig.suptitle("RPF day percentage by site", fontsize=16)
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
        heat = (
            month_hour.pivot(index="month", columns="hour", values="rpf_interval_pct")
            .reindex(index=range(1, 13), columns=range(0, 24))
            .fillna(0)
        )
        image = ax.imshow(
            heat.to_numpy(), aspect="auto", origin="lower", cmap=journal_colormap("rpf_heat")
        )
        hour_ticks = list(range(0, 24, 4))
        ax.set_xticks(hour_ticks)
        ax.set_xticklabels([str(hour) for hour in hour_ticks], fontsize=11)
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels([str(int(month)) for month in heat.index], fontsize=11)
        ax.set_xlabel("Hour", fontsize=13)
        ax.set_title(dataset, fontsize=15)
        ax.grid(False)
    axes[0].set_ylabel("Month", fontsize=13)
    fig.suptitle("RPF interval percentage\nby month and hour", fontsize=16, y=1.08)
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), label="% intervals")
    colorbar.ax.tick_params(labelsize=11)
    colorbar.set_label("% intervals", fontsize=13)
    path = figures_dir / "fig02_month_hour_heatmap_alpha_beta.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    figure_paths.append(path)
    plt.close(fig)

    if not events.empty:
        from matplotlib.ticker import MaxNLocator

        fig, ax = plt.subplots(figsize=(8, 4))
        max_hours = max(1.0, float(events["duration_hours"].max()))
        bins = np.linspace(0, max_hours, min(7, max(1, math.ceil(max_hours))) + 1)
        for idx, (dataset, group) in enumerate(events.groupby("dataset")):
            weights = np.ones(len(group), dtype=float) / len(group) * 100.0
            ax.hist(
                group["duration_hours"],
                bins=bins,
                weights=weights,
                alpha=0.55,
                label=dataset,
                color=JOURNAL_BAR_COLORS[idx % len(JOURNAL_BAR_COLORS)],
            )
        ax.set_xlabel("Contiguous RPF event duration (hours)", fontsize=13)
        ax.set_ylabel("Share of events (%)", fontsize=13)
        ax.set_title("RPF event duration distribution", fontsize=15)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(fontsize=11)
        style_axis_grid(ax)
        fig.tight_layout()
        path = figures_dir / "fig03_event_duration_distribution_alpha_beta.png"
        fig.savefig(path, dpi=200)
        figure_paths.append(path)
        plt.close(fig)

    if not day_of_month.empty:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.8), sharey=True)
        cmap = journal_colormap("rpf_heat").copy()
        cmap.set_bad(color=JOURNAL_COLORS["light_white"])
        for ax, dataset in zip(axes, ["Alpha", "Beta"]):
            subset = day_of_month.loc[day_of_month["dataset"] == dataset].copy()
            heat = subset.pivot(index="month", columns="day", values="rpf_site_day_pct")
            heat = heat.reindex(index=range(1, 13), columns=range(1, 32))
            masked = np.ma.masked_invalid(heat.to_numpy(dtype=float))
            image = ax.imshow(masked, aspect="auto", origin="lower", cmap=cmap, vmin=0)
            day_ticks = list(range(0, 31, 5))
            ax.set_xticks(day_ticks)
            ax.set_xticklabels([str(day + 1) for day in day_ticks], fontsize=10)
            ax.set_yticks(range(12))
            ax.set_yticklabels([str(month) for month in range(1, 13)], fontsize=10)
            ax.set_xlabel("Day of month", fontsize=12)
            ax.set_title(dataset, fontsize=14)
            ax.grid(False)
        axes[0].set_ylabel("Month", fontsize=12)
        fig.suptitle("RPF site-day percentage by calendar day", fontsize=15, y=1.02)
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), label="RPF site-days (%)")
        colorbar.ax.tick_params(labelsize=10)
        colorbar.set_label("RPF site-days (%)", fontsize=12)
        path = figures_dir / "fig04_day_of_month_rpf_heatmap_alpha_beta.png"
        fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0.15)
        figure_paths.append(path)
        plt.close(fig)

    if not event_count_distribution.empty:
        category_order = ["1", "2", "3", "4", "5+"]
        colours = [
            JOURNAL_COLORS["dark_blue"],
            JOURNAL_COLORS["orange"],
            JOURNAL_COLORS["grey"],
            JOURNAL_COLORS["light_grey"],
            "#8f5f2a",
        ]
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9.2, 4.4))
        for ax, dataset in zip(axes, ["Alpha", "Beta"]):
            subset = event_count_distribution.loc[
                event_count_distribution["dataset"] == dataset
            ].copy()
            grouped = (
                subset.groupby("plot_category", as_index=False)["n_rpf_site_days"]
                .sum()
                .set_index("plot_category")
                .reindex(category_order, fill_value=0)
            )
            values = grouped["n_rpf_site_days"].to_numpy(dtype=float)
            if values.sum() == 0:
                ax.text(0.5, 0.5, "No RPF days", ha="center", va="center")
                ax.axis("off")
                continue
            wedges, _ = ax.pie(
                values,
                labels=None,
                colors=colours,
                startangle=90,
                counterclock=False,
                wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.2},
            )
            total = int(values.sum())
            ax.text(
                0,
                0,
                f"{total:,}\nRPF days",
                ha="center",
                va="center",
                fontsize=11,
                color=JOURNAL_COLORS["dark_blue"],
            )
            ax.set_title(dataset, fontsize=13)
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=colour, label=label)
            for colour, label in zip(colours, category_order)
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=len(category_order),
            frameon=False,
            fontsize=10,
            title="Contiguous RPF events per RPF day",
            title_fontsize=10,
        )
        fig.suptitle("RPF event-count structure by day", fontsize=15, y=0.98)
        fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.25, wspace=0.10)
        path = figures_dir / "fig05_rpf_events_per_day_doughnut_alpha_beta.png"
        fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0.12)
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
    interval_mask = daytime_mask(pred_df, cfg)
    day_df = (
        pred_df.groupby(["substation_id", "date"])
        .agg(label_day=("label_interval", "any"), pred_day=("pred_interval", "any"))
        .reset_index()
    )
    rows = []
    for level, values in [
        ("day", binary_metrics(day_df["label_day"], day_df["pred_day"])),
        (
            "interval",
            binary_metrics(
                pred_df.loc[interval_mask, "label_interval"],
                pred_df.loc[interval_mask, "pred_interval"],
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


CORRECTION_PREDICTION_COLUMNS = EXPECTED_COLUMNS + [
    "pred_interval",
    "corrected_net_load_MW",
]


def reusable_prediction_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("execution", {}).get("reuse_correction_prediction_files", False))


def load_reusable_correction_prediction(
    path: Path,
    eval_df: pd.DataFrame,
    require_existing: bool = False,
) -> pd.DataFrame | None:
    if not path.exists():
        if require_existing:
            raise FileNotFoundError(f"Reusable correction prediction file not found: {path}")
        return None
    candidate = pd.read_csv(path)
    missing = [col for col in CORRECTION_PREDICTION_COLUMNS if col not in candidate.columns]
    if missing:
        if require_existing:
            raise ValueError(f"Reusable prediction {path} is missing columns: {missing}")
        return None
    if len(candidate) != len(eval_df):
        if require_existing:
            raise ValueError(
                f"Reusable prediction {path} has {len(candidate)} rows; expected {len(eval_df)}."
            )
        return None
    actual_keys = candidate[["substation_id", "timestamp"]].astype("string").reset_index(drop=True)
    expected_keys = eval_df[["substation_id", "timestamp"]].astype("string").reset_index(drop=True)
    if not actual_keys.equals(expected_keys):
        if require_existing:
            raise ValueError(f"Reusable prediction {path} has mismatched site/timestamp keys.")
        return None

    out = eval_df.copy()
    out["pred_interval"] = _coerce_bool(candidate["pred_interval"]).to_numpy()
    out["corrected_net_load_MW"] = pd.to_numeric(
        candidate["corrected_net_load_MW"], errors="coerce"
    ).to_numpy()
    if "prob_interval" in candidate.columns:
        out["prob_interval"] = pd.to_numeric(candidate["prob_interval"], errors="coerce").to_numpy()
    return out


def write_correction_prediction(pred: pd.DataFrame, path: Path) -> Path:
    write_cols = list(CORRECTION_PREDICTION_COLUMNS)
    if "prob_interval" in pred.columns:
        write_cols.append("prob_interval")
    return write_csv(pred[write_cols], path)


def correction_prediction_path(
    dirs: dict[str, Path],
    index: int,
    fold_id: str,
    method: str,
) -> Path:
    return dirs["intermediate"] / f"{index:02d}_correction_predictions_{fold_id}_{method}.csv"


def get_or_make_correction_prediction(
    eval_df: pd.DataFrame,
    cfg: dict[str, Any],
    path: Path,
    make_prediction: Any,
) -> tuple[pd.DataFrame, bool]:
    if reusable_prediction_enabled(cfg):
        reused = load_reusable_correction_prediction(path, eval_df)
        if reused is not None:
            return reused, True
    pred = make_prediction()
    write_correction_prediction(pred, path)
    return pred, False


CORRECTION_PLACEHOLDER_TARGETS: dict[tuple[str, str, str], tuple[float, float]] = {
    ("Alpha", "m8_xgb", "day"): (0.94, 0.91),
    ("Alpha", "m8_xgb", "interval"): (0.89, 0.86),
    ("Alpha", "m7_dtr", "day"): (0.76, 0.70),
    ("Alpha", "m7_dtr", "interval"): (0.67, 0.62),
    ("Beta", "m8_xgb", "day"): (0.87, 0.81),
    ("Beta", "m8_xgb", "interval"): (0.79, 0.74),
    ("Beta", "m7_dtr", "day"): (0.69, 0.61),
    ("Beta", "m7_dtr", "interval"): (0.59, 0.53),
}


def placeholder_binary_metrics(
    support: int,
    positive_support: int,
    target_precision: float,
    target_recall: float,
) -> dict[str, Any]:
    support = int(support)
    positive_support = int(positive_support)
    negative_support = max(0, support - positive_support)
    if support <= 0 or positive_support <= 0:
        return {
            "support": support,
            "positive_support": positive_support,
            "tp": 0,
            "fp": 0,
            "fn": positive_support,
            "tn": negative_support,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    target_precision = float(np.clip(target_precision, 0.01, 0.99))
    target_recall = float(np.clip(target_recall, 0.01, 0.99))
    tp = int(round(positive_support * target_recall))
    tp = min(max(tp, 0), positive_support)
    fn = positive_support - tp
    predicted_positive = int(round(tp / target_precision)) if tp else 0
    fp = min(max(predicted_positive - tp, 0), negative_support)
    tn = negative_support - fp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "support": support,
        "positive_support": positive_support,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def correction_metric_supports(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, tuple[int, int]]:
    day_df = (
        df.groupby(["substation_id", "date"])["label_interval"]
        .any()
        .reset_index(name="label_day")
    )
    interval_df = df.loc[daytime_mask(df, cfg)]
    return {
        "day": (int(len(day_df)), int(day_df["label_day"].sum())),
        "interval": (
            int(len(interval_df)),
            int(interval_df["label_interval"].sum()),
        ),
    }


def correction_placeholder_target(
    dataset: str,
    method: str,
    level: str,
    fold_index: int = 0,
) -> tuple[float, float]:
    precision, recall = CORRECTION_PLACEHOLDER_TARGETS.get(
        (dataset, method, level), (0.70, 0.65)
    )
    if dataset == "Alpha":
        offset = 0.015 * fold_index
        precision -= offset
        recall -= offset
    return float(np.clip(precision, 0.01, 0.99)), float(np.clip(recall, 0.01, 0.99))


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
        result["pred_interval"] = (
            merged["pred_interval"].fillna(False).infer_objects(copy=False).astype(bool).to_numpy()
        )
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
            {"experiment": "alpha_complete_loso", "fold_id": f"holdout_{site}", "status": "planned"}
        )
    rows.append(
        {"experiment": "beta_transfer", "fold_id": "alpha_train_to_beta", "status": "planned"}
    )
    rows.append({"experiment": "beta_transfer", "fold_id": "m7_dtr_beta", "status": "planned"})
    rows.append({"experiment": "beta_rows", "fold_id": "beta_filtered", "status": str(len(beta))})
    return pd.DataFrame(rows)


def correction_smoke_metrics(
    alpha: pd.DataFrame, beta: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    methods = cfg["correction"]["methods"]
    for fold_index, holdout_site in enumerate(alpha_loso_sites(alpha, cfg)):
        eval_df = filter_date_window(
            alpha.loc[alpha["substation_id"] == holdout_site].copy(),
            cfg["windows"]["test_start"],
            cfg["windows"]["test_end"],
        )
        supports = correction_metric_supports(eval_df, cfg)
        for method in methods:
            for level, (support, positive_support) in supports.items():
                target_precision, target_recall = correction_placeholder_target(
                    "Alpha", method, level, fold_index
                )
                rows.append(
                    {
                        "dataset": "Alpha",
                        "fold_id": f"alpha_holdout_{holdout_site}",
                        "method": method,
                        "level": level,
                        **placeholder_binary_metrics(
                            support,
                            positive_support,
                            target_precision,
                            target_recall,
                        ),
                        "is_placeholder": True,
                        "status": "placeholder_smoke_only",
                    }
                )
    supports = correction_metric_supports(beta, cfg)
    for method in methods:
        for level, (support, positive_support) in supports.items():
            target_precision, target_recall = correction_placeholder_target(
                "Beta", method, level
            )
            rows.append(
                {
                    "dataset": "Beta",
                    "fold_id": "beta_transfer",
                    "method": method,
                    "level": level,
                    **placeholder_binary_metrics(
                        support,
                        positive_support,
                        target_precision,
                        target_recall,
                    ),
                    "is_placeholder": True,
                    "status": "placeholder_smoke_only",
                }
            )
    return pd.DataFrame(rows)


COUNT_COLUMNS = ["support", "positive_support", "tp", "fp", "fn", "tn"]
SCORE_COLUMNS = ["precision", "recall", "f1"]
CORRECTION_METHOD_ORDER = ["m8_xgb", "m7_dtr"]
CORRECTION_LEVEL_ORDER = ["day", "interval"]


def metric_scores_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def beta_top_rpf_sites(beta: pd.DataFrame, top_n: int = 3) -> list[str]:
    return beta_rpf_site_order(beta)[:top_n]


def beta_rpf_site_order(beta: pd.DataFrame) -> list[str]:
    rankings = site_rpf_summary(beta, "Beta").sort_values(
        ["rpf_days", "rpf_intervals", "substation_id"],
        ascending=[False, False, True],
    )
    return rankings["substation_id"].tolist()


def correction_smoke_beta_site_metrics(
    beta: pd.DataFrame,
    cfg: dict[str, Any],
    sites: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    methods = cfg["correction"]["methods"]
    for site in (sites or beta_rpf_site_order(beta)):
        site_df = beta.loc[beta["substation_id"] == site].copy()
        supports = correction_metric_supports(site_df, cfg)
        for method in methods:
            for level, (support, positive_support) in supports.items():
                target_precision, target_recall = correction_placeholder_target(
                    "Beta", method, level
                )
                rows.append(
                    {
                        "dataset": "Beta",
                        "substation_id": site,
                        "fold_id": f"beta_site_{site}",
                        "method": method,
                        "level": level,
                        **placeholder_binary_metrics(
                            support,
                            positive_support,
                            target_precision,
                            target_recall,
                        ),
                        "is_placeholder": True,
                        "status": "placeholder_smoke_only",
                    }
                )
    return pd.DataFrame(rows)


def correction_beta_site_metrics_from_predictions(
    beta: pd.DataFrame,
    cfg: dict[str, Any],
    predictions_by_method: dict[str, pd.DataFrame],
    sites: list[str] | None = None,
) -> pd.DataFrame:
    frames = []
    for site in (sites or beta_rpf_site_order(beta)):
        for method in cfg["correction"]["methods"]:
            if method not in predictions_by_method:
                continue
            pred = predictions_by_method[method]
            site_pred = pred.loc[pred["substation_id"] == site].copy()
            site_metrics = evaluate_prediction_frame(
                site_pred, cfg, "Beta", f"beta_site_{site}", method
            )
            site_metrics.insert(1, "substation_id", site)
            frames.append(site_metrics)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["is_placeholder"] = False
    out["status"] = "complete"
    return out


def correction_alpha_site_metrics_from_loso_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    alpha_metrics = metrics.loc[metrics["dataset"] == "Alpha"].copy()
    if alpha_metrics.empty:
        return pd.DataFrame()
    alpha_metrics["substation_id"] = (
        alpha_metrics["fold_id"].astype(str).str.replace("alpha_holdout_", "", regex=False)
    )
    return alpha_metrics[
        [
            "dataset",
            "substation_id",
            "fold_id",
            "method",
            "level",
            *COUNT_COLUMNS,
            *SCORE_COLUMNS,
            "is_placeholder",
            "status",
        ]
    ].reset_index(drop=True)


def correction_pooled_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    usable = metrics.copy()
    for col in COUNT_COLUMNS:
        usable[col] = pd.to_numeric(usable[col], errors="coerce").fillna(0).astype(int)
    usable["summary_group"] = usable["dataset"].map(
        {"Alpha": "Alpha CV pooled", "Beta": "Beta overall"}
    ).fillna(usable["dataset"].astype(str))
    grouped = (
        usable.groupby(["summary_group", "dataset", "method", "level"], as_index=False)[
            COUNT_COLUMNS
        ]
        .sum()
        .sort_values(["summary_group", "method", "level"])
    )
    score_rows = []
    for _, row in grouped.iterrows():
        scores = metric_scores_from_counts(int(row["tp"]), int(row["fp"]), int(row["fn"]))
        score_rows.append({**row.to_dict(), **scores})
    return pd.DataFrame(score_rows)


def correction_metrics_table(
    metrics: pd.DataFrame,
    beta_site_metrics: pd.DataFrame,
) -> pd.DataFrame:
    base = metrics.copy()
    base["summary_scope"] = np.where(
        base["dataset"].eq("Alpha"), "alpha_loso_fold", "beta_overall"
    )
    base["substation_id"] = ""
    alpha_mask = base["summary_scope"].eq("alpha_loso_fold")
    base.loc[alpha_mask, "substation_id"] = (
        base.loc[alpha_mask, "fold_id"].astype(str).str.replace("alpha_holdout_", "", regex=False)
    )

    if beta_site_metrics.empty:
        site_rows = pd.DataFrame(columns=base.columns)
    else:
        site_rows = beta_site_metrics.copy()
        site_rows["summary_scope"] = "beta_site"

    combined = pd.concat([base, site_rows], ignore_index=True, sort=False)
    combined["_scope_order"] = combined["summary_scope"].map(
        {"alpha_loso_fold": 0, "beta_overall": 1, "beta_site": 2}
    ).fillna(99)
    combined["_row_order"] = np.arange(len(combined))
    ordered_cols = [
        "summary_scope",
        "dataset",
        "substation_id",
        "fold_id",
        "method",
        "level",
        *COUNT_COLUMNS,
        *SCORE_COLUMNS,
        "is_placeholder",
        "status",
    ]
    for col in ordered_cols:
        if col not in combined.columns:
            combined[col] = ""
    return (
        combined.sort_values(["_scope_order", "_row_order"])
        .reset_index(drop=True)[ordered_cols]
    )


def correction_confusion_matrices(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = ["dataset", "fold_id", "method", "level", "tp", "fp", "fn", "tn"]
    return metrics[[col for col in cols if col in metrics.columns]].copy()


def _format_plot_count(value: float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M".replace(".0M", "M")
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k".replace(".0k", "k")
    return f"{int(round(value))}"


def _plot_confusion_panel(ax: Any, fig: Any, row: pd.Series, title: str) -> None:
    from matplotlib.ticker import FuncFormatter

    cm = np.array(
        [[row["tp"], row["fn"]], [row["fp"], row["tn"]]],
        dtype=float,
    )
    image = ax.imshow(cm, interpolation="nearest", cmap=journal_colormap("correction_confusion"))
    colorbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.025)
    colorbar.set_label("Count", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.formatter = FuncFormatter(lambda x, pos: _format_plot_count(x))
    colorbar.update_ticks()

    vmax = float(cm.max()) if cm.size else 0.0
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            colour = "white" if vmax and value > vmax * 0.55 else JOURNAL_COLORS["dark_blue"]
            ax.text(
                j,
                i,
                _format_plot_count(value),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=colour,
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["RPF", "No RPF"], fontsize=8)
    ax.set_yticklabels(["RPF", "No RPF"], fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_title(
        f"{title}\nP={row['precision']:.3f}  R={row['recall']:.3f}  F1={row['f1']:.3f}",
        fontsize=9,
        pad=6,
    )
    ax.grid(False)


def write_site_score_boxplot(
    site_metrics: pd.DataFrame,
    figures_dir: Path,
    dataset_label: str,
    output_name: str,
) -> Path | None:
    if site_metrics.empty:
        return None
    plot_df = site_metrics.dropna(subset=SCORE_COLUMNS, how="all").copy()
    if plot_df.empty:
        return None
    plt = _load_matplotlib()
    figures_dir.mkdir(parents=True, exist_ok=True)
    metric_labels = {"precision": "Precision", "recall": "Recall", "f1": "F1"}
    method_labels = {"m8_xgb": "m8_xgb", "m7_dtr": "m7_dtr"}
    method_colors = {
        "m8_xgb": JOURNAL_COLORS["dark_blue"],
        "m7_dtr": JOURNAL_COLORS["orange"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), sharey=True)
    for ax, level, title in zip(axes, CORRECTION_LEVEL_ORDER, ["Day level", "Interval level"]):
        level_df = plot_df.loc[plot_df["level"] == level].copy()
        centers = np.arange(len(SCORE_COLUMNS))
        offsets = {"m8_xgb": -0.18, "m7_dtr": 0.18}
        for method in CORRECTION_METHOD_ORDER:
            method_df = level_df.loc[level_df["method"] == method].copy()
            if method_df.empty:
                continue
            positions = centers + offsets.get(method, 0.0)
            data = [
                pd.to_numeric(method_df[metric], errors="coerce").dropna().to_numpy()
                for metric in SCORE_COLUMNS
            ]
            box = ax.boxplot(
                data,
                positions=positions,
                widths=0.28,
                patch_artist=True,
                showfliers=False,
            )
            for patch in box["boxes"]:
                patch.set_facecolor(method_colors.get(method, JOURNAL_COLORS["grey"]))
                patch.set_alpha(0.70)
                patch.set_edgecolor(JOURNAL_COLORS["dark_blue"])
            for element in ["whiskers", "caps", "medians"]:
                for item in box[element]:
                    item.set_color(JOURNAL_COLORS["dark_blue"])
            for metric_idx, metric in enumerate(SCORE_COLUMNS):
                values = pd.to_numeric(method_df[metric], errors="coerce").dropna().to_numpy()
                if len(values) == 0:
                    continue
                jitter = np.linspace(-0.045, 0.045, len(values))
                ax.scatter(
                    np.full(len(values), positions[metric_idx]) + jitter,
                    values,
                    s=16,
                    color=method_colors.get(method, JOURNAL_COLORS["grey"]),
                    edgecolors="white",
                    linewidths=0.35,
                    zorder=3,
                )
        ax.set_title(title, fontsize=12)
        ax.set_xticks(centers)
        ax.set_xticklabels([metric_labels[metric] for metric in SCORE_COLUMNS], fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="y", labelsize=9)
        style_axis_grid(ax)
    axes[0].set_ylabel(f"Score across {dataset_label} sites", fontsize=10)
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=method_colors[method], label=method_labels[method], alpha=0.70)
        for method in CORRECTION_METHOD_ORDER
        if method in set(plot_df["method"])
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"{dataset_label} site-level correction score distribution", fontsize=13, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.22, wspace=0.10)
    path = figures_dir / output_name
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path


def write_beta_site_score_boxplot(beta_site_metrics: pd.DataFrame, figures_dir: Path) -> Path | None:
    return write_site_score_boxplot(
        beta_site_metrics,
        figures_dir,
        "Beta",
        "fig03_beta_site_precision_recall_f1_boxplot.png",
    )


def write_alpha_site_score_boxplot(alpha_site_metrics: pd.DataFrame, figures_dir: Path) -> Path | None:
    return write_site_score_boxplot(
        alpha_site_metrics,
        figures_dir,
        "Alpha",
        "fig04_alpha_site_precision_recall_f1_boxplot.png",
    )


def write_correction_figures(
    metrics: pd.DataFrame,
    figures_dir: Path,
    alpha_site_metrics: pd.DataFrame | None = None,
    beta_site_metrics: pd.DataFrame | None = None,
) -> list[Path]:
    usable = metrics.dropna(subset=["precision", "recall", "f1"], how="all").copy()
    if usable.empty:
        return []
    plt = _load_matplotlib()
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    for stale_name in [
        "fig01_correction_confusion_matrices.png",
        "fig02_correction_precision_recall_f1.png",
    ]:
        stale_path = figures_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    pooled = correction_pooled_metrics(usable)
    summary_groups = ["Alpha CV pooled", "Beta overall"]
    methods = [method for method in CORRECTION_METHOD_ORDER if method in set(pooled["method"])]

    for level, suffix, title_level in [
        ("day", "day", "Day-level"),
        ("interval", "interval", "Interval-level"),
    ]:
        plot_df = pooled.loc[pooled["level"] == level].copy()
        if plot_df.empty:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.25))
        for row_idx, group in enumerate(summary_groups):
            for col_idx, method in enumerate(methods[:2]):
                ax = axes[row_idx, col_idx]
                match = plot_df.loc[
                    (plot_df["summary_group"] == group) & (plot_df["method"] == method)
                ]
                if match.empty:
                    ax.axis("off")
                    continue
                _plot_confusion_panel(ax, fig, match.iloc[0], f"{group} - {method}")
        fig.suptitle(f"{title_level} correction confusion matrices", fontsize=12, y=0.98)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.08, wspace=0.40, hspace=0.62)
        path = figures_dir / f"fig01{'a' if level == 'day' else 'b'}_confusion_matrices_{suffix}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
        figure_paths.append(path)
        plt.close(fig)

    metric_labels = {"precision": "Precision", "recall": "Recall", "f1": "F1"}
    for level, suffix, title_level in [
        ("day", "day", "Day-level"),
        ("interval", "interval", "Interval-level"),
    ]:
        plot_df = pooled.loc[pooled["level"] == level].copy()
        if plot_df.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
        for ax, group in zip(axes, summary_groups):
            group_df = plot_df.loc[plot_df["summary_group"] == group].copy()
            group_df = group_df.set_index("method").reindex(methods).reset_index()
            x = np.arange(len(methods))
            width = 0.24
            for idx, metric in enumerate(SCORE_COLUMNS):
                values = pd.to_numeric(group_df[metric], errors="coerce").fillna(0).to_numpy()
                bars = ax.bar(
                    x + (idx - 1) * width,
                    values,
                    width,
                    label=metric_labels[metric],
                    color=JOURNAL_BAR_COLORS[idx],
                )
                ax.bar_label(
                    bars,
                    labels=[f"{value:.2f}" for value in values],
                    padding=2,
                    fontsize=7,
                )
            ax.set_title(group, fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, fontsize=8)
            ax.set_ylim(0, 1.08)
            style_axis_grid(ax)
            ax.tick_params(axis="y", labelsize=8)
        axes[0].set_ylabel("Score", fontsize=9)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, 0.02),
        )
        fig.suptitle(f"{title_level} correction scores", fontsize=12, y=0.97)
        fig.subplots_adjust(left=0.08, right=0.99, top=0.78, bottom=0.28, wspace=0.08)
        path = (
            figures_dir
            / f"fig02{'a' if level == 'day' else 'b'}_precision_recall_f1_{suffix}.png"
        )
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
        figure_paths.append(path)
        plt.close(fig)
    if beta_site_metrics is not None:
        beta_site_path = write_beta_site_score_boxplot(beta_site_metrics, figures_dir)
        if beta_site_path is not None:
            figure_paths.append(beta_site_path)
    if alpha_site_metrics is not None:
        alpha_site_path = write_alpha_site_score_boxplot(alpha_site_metrics, figures_dir)
        if alpha_site_path is not None:
            figure_paths.append(alpha_site_path)
    return figure_paths


def correction_validation_preflight(article_root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = find_article_root(article_root)
    cfg = load_config(root)
    paths = article_paths(root, cfg)
    alpha = load_dataset(root, cfg, "alpha")
    beta = load_dataset(root, cfg, "beta")
    alpha_sites = alpha_loso_sites(alpha, cfg)
    beta_rankings = site_rpf_summary(beta, "Beta").sort_values(
        ["rpf_days", "rpf_intervals", "substation_id"],
        ascending=[False, False, True],
    )
    dependency_rows = []
    for package in ["pandas", "numpy", "sklearn", "xgboost", "pyarrow", "matplotlib"]:
        try:
            module = __import__(package)
            dependency_rows.append(
                {
                    "package": package,
                    "available": True,
                    "version": getattr(module, "__version__", "installed"),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive notebook report
            dependency_rows.append(
                {"package": package, "available": False, "version": type(exc).__name__}
            )
    readiness = pd.DataFrame(
        [
            {
                "check": "run_full_correction_validation",
                "value": bool(cfg["execution"]["run_full_correction_validation"]),
            },
            {
                "check": "reuse_correction_prediction_files",
                "value": reusable_prediction_enabled(cfg),
            },
            {"check": "alpha_rows", "value": int(len(alpha))},
            {"check": "alpha_sites", "value": int(alpha["substation_id"].nunique())},
            {"check": "alpha_loso_sites", "value": ", ".join(alpha_sites)},
            {"check": "beta_rows", "value": int(len(beta))},
            {"check": "beta_sites", "value": int(beta["substation_id"].nunique())},
            {
                "check": "beta_date_window",
                "value": f"{beta['date'].min()} to {beta['date'].max()}",
            },
        ]
    )
    workload = pd.DataFrame(
        [
            {"step": "Alpha LOSO m8_xgb training", "count": len(alpha_sites)},
            {"step": "Alpha LOSO m8_xgb prediction", "count": len(alpha_sites)},
            {"step": "Alpha LOSO m7_dtr batched inference", "count": 1},
            {"step": "Alpha-to-Beta m8_xgb training", "count": 1},
            {"step": "Beta m8_xgb prediction", "count": 1},
            {"step": "Beta m7_dtr inference", "count": 1},
            {"step": "Primary metric rows", "count": len(alpha_sites) * 4 + 4},
            {"step": "Alpha site metric rows", "count": len(alpha_sites) * 4},
            {"step": "Beta site metric rows", "count": int(beta["substation_id"].nunique()) * 4},
        ]
    )
    expected_outputs = pd.DataFrame(
        {
            "artifact": [
                "01_correction_validation_plan.csv",
                "01_correction_metrics.csv",
                "02_correction_confusion_matrices.csv",
                "table01_correction_metrics_summary.csv",
                "table02_beta_transfer_key_metrics.csv",
                "fig01a_confusion_matrices_day.png",
                "fig01b_confusion_matrices_interval.png",
                "fig02a_precision_recall_f1_day.png",
                "fig02b_precision_recall_f1_interval.png",
                "fig03_beta_site_precision_recall_f1_boxplot.png",
                "fig04_alpha_site_precision_recall_f1_boxplot.png",
            ]
        }
    )
    return {
        "readiness": readiness,
        "dependencies": pd.DataFrame(dependency_rows),
        "workload": workload,
        "beta_rankings": beta_rankings,
        "expected_outputs": expected_outputs,
    }


def run_correction_validation(article_root: Path | None = None) -> dict[str, Any]:
    root = find_article_root(article_root)
    cfg = load_config(root)
    paths = article_paths(root, cfg)
    ensure_output_dirs(paths)
    dirs = notebook_output_dirs(paths, "02_correction_validation")
    alpha = load_dataset(root, cfg, "alpha")
    beta = load_dataset(root, cfg, "beta")
    beta_sites = beta_rpf_site_order(beta)
    stale_beta_top3_table = dirs["tables"] / "table03_beta_top3_site_metrics.csv"
    remove_file_if_exists(stale_beta_top3_table)

    if not bool(cfg["execution"]["run_full_correction_validation"]):
        plan = correction_smoke_plan(alpha, beta, cfg)
        metrics = correction_smoke_metrics(alpha, beta, cfg)
        alpha_site_metrics = correction_alpha_site_metrics_from_loso_metrics(metrics)
        beta_site_metrics = correction_smoke_beta_site_metrics(beta, cfg, sites=beta_sites)
        table_metrics = correction_metrics_table(metrics, beta_site_metrics)
        confusion = correction_confusion_matrices(metrics)
        write_csv(plan, dirs["intermediate"] / "01_correction_validation_plan.csv")
        write_csv(metrics, dirs["metrics"] / "01_correction_metrics.csv")
        write_csv(confusion, dirs["metrics"] / "02_correction_confusion_matrices.csv")
        write_csv(table_metrics, dirs["tables"] / "table01_correction_metrics_summary.csv")
        write_csv(
            table_metrics.loc[table_metrics["summary_scope"] == "beta_overall"].copy(),
            dirs["tables"] / "table02_beta_transfer_key_metrics.csv",
        )
        figure_paths = write_correction_figures(
            metrics,
            dirs["figures"],
            alpha_site_metrics=alpha_site_metrics,
            beta_site_metrics=beta_site_metrics,
        )
        write_manifest(
            paths,
            "02_correction_validation.json",
            {
                "notebook": "02_correction_validation.ipynb",
                "schema_version": cfg["schema_version"],
                "status": "placeholder_smoke_only",
                "publication_ready": False,
                "contains_placeholder_metrics": True,
                "output_subfolders": output_dir_manifest_payload(dirs),
                "row_counts": {"alpha": int(len(alpha)), "beta": int(len(beta))},
                "tables": [
                    "table01_correction_metrics_summary.csv",
                    "table02_beta_transfer_key_metrics.csv",
                ],
                "figures": [path.name for path in figure_paths],
            },
        )
        return {
            "status": "placeholder_smoke_only",
            "plan": plan,
            "metrics": metrics,
            "alpha_site_metrics": alpha_site_metrics,
            "beta_site_metrics": beta_site_metrics,
            "table_metrics": table_metrics,
            "figure_paths": figure_paths,
        }

    metric_frames = []
    alpha_sites = alpha_loso_sites(alpha, cfg)
    alpha_eval_frames = []
    for holdout_site in alpha_sites:
        eval_df = filter_date_window(
            alpha.loc[alpha["substation_id"] == holdout_site].copy(),
            cfg["windows"]["test_start"],
            cfg["windows"]["test_end"],
        )
        eval_df["_fold_id"] = f"alpha_holdout_{holdout_site}"
        alpha_eval_frames.append(eval_df)

    alpha_m7_by_fold: dict[str, pd.DataFrame] = {}
    alpha_m7_missing_frames = []
    for fold_index, eval_df_with_fold in enumerate(alpha_eval_frames):
        fold_id = str(eval_df_with_fold["_fold_id"].iloc[0])
        pred_path = correction_prediction_path(dirs, 3 + fold_index * 2, fold_id, "m7_dtr")
        reusable = (
            load_reusable_correction_prediction(
                pred_path,
                eval_df_with_fold.drop(columns=["_fold_id"]).reset_index(drop=True),
            )
            if reusable_prediction_enabled(cfg)
            else None
        )
        if reusable is not None:
            alpha_m7_by_fold[fold_id] = reusable
        else:
            alpha_m7_missing_frames.append(eval_df_with_fold)
    if alpha_m7_missing_frames:
        alpha_m7_all = predict_m7(pd.concat(alpha_m7_missing_frames, ignore_index=True), cfg, root)
        for fold_id, frame in alpha_m7_all.groupby("_fold_id", sort=False):
            alpha_m7_by_fold[str(fold_id)] = frame.drop(columns=["_fold_id"]).reset_index(drop=True)

    pred_index = 2
    for holdout_site, eval_df_with_fold in zip(alpha_sites, alpha_eval_frames):
        train_source = alpha.loc[alpha["substation_id"] != holdout_site].copy()
        fold_id = f"alpha_holdout_{holdout_site}"
        eval_df = eval_df_with_fold.drop(columns=["_fold_id"]).reset_index(drop=True)
        bundle = train_m8_bundle(train_source, cfg, root)
        for method, make_prediction in [
            (
                "m8_xgb",
                lambda eval_df=eval_df, bundle=bundle: predict_m8_bundle(
                    eval_df, bundle, cfg, root
                ),
            ),
            ("m7_dtr", lambda fold_id=fold_id: alpha_m7_by_fold[fold_id]),
        ]:
            pred_path = correction_prediction_path(dirs, pred_index, fold_id, method)
            pred, _ = get_or_make_correction_prediction(
                eval_df, cfg, pred_path, make_prediction
            )
            pred_index += 1
            metric_frames.append(evaluate_prediction_frame(pred, cfg, "Alpha", fold_id, method))

    alpha_train = alpha.copy()
    beta_bundle = train_m8_bundle(alpha_train, cfg, root)
    beta_predictions_by_method: dict[str, pd.DataFrame] = {}
    for method, make_prediction in [
        ("m8_xgb", lambda: predict_m8_bundle(beta, beta_bundle, cfg, root)),
        ("m7_dtr", lambda: predict_m7(beta, cfg, root)),
    ]:
        fold_id = "beta_transfer"
        pred_path = correction_prediction_path(dirs, pred_index, fold_id, method)
        pred, _ = get_or_make_correction_prediction(beta, cfg, pred_path, make_prediction)
        beta_predictions_by_method[method] = pred
        pred_index += 1
        metric_frames.append(evaluate_prediction_frame(pred, cfg, "Beta", fold_id, method))

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics["is_placeholder"] = False
    metrics["status"] = "complete"
    alpha_site_metrics = correction_alpha_site_metrics_from_loso_metrics(metrics)
    beta_site_metrics = correction_beta_site_metrics_from_predictions(
        beta, cfg, beta_predictions_by_method, sites=beta_sites
    )
    table_metrics = correction_metrics_table(metrics, beta_site_metrics)
    plan = correction_smoke_plan(alpha, beta, cfg)
    confusion = correction_confusion_matrices(metrics)
    write_csv(plan, dirs["intermediate"] / "01_correction_validation_plan.csv")
    write_csv(metrics, dirs["metrics"] / "01_correction_metrics.csv")
    write_csv(confusion, dirs["metrics"] / "02_correction_confusion_matrices.csv")
    write_csv(table_metrics, dirs["tables"] / "table01_correction_metrics_summary.csv")
    write_csv(
        table_metrics.loc[table_metrics["summary_scope"] == "beta_overall"].copy(),
        dirs["tables"] / "table02_beta_transfer_key_metrics.csv",
    )
    figure_paths = write_correction_figures(
        metrics,
        dirs["figures"],
        alpha_site_metrics=alpha_site_metrics,
        beta_site_metrics=beta_site_metrics,
    )
    write_manifest(
        paths,
        "02_correction_validation.json",
        {
            "notebook": "02_correction_validation.ipynb",
            "schema_version": cfg["schema_version"],
            "status": "complete",
            "publication_ready": True,
            "contains_placeholder_metrics": False,
            "output_subfolders": output_dir_manifest_payload(dirs),
            "row_counts": {"alpha": int(len(alpha)), "beta": int(len(beta))},
            "tables": [
                "table01_correction_metrics_summary.csv",
                "table02_beta_transfer_key_metrics.csv",
            ],
            "figures": [path.name for path in figure_paths],
        },
    )
    return {
        "status": "complete",
        "metrics": metrics,
        "alpha_site_metrics": alpha_site_metrics,
        "beta_site_metrics": beta_site_metrics,
        "table_metrics": table_metrics,
        "figure_paths": figure_paths,
    }


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
        is_placeholder = (
            bool(group["is_placeholder"].fillna(False).astype(bool).any())
            if "is_placeholder" in group.columns
            else False
        )
        rows.append(
            {
                "data_condition": condition,
                "data_condition_label": FORECAST_DISPLAY_LABELS.get(condition, condition),
                "model": model,
                "model_label": FORECAST_DISPLAY_LABELS.get(model, model),
                "n_targets": int(len(group)),
                "rmse_MW": rmse(group["y_reference"], group["y_pred"]),
                "mae_MW": mae(group["y_reference"], group["y_pred"]),
                "is_placeholder": is_placeholder,
                "status": "placeholder_smoke_only" if is_placeholder else "complete",
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


GAMMA_DATA_CONDITION_COLUMNS = {
    "raw_uncorrected": "raw_uncorrected_MW",
    "m8_xgb_corrected": "m8_xgb_corrected_MW",
    "reference_corrected": "reference_corrected_MW",
}


def perfect_model_baseline(gamma: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    start = cfg["windows"]["gamma_forecast_test_start"]
    end = cfg["windows"]["gamma_forecast_test_end"]
    frames = []
    forecast_cols = ["target_timestamp", "data_condition", "model", "y_reference", "y_pred"]
    for condition, column in GAMMA_DATA_CONDITION_COLUMNS.items():
        examples = build_forecast_examples(gamma, column, cfg, start, end)
        out = examples[["target_timestamp", "y_reference", "y_condition"]].copy()
        out["data_condition"] = condition
        out["model"] = "perfect_model_baseline"
        out["y_pred"] = out["y_condition"]
        frames.append(out[forecast_cols])
    baseline = pd.concat(frames, ignore_index=True)
    baseline["is_placeholder"] = False
    baseline["status"] = "complete"
    return baseline[forecast_cols + ["is_placeholder", "status"]]


def placeholder_m8_corrected_series(gamma: pd.DataFrame) -> pd.Series:
    raw = gamma["net_load_MW"].astype(float)
    reference = gamma["reference_net_load_MW"].astype(float)
    flags = gamma["label_interval"].astype(bool)
    corrected = np.where(flags, reference * 0.9 + raw * 0.1, raw)
    return pd.Series(corrected, index=gamma.index, name="m8_xgb_corrected_MW")


def placeholder_forecast_predictions(
    examples: pd.DataFrame,
    condition: str,
    model: str,
) -> pd.Series:
    seasonal = examples["origin_value"].astype(float)
    reference = examples["y_reference"].astype(float)
    residual = seasonal - reference
    phase = np.arange(len(examples), dtype=float)
    wiggle = 0.015 * reference.abs().median() * np.sin(phase / 17.0)
    factors = {
        ("raw_uncorrected", "linear_regression"): 1.08,
        ("raw_uncorrected", "xgboost"): 1.03,
        ("m8_xgb_corrected", "linear_regression"): 0.55,
        ("m8_xgb_corrected", "xgboost"): 0.42,
        ("reference_corrected", "linear_regression"): 0.35,
        ("reference_corrected", "xgboost"): 0.24,
    }
    factor = factors.get((condition, model), 1.0)
    return reference + residual * factor + wiggle


def enforce_placeholder_forecast_baseline_floor(
    forecasts: pd.DataFrame, baseline: pd.DataFrame
) -> pd.DataFrame:
    out = forecasts.copy()
    baseline_metrics = forecast_metric_rows(baseline).set_index("data_condition")
    model_floor_multipliers = {
        "xgboost": 1.04,
        "linear_regression": 1.08,
        "seasonal_naive": 1.12,
    }
    for condition, baseline_row in baseline_metrics.iterrows():
        baseline_rmse = float(baseline_row["rmse_MW"])
        for model, multiplier in model_floor_multipliers.items():
            mask = (out["data_condition"] == condition) & (out["model"] == model)
            if not mask.any():
                continue
            residual = (out.loc[mask, "y_pred"] - out.loc[mask, "y_reference"]).to_numpy(dtype=float)
            finite = np.isfinite(residual)
            if not finite.any():
                continue
            current_rmse = float(np.sqrt(np.mean(residual[finite] ** 2)))
            target_rmse = max(baseline_rmse * multiplier, 1e-6)
            if current_rmse > target_rmse:
                continue
            if current_rmse == 0.0:
                phase = np.linspace(0.0, 2.0 * np.pi, int(finite.sum()), endpoint=False)
                residual[finite] = target_rmse * np.sin(phase)
                current_rmse = float(np.sqrt(np.mean(residual[finite] ** 2)))
            scale = target_rmse / current_rmse
            residual[finite] = residual[finite] * scale
            out.loc[mask, "y_pred"] = out.loc[mask, "y_reference"].to_numpy(dtype=float) + residual
    return out


def placeholder_forecast_rows(gamma: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    test_start = cfg["windows"]["gamma_forecast_test_start"]
    test_end = cfg["windows"]["gamma_forecast_test_end"]
    forecast_frames = []
    for condition, column in GAMMA_DATA_CONDITION_COLUMNS.items():
        examples = build_forecast_examples(gamma, column, cfg, test_start, test_end)
        forecast_cols = ["target_timestamp", "data_condition", "model", "y_reference", "y_pred"]
        seasonal = examples[["target_timestamp", "y_reference", "origin_value"]].copy()
        seasonal["data_condition"] = condition
        seasonal["model"] = "seasonal_naive"
        seasonal["y_pred"] = seasonal["origin_value"]
        forecast_frames.append(seasonal[forecast_cols])
        for model in ["linear_regression", "xgboost"]:
            pred = examples[["target_timestamp", "y_reference"]].copy()
            pred["data_condition"] = condition
            pred["model"] = model
            pred["y_pred"] = placeholder_forecast_predictions(examples, condition, model)
            forecast_frames.append(pred[forecast_cols])
    forecasts = pd.concat(forecast_frames, ignore_index=True)
    forecasts["is_placeholder"] = True
    forecasts["status"] = "placeholder_smoke_only"
    return enforce_placeholder_forecast_baseline_floor(
        forecasts, perfect_model_baseline(gamma, cfg)
    )


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
    run_full_forecast = bool(cfg["execution"]["run_full_forecast"])
    for stale_path in [
        dirs["metrics"] / "01_gamma_data_error_benchmark.csv",
        dirs["tables"] / "table02_gamma_data_error_benchmark.csv",
        dirs["figures"] / "fig02a_gamma_perfect_model_baseline_rmse.png",
        dirs["figures"] / "fig02b_gamma_forecast_rmse.png",
    ]:
        remove_file_if_exists(stale_path)

    if not run_full_forecast:
        gamma["m8_xgb_corrected_MW"] = placeholder_m8_corrected_series(gamma)
        alpha = None
    else:
        from sklearn.linear_model import LinearRegression
        from xgboost import XGBRegressor

        alpha = load_dataset(root, cfg, "alpha")
        m8_bundle = train_m8_bundle(alpha, cfg, root)
        gamma_pred = predict_m8_bundle(gamma, m8_bundle, cfg, root)
        gamma["m8_xgb_corrected_MW"] = gamma_pred["corrected_net_load_MW"].to_numpy()

    baseline = perfect_model_baseline(gamma, cfg)
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
    write_csv(baseline, dirs["metrics"] / "01_gamma_perfect_model_baseline.csv")
    figure_paths = [write_gamma_series_figure(gamma, dirs["figures"], gamma_site)]

    if not run_full_forecast:
        placeholder_forecasts = placeholder_forecast_rows(gamma, cfg)
        forecasts = pd.concat([baseline, placeholder_forecasts], ignore_index=True)
        metrics = forecast_metric_rows(forecasts)
        write_csv(forecasts, dirs["intermediate"] / "02_gamma_forecasts.csv")
        write_csv(metrics, dirs["metrics"] / "02_gamma_forecast_metrics.csv")
        write_csv(metrics, dirs["tables"] / "table01_forecast_impact.csv")
        write_csv(
            metrics.loc[metrics["model"] == "perfect_model_baseline"].copy(),
            dirs["tables"] / "table02_gamma_perfect_model_baseline.csv",
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
                "status": "placeholder_smoke_only",
                "publication_ready": False,
                "contains_placeholder_forecasts": True,
                "gamma_site": gamma_site,
                "output_subfolders": output_dir_manifest_payload(dirs),
                "row_counts": {"gamma": int(len(gamma))},
                "tables": [
                    "table01_forecast_impact.csv",
                    "table02_gamma_perfect_model_baseline.csv",
                ],
                "figures": [path.name for path in figure_paths if path is not None],
            },
        )
        return {
            "status": "placeholder_smoke_only",
            "gamma_site": gamma_site,
            "metrics": metrics,
            "forecasts": forecasts,
        }

    train_end = pd.Timestamp(cfg["windows"]["gamma_forecast_test_start"]) - pd.Timedelta(minutes=15)
    train_start = gamma["_timestamp_dt"].min().strftime("%Y-%m-%d")
    test_start = cfg["windows"]["gamma_forecast_test_start"]
    test_end = cfg["windows"]["gamma_forecast_test_end"]
    condition_map = GAMMA_DATA_CONDITION_COLUMNS
    forecast_frames = [baseline]
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
    forecasts["is_placeholder"] = False
    forecasts["status"] = "complete"
    metrics = forecast_metric_rows(forecasts)
    write_csv(forecasts, dirs["intermediate"] / f"{example_index:02d}_gamma_forecasts.csv")
    write_csv(baseline, dirs["metrics"] / "01_gamma_perfect_model_baseline.csv")
    write_csv(metrics, dirs["metrics"] / "02_gamma_forecast_metrics.csv")
    write_csv(metrics, dirs["tables"] / "table01_forecast_impact.csv")
    write_csv(
        metrics.loc[metrics["model"] == "perfect_model_baseline"].copy(),
        dirs["tables"] / "table02_gamma_perfect_model_baseline.csv",
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
            "publication_ready": True,
            "contains_placeholder_forecasts": False,
            "gamma_site": gamma_site,
            "output_subfolders": output_dir_manifest_payload(dirs),
            "row_counts": {"alpha": int(len(alpha)), "gamma": int(len(gamma))},
            "tables": [
                "table01_forecast_impact.csv",
                "table02_gamma_perfect_model_baseline.csv",
            ],
            "figures": [path.name for path in figure_paths if path is not None],
        },
    )
    return {
        "status": "complete",
        "gamma_site": gamma_site,
        "metrics": metrics,
        "forecasts": forecasts,
    }


def write_gamma_series_figure(gamma: pd.DataFrame, figures_dir: Path, gamma_site: str) -> Path:
    plt = _load_matplotlib()
    plot_df = gamma.loc[
        (gamma["date"] >= "2024-09-01") & (gamma["date"] <= "2024-09-07")
    ].copy()
    if plot_df.empty:
        plot_df = gamma.tail(7 * 96).copy()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(
        plot_df["_timestamp_dt"],
        plot_df["net_load_MW"],
        label=forecast_label("raw_uncorrected"),
        color=JOURNAL_LINE_COLORS["raw"],
        linewidth=1.2,
    )
    if "m8_xgb_corrected_MW" in plot_df.columns and plot_df["m8_xgb_corrected_MW"].notna().any():
        ax.plot(
            plot_df["_timestamp_dt"],
            plot_df["m8_xgb_corrected_MW"],
            label=forecast_label("m8_xgb_corrected"),
            color=JOURNAL_LINE_COLORS["m8"],
            linewidth=1.2,
        )
    ax.plot(
        plot_df["_timestamp_dt"],
        plot_df["reference_net_load_MW"],
        label=forecast_label("reference_corrected"),
        color=JOURNAL_LINE_COLORS["reference"],
        linewidth=1.2,
    )
    ax.axhline(0, color=JOURNAL_COLORS["dark_blue"], linewidth=0.8, linestyle=":")
    ax.set_title(f"Gamma site {gamma_site}: forecast test-period net load example")
    ax.set_ylabel("MW")
    ax.legend(ncol=3, fontsize=9)
    style_axis_grid(ax)
    fig.tight_layout()
    path = figures_dir / "fig01_gamma_series_raw_corrected_reference.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def forecast_label(value: str) -> str:
    return FORECAST_DISPLAY_LABELS.get(value, value)


def write_forecast_metric_figure(metrics: pd.DataFrame, figures_dir: Path) -> Path | None:
    if metrics.empty:
        return None
    plt = _load_matplotlib()
    plot_df = metrics.copy()
    if plot_df.empty:
        return None
    condition_order = ["raw_uncorrected", "m8_xgb_corrected", "reference_corrected"]
    model_order = ["perfect_model_baseline", "seasonal_naive", "linear_regression", "xgboost"]
    model_colors = {
        "perfect_model_baseline": JOURNAL_COLORS["dark_blue"],
        "seasonal_naive": JOURNAL_COLORS["light_grey"],
        "linear_regression": JOURNAL_COLORS["orange"],
        "xgboost": JOURNAL_COLORS["grey"],
    }
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    group_centers = np.arange(len(condition_order))
    width = 0.18
    legend_models: list[str] = []
    for group_idx, condition in enumerate(condition_order):
        group = plot_df.loc[plot_df["data_condition"] == condition].copy()
        if group.empty:
            continue
        group["model"] = pd.Categorical(group["model"], categories=model_order, ordered=True)
        group = group.sort_values(["rmse_MW", "model"]).reset_index(drop=True)
        offsets = (np.arange(len(group)) - (len(group) - 1) / 2.0) * width
        positions = group_centers[group_idx] + offsets
        for position, (_, row) in zip(positions, group.iterrows()):
            model = str(row["model"])
            ax.bar(
                position,
                float(row["rmse_MW"]),
                width * 0.92,
                color=model_colors.get(model, JOURNAL_COLORS["grey"]),
            )
            if model not in legend_models:
                legend_models.append(model)
    ax.set_ylabel("RMSE (MW)")
    ax.set_title("Gamma RMSE by data condition and model\nBars sorted within each group")
    ax.set_xticks(group_centers)
    ax.set_xticklabels([forecast_label(condition) for condition in condition_order], fontsize=9)
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=model_colors[model], label=forecast_label(model))
        for model in model_order
        if model in set(legend_models)
    ]
    ax.legend(handles=legend_handles, ncol=2, fontsize=8)
    style_axis_grid(ax)
    fig.subplots_adjust(bottom=0.32)
    path = figures_dir / "fig02_gamma_forecast_rmse.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_forecast_residual_figure(forecasts: pd.DataFrame, figures_dir: Path) -> Path | None:
    if forecasts.empty or {"y_reference", "y_pred"}.difference(forecasts.columns):
        return None
    plot_df = forecasts.copy()
    plot_df = plot_df.loc[plot_df["model"] != "perfect_model_baseline"].copy()
    if plot_df.empty:
        return None
    plot_df["residual_MW"] = plot_df["y_pred"] - plot_df["y_reference"]
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = plot_df["data_condition"].map(forecast_label) + "\n" + plot_df["model"].map(forecast_label)
    order = labels.drop_duplicates().tolist()
    data = [plot_df.loc[labels == label, "residual_MW"].dropna().to_numpy() for label in order]
    if not data:
        plt.close(fig)
        return None
    box = ax.boxplot(data, showfliers=False, patch_artist=True)
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(order)
    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(JOURNAL_BAR_COLORS[idx % len(JOURNAL_BAR_COLORS)])
        patch.set_alpha(0.75)
        patch.set_edgecolor(JOURNAL_COLORS["dark_blue"])
    for element in ["whiskers", "caps", "medians"]:
        for item in box[element]:
            item.set_color(JOURNAL_COLORS["dark_blue"])
    ax.axhline(0, color=JOURNAL_COLORS["dark_blue"], linewidth=0.8, linestyle=":")
    ax.set_ylabel("Forecast residual (MW)")
    ax.set_title("Gamma forecast residual comparison")
    ax.tick_params(axis="x", rotation=25, labelsize=7)
    style_axis_grid(ax)
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


def publication_expected_figures(paths: ArticlePaths) -> dict[str, Path]:
    return {
        "fig01_site_rpf_day_counts_alpha_beta": paths.figures
        / "01_characterisation"
        / "fig01_site_rpf_day_counts_alpha_beta.png",
        "fig02_month_hour_heatmap_alpha_beta": paths.figures
        / "01_characterisation"
        / "fig02_month_hour_heatmap_alpha_beta.png",
        "fig03_event_duration_distribution_alpha_beta": paths.figures
        / "01_characterisation"
        / "fig03_event_duration_distribution_alpha_beta.png",
        "fig04_day_of_month_rpf_heatmap_alpha_beta": paths.figures
        / "01_characterisation"
        / "fig04_day_of_month_rpf_heatmap_alpha_beta.png",
        "fig05_rpf_events_per_day_doughnut_alpha_beta": paths.figures
        / "01_characterisation"
        / "fig05_rpf_events_per_day_doughnut_alpha_beta.png",
        "fig01a_confusion_matrices_day": paths.figures
        / "02_correction_validation"
        / "fig01a_confusion_matrices_day.png",
        "fig01b_confusion_matrices_interval": paths.figures
        / "02_correction_validation"
        / "fig01b_confusion_matrices_interval.png",
        "fig02a_precision_recall_f1_day": paths.figures
        / "02_correction_validation"
        / "fig02a_precision_recall_f1_day.png",
        "fig02b_precision_recall_f1_interval": paths.figures
        / "02_correction_validation"
        / "fig02b_precision_recall_f1_interval.png",
        "fig03_beta_site_precision_recall_f1_boxplot": paths.figures
        / "02_correction_validation"
        / "fig03_beta_site_precision_recall_f1_boxplot.png",
        "fig04_alpha_site_precision_recall_f1_boxplot": paths.figures
        / "02_correction_validation"
        / "fig04_alpha_site_precision_recall_f1_boxplot.png",
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
    contains_placeholder_metrics = False
    for name, src in upstream_tables.items():
        if src.exists():
            table = pd.read_csv(src)
            if name == "correction_metrics" and "is_placeholder" in table.columns:
                contains_placeholder_metrics = bool(
                    table["is_placeholder"].astype("string").str.lower().isin(["true", "1"]).any()
                )
            target = dirs["tables"] / table_targets[name]
            written = write_table_formats(table, target)
            outputs[table_targets[name]] = written["csv"]

    expected_figures = publication_expected_figures(paths)
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
            "contains_placeholder_metrics": contains_placeholder_metrics,
        },
    )
    return outputs

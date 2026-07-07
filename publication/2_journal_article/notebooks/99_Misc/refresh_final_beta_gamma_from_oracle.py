from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

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
CONFIDENCE_COLUMN = "confidence"
FINAL_BETA_COLUMNS = EXPECTED_COLUMNS + [CONFIDENCE_COLUMN]
REVIEW_START = "2023-10-01"
REVIEW_END = "2024-09-30"
EXPECTED_BETA_ROWS = 280_800
EXPECTED_BETA_SITE_DAYS = 2_928
EXPECTED_GAMMA_ROWS = 35_136


def find_repo_root() -> Path:
    start = Path(__file__).resolve()
    marker = Path("publication/2_journal_article/config/experiment_config.yaml")
    for candidate in [start.parent, *start.parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"Could not find repo root containing {marker}")


ROOT = find_repo_root()
ARTICLE = ROOT / "publication/2_journal_article"
ORACLE_DIR = ARTICLE / "dataset/oracle_data_creation"
ORACLE_ARCHIVE_CLEANUP = ORACLE_DIR / "archive/2026-07-03_oracle_workspace_cleanup"
ORACLE_CORE = ORACLE_ARCHIVE_CLEANUP / "oracle_review_core.py"
DEFAULT_ANNOTATIONS = ORACLE_ARCHIVE_CLEANUP / "manual_oracle_annotations_final_review.csv"
DEFAULT_SOURCE = ARTICLE / "dataset/processed/actual_pynrpf_dataset.parquet"
DEFAULT_CONFIG = ARTICLE / "config/experiment_config.yaml"
DEFAULT_FINAL_DIR = ARTICLE / "dataset/final"


def import_oracle_core(core_path: Path):
    if not core_path.exists():
        raise FileNotFoundError(f"Oracle review core not found: {core_path}")
    sys.path.insert(0, str(core_path.parent))
    import oracle_review_core  # type: ignore

    return oracle_review_core


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def write_parquet(df: pd.DataFrame, path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def write_csv(df: pd.DataFrame, path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def write_json(data: dict[str, Any], path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sha256(paths: list[Path], output_path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    lines = []
    for path in paths:
        if path.exists():
            lines.append(f"{sha256(path)}  {path.name}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_final_dates(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="raise").dt.strftime("%Y-%m-%d")
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="raise").astype(str)
    work["label_interval"] = coerce_bool(work["label_interval"])
    work["label_day"] = work.groupby(["substation_id", "date"])["label_interval"].transform("any")
    if CONFIDENCE_COLUMN in work.columns:
        work[CONFIDENCE_COLUMN] = (
            work[CONFIDENCE_COLUMN]
            .astype("string")
            .fillna("sure")
            .str.strip()
            .str.lower()
            .replace("", "sure")
        )
    return work.sort_values(["substation_id", "timestamp"]).reset_index(drop=True)


def rename_beta_sites(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    site = work["substation_id"].astype(str)
    if not site.str.startswith("act_").all():
        bad = sorted(site.loc[~site.str.startswith("act_")].unique())
        raise ValueError(f"Expected raw actual site IDs to start with act_, found: {bad}")
    work["substation_id"] = site.str.replace(r"^act_", "beta_", regex=True)
    return work


def fast_apply_annotations(scoped: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """Vectorized equivalent of oracle_review_core.apply_annotations for complete reviews."""
    required = {"substation_id", "date", "review_action", "rpf_start_time", "rpf_end_time", CONFIDENCE_COLUMN}
    missing = required.difference(annotations.columns)
    if missing:
        raise ValueError(f"Annotations are missing required columns: {sorted(missing)}")

    valid_keys = scoped[["substation_id", "date"]].drop_duplicates()
    ann_keys = annotations[["substation_id", "date"]].drop_duplicates()
    missing_annotations = valid_keys.merge(ann_keys, on=["substation_id", "date"], how="left", indicator=True)
    missing_annotations = missing_annotations.loc[missing_annotations["_merge"].eq("left_only")]
    if not missing_annotations.empty:
        raise ValueError(
            "Fast refresh expects complete final-review annotations. "
            f"Missing first rows: {missing_annotations.head(5).to_dict(orient='records')}"
        )
    unknown = ann_keys.merge(valid_keys, on=["substation_id", "date"], how="left", indicator=True)
    unknown = unknown.loc[unknown["_merge"].eq("left_only")]
    if not unknown.empty:
        raise ValueError(f"Annotations outside review scope: {unknown.head(5).to_dict(orient='records')}")

    work = scoped.copy()
    work["_old_label_interval"] = coerce_bool(work["label_interval"])
    merged = work.merge(
        annotations,
        on=["substation_id", "date"],
        how="left",
        validate="many_to_one",
    )
    action = merged["review_action"].astype(str)
    manual_window = action.eq("manual_window")
    accept_old = action.eq("accept_old")
    no_rpf = action.eq("no_rpf")
    if not (manual_window | accept_old | no_rpf).all():
        bad = sorted(set(action.loc[~(manual_window | accept_old | no_rpf)]))
        raise ValueError(f"Unknown review_action after merge: {bad}")

    new_label = np.where(accept_old, merged["_old_label_interval"].to_numpy(dtype=bool), False)
    manual_mask = (
        manual_window
        & merged["_time_hhmm"].astype(str).ge(merged["rpf_start_time"].astype(str))
        & merged["_time_hhmm"].astype(str).le(merged["rpf_end_time"].astype(str))
    )
    new_label = np.where(manual_mask, True, new_label)
    merged["label_interval"] = new_label.astype(bool)
    merged["label_day"] = merged.groupby(["substation_id", "date"])["label_interval"].transform("any")
    merged[CONFIDENCE_COLUMN] = (
        merged[CONFIDENCE_COLUMN]
        .astype("string")
        .fillna("sure")
        .str.strip()
        .str.lower()
        .replace("", "sure")
    )
    return merged[FINAL_BETA_COLUMNS].copy()


def reference_net_load(df: pd.DataFrame) -> pd.Series:
    flags = coerce_bool(df["label_interval"])
    values = np.where(flags, -df["net_load_MW"], df["net_load_MW"])
    out = pd.Series(values, index=df.index, name="reference_net_load_MW")
    out.loc[df["net_load_MW"].isna()] = np.nan
    return out


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    err = pd.to_numeric(y_true, errors="coerce") - pd.to_numeric(y_pred, errors="coerce")
    return float(np.sqrt(np.nanmean(np.square(err))))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    err = pd.to_numeric(y_true, errors="coerce") - pd.to_numeric(y_pred, errors="coerce")
    return float(np.nanmean(np.abs(err)))


def dataset_summary(df: pd.DataFrame, dataset_name: str, source_file: str) -> dict[str, Any]:
    site_day = df[["substation_id", "date", "label_day"]].drop_duplicates()
    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "n_rows": int(len(df)),
        "n_stations": int(df["substation_id"].nunique()),
        "min_timestamp": str(df["timestamp"].min()),
        "max_timestamp": str(df["timestamp"].max()),
        "n_dates": int(df["date"].nunique()),
        "null_timestamp": int(df["timestamp"].isna().sum()),
        "null_net_load_MW": int(df["net_load_MW"].isna().sum()),
        "null_solar_MW": int(df["solar_MW"].isna().sum()),
        "duplicate_substation_timestamp_keys": int(
            df.duplicated(["substation_id", "timestamp"], keep=False).sum()
        ),
        "positive_label_interval": int(coerce_bool(df["label_interval"]).sum()),
        "positive_label_day_site_days": int(coerce_bool(site_day["label_day"]).sum()),
        "source_stage": "final",
        "source_file": source_file,
    }
    if CONFIDENCE_COLUMN in df.columns:
        confidence_site_days = df[["substation_id", "date", CONFIDENCE_COLUMN]].drop_duplicates()
        summary["sure_confidence_site_days"] = int(confidence_site_days[CONFIDENCE_COLUMN].eq("sure").sum())
        summary["unsure_confidence_site_days"] = int(
            confidence_site_days[CONFIDENCE_COLUMN].eq("unsure").sum()
        )
    return summary


def gamma_site_rankings(beta: pd.DataFrame, ranking_metric: str) -> pd.DataFrame:
    work = beta.copy()
    work["reference_net_load_MW"] = reference_net_load(work)
    rows: list[dict[str, Any]] = []
    for site, group in work.groupby("substation_id", sort=True):
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
                "max_raw_flagged_MW": float(flagged["net_load_MW"].max()) if not flagged.empty else 0.0,
                "mean_raw_flagged_MW": float(flagged["net_load_MW"].mean()) if not flagged.empty else 0.0,
                "min_reference_net_load_MW": min_reference,
                "abs_min_reference_net_load_MW": abs(min_reference),
                "raw_reference_error_MW_sum": float(err.abs().sum()),
            }
        )
    ranking = pd.DataFrame(rows)
    if ranking_metric not in ranking.columns:
        raise ValueError(f"Unknown Gamma ranking metric: {ranking_metric}")
    ranking = ranking.sort_values(
        [ranking_metric, "rpf_days", "rpf_intervals", "substation_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def select_gamma_site(beta: pd.DataFrame, cfg: dict[str, Any], gamma_rank: pd.DataFrame) -> str:
    gamma_cfg = cfg.get("gamma", {})
    if gamma_cfg.get("selection_mode") == "manual" and gamma_cfg.get("manual_site"):
        site = str(gamma_cfg["manual_site"])
        if site not in set(beta["substation_id"]):
            raise ValueError(f"Configured manual gamma site is not present in Beta: {site}")
        return site
    return str(gamma_rank.iloc[0]["substation_id"])


def validate_beta(beta: pd.DataFrame, cfg: dict[str, Any]) -> None:
    if list(beta.columns) != FINAL_BETA_COLUMNS:
        raise ValueError(f"Unexpected Beta columns: {list(beta.columns)}")
    if len(beta) != EXPECTED_BETA_ROWS:
        raise ValueError(f"Expected {EXPECTED_BETA_ROWS} Beta rows, found {len(beta)}")
    site_days = beta[["substation_id", "date"]].drop_duplicates()
    if len(site_days) != EXPECTED_BETA_SITE_DAYS:
        raise ValueError(f"Expected {EXPECTED_BETA_SITE_DAYS} Beta site-days, found {len(site_days)}")
    if beta["date"].min() != cfg["windows"]["beta_start"] or beta["date"].max() != cfg["windows"]["beta_end"]:
        raise ValueError("Beta date range does not match config.")
    expected_sites = [f"beta_{letter}" for letter in "ABCDEFGH"]
    sites = sorted(beta["substation_id"].unique().tolist())
    if sites != expected_sites:
        raise ValueError(f"Unexpected Beta sites: {sites}")
    confidence_counts = beta[["substation_id", "date", CONFIDENCE_COLUMN]].drop_duplicates()[
        CONFIDENCE_COLUMN
    ].value_counts().to_dict()
    unknown_confidence = set(confidence_counts) - {"sure", "unsure"}
    if unknown_confidence:
        raise ValueError(f"Unknown confidence values in final Beta: {sorted(unknown_confidence)}")


def validate_gamma(gamma: pd.DataFrame, beta: pd.DataFrame) -> None:
    if list(gamma.columns) != FINAL_BETA_COLUMNS:
        raise ValueError(f"Unexpected Gamma columns: {list(gamma.columns)}")
    if gamma["substation_id"].nunique() != 1:
        raise ValueError("Gamma must contain exactly one site.")
    if len(gamma) != EXPECTED_GAMMA_ROWS:
        raise ValueError(f"Expected {EXPECTED_GAMMA_ROWS} Gamma rows, found {len(gamma)}")
    if gamma["date"].min() != beta["date"].min() or gamma["date"].max() != beta["date"].max():
        raise ValueError("Gamma date range must match Beta date range.")


def refresh(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    cfg = read_config(args.config)
    oracle_core = import_oracle_core(args.oracle_core)

    source = oracle_core.load_source_dataset(args.source)
    scoped = oracle_core.filter_review_scope(source)
    oracle_core.assert_expected_review_scope(scoped)
    annotations = oracle_core.read_annotations(args.annotations)
    status = oracle_core.review_status(scoped, annotations)
    if not status["complete"] and not args.allow_incomplete:
        raise ValueError(
            "Final-review annotations are incomplete. "
            "Pass --allow-incomplete only for deliberate partial refreshes."
        )

    oracle_reviewed = fast_apply_annotations(scoped, annotations)
    oracle_reviewed = prepare_final_dates(oracle_reviewed)

    beta = prepare_final_dates(rename_beta_sites(oracle_reviewed))[FINAL_BETA_COLUMNS]
    beta = beta.loc[(beta["date"] >= REVIEW_START) & (beta["date"] <= REVIEW_END)].copy()
    gamma_rank = gamma_site_rankings(beta, cfg.get("gamma", {}).get("ranking_metric", "data_error_rmse_MW"))
    gamma_site = select_gamma_site(beta, cfg, gamma_rank)
    gamma = beta.loc[beta["substation_id"].eq(gamma_site)].copy().reset_index(drop=True)

    validate_beta(beta, cfg)
    validate_gamma(gamma, beta)

    final_dir = args.final_dir
    beta_path = final_dir / "dataset_beta.parquet"
    gamma_path = final_dir / "dataset_gamma.parquet"
    final_summary_path = final_dir / "dataset_final_summary.csv"
    gamma_rank_path = final_dir / "gamma_selection_summary.csv"
    final_sha_path = final_dir / "sha256.txt"

    oracle_output_dir = args.oracle_output_dir
    oracle_parquet_path = oracle_output_dir / "actual_pynrpf_dataset_reflagged.parquet"
    oracle_summary_path = oracle_output_dir / "dataset_summary.csv"
    oracle_status_path = oracle_output_dir / "review_status.json"
    oracle_sha_path = oracle_output_dir / "sha256.txt"

    final_summary_rows = []
    alpha_path = final_dir / "dataset_alpha.parquet"
    if alpha_path.exists():
        alpha = pd.read_parquet(alpha_path)
        final_summary_rows.append(
            dataset_summary(alpha, "Alpha", "dataset/final/dataset_alpha.parquet")
        )
    final_summary_rows.extend(
        [
            dataset_summary(beta, "Beta", "dataset/final/dataset_beta.parquet"),
            dataset_summary(gamma, "Gamma", "dataset/final/dataset_gamma.parquet"),
        ]
    )
    final_summary = pd.DataFrame(final_summary_rows)

    oracle_summary = pd.DataFrame([oracle_core.dataset_summary(oracle_reviewed, "Reviewed actual oracle")])

    write_parquet(oracle_reviewed[FINAL_BETA_COLUMNS], oracle_parquet_path, args.dry_run)
    write_csv(oracle_summary, oracle_summary_path, args.dry_run)
    write_json(status, oracle_status_path, args.dry_run)
    if args.write_oracle_csv and not args.dry_run:
        oracle_reviewed[FINAL_BETA_COLUMNS].to_csv(
            oracle_output_dir / "actual_pynrpf_dataset_reflagged.csv",
            index=False,
        )

    write_parquet(beta, beta_path, args.dry_run)
    write_parquet(gamma, gamma_path, args.dry_run)
    write_csv(final_summary, final_summary_path, args.dry_run)
    write_csv(gamma_rank, gamma_rank_path, args.dry_run)
    write_sha256([oracle_parquet_path, oracle_summary_path, oracle_status_path], oracle_sha_path, args.dry_run)
    write_sha256(
        [path for path in [alpha_path, beta_path, gamma_path, final_summary_path, gamma_rank_path] if path.exists()],
        final_sha_path,
        args.dry_run,
    )

    return {
        "dry_run": bool(args.dry_run),
        "elapsed_seconds": time.time() - started,
        "review_status": status,
        "oracle_rows": int(len(oracle_reviewed)),
        "beta_rows": int(len(beta)),
        "beta_rpf_days": int(beta[["substation_id", "date", "label_day"]].drop_duplicates()["label_day"].sum()),
        "beta_rpf_intervals": int(beta["label_interval"].sum()),
        "beta_confidence_counts": beta[["substation_id", "date", CONFIDENCE_COLUMN]]
        .drop_duplicates()[CONFIDENCE_COLUMN]
        .value_counts()
        .to_dict(),
        "gamma_site": gamma_site,
        "gamma_rows": int(len(gamma)),
        "gamma_rpf_days": int(
            gamma[["substation_id", "date", "label_day"]].drop_duplicates()["label_day"].sum()
        ),
        "outputs": {
            "oracle_parquet": str(oracle_parquet_path.relative_to(ROOT)),
            "beta_parquet": str(beta_path.relative_to(ROOT)),
            "gamma_parquet": str(gamma_path.relative_to(ROOT)),
            "final_summary": str(final_summary_path.relative_to(ROOT)),
            "gamma_rank": str(gamma_rank_path.relative_to(ROOT)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast-refresh reviewed oracle labels into active oracle parquet and "
            "journal final Beta/Gamma parquet outputs. CSV export is skipped by default."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--oracle-core", type=Path, default=ORACLE_CORE)
    parser.add_argument("--oracle-output-dir", type=Path, default=ORACLE_DIR)
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print outputs without writing files.")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--write-oracle-csv",
        action="store_true",
        help="Also write the large active oracle CSV. Off by default for speed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = refresh(args)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

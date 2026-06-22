from __future__ import annotations

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
            and (candidate / "dataset" / "processed").exists()
        ):
            return candidate
        nested = candidate / "publication" / "2_journal_article"
        if (nested / "dataset" / "processed").exists():
            return nested.resolve()
    raise RuntimeError(f"Could not locate publication/2_journal_article from {start}")


def article_path(article_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else article_root / path


def load_config(article_root: Path) -> dict[str, Any]:
    path = article_root / "config" / "experiment_config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if cfg.get("schema_version") != "journal_alpha_beta_gamma_v1":
        raise ValueError("Expected journal_alpha_beta_gamma_v1 config schema.")
    cfg["_config_path"] = str(path)
    return cfg


def article_paths(article_root: Path, cfg: dict[str, Any]) -> ArticlePaths:
    output_root = article_path(article_root, cfg["paths"]["output_base_dir"])
    outputs = cfg["outputs"]
    return ArticlePaths(
        root=article_root,
        config_path=Path(cfg["_config_path"]),
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
        paths.intermediate,
        paths.metrics,
        paths.tables,
        paths.figures,
        paths.manifests,
    ]:
        path.mkdir(parents=True, exist_ok=True)


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
        path = article_path(article_root, cfg["paths"]["alpha_dataset_csv"])
        return prepare_dataset(pd.read_csv(path), "Alpha")
    if dataset_key == "beta":
        use_reviewed = bool(cfg.get("datasets", {}).get("use_reviewed_beta", False))
        key = "beta_reviewed_dataset_csv" if use_reviewed else "beta_dataset_csv"
        path = article_path(article_root, cfg["paths"][key])
        beta = prepare_dataset(pd.read_csv(path), "Beta")
        return filter_date_window(beta, cfg["windows"]["beta_start"], cfg["windows"]["beta_end"])
    raise ValueError(f"Unknown dataset_key: {dataset_key}")


def filter_date_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].copy().reset_index(drop=True)


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


def gamma_site_rankings(beta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site, group in beta.groupby("substation_id"):
        err = group["net_load_MW"] - group["reference_net_load_MW"]
        site_days = group.groupby("date")["label_interval"].any()
        flagged = group.loc[group["label_interval"]]
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
                "min_reference_net_load_MW": float(group["reference_net_load_MW"].min()),
                "raw_reference_error_MW_sum": float(err.abs().sum()),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["data_error_rmse_MW", "rpf_days", "rpf_intervals", "substation_id"],
        ascending=[False, False, False, True],
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking.reset_index(drop=True)


def select_gamma_site(beta: pd.DataFrame) -> str:
    return str(gamma_site_rankings(beta).iloc[0]["substation_id"])


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


def run_prepare_datasets(article_root: Path | None = None) -> dict[str, Any]:
    root = find_article_root(article_root)
    cfg = load_config(root)
    paths = article_paths(root, cfg)
    ensure_output_dirs(paths)
    alpha = load_dataset(root, cfg, "alpha")
    beta = load_dataset(root, cfg, "beta")

    dataset_rows = [
        dataset_summary(alpha, "Alpha"),
        dataset_summary(beta, "Beta"),
    ]
    gamma_rank = gamma_site_rankings(beta)
    gamma_site = str(gamma_rank.iloc[0]["substation_id"])
    gamma = beta.loc[beta["substation_id"] == gamma_site].copy()
    dataset_rows.append(dataset_summary(gamma, "Gamma"))

    alpha_rank = site_rpf_summary(alpha, "Alpha").sort_values(
        ["rpf_days", "rpf_intervals", "substation_id"],
        ascending=[False, False, True],
    )
    alpha_rank.insert(0, "rank", np.arange(1, len(alpha_rank) + 1))
    alpha_sites = alpha_loso_sites(alpha, cfg)

    write_csv(pd.DataFrame(dataset_rows), paths.intermediate / "dataset_summary.csv")
    write_csv(alpha_rank, paths.intermediate / "alpha_site_rankings.csv")
    write_csv(gamma_rank, paths.intermediate / "beta_gamma_site_rankings.csv")
    write_manifest(
        paths,
        "00_prepare_datasets.json",
        {
            "notebook": "00_prepare_datasets.ipynb",
            "alpha_loso_sites": alpha_sites,
            "gamma_site": gamma_site,
            "beta_uses_reviewed_oracle": bool(cfg["datasets"]["use_reviewed_beta"]),
        },
    )
    return {"alpha": alpha, "beta": beta, "gamma_site": gamma_site, "alpha_sites": alpha_sites}


def run_characterisation(article_root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = find_article_root(article_root)
    cfg = load_config(root)
    paths = article_paths(root, cfg)
    ensure_output_dirs(paths)
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
    write_csv(occurrence, paths.intermediate / "rpf_occurrence_by_site.csv")
    write_csv(temporal, paths.intermediate / "rpf_temporal_summary.csv")
    write_csv(events, paths.intermediate / "rpf_event_summary.csv")
    write_characterisation_figures(occurrence, temporal, events, paths)
    write_manifest(
        paths,
        "01_characterisation.json",
        {"notebook": "01_characterisation.ipynb", "n_events": int(len(events))},
    )
    return {"occurrence": occurrence, "temporal": temporal, "events": events}


def _load_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_characterisation_figures(
    occurrence: pd.DataFrame,
    temporal: pd.DataFrame,
    events: pd.DataFrame,
    paths: ArticlePaths,
) -> None:
    plt = _load_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 4))
    pivot = occurrence.pivot(index="substation_id", columns="dataset", values="rpf_days").fillna(0)
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("RPF days")
    ax.set_xlabel("Site")
    ax.set_title("RPF day counts by site")
    fig.tight_layout()
    fig.savefig(paths.figures / "site_rpf_day_counts.png", dpi=200)
    plt.close(fig)

    for dataset in ["Alpha", "Beta"]:
        month_hour = temporal[
            (temporal["dataset"] == dataset) & (temporal["level"] == "month_hour")
        ]
        heat = month_hour.pivot(index="month", columns="hour", values="rpf_interval_pct").fillna(0)
        fig, ax = plt.subplots(figsize=(10, 4))
        image = ax.imshow(heat.to_numpy(), aspect="auto", origin="lower")
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels(heat.columns)
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(heat.index)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Month")
        ax.set_title(f"{dataset} RPF interval percentage by month and hour")
        fig.colorbar(image, ax=ax, label="% intervals")
        fig.tight_layout()
        fig.savefig(paths.figures / f"month_hour_heatmap_{dataset.lower()}.png", dpi=200)
        plt.close(fig)

    if not events.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        events["duration_minutes"].hist(ax=ax, bins=30)
        ax.set_xlabel("Event duration (minutes)")
        ax.set_ylabel("Count")
        ax.set_title("RPF event duration distribution")
        fig.tight_layout()
        fig.savefig(paths.figures / "event_duration_distribution.png", dpi=200)
        plt.close(fig)


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


def run_correction_validation(article_root: Path | None = None) -> dict[str, Any]:
    root = find_article_root(article_root)
    cfg = load_config(root)
    paths = article_paths(root, cfg)
    ensure_output_dirs(paths)
    alpha = load_dataset(root, cfg, "alpha")
    beta = load_dataset(root, cfg, "beta")

    if not bool(cfg["execution"]["run_full_correction_validation"]):
        plan = correction_smoke_plan(alpha, beta, cfg)
        metrics = correction_smoke_metrics(alpha, cfg)
        write_csv(plan, paths.intermediate / "correction_validation_plan.csv")
        write_csv(metrics, paths.metrics / "correction_metrics.csv")
        write_manifest(
            paths,
            "02_correction_validation.json",
            {"notebook": "02_correction_validation.ipynb", "status": "smoke_only"},
        )
        return {"status": "smoke_only", "plan": plan, "metrics": metrics}

    metric_frames = []
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
            pred_path = paths.intermediate / f"correction_predictions_{fold_id}_{method}.csv"
            write_csv(pred[pred_cols], pred_path)
            metric_frames.append(evaluate_prediction_frame(pred, cfg, "Alpha", fold_id, method))

    alpha_train = alpha.copy()
    beta_bundle = train_m8_bundle(alpha_train, cfg, root)
    for method, pred in [
        ("m8_xgb", predict_m8_bundle(beta, beta_bundle, cfg, root)),
        ("m7_dtr", predict_m7(beta, cfg, root)),
    ]:
        fold_id = "beta_transfer"
        pred_cols = EXPECTED_COLUMNS + ["pred_interval", "corrected_net_load_MW"]
        pred_path = paths.intermediate / f"correction_predictions_{fold_id}_{method}.csv"
        write_csv(pred[pred_cols], pred_path)
        metric_frames.append(evaluate_prediction_frame(pred, cfg, "Beta", fold_id, method))

    metrics = pd.concat(metric_frames, ignore_index=True)
    write_csv(metrics, paths.metrics / "correction_metrics.csv")
    write_manifest(
        paths,
        "02_correction_validation.json",
        {"notebook": "02_correction_validation.ipynb", "status": "complete"},
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
    beta = load_dataset(root, cfg, "beta")
    gamma_site = select_gamma_site(beta)
    gamma = beta.loc[beta["substation_id"] == gamma_site].copy()
    gamma["raw_uncorrected_MW"] = gamma["net_load_MW"]
    gamma["reference_corrected_MW"] = gamma["reference_net_load_MW"]

    benchmark = data_error_benchmark(gamma, cfg)
    write_csv(benchmark, paths.metrics / "gamma_data_error_benchmark.csv")
    write_gamma_series_figure(gamma, paths, gamma_site)

    if not bool(cfg["execution"]["run_full_forecast"]):
        metrics = forecast_metric_rows(benchmark)
        write_csv(metrics, paths.metrics / "gamma_forecast_metrics.csv")
        write_forecast_metric_figure(metrics, paths)
        write_manifest(
            paths,
            "03_gamma_forecast_impact.json",
            {
                "notebook": "03_gamma_forecast_impact.ipynb",
                "status": "smoke_only",
                "gamma_site": gamma_site,
            },
        )
        return {"status": "smoke_only", "gamma_site": gamma_site, "metrics": metrics}

    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor

    train_end = pd.Timestamp(cfg["windows"]["gamma_forecast_test_start"]) - pd.Timedelta(minutes=15)
    train_start = gamma["_timestamp_dt"].min().strftime("%Y-%m-%d")
    test_start = cfg["windows"]["gamma_forecast_test_start"]
    test_end = cfg["windows"]["gamma_forecast_test_end"]
    condition_map = {
        "raw_uncorrected": "raw_uncorrected_MW",
        "reference_corrected": "reference_corrected_MW",
    }
    forecast_frames = [benchmark]
    for condition, column in condition_map.items():
        train_examples = build_forecast_examples(
            gamma, column, cfg, train_start, train_end.strftime("%Y-%m-%d")
        )
        test_examples = build_forecast_examples(gamma, column, cfg, test_start, test_end)
        train_examples = train_examples.loc[train_examples["target_timestamp"] <= train_end]
        train_path = paths.intermediate / f"gamma_forecast_examples_train_{condition}.csv"
        test_path = paths.intermediate / f"gamma_forecast_examples_test_{condition}.csv"
        write_csv(train_examples, train_path)
        write_csv(test_examples, test_path)

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
    write_csv(forecasts, paths.intermediate / "gamma_forecasts.csv")
    write_csv(metrics, paths.metrics / "gamma_forecast_metrics.csv")
    write_forecast_metric_figure(metrics, paths)
    write_manifest(
        paths,
        "03_gamma_forecast_impact.json",
        {
            "notebook": "03_gamma_forecast_impact.ipynb",
            "status": "complete",
            "gamma_site": gamma_site,
        },
    )
    return {"status": "complete", "gamma_site": gamma_site, "metrics": metrics}


def write_gamma_series_figure(gamma: pd.DataFrame, paths: ArticlePaths, gamma_site: str) -> None:
    plt = _load_matplotlib()
    plot_df = gamma.loc[
        (gamma["date"] >= "2024-09-01") & (gamma["date"] <= "2024-09-07")
    ].copy()
    if plot_df.empty:
        plot_df = gamma.tail(7 * 96).copy()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(plot_df["_timestamp_dt"], plot_df["net_load_MW"], label="Raw")
    ax.plot(plot_df["_timestamp_dt"], plot_df["reference_net_load_MW"], label="Reference")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title(f"Gamma site {gamma_site}: raw vs reference net load")
    ax.set_ylabel("MW")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.figures / "gamma_series_raw_corrected_reference.png", dpi=200)
    plt.close(fig)


def write_forecast_metric_figure(metrics: pd.DataFrame, paths: ArticlePaths) -> None:
    if metrics.empty:
        return
    plt = _load_matplotlib()
    labels = metrics["data_condition"] + " / " + metrics["model"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, metrics["rmse_MW"])
    ax.set_ylabel("RMSE (MW)")
    ax.set_title("Gamma forecast impact")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(paths.figures / "gamma_forecast_rmse.png", dpi=200)
    plt.close(fig)


def run_publication_tables(article_root: Path | None = None) -> dict[str, Path]:
    root = find_article_root(article_root)
    cfg = load_config(root)
    paths = article_paths(root, cfg)
    ensure_output_dirs(paths)
    outputs: dict[str, Path] = {}
    mapping = {
        "dataset_summary.csv": "table1_dataset_summary.csv",
        "rpf_occurrence_by_site.csv": "table2_characterisation_summary.csv",
    }
    for source, target in mapping.items():
        src = paths.intermediate / source
        if src.exists():
            outputs[target] = write_csv(pd.read_csv(src), paths.tables / target)
    metrics_src = paths.metrics / "correction_metrics.csv"
    if metrics_src.exists():
        outputs["table3_correction_metrics.csv"] = write_csv(
            pd.read_csv(metrics_src), paths.tables / "table3_correction_metrics.csv"
        )
    forecast_src = paths.metrics / "gamma_forecast_metrics.csv"
    if forecast_src.exists():
        outputs["table4_forecast_impact.csv"] = write_csv(
            pd.read_csv(forecast_src), paths.tables / "table4_forecast_impact.csv"
        )
    write_manifest(
        paths,
        "04_publication_tables_figures.json",
        {"notebook": "04_publication_tables_figures.ipynb", "tables": sorted(outputs)},
    )
    return outputs

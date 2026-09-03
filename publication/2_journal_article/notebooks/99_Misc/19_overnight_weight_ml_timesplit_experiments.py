from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "13_260707_physical_score_loso_experiments.py"
OUTPUT_FOLDER_NAME = "19_overnight_weight_ml_timesplit_experiments"
SEED = 9
FULL_WEIGHT_CONFIGS = 3000
WEIGHT_BATCH_SIZE = 100
SMOKE_WEIGHT_CONFIGS = 5
SMOKE_ML_CONFIGS_PER_FAMILY = 2
TEST_START = "2024-07-01"
VALIDATION_START = "2024-06-01"
TEST_END = "2024-09-30"


def load_base_module():
    spec = importlib.util.spec_from_file_location("physical_score_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import base physical-score script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()
ROOT: Path = base.ROOT
JOURNAL: Path = base.JOURNAL
FINAL_DATASET_DIR: Path = base.FINAL_DATASET_DIR
OUT_ROOT = SCRIPT_DIR / "outputs" / OUTPUT_FOLDER_NAME
FEATURE_COLUMNS = list(base.FEATURE_COLUMNS)
CALENDAR_COLUMNS = [
    "month_sin",
    "month_cos",
    "is_weekend",
    "season_autumn",
    "season_spring",
    "season_summer",
    "season_winter",
]
FEATURE_SETS = {
    "phys9": FEATURE_COLUMNS,
    "phys9_calendar": FEATURE_COLUMNS + CALENDAR_COLUMNS,
}

FAILURE_COLUMNS = ["timestamp", "mode", "stage", "error"]
BETA_SURE = "sure"
REGIME_R1 = "R1_beta_loso"
REGIME_R2 = "R2_beta_loso_plus_alpha"
REGIME_R3 = "R3_alpha_only_to_beta"

STAGE_DEFS = {
    "cache": (1, "Refresh daily feature cache"),
    "weight_grid": (2, "Build weight candidates"),
    "strict_weights": (3, "Strict physical-weight LOSO"),
    "ml_grid": (4, "Build ML config grid"),
    "strict_ml": (5, "Strict ML LOSO"),
    "timesplit": (6, "Last-3-month time split"),
    "optimistic": (7, "Optimistic Beta-guided upper bound"),
    "selection": (8, "Model selection and confidence coverage"),
    "final": (9, "Final HTML/checks"),
}
STAGE_COUNT = len(STAGE_DEFS)
PROGRESS_COLUMNS = [
    "timestamp",
    "mode",
    "stage_number",
    "stage_key",
    "stage_name",
    "status",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "outputs",
    "best_metric_note",
    "error",
]


@dataclass(frozen=True)
class RunContext:
    mode: str
    out_dir: Path
    force: bool
    weight_limit: int
    ml_limit_per_family: int | None
    feature_sets: list[str]
    beta_folds: list[str] | None


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now_stamp() -> str:
    return pd.Timestamp.now().isoformat(timespec="seconds")


def safe_bool(values: pd.Series) -> pd.Series:
    return base.safe_bool(values)


def date_key(values: pd.Series) -> pd.Series:
    return base.date_key(values)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def append_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    prior = read_csv_if_exists(path)
    write_csv(path, frame.copy() if prior.empty else pd.concat([prior, frame], ignore_index=True))


def ensure_failure_log(path: Path) -> None:
    if not path.exists():
        write_csv(path, pd.DataFrame(columns=FAILURE_COLUMNS))


def ensure_progress_log(ctx: RunContext) -> None:
    path = ctx.out_dir / "15_stage_progress.csv"
    if not path.exists():
        write_csv(path, pd.DataFrame(columns=PROGRESS_COLUMNS))


def stage_output_paths(ctx: RunContext, stage_key: str) -> list[Path]:
    html_path = SCRIPT_DIR / "2026-07-08_overnight_weight_ml_experiment_summary.html"
    mapping = {
        "cache": [ctx.out_dir / "01_refreshed_daily_feature_cache.csv"],
        "weight_grid": [ctx.out_dir / "02_weight_grid_candidates.csv"],
        "strict_weights": [ctx.out_dir / "03_strict_weight_fold_results.csv", ctx.out_dir / "04_strict_weight_summary.csv"],
        "ml_grid": [ctx.out_dir / "05_ml_config_grid.csv"],
        "strict_ml": [ctx.out_dir / "06_strict_ml_fold_results.csv", ctx.out_dir / "07_strict_ml_summary.csv"],
        "timesplit": [ctx.out_dir / "08_time_split_results.csv"],
        "optimistic": [ctx.out_dir / "09_optimistic_beta_guided_results.csv"],
        "selection": [ctx.out_dir / "10_model_selection_summary.csv", ctx.out_dir / "13_confidence_coverage_summary.csv"],
        "final": [html_path, ctx.out_dir / "14_stage_research_journal.md", ctx.out_dir / "15_stage_progress.csv"],
    }
    return mapping[stage_key]


def stage_is_complete(ctx: RunContext, stage_key: str) -> bool:
    if ctx.force:
        return False
    if stage_key == "final":
        return False
    paths = stage_output_paths(ctx, stage_key)
    required = [path for path in paths if path.name != "13_confidence_coverage_summary.csv"]
    return all(path.exists() for path in required)


def format_metric_note_from_frame(frame: pd.DataFrame, subset: str | None = None) -> str:
    if frame.empty or "summary_scope" not in frame.columns:
        return "No metric rows available yet."
    data = frame.loc[frame["summary_scope"].eq("pooled")].copy()
    if subset is not None and "subset" in data.columns:
        data = data.loc[data["subset"].eq(subset)].copy()
    if data.empty:
        return "No pooled metric row available for the target subset."
    sort_cols = [col for col in ["f1", "precision", "recall"] if col in data.columns]
    data = data.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    row = data.iloc[0]
    model = " / ".join(
        str(row.get(col, ""))
        for col in ["regime", "model_family", "feature_set", "model_variant"]
        if str(row.get(col, "")) != ""
    )
    return (
        f"Best {row.get('subset', subset or 'pooled')} row: {model}; "
        f"P={float(row.get('precision', 0.0)):.4f}, "
        f"R={float(row.get('recall', 0.0)):.4f}, "
        f"F1={float(row.get('f1', 0.0)):.4f}, "
        f"support={int(row.get('support', 0))}."
    )


def stage_metric_note(ctx: RunContext, stage_key: str) -> str:
    if stage_key == "cache":
        cache = read_csv_if_exists(ctx.out_dir / "01_refreshed_daily_feature_cache.csv")
        if cache.empty:
            return "Feature cache is not available yet."
        beta = cache.loc[cache["dataset"].eq("beta")]
        return f"Cached {len(cache)} site-days; Beta={len(beta)}, Beta sure={int(beta['confidence'].eq(BETA_SURE).sum())}."
    if stage_key == "weight_grid":
        grid = read_csv_if_exists(ctx.out_dir / "02_weight_grid_candidates.csv")
        return f"Prepared {len(grid)} physical weight candidates."
    if stage_key == "strict_weights":
        return format_metric_note_from_frame(read_csv_if_exists(ctx.out_dir / "04_strict_weight_summary.csv"), "beta_sure_only")
    if stage_key == "ml_grid":
        grid = read_csv_if_exists(ctx.out_dir / "05_ml_config_grid.csv")
        if grid.empty or "model_family" not in grid.columns:
            return "ML config grid is not available yet."
        counts = grid["model_family"].value_counts().sort_index()
        return "Prepared ML configs: " + ", ".join(f"{family}={count}" for family, count in counts.items()) + "."
    if stage_key == "strict_ml":
        return format_metric_note_from_frame(read_csv_if_exists(ctx.out_dir / "07_strict_ml_summary.csv"), "beta_sure_only")
    if stage_key == "timesplit":
        return format_metric_note_from_frame(read_csv_if_exists(ctx.out_dir / "08_time_split_results.csv"), "beta_test_sure_only")
    if stage_key == "optimistic":
        return format_metric_note_from_frame(read_csv_if_exists(ctx.out_dir / "09_optimistic_beta_guided_results.csv"), "beta_sure_only")
    if stage_key == "selection":
        summary = read_csv_if_exists(ctx.out_dir / "10_model_selection_summary.csv")
        strict = summary.loc[summary["stage"].astype(str).str.startswith("strict")] if not summary.empty and "stage" in summary.columns else pd.DataFrame()
        return format_metric_note_from_frame(strict, "beta_sure_only")
    if stage_key == "final":
        html_path = SCRIPT_DIR / "2026-07-08_overnight_weight_ml_experiment_summary.html"
        return f"Summary HTML exists={html_path.exists()}; progress rows={len(read_csv_if_exists(ctx.out_dir / '15_stage_progress.csv'))}."
    return "No stage note available."


def append_stage_progress(
    ctx: RunContext,
    *,
    stage_key: str,
    status: str,
    started_at: str,
    finished_at: str,
    elapsed_seconds: float,
    outputs: list[Path],
    best_metric_note: str,
    error: str = "",
) -> None:
    ensure_progress_log(ctx)
    number, name = STAGE_DEFS[stage_key]
    path = ctx.out_dir / "15_stage_progress.csv"
    prior = read_csv_if_exists(path)
    row = {
        "timestamp": now_stamp(),
        "mode": ctx.mode,
        "stage_number": number,
        "stage_key": stage_key,
        "stage_name": name,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "outputs": "; ".join(rel(path) for path in outputs),
        "best_metric_note": best_metric_note,
        "error": error,
    }
    row_frame = pd.DataFrame([row])
    write_csv(path, row_frame if prior.empty else pd.concat([prior, row_frame], ignore_index=True))


def append_stage_journal(
    ctx: RunContext,
    *,
    stage_key: str,
    status: str,
    started_at: str,
    finished_at: str,
    elapsed_seconds: float,
    outputs: list[Path],
    best_metric_note: str,
    error: str = "",
) -> None:
    number, name = STAGE_DEFS[stage_key]
    next_stage = next((value for value in STAGE_DEFS.values() if value[0] == number + 1), None)
    next_text = f"Stage {number + 1}/{STAGE_COUNT}: {next_stage[1]}" if next_stage else "Run complete"
    existing_outputs = [path for path in outputs if path.exists()]
    lines = [
        f"## {finished_at} - Stage {number}/{STAGE_COUNT}: {name}",
        "",
        f"- Status: `{status}`",
        f"- Started: `{started_at}`",
        f"- Finished: `{finished_at}`",
        f"- Elapsed seconds: `{elapsed_seconds:.3f}`",
        f"- Outputs present: `{len(existing_outputs)}/{len(outputs)}`",
        f"- Key note: {best_metric_note}",
        f"- Next: {next_text}",
    ]
    if error:
        lines.append(f"- Error: `{error}`")
    if existing_outputs:
        lines.append("- Output files:")
        lines.extend(f"  - `{rel(path)}`" for path in existing_outputs)
    lines.append("")
    path = ctx.out_dir / "14_stage_research_journal.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        header = [
            "# Overnight Weight/ML Experiment Stage Journal",
            "",
            "This journal is appended automatically after each stage completes, skips, or fails.",
            "",
        ]
        path.write_text("\n".join(header), encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def record_stage_event(
    ctx: RunContext,
    *,
    stage_key: str,
    status: str,
    started_at: str,
    started_time: float,
    error: str = "",
) -> None:
    finished_at = now_stamp()
    elapsed = time.time() - started_time
    outputs = stage_output_paths(ctx, stage_key)
    note = stage_metric_note(ctx, stage_key)
    append_stage_progress(
        ctx,
        stage_key=stage_key,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed,
        outputs=outputs,
        best_metric_note=note,
        error=error,
    )
    append_stage_journal(
        ctx,
        stage_key=stage_key,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed,
        outputs=outputs,
        best_metric_note=note,
        error=error,
    )


def stage_complete(path: Path, *, force: bool) -> bool:
    return path.exists() and not force


def clean_numeric(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def load_day_truth(dataset: str) -> pd.DataFrame:
    path = FINAL_DATASET_DIR / f"dataset_{dataset}.parquet"
    truth = pd.read_parquet(path, columns=["substation_id", "date", "label_day"])
    truth["substation_id"] = truth["substation_id"].astype(str)
    truth["date"] = date_key(truth["date"])
    truth["true_day"] = safe_bool(truth["label_day"])
    return (
        truth.groupby(["substation_id", "date"], as_index=False)
        .agg(true_day=("true_day", "max"))
        .assign(dataset=dataset)
        [["dataset", "substation_id", "date", "true_day"]]
    )


def load_beta_confidence() -> pd.DataFrame:
    conf = pd.read_csv(base.REVIEWER_B_PATH, usecols=["substation_id", "date", "confidence"])
    conf["substation_id"] = conf["substation_id"].astype(str).str.replace("act_", "beta_", regex=False)
    conf["date"] = date_key(conf["date"])
    conf["confidence"] = conf["confidence"].fillna("missing").astype(str).str.strip().str.lower()
    conf = conf.drop_duplicates(["substation_id", "date"], keep="last")
    return conf


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dt = pd.to_datetime(out["date"], errors="coerce")
    month = dt.dt.month.fillna(1).astype(int)
    out["month"] = month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["is_weekend"] = dt.dt.dayofweek.fillna(0).isin([5, 6]).astype(int)
    season = np.select(
        [
            month.isin([12, 1, 2]),
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8]),
            month.isin([9, 10, 11]),
        ],
        ["summer", "autumn", "winter", "spring"],
        default="unknown",
    )
    out["season"] = season
    for name in ["autumn", "spring", "summer", "winter"]:
        out[f"season_{name}"] = (out["season"].eq(name)).astype(int)
    return out


def load_or_build_refreshed_cache(ctx: RunContext) -> pd.DataFrame:
    out_path = ctx.out_dir / "01_refreshed_daily_feature_cache.csv"
    if stage_complete(out_path, force=ctx.force):
        return pd.read_csv(out_path)

    source_path = base.OUT_ROOT / "C1_cached_daily_features/02_c1_daily_feature_cache.csv"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing source C1 cache: {source_path}. Run 13_260707_physical_score_loso_experiments.py --chunk C1 first."
        )
    cache = pd.read_csv(source_path)
    cache["dataset"] = cache["dataset"].astype(str)
    cache["substation_id"] = cache["substation_id"].astype(str)
    cache["date"] = date_key(cache["date"])
    cache = cache.drop(columns=[col for col in ["true_day", "confidence"] if col in cache.columns])

    truth = pd.concat([load_day_truth("alpha"), load_day_truth("beta")], ignore_index=True)
    merged = cache.merge(truth, on=["dataset", "substation_id", "date"], how="inner", validate="one_to_one")
    if len(merged) != len(cache):
        raise ValueError(f"Refreshed truth row mismatch: cache={len(cache)}, merged={len(merged)}")

    beta_conf = load_beta_confidence()
    merged = merged.merge(beta_conf, on=["substation_id", "date"], how="left")
    merged["confidence"] = np.where(
        merged["dataset"].eq("beta"),
        merged["confidence"].fillna("missing").astype(str).str.strip().str.lower(),
        "not_applicable",
    )
    merged = add_calendar_features(merged)
    merged = clean_numeric(merged, FEATURE_COLUMNS + CALENDAR_COLUMNS)
    merged["true_day"] = safe_bool(merged["true_day"])
    write_csv(out_path, merged)
    return merged


def compute_metrics(true_values: pd.Series | np.ndarray, pred_values: pd.Series | np.ndarray) -> dict[str, float | int]:
    true = pd.Series(true_values).fillna(False).astype(bool).to_numpy()
    pred = pd.Series(pred_values).fillna(False).astype(bool).to_numpy()
    tp = int((true & pred).sum())
    fp = int((~true & pred).sum())
    fn = int((true & ~pred).sum())
    tn = int((~true & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
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


def metric_rows(
    frame: pd.DataFrame,
    *,
    stage: str,
    regime: str,
    model_family: str,
    model_variant: str,
    feature_set: str,
    fold: str,
    subset: str,
    pred_col: str = "pred_day",
    threshold: float = np.nan,
    notes: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    base_meta = {
        "stage": stage,
        "regime": regime,
        "model_family": model_family,
        "model_variant": model_variant,
        "feature_set": feature_set,
        "fold": fold,
        "dataset": str(frame["dataset"].iloc[0]) if "dataset" in frame.columns and frame["dataset"].nunique() == 1 else "mixed",
        "subset": subset,
        "threshold": threshold,
        "notes": notes,
    }
    rows.append({**base_meta, "summary_scope": "pooled", "substation_id": "", **compute_metrics(frame["true_day"], frame[pred_col])})

    site_metrics = []
    for site, site_frame in frame.groupby("substation_id", sort=True):
        m = compute_metrics(site_frame["true_day"], site_frame[pred_col])
        rows.append({**base_meta, "summary_scope": "site", "substation_id": site, **m})
        site_metrics.append({"substation_id": site, **m})
    site_df = pd.DataFrame(site_metrics)
    if not site_df.empty:
        rows.append(
            {
                **base_meta,
                "summary_scope": "macro_site_average",
                "substation_id": "",
                "support": int(site_df["support"].sum()),
                "positive_support": int(site_df["positive_support"].sum()),
                "tp": int(site_df["tp"].sum()),
                "fp": int(site_df["fp"].sum()),
                "fn": int(site_df["fn"].sum()),
                "tn": int(site_df["tn"].sum()),
                "precision": float(site_df["precision"].mean()),
                "recall": float(site_df["recall"].mean()),
                "f1": float(site_df["f1"].mean()),
            }
        )
        positive_sites = site_df.loc[site_df["positive_support"] > 0].copy()
        if not positive_sites.empty:
            rows.append(
                {
                    **base_meta,
                    "summary_scope": "positive_site_macro_average",
                    "substation_id": "",
                    "support": int(positive_sites["support"].sum()),
                    "positive_support": int(positive_sites["positive_support"].sum()),
                    "tp": int(positive_sites["tp"].sum()),
                    "fp": int(positive_sites["fp"].sum()),
                    "fn": int(positive_sites["fn"].sum()),
                    "tn": int(positive_sites["tn"].sum()),
                    "precision": float(positive_sites["precision"].mean()),
                    "recall": float(positive_sites["recall"].mean()),
                    "f1": float(positive_sites["f1"].mean()),
                }
            )
    return rows


def beta_subsets(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("beta_all", frame.copy()),
        ("beta_sure_only", frame.loc[frame["confidence"].eq(BETA_SURE)].copy()),
    ]


def select_threshold_macro(frame: pd.DataFrame, *, score_col: str, dataset_balanced: bool = False) -> tuple[float, dict[str, Any]]:
    data = frame[["dataset", "substation_id", "true_day", score_col]].dropna().copy()
    if data.empty:
        return float("inf"), {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    _, selected, _ = base.select_threshold_weighted_macro_site(data, score_col, dataset_balanced=dataset_balanced)
    return float(selected["threshold"]), selected


def score_weight(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    x = frame[FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=float)
    w = np.array([weights.get(col, 0.0) for col in FEATURE_COLUMNS], dtype=float)
    return x @ w


def weight_config_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:04d}"


def generate_weight_grid(limit: int) -> pd.DataFrame:
    bounds = {
        "F1_bridge_improvement": (0.0, 3.0),
        "F2_roughness_improvement": (0.0, 3.0),
        "F3_slope_continuity_improvement": (0.0, 2.0),
        "F4_duration_plausibility": (0.0, 3.0),
        "F5_n_height_ratio": (0.0, 3.0),
        "F6_solar_strength_ratio": (0.0, 2.0),
        "F7_solar_peak_alignment": (0.0, 2.0),
        "F8_site_centered_core_score": (0.0, 1.5),
        "F9_site_rank_core_score": (0.0, 1.5),
    }
    c14_best = {
        "F1_bridge_improvement": 1.0,
        "F2_roughness_improvement": 1.5,
        "F3_slope_continuity_improvement": 1.0,
        "F4_duration_plausibility": 1.0,
        "F5_n_height_ratio": 1.0,
        "F6_solar_strength_ratio": 1.0,
        "F7_solar_peak_alignment": 1.0,
        "F8_site_centered_core_score": 0.5,
        "F9_site_rank_core_score": 0.0,
    }
    rows: list[dict[str, Any]] = []

    def add_row(config_id: str, source: str, weights: dict[str, float]) -> None:
        rounded = {col: round(float(weights.get(col, 0.0)), 4) for col in FEATURE_COLUMNS}
        rows.append({"config_id": config_id, "source": source, **{f"weight_{col}": rounded[col] for col in FEATURE_COLUMNS}})

    add_row("W_anchor_all_equal", "anchor", {col: 1.0 for col in FEATURE_COLUMNS})
    add_row("W_anchor_c14_best", "anchor", c14_best)
    for idx, col in enumerate(FEATURE_COLUMNS, start=1):
        weights = {feature: 1.0 for feature in FEATURE_COLUMNS}
        weights[col] = 0.0
        add_row(weight_config_id("W_drop_one", idx), "drop_one", weights)

    rng = np.random.default_rng(SEED)
    remaining = max(limit - len(rows), 0)
    uniform_n = remaining // 2
    local_n = remaining - uniform_n
    low = np.array([bounds[col][0] for col in FEATURE_COLUMNS], dtype=float)
    high = np.array([bounds[col][1] for col in FEATURE_COLUMNS], dtype=float)
    center = np.array([c14_best[col] for col in FEATURE_COLUMNS], dtype=float)
    span = high - low
    for i, values in enumerate(rng.uniform(low, high, size=(uniform_n, len(FEATURE_COLUMNS))), start=1):
        add_row(weight_config_id("W_uniform", i), "uniform_random", dict(zip(FEATURE_COLUMNS, values)))
    for i, values in enumerate(center + rng.normal(0.0, span * 0.18, size=(local_n, len(FEATURE_COLUMNS))), start=1):
        add_row(weight_config_id("W_local", i), "local_random", dict(zip(FEATURE_COLUMNS, np.clip(values, low, high))))

    grid = pd.DataFrame(rows)
    weight_cols = [f"weight_{col}" for col in FEATURE_COLUMNS]
    grid = grid.drop_duplicates(weight_cols, keep="first").reset_index(drop=True)
    return grid.head(limit).copy()


def weights_from_row(row: pd.Series) -> dict[str, float]:
    return {col: float(row[f"weight_{col}"]) for col in FEATURE_COLUMNS}


def build_weight_grid(ctx: RunContext) -> pd.DataFrame:
    path = ctx.out_dir / "02_weight_grid_candidates.csv"
    if stage_complete(path, force=ctx.force):
        return pd.read_csv(path)
    grid = generate_weight_grid(ctx.weight_limit)
    write_csv(path, grid)
    return grid


def metric_value(frame: pd.DataFrame, metric: str = "f1") -> float:
    if frame.empty:
        return 0.0
    return float(compute_metrics(frame["true_day"], frame["pred_day"])[metric])


def choose_best_result(results: pd.DataFrame) -> pd.Series:
    sort_cols = ["f1", "precision", "recall", "config_id"]
    ascending = [False, False, False, True]
    return results.sort_values(sort_cols, ascending=ascending).iloc[0]


def select_weight_by_inner_beta_cv(
    *,
    daily: pd.DataFrame,
    weight_grid: pd.DataFrame,
    heldout_site: str,
    regime: str,
) -> tuple[pd.Series, pd.DataFrame]:
    alpha = daily.loc[daily["dataset"].eq("alpha")].copy()
    beta = daily.loc[daily["dataset"].eq("beta")].copy()
    train_sites = [site for site in sorted(beta["substation_id"].unique()) if site != heldout_site]
    rows: list[dict[str, Any]] = []
    for _, config in weight_grid.iterrows():
        weights = weights_from_row(config)
        cv_parts: list[pd.DataFrame] = []
        for inner_site in train_sites:
            beta_inner_train = beta.loc[
                beta["confidence"].eq(BETA_SURE)
                & ~beta["substation_id"].isin([heldout_site, inner_site])
            ].copy()
            if regime == REGIME_R2:
                train = pd.concat([alpha, beta_inner_train], ignore_index=True)
                dataset_balanced = True
            else:
                train = beta_inner_train
                dataset_balanced = False
            val = beta.loc[beta["substation_id"].eq(inner_site) & beta["confidence"].eq(BETA_SURE)].copy()
            if train.empty or val.empty:
                continue
            train["score"] = score_weight(train, weights)
            val["score"] = score_weight(val, weights)
            threshold, _ = select_threshold_macro(train, score_col="score", dataset_balanced=dataset_balanced)
            val["pred_day"] = val["score"] >= threshold
            cv_parts.append(val)
        cv_pred = pd.concat(cv_parts, ignore_index=True) if cv_parts else pd.DataFrame()
        metrics = compute_metrics(cv_pred["true_day"], cv_pred["pred_day"]) if not cv_pred.empty else compute_metrics([], [])
        rows.append(
            {
                "stage": "strict_weight_inner_cv",
                "regime": regime,
                "model_family": "physical_weight_grid",
                "model_variant": "weight_search_candidate",
                "feature_set": "phys9",
                "fold": heldout_site,
                "config_id": config["config_id"],
                "config_source": config["source"],
                "subset": "inner_beta_sure_cv",
                "summary_scope": "pooled",
                "substation_id": "",
                **metrics,
            }
        )
    result = pd.DataFrame(rows)
    return choose_best_result(result), result


def select_weight_by_alpha_loso(
    *,
    daily: pd.DataFrame,
    weight_grid: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    alpha = daily.loc[daily["dataset"].eq("alpha")].copy()
    sites = sorted(alpha["substation_id"].unique())
    rows: list[dict[str, Any]] = []
    for _, config in weight_grid.iterrows():
        weights = weights_from_row(config)
        parts: list[pd.DataFrame] = []
        for inner_site in sites:
            train = alpha.loc[~alpha["substation_id"].eq(inner_site)].copy()
            val = alpha.loc[alpha["substation_id"].eq(inner_site)].copy()
            train["score"] = score_weight(train, weights)
            val["score"] = score_weight(val, weights)
            threshold, _ = select_threshold_macro(train, score_col="score", dataset_balanced=False)
            val["pred_day"] = val["score"] >= threshold
            parts.append(val)
        cv_pred = pd.concat(parts, ignore_index=True)
        metrics = compute_metrics(cv_pred["true_day"], cv_pred["pred_day"])
        rows.append(
            {
                "stage": "strict_weight_alpha_loso_cv",
                "regime": REGIME_R3,
                "model_family": "physical_weight_grid",
                "model_variant": "weight_search_candidate",
                "feature_set": "phys9",
                "fold": "alpha_loso",
                "config_id": config["config_id"],
                "config_source": config["source"],
                "subset": "alpha_loso_cv",
                "summary_scope": "pooled",
                "substation_id": "",
                **metrics,
            }
        )
    result = pd.DataFrame(rows)
    return choose_best_result(result), result


def run_strict_weight_stage(ctx: RunContext) -> None:
    fold_path = ctx.out_dir / "03_strict_weight_fold_results.csv"
    summary_path = ctx.out_dir / "04_strict_weight_summary.csv"
    audit_path = ctx.out_dir / "11_prediction_audit.csv"
    cv_batch_path = ctx.out_dir / "03a_strict_weight_cv_batches.csv"
    pred_partial_path = ctx.out_dir / "03b_strict_weight_prediction_parts.csv"
    if stage_complete(summary_path, force=ctx.force):
        print(f"Skipping strict weight stage, found {rel(summary_path)}")
        return
    if ctx.force:
        for path in [cv_batch_path, pred_partial_path, fold_path, summary_path, audit_path]:
            if path.exists():
                path.unlink()
    started = time.time()
    daily = load_or_build_refreshed_cache(ctx)
    grid = build_weight_grid(ctx)
    alpha = daily.loc[daily["dataset"].eq("alpha")].copy()
    beta = daily.loc[daily["dataset"].eq("beta")].copy()
    beta_sites = ctx.beta_folds or sorted(beta["substation_id"].unique())
    batches = [
        (batch_id, grid.iloc[start : start + WEIGHT_BATCH_SIZE].copy())
        for batch_id, start in enumerate(range(0, len(grid), WEIGHT_BATCH_SIZE), start=1)
    ]
    total_work_units = len([REGIME_R1, REGIME_R2]) * len(beta_sites) * len(batches) + len(batches)

    def existing_keys(path: Path, cols: list[str]) -> set[tuple[Any, ...]]:
        frame = read_csv_if_exists(path)
        if frame.empty or not set(cols).issubset(frame.columns):
            return set()
        return set(map(tuple, frame[cols].drop_duplicates().to_numpy()))

    completed_cv = existing_keys(cv_batch_path, ["regime", "fold", "batch_id"])
    completed_pred = existing_keys(pred_partial_path, ["regime", "fold"])
    work_unit = 0
    for regime in [REGIME_R1, REGIME_R2]:
        for heldout_site in beta_sites:
            for batch_id, batch in batches:
                work_unit += 1
                key = (regime, heldout_site, batch_id)
                if key in completed_cv:
                    continue
                print(
                    f"  strict weight batch {work_unit}/{total_work_units}: "
                    f"{regime} {heldout_site} batch {batch_id}/{len(batches)} ({len(batch)} configs)"
                )
                _, cv_results = select_weight_by_inner_beta_cv(
                    daily=daily,
                    weight_grid=batch,
                    heldout_site=heldout_site,
                    regime=regime,
                )
                cv_results["batch_id"] = batch_id
                append_csv(cv_batch_path, cv_results)
                completed_cv.add(key)
            if (regime, heldout_site) in completed_pred:
                continue
            cv_all = read_csv_if_exists(cv_batch_path)
            pair_results = cv_all.loc[cv_all["regime"].eq(regime) & cv_all["fold"].eq(heldout_site)].copy()
            best = choose_best_result(pair_results)
            config = grid.loc[grid["config_id"].eq(best["config_id"])].iloc[0]
            weights = weights_from_row(config)
            beta_train = beta.loc[
                beta["confidence"].eq(BETA_SURE) & ~beta["substation_id"].eq(heldout_site)
            ].copy()
            if regime == REGIME_R2:
                train = pd.concat([alpha, beta_train], ignore_index=True)
                dataset_balanced = True
                train_subset = "all_alpha_plus_other_beta_sure"
            else:
                train = beta_train
                dataset_balanced = False
                train_subset = "other_beta_sure"
            train["score"] = score_weight(train, weights)
            threshold, selected = select_threshold_macro(train, score_col="score", dataset_balanced=dataset_balanced)
            eval_frame = beta.loc[beta["substation_id"].eq(heldout_site)].copy()
            eval_frame["score"] = score_weight(eval_frame, weights)
            eval_frame["pred_day"] = eval_frame["score"] >= threshold
            eval_frame["threshold"] = threshold
            eval_frame["confidence_score"] = (eval_frame["score"] - threshold).abs()
            eval_frame["stage"] = "strict_weight"
            eval_frame["regime"] = regime
            eval_frame["model_family"] = "physical_weight_grid"
            eval_frame["model_variant"] = "selected_weight_grid"
            eval_frame["feature_set"] = "phys9"
            eval_frame["fold"] = heldout_site
            eval_frame["config_id"] = config["config_id"]
            eval_frame["training_subset"] = train_subset
            eval_frame["selection_f1"] = float(best["f1"])
            eval_frame["threshold_selection_f1"] = float(selected.get("weighted_macro_f1", selected.get("macro_f1", np.nan)))
            append_csv(pred_partial_path, eval_frame)
            completed_pred.add((regime, heldout_site))

    for batch_id, batch in batches:
        work_unit += 1
        key = (REGIME_R3, "alpha_loso", batch_id)
        if key in completed_cv:
            continue
        print(
            f"  strict weight batch {work_unit}/{total_work_units}: "
            f"{REGIME_R3} alpha_loso batch {batch_id}/{len(batches)} ({len(batch)} configs)"
        )
        _, r3_batch_results = select_weight_by_alpha_loso(daily=daily, weight_grid=batch)
        r3_batch_results["batch_id"] = batch_id
        append_csv(cv_batch_path, r3_batch_results)
        completed_cv.add(key)
    cv_all = read_csv_if_exists(cv_batch_path)
    r3_results = cv_all.loc[cv_all["regime"].eq(REGIME_R3)].copy()
    best_r3 = choose_best_result(r3_results)
    r3_config = grid.loc[grid["config_id"].eq(best_r3["config_id"])].iloc[0]
    r3_weights = weights_from_row(r3_config)
    if (REGIME_R3, "all_beta") not in completed_pred:
        alpha_train = alpha.copy()
        alpha_train["score"] = score_weight(alpha_train, r3_weights)
        r3_threshold, r3_selected = select_threshold_macro(alpha_train, score_col="score", dataset_balanced=False)
        r3_eval = beta.copy()
        r3_eval["score"] = score_weight(r3_eval, r3_weights)
        r3_eval["pred_day"] = r3_eval["score"] >= r3_threshold
        r3_eval["threshold"] = r3_threshold
        r3_eval["confidence_score"] = (r3_eval["score"] - r3_threshold).abs()
        r3_eval["stage"] = "strict_weight"
        r3_eval["regime"] = REGIME_R3
        r3_eval["model_family"] = "physical_weight_grid"
        r3_eval["model_variant"] = "selected_weight_grid"
        r3_eval["feature_set"] = "phys9"
        r3_eval["fold"] = "all_beta"
        r3_eval["config_id"] = r3_config["config_id"]
        r3_eval["training_subset"] = "all_alpha"
        r3_eval["selection_f1"] = float(best_r3["f1"])
        r3_eval["threshold_selection_f1"] = float(r3_selected.get("macro_f1", np.nan))
        append_csv(pred_partial_path, r3_eval)

    fold_results = read_csv_if_exists(cv_batch_path)
    predictions = read_csv_if_exists(pred_partial_path)
    summary_rows: list[dict[str, Any]] = []
    for (regime, model_variant), group in predictions.groupby(["regime", "model_variant"], sort=True):
        for subset, subset_frame in beta_subsets(group):
            summary_rows.extend(
                metric_rows(
                    subset_frame,
                    stage="strict_weight",
                    regime=regime,
                    model_family="physical_weight_grid",
                    model_variant=model_variant,
                    feature_set="phys9",
                    fold="combined",
                    subset=subset,
                    threshold=np.nan,
                    notes="Selected by leakage-safe inner validation.",
                )
            )
    summary = pd.DataFrame(summary_rows)
    keep_cols = [
        "stage",
        "regime",
        "model_family",
        "model_variant",
        "feature_set",
        "fold",
        "config_id",
        "training_subset",
        "dataset",
        "substation_id",
        "date",
        "confidence",
        "true_day",
        "score",
        "threshold",
        "pred_day",
        "confidence_score",
        "selection_f1",
        "threshold_selection_f1",
    ]
    write_csv(fold_path, fold_results)
    write_csv(summary_path, summary)
    write_csv(audit_path, predictions[[col for col in keep_cols if col in predictions.columns]])
    append_manifest(ctx, "strict_weight", started, {"configs": len(grid), "folds": len(beta_sites), "rows": len(predictions)})
    print_summary(summary, "Strict weight summary")


def sample_weights(frame: pd.DataFrame, *, dataset_balanced: bool) -> np.ndarray:
    data = frame[["dataset", "substation_id", "true_day"]].reset_index(drop=True).copy()
    data["group"] = data["dataset"].astype(str) + "|" + data["substation_id"].astype(str)
    weights = np.zeros(len(data), dtype=float)
    if dataset_balanced and data["dataset"].nunique() > 1:
        dataset_values = sorted(data["dataset"].unique())
        dataset_weight = {dataset: 1.0 / len(dataset_values) for dataset in dataset_values}
        for dataset, ds_frame in data.groupby("dataset", sort=True):
            groups = ds_frame["group"].unique()
            for group in groups:
                idx = data.index[data["group"].eq(group)].to_numpy()
                weights[idx] = dataset_weight[dataset] / len(groups) / len(idx)
    else:
        groups = data["group"].unique()
        for group in groups:
            idx = data.index[data["group"].eq(group)].to_numpy()
            weights[idx] = 1.0 / len(groups) / len(idx)
    y = data["true_day"].astype(bool).to_numpy()
    pos = max(int(y.sum()), 1)
    neg = max(int((~y).sum()), 1)
    class_weight = np.where(y, len(y) / (2 * pos), len(y) / (2 * neg))
    weights = weights * class_weight
    return weights / max(weights.mean(), 1e-12)


def generate_ml_configs(limit_per_family: int | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    rf_configs = [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf, "max_features": mf}
        for n in [200, 400]
        for d in [3, 5, 8, None]
        for leaf in [2, 5]
        for mf in ["sqrt", 0.7]
    ]
    xgb_configs = [
        {
            "n_estimators": n,
            "max_depth": d,
            "learning_rate": lr,
            "subsample": ss,
            "colsample_bytree": cs,
            "min_child_weight": mcw,
            "reg_lambda": reg,
        }
        for n in [100, 200, 350]
        for d in [2, 3, 4]
        for lr in [0.02, 0.05, 0.1]
        for ss in [0.8, 1.0]
        for cs in [0.8, 1.0]
        for mcw in [1, 5]
        for reg in [1.0, 3.0]
    ]
    mlp_configs = [
        {"hidden_layer_sizes": h, "alpha": a, "learning_rate_init": lr, "activation": act}
        for h in [(8,), (16,), (16, 8), (32, 16)]
        for a in [0.0001, 0.001, 0.01]
        for lr in [0.0005, 0.001]
        for act in ["relu", "tanh"]
    ]

    rng = np.random.default_rng(SEED)
    for family, configs in [
        ("rf", rf_configs),
        ("xgb", xgb_configs),
        ("mlp", mlp_configs),
    ]:
        if limit_per_family is None:
            chosen = configs
            if family == "xgb":
                chosen = [configs[i] for i in rng.choice(len(configs), size=min(24, len(configs)), replace=False)]
            if family == "rf":
                chosen = [configs[i] for i in rng.choice(len(configs), size=min(18, len(configs)), replace=False)]
            if family == "mlp":
                chosen = [configs[i] for i in rng.choice(len(configs), size=min(12, len(configs)), replace=False)]
        else:
            chosen = configs[:limit_per_family]
        for idx, params in enumerate(chosen, start=1):
            rows.append(
                {
                    "config_id": f"{family.upper()}_{idx:03d}",
                    "model_family": family,
                    "params_json": json.dumps(params, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def build_ml_config_grid(ctx: RunContext) -> pd.DataFrame:
    path = ctx.out_dir / "05_ml_config_grid.csv"
    if stage_complete(path, force=ctx.force):
        return pd.read_csv(path)
    configs = generate_ml_configs(ctx.ml_limit_per_family)
    write_csv(path, configs)
    return configs


def balanced_mlp_training_frame(train: pd.DataFrame, seed: int) -> pd.DataFrame:
    y = train["true_day"].astype(bool)
    pos = train.loc[y].copy()
    neg = train.loc[~y].copy()
    if pos.empty or neg.empty:
        return train.copy()
    target = max(len(pos), min(len(neg), len(pos) * 3))
    rng = np.random.default_rng(seed)
    pos_sample = pos.iloc[rng.choice(len(pos), size=target, replace=True)]
    neg_sample = neg.iloc[rng.choice(len(neg), size=target, replace=False if target <= len(neg) else True)]
    return pd.concat([pos_sample, neg_sample], ignore_index=True).sample(frac=1, random_state=seed)


def fit_predict_ml(
    *,
    family: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    predict_frame: pd.DataFrame,
    feature_cols: list[str],
    dataset_balanced: bool,
    seed: int,
) -> np.ndarray:
    if train["true_day"].nunique() < 2:
        return np.full(len(predict_frame), float(train["true_day"].mean()) if len(train) else 0.0)
    x_pred = predict_frame[feature_cols].fillna(0.0).to_numpy(dtype=float)
    if family == "rf":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=seed,
            n_jobs=-1,
        )
        x = train[feature_cols].fillna(0.0).to_numpy(dtype=float)
        y = train["true_day"].astype(bool).to_numpy(dtype=int)
        model.fit(x, y, sample_weight=sample_weights(train, dataset_balanced=dataset_balanced))
        return model.predict_proba(x_pred)[:, 1]
    if family == "xgb":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            min_child_weight=float(params["min_child_weight"]),
            reg_lambda=float(params["reg_lambda"]),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=max(1, (os.cpu_count() or 2) - 1),
        )
        x = train[feature_cols].fillna(0.0).to_numpy(dtype=float)
        y = train["true_day"].astype(bool).to_numpy(dtype=int)
        model.fit(x, y, sample_weight=sample_weights(train, dataset_balanced=dataset_balanced), verbose=False)
        return model.predict_proba(x_pred)[:, 1]
    if family == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        train_balanced = balanced_mlp_training_frame(train, seed)
        x = train_balanced[feature_cols].fillna(0.0).to_numpy(dtype=float)
        y = train_balanced["true_day"].astype(bool).to_numpy(dtype=int)
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
                alpha=float(params["alpha"]),
                learning_rate_init=float(params["learning_rate_init"]),
                activation=str(params["activation"]),
                max_iter=350,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=15,
                random_state=seed,
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x, y)
        return model.predict_proba(x_pred)[:, 1]
    raise ValueError(f"Unknown ML family: {family}")


def select_ml_by_inner_beta_cv(
    *,
    daily: pd.DataFrame,
    configs: pd.DataFrame,
    feature_set: str,
    heldout_site: str,
    regime: str,
) -> tuple[pd.Series, pd.DataFrame]:
    feature_cols = FEATURE_SETS[feature_set]
    alpha = daily.loc[daily["dataset"].eq("alpha")].copy()
    beta = daily.loc[daily["dataset"].eq("beta")].copy()
    train_sites = [site for site in sorted(beta["substation_id"].unique()) if site != heldout_site]
    rows: list[dict[str, Any]] = []
    for _, config in configs.iterrows():
        family = str(config["model_family"])
        params = json.loads(config["params_json"])
        cv_parts: list[pd.DataFrame] = []
        for inner_site in train_sites:
            beta_inner_train = beta.loc[
                beta["confidence"].eq(BETA_SURE)
                & ~beta["substation_id"].isin([heldout_site, inner_site])
            ].copy()
            if regime == REGIME_R2:
                train = pd.concat([alpha, beta_inner_train], ignore_index=True)
                dataset_balanced = True
            else:
                train = beta_inner_train
                dataset_balanced = False
            val = beta.loc[beta["substation_id"].eq(inner_site) & beta["confidence"].eq(BETA_SURE)].copy()
            if train.empty or val.empty:
                continue
            val["score"] = fit_predict_ml(
                family=family,
                params=params,
                train=train,
                predict_frame=val,
                feature_cols=feature_cols,
                dataset_balanced=dataset_balanced,
                seed=SEED,
            )
            cv_parts.append(val)
        cv_pred = pd.concat(cv_parts, ignore_index=True) if cv_parts else pd.DataFrame()
        if cv_pred.empty:
            threshold = float("inf")
            metrics = compute_metrics([], [])
        else:
            threshold, _ = select_threshold_macro(cv_pred, score_col="score", dataset_balanced=False)
            cv_pred["pred_day"] = cv_pred["score"] >= threshold
            metrics = compute_metrics(cv_pred["true_day"], cv_pred["pred_day"])
        rows.append(
            {
                "stage": "strict_ml_inner_cv",
                "regime": regime,
                "model_family": family,
                "model_variant": "ml_config_candidate",
                "feature_set": feature_set,
                "fold": heldout_site,
                "config_id": config["config_id"],
                "subset": "inner_beta_sure_cv",
                "summary_scope": "pooled",
                "substation_id": "",
                "threshold": threshold,
                **metrics,
            }
        )
    result = pd.DataFrame(rows)
    return choose_best_result(result), result


def select_ml_by_alpha_loso(
    *,
    daily: pd.DataFrame,
    configs: pd.DataFrame,
    feature_set: str,
) -> tuple[pd.Series, pd.DataFrame]:
    feature_cols = FEATURE_SETS[feature_set]
    alpha = daily.loc[daily["dataset"].eq("alpha")].copy()
    sites = sorted(alpha["substation_id"].unique())
    rows: list[dict[str, Any]] = []
    for _, config in configs.iterrows():
        family = str(config["model_family"])
        params = json.loads(config["params_json"])
        parts: list[pd.DataFrame] = []
        for inner_site in sites:
            train = alpha.loc[~alpha["substation_id"].eq(inner_site)].copy()
            val = alpha.loc[alpha["substation_id"].eq(inner_site)].copy()
            val["score"] = fit_predict_ml(
                family=family,
                params=params,
                train=train,
                predict_frame=val,
                feature_cols=feature_cols,
                dataset_balanced=False,
                seed=SEED,
            )
            parts.append(val)
        cv_pred = pd.concat(parts, ignore_index=True)
        threshold, _ = select_threshold_macro(cv_pred, score_col="score", dataset_balanced=False)
        cv_pred["pred_day"] = cv_pred["score"] >= threshold
        rows.append(
            {
                "stage": "strict_ml_alpha_loso_cv",
                "regime": REGIME_R3,
                "model_family": family,
                "model_variant": "ml_config_candidate",
                "feature_set": feature_set,
                "fold": "alpha_loso",
                "config_id": config["config_id"],
                "subset": "alpha_loso_cv",
                "summary_scope": "pooled",
                "substation_id": "",
                "threshold": threshold,
                **compute_metrics(cv_pred["true_day"], cv_pred["pred_day"]),
            }
        )
    result = pd.DataFrame(rows)
    return choose_best_result(result), result


def run_strict_ml_stage(ctx: RunContext) -> None:
    fold_path = ctx.out_dir / "06_strict_ml_fold_results.csv"
    summary_path = ctx.out_dir / "07_strict_ml_summary.csv"
    audit_path = ctx.out_dir / "11_prediction_audit.csv"
    cv_config_path = ctx.out_dir / "06a_strict_ml_cv_configs.csv"
    pred_partial_path = ctx.out_dir / "06b_strict_ml_prediction_parts.csv"
    if stage_complete(summary_path, force=ctx.force):
        print(f"Skipping strict ML stage, found {rel(summary_path)}")
        return
    if ctx.force:
        for path in [cv_config_path, pred_partial_path, fold_path, summary_path]:
            if path.exists():
                path.unlink()
    started = time.time()
    daily = load_or_build_refreshed_cache(ctx)
    configs = build_ml_config_grid(ctx)
    alpha = daily.loc[daily["dataset"].eq("alpha")].copy()
    beta = daily.loc[daily["dataset"].eq("beta")].copy()
    beta_sites = ctx.beta_folds or sorted(beta["substation_id"].unique())

    def existing_keys(path: Path, cols: list[str]) -> set[tuple[Any, ...]]:
        frame = read_csv_if_exists(path)
        if frame.empty or not set(cols).issubset(frame.columns):
            return set()
        return set(map(tuple, frame[cols].drop_duplicates().to_numpy()))

    completed_cv = existing_keys(cv_config_path, ["feature_set", "model_family", "regime", "fold", "config_id"])
    completed_pred = existing_keys(pred_partial_path, ["feature_set", "model_family", "regime", "fold"])
    total_work_units = 0
    for feature_set in ctx.feature_sets:
        for family in sorted(configs["model_family"].unique()):
            n_configs = len(configs.loc[configs["model_family"].eq(family)])
            total_work_units += n_configs * (2 * len(beta_sites) + 1)
    work_unit = 0

    for feature_set in ctx.feature_sets:
        feature_cols = FEATURE_SETS[feature_set]
        for family in sorted(configs["model_family"].unique()):
            family_configs = configs.loc[configs["model_family"].eq(family)].copy()
            for regime in [REGIME_R1, REGIME_R2]:
                for heldout_site in beta_sites:
                    for _, candidate in family_configs.iterrows():
                        work_unit += 1
                        key = (feature_set, family, regime, heldout_site, candidate["config_id"])
                        if key in completed_cv:
                            continue
                        print(
                            f"  strict ML config {work_unit}/{total_work_units}: "
                            f"{feature_set} {family} {regime} {heldout_site} {candidate['config_id']}"
                        )
                        _, cv_results = select_ml_by_inner_beta_cv(
                            daily=daily,
                            configs=pd.DataFrame([candidate]),
                            feature_set=feature_set,
                            heldout_site=heldout_site,
                            regime=regime,
                        )
                        append_csv(cv_config_path, cv_results)
                        completed_cv.add(key)
                    if (feature_set, family, regime, heldout_site) in completed_pred:
                        continue
                    cv_all = read_csv_if_exists(cv_config_path)
                    pair_results = cv_all.loc[
                        cv_all["feature_set"].eq(feature_set)
                        & cv_all["model_family"].eq(family)
                        & cv_all["regime"].eq(regime)
                        & cv_all["fold"].eq(heldout_site)
                    ].copy()
                    best = choose_best_result(pair_results)
                    config = family_configs.loc[family_configs["config_id"].eq(best["config_id"])].iloc[0]
                    params = json.loads(config["params_json"])
                    beta_train = beta.loc[
                        beta["confidence"].eq(BETA_SURE) & ~beta["substation_id"].eq(heldout_site)
                    ].copy()
                    if regime == REGIME_R2:
                        train = pd.concat([alpha, beta_train], ignore_index=True)
                        dataset_balanced = True
                        train_subset = "all_alpha_plus_other_beta_sure"
                    else:
                        train = beta_train
                        dataset_balanced = False
                        train_subset = "other_beta_sure"
                    eval_frame = beta.loc[beta["substation_id"].eq(heldout_site)].copy()
                    eval_frame["score"] = fit_predict_ml(
                        family=family,
                        params=params,
                        train=train,
                        predict_frame=eval_frame,
                        feature_cols=feature_cols,
                        dataset_balanced=dataset_balanced,
                        seed=SEED,
                    )
                    threshold = float(best["threshold"])
                    eval_frame["pred_day"] = eval_frame["score"] >= threshold
                    eval_frame["threshold"] = threshold
                    eval_frame["confidence_score"] = (eval_frame["score"] - threshold).abs()
                    eval_frame["stage"] = "strict_ml"
                    eval_frame["regime"] = regime
                    eval_frame["model_family"] = family
                    eval_frame["model_variant"] = "selected_ml_config"
                    eval_frame["feature_set"] = feature_set
                    eval_frame["fold"] = heldout_site
                    eval_frame["config_id"] = config["config_id"]
                    eval_frame["training_subset"] = train_subset
                    eval_frame["selection_f1"] = float(best["f1"])
                    append_csv(pred_partial_path, eval_frame)
                    completed_pred.add((feature_set, family, regime, heldout_site))

            for _, candidate in family_configs.iterrows():
                work_unit += 1
                key = (feature_set, family, REGIME_R3, "alpha_loso", candidate["config_id"])
                if key in completed_cv:
                    continue
                print(
                    f"  strict ML config {work_unit}/{total_work_units}: "
                    f"{feature_set} {family} {REGIME_R3} alpha_loso {candidate['config_id']}"
                )
                _, r3_results = select_ml_by_alpha_loso(
                    daily=daily,
                    configs=pd.DataFrame([candidate]),
                    feature_set=feature_set,
                )
                append_csv(cv_config_path, r3_results)
                completed_cv.add(key)
            if (feature_set, family, REGIME_R3, "all_beta") in completed_pred:
                continue
            cv_all = read_csv_if_exists(cv_config_path)
            family_r3_results = cv_all.loc[
                cv_all["feature_set"].eq(feature_set)
                & cv_all["model_family"].eq(family)
                & cv_all["regime"].eq(REGIME_R3)
            ].copy()
            best_r3 = choose_best_result(family_r3_results)
            r3_config = family_configs.loc[family_configs["config_id"].eq(best_r3["config_id"])].iloc[0]
            r3_params = json.loads(r3_config["params_json"])
            r3_eval = beta.copy()
            r3_eval["score"] = fit_predict_ml(
                family=family,
                params=r3_params,
                train=alpha,
                predict_frame=r3_eval,
                feature_cols=feature_cols,
                dataset_balanced=False,
                seed=SEED,
            )
            r3_threshold = float(best_r3["threshold"])
            r3_eval["pred_day"] = r3_eval["score"] >= r3_threshold
            r3_eval["threshold"] = r3_threshold
            r3_eval["confidence_score"] = (r3_eval["score"] - r3_threshold).abs()
            r3_eval["stage"] = "strict_ml"
            r3_eval["regime"] = REGIME_R3
            r3_eval["model_family"] = family
            r3_eval["model_variant"] = "selected_ml_config"
            r3_eval["feature_set"] = feature_set
            r3_eval["fold"] = "all_beta"
            r3_eval["config_id"] = r3_config["config_id"]
            r3_eval["training_subset"] = "all_alpha"
            r3_eval["selection_f1"] = float(best_r3["f1"])
            append_csv(pred_partial_path, r3_eval)
            completed_pred.add((feature_set, family, REGIME_R3, "all_beta"))

    fold_results = read_csv_if_exists(cv_config_path)
    predictions = read_csv_if_exists(pred_partial_path)
    summary_rows: list[dict[str, Any]] = []
    for (regime, family, feature_set), group in predictions.groupby(["regime", "model_family", "feature_set"], sort=True):
        for subset, subset_frame in beta_subsets(group):
            summary_rows.extend(
                metric_rows(
                    subset_frame,
                    stage="strict_ml",
                    regime=regime,
                    model_family=family,
                    model_variant="selected_ml_config",
                    feature_set=feature_set,
                    fold="combined",
                    subset=subset,
                    threshold=np.nan,
                    notes="Selected by leakage-safe inner validation.",
                )
            )
    summary = pd.DataFrame(summary_rows)
    keep_cols = [
        "stage",
        "regime",
        "model_family",
        "model_variant",
        "feature_set",
        "fold",
        "config_id",
        "training_subset",
        "dataset",
        "substation_id",
        "date",
        "confidence",
        "true_day",
        "score",
        "threshold",
        "pred_day",
        "confidence_score",
        "selection_f1",
    ]
    write_csv(fold_path, fold_results)
    write_csv(summary_path, summary)
    previous = read_csv_if_exists(audit_path)
    if not previous.empty and "stage" in previous.columns:
        previous = previous.loc[~previous["stage"].eq("strict_ml")].copy()
    audit = pd.concat([previous, predictions[[col for col in keep_cols if col in predictions.columns]]], ignore_index=True)
    audit = audit.drop_duplicates(["stage", "regime", "model_family", "feature_set", "fold", "substation_id", "date"], keep="last")
    write_csv(audit_path, audit)
    append_manifest(ctx, "strict_ml", started, {"ml_configs": len(configs), "folds": len(beta_sites), "rows": len(predictions)})
    print_summary(summary, "Strict ML summary")


def time_split_frames(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = daily.copy()
    dt = pd.to_datetime(frame["date"], errors="coerce")
    train = frame.loc[dt < pd.Timestamp(VALIDATION_START)].copy()
    val = frame.loc[(dt >= pd.Timestamp(VALIDATION_START)) & (dt < pd.Timestamp(TEST_START))].copy()
    test = frame.loc[(dt >= pd.Timestamp(TEST_START)) & (dt <= pd.Timestamp(TEST_END))].copy()
    train = train.loc[train["dataset"].eq("alpha") | train["confidence"].eq(BETA_SURE)].copy()
    val = val.loc[val["dataset"].eq("alpha") | val["confidence"].eq(BETA_SURE)].copy()
    return train, val, test


def run_time_split_stage(ctx: RunContext) -> None:
    path = ctx.out_dir / "08_time_split_results.csv"
    if stage_complete(path, force=ctx.force):
        print(f"Skipping time split stage, found {rel(path)}")
        return
    started = time.time()
    daily = load_or_build_refreshed_cache(ctx)
    train, val, test = time_split_frames(daily)
    weight_grid = build_weight_grid(ctx)
    ml_configs = build_ml_config_grid(ctx)
    metric_out: list[dict[str, Any]] = []
    pred_parts: list[pd.DataFrame] = []

    # Weight configs.
    weight_selection_rows = []
    for _, config in weight_grid.iterrows():
        weights = weights_from_row(config)
        val_scored = val.copy()
        val_scored["score"] = score_weight(val_scored, weights)
        threshold, _ = select_threshold_macro(val_scored, score_col="score", dataset_balanced=True)
        val_scored["pred_day"] = val_scored["score"] >= threshold
        m = compute_metrics(val_scored["true_day"], val_scored["pred_day"])
        weight_selection_rows.append({"config_id": config["config_id"], "threshold": threshold, **m})
    weight_selection = pd.DataFrame(weight_selection_rows)
    if not weight_selection.empty:
        best = choose_best_result(weight_selection)
        config = weight_grid.loc[weight_grid["config_id"].eq(best["config_id"])].iloc[0]
        weights = weights_from_row(config)
        eval_frame = test.copy()
        eval_frame["score"] = score_weight(eval_frame, weights)
        eval_frame["threshold"] = float(best["threshold"])
        eval_frame["pred_day"] = eval_frame["score"] >= eval_frame["threshold"]
        eval_frame["confidence_score"] = (eval_frame["score"] - eval_frame["threshold"]).abs()
        eval_frame["model_family"] = "physical_weight_grid"
        eval_frame["model_variant"] = "time_split_selected_weight"
        eval_frame["feature_set"] = "phys9"
        eval_frame["config_id"] = config["config_id"]
        pred_parts.append(eval_frame)

    # ML configs.
    for feature_set in ctx.feature_sets:
        feature_cols = FEATURE_SETS[feature_set]
        for family in sorted(ml_configs["model_family"].unique()):
            family_configs = ml_configs.loc[ml_configs["model_family"].eq(family)].copy()
            selection_rows = []
            val_scores_by_config: dict[str, tuple[float, dict[str, Any]]] = {}
            for _, config in family_configs.iterrows():
                params = json.loads(config["params_json"])
                val_scored = val.copy()
                val_scored["score"] = fit_predict_ml(
                    family=family,
                    params=params,
                    train=train,
                    predict_frame=val_scored,
                    feature_cols=feature_cols,
                    dataset_balanced=True,
                    seed=SEED,
                )
                threshold, _ = select_threshold_macro(val_scored, score_col="score", dataset_balanced=True)
                val_scored["pred_day"] = val_scored["score"] >= threshold
                m = compute_metrics(val_scored["true_day"], val_scored["pred_day"])
                selection_rows.append({"config_id": config["config_id"], "threshold": threshold, **m})
                val_scores_by_config[str(config["config_id"])] = (threshold, params)
            selection = pd.DataFrame(selection_rows)
            if selection.empty:
                continue
            best = choose_best_result(selection)
            threshold, params = val_scores_by_config[str(best["config_id"])]
            eval_frame = test.copy()
            eval_frame["score"] = fit_predict_ml(
                family=family,
                params=params,
                train=train,
                predict_frame=eval_frame,
                feature_cols=feature_cols,
                dataset_balanced=True,
                seed=SEED,
            )
            eval_frame["threshold"] = threshold
            eval_frame["pred_day"] = eval_frame["score"] >= threshold
            eval_frame["confidence_score"] = (eval_frame["score"] - threshold).abs()
            eval_frame["model_family"] = family
            eval_frame["model_variant"] = "time_split_selected_ml"
            eval_frame["feature_set"] = feature_set
            eval_frame["config_id"] = best["config_id"]
            pred_parts.append(eval_frame)

    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    for (family, variant, feature_set), group in predictions.groupby(["model_family", "model_variant", "feature_set"], sort=True):
        for dataset, dataset_frame in group.groupby("dataset", sort=True):
            subset_name = f"{dataset}_test_all"
            metric_out.extend(
                metric_rows(
                    dataset_frame,
                    stage="time_split",
                    regime="time_split_last_3_months",
                    model_family=family,
                    model_variant=variant,
                    feature_set=feature_set,
                    fold="2024-07-01_to_2024-09-30",
                    subset=subset_name,
                    threshold=np.nan,
                    notes="Train before June 2024; tune June 2024; test July-September 2024.",
                )
            )
            if dataset == "beta":
                sure = dataset_frame.loc[dataset_frame["confidence"].eq(BETA_SURE)].copy()
                metric_out.extend(
                    metric_rows(
                        sure,
                        stage="time_split",
                        regime="time_split_last_3_months",
                        model_family=family,
                        model_variant=variant,
                        feature_set=feature_set,
                        fold="2024-07-01_to_2024-09-30",
                        subset="beta_test_sure_only",
                        threshold=np.nan,
                        notes="Train before June 2024; tune June 2024; test July-September 2024.",
                    )
                )
    results = pd.DataFrame(metric_out)
    write_csv(path, results)
    if not predictions.empty:
        write_csv(ctx.out_dir / "08_time_split_prediction_audit.csv", predictions)
    append_manifest(ctx, "time_split", started, {"rows": len(results), "predictions": len(predictions)})
    print_summary(results, "Time split summary")


def run_optimistic_stage(ctx: RunContext) -> None:
    path = ctx.out_dir / "09_optimistic_beta_guided_results.csv"
    if stage_complete(path, force=ctx.force):
        print(f"Skipping optimistic stage, found {rel(path)}")
        return
    started = time.time()
    daily = load_or_build_refreshed_cache(ctx)
    beta = daily.loc[daily["dataset"].eq("beta")].copy()
    beta_sure = beta.loc[beta["confidence"].eq(BETA_SURE)].copy()
    weight_grid = build_weight_grid(ctx)
    ml_configs = build_ml_config_grid(ctx)
    metric_out: list[dict[str, Any]] = []

    best_rows = []
    for _, config in weight_grid.iterrows():
        weights = weights_from_row(config)
        scored = beta_sure.copy()
        scored["score"] = score_weight(scored, weights)
        threshold, _ = select_threshold_macro(scored, score_col="score", dataset_balanced=False)
        scored["pred_day"] = scored["score"] >= threshold
        best_rows.append({"config_id": config["config_id"], "threshold": threshold, **compute_metrics(scored["true_day"], scored["pred_day"])})
    best_table = pd.DataFrame(best_rows)
    if not best_table.empty:
        best = choose_best_result(best_table)
        weights = weights_from_row(weight_grid.loc[weight_grid["config_id"].eq(best["config_id"])].iloc[0])
        pred = beta.copy()
        pred["score"] = score_weight(pred, weights)
        pred["threshold"] = float(best["threshold"])
        pred["pred_day"] = pred["score"] >= pred["threshold"]
        for subset, subset_frame in beta_subsets(pred):
            metric_out.extend(
                metric_rows(
                    subset_frame,
                    stage="optimistic_beta_guided",
                    regime="all_beta_sure_tuned",
                    model_family="physical_weight_grid",
                    model_variant="optimistic_selected_weight",
                    feature_set="phys9",
                    fold="all_beta",
                    subset=subset,
                    threshold=float(best["threshold"]),
                    notes="Optimistic: Beta sure labels used for model/threshold selection.",
                )
            )

    for feature_set in ctx.feature_sets:
        feature_cols = FEATURE_SETS[feature_set]
        for family in sorted(ml_configs["model_family"].unique()):
            family_configs = ml_configs.loc[ml_configs["model_family"].eq(family)].copy()
            selection_rows = []
            scored_by_config: dict[str, tuple[float, dict[str, Any]]] = {}
            for _, config in family_configs.iterrows():
                params = json.loads(config["params_json"])
                scored = beta_sure.copy()
                scored["score"] = fit_predict_ml(
                    family=family,
                    params=params,
                    train=beta_sure,
                    predict_frame=scored,
                    feature_cols=feature_cols,
                    dataset_balanced=False,
                    seed=SEED,
                )
                threshold, _ = select_threshold_macro(scored, score_col="score", dataset_balanced=False)
                scored["pred_day"] = scored["score"] >= threshold
                selection_rows.append({"config_id": config["config_id"], "threshold": threshold, **compute_metrics(scored["true_day"], scored["pred_day"])})
                scored_by_config[str(config["config_id"])] = (threshold, params)
            selection = pd.DataFrame(selection_rows)
            if selection.empty:
                continue
            best = choose_best_result(selection)
            threshold, params = scored_by_config[str(best["config_id"])]
            pred = beta.copy()
            pred["score"] = fit_predict_ml(
                family=family,
                params=params,
                train=beta_sure,
                predict_frame=pred,
                feature_cols=feature_cols,
                dataset_balanced=False,
                seed=SEED,
            )
            pred["threshold"] = threshold
            pred["pred_day"] = pred["score"] >= threshold
            for subset, subset_frame in beta_subsets(pred):
                metric_out.extend(
                    metric_rows(
                        subset_frame,
                        stage="optimistic_beta_guided",
                        regime="all_beta_sure_tuned",
                        model_family=family,
                        model_variant="optimistic_selected_ml",
                        feature_set=feature_set,
                        fold="all_beta",
                        subset=subset,
                        threshold=threshold,
                        notes="Optimistic: Beta sure labels used for model/threshold selection.",
                    )
                )
    results = pd.DataFrame(metric_out)
    write_csv(path, results)
    append_manifest(ctx, "optimistic_beta_guided", started, {"rows": len(results)})
    print_summary(results, "Optimistic beta-guided summary")


def confidence_coverage(predictions: pd.DataFrame, meta_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(meta_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group = group.copy().sort_values("confidence_score", ascending=False)
        for coverage in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            n = max(1, int(round(len(group) * coverage)))
            head = group.head(n)
            rows.append(
                {
                    **dict(zip(meta_cols, keys)),
                    "coverage_pct": int(round(coverage * 100)),
                    **compute_metrics(head["true_day"], head["pred_day"]),
                }
            )
    return pd.DataFrame(rows)


def run_model_selection_summary(ctx: RunContext) -> None:
    started = time.time()
    rows: list[pd.DataFrame] = []
    for rel_path in [
        "04_strict_weight_summary.csv",
        "07_strict_ml_summary.csv",
        "08_time_split_results.csv",
        "09_optimistic_beta_guided_results.csv",
    ]:
        path = ctx.out_dir / rel_path
        if path.exists():
            rows.append(pd.read_csv(path))
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if combined.empty:
        write_csv(ctx.out_dir / "10_model_selection_summary.csv", combined)
        return
    pooled = combined.loc[combined["summary_scope"].eq("pooled")].copy()
    rank = pooled.sort_values(
        ["subset", "f1", "precision", "recall"],
        ascending=[True, False, False, False],
    )
    write_csv(ctx.out_dir / "10_model_selection_summary.csv", rank)

    audit = read_csv_if_exists(ctx.out_dir / "11_prediction_audit.csv")
    if not audit.empty and "confidence_score" in audit.columns:
        coverage = confidence_coverage(
            audit.loc[audit["confidence"].eq(BETA_SURE)].copy(),
            ["stage", "regime", "model_family", "model_variant", "feature_set"],
        )
        write_csv(ctx.out_dir / "13_confidence_coverage_summary.csv", coverage)
    append_manifest(ctx, "model_selection_summary", started, {"rows": len(rank)})


def print_summary(frame: pd.DataFrame, title: str) -> None:
    if frame.empty:
        print(f"\n{title}: no rows")
        return
    view = frame.loc[frame["summary_scope"].eq("pooled")].copy()
    if "subset" in view.columns:
        view = view.loc[view["subset"].astype(str).str.contains("sure|all|test", regex=True, case=False)]
    cols = [
        col
        for col in ["stage", "regime", "model_family", "model_variant", "feature_set", "subset", "precision", "recall", "f1", "support", "positive_support"]
        if col in view.columns
    ]
    print(f"\n{title}")
    if cols:
        print(view[cols].sort_values("f1", ascending=False).head(20).round(4).to_string(index=False))


def append_manifest(ctx: RunContext, stage: str, started: float, extra: dict[str, Any]) -> None:
    path = ctx.out_dir / "00_run_manifest.csv"
    prior = read_csv_if_exists(path)
    row = {
        "timestamp": now_stamp(),
        "mode": ctx.mode,
        "stage": stage,
        "elapsed_seconds": round(time.time() - started, 3),
        "force": ctx.force,
        "output_dir": rel(ctx.out_dir),
        **extra,
    }
    frame = pd.concat([prior, pd.DataFrame([row])], ignore_index=True)
    write_csv(path, frame)


def html_table(frame: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
    if frame.empty:
        return "<p>No rows available.</p>"
    available = [col for col in cols if col in frame.columns]
    return frame[available].head(max_rows).round(4).to_html(index=False, classes="compact")


def generate_html_summary(ctx: RunContext) -> Path:
    out_path = SCRIPT_DIR / "2026-07-08_overnight_weight_ml_experiment_summary.html"
    summary = read_csv_if_exists(ctx.out_dir / "10_model_selection_summary.csv")
    strict_weight = read_csv_if_exists(ctx.out_dir / "04_strict_weight_summary.csv")
    strict_ml = read_csv_if_exists(ctx.out_dir / "07_strict_ml_summary.csv")
    time_split = read_csv_if_exists(ctx.out_dir / "08_time_split_results.csv")
    optimistic = read_csv_if_exists(ctx.out_dir / "09_optimistic_beta_guided_results.csv")
    manifest = read_csv_if_exists(ctx.out_dir / "00_run_manifest.csv")
    progress = read_csv_if_exists(ctx.out_dir / "15_stage_progress.csv")

    def best_beta_sure(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        f = frame.loc[frame["summary_scope"].eq("pooled") & frame["subset"].astype(str).str.contains("sure", case=False, regex=False)].copy()
        return f.sort_values(["f1", "precision", "recall"], ascending=[False, False, False])

    best = best_beta_sure(summary)
    css = """
    body{font-family:Arial,sans-serif;line-height:1.45;color:#22303d;margin:0;background:#f8f6f4}
    main{max-width:1180px;margin:auto;background:white;padding:32px 42px}
    h1,h2{color:#22303d} h2{border-top:2px solid #ebe3e3;padding-top:18px}
    .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:#ebe3e3;padding:14px;border-left:5px solid #eb932c}
    table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}th,td{border:1px solid #d8d8d8;padding:6px;text-align:left}th{background:#22303d;color:white}
    .note{background:#fff4e8;border-left:5px solid #eb932c;padding:12px}.small{font-size:12px;color:#5C7D99}
    """
    headline = best.head(1)
    card_html = ""
    if not headline.empty:
        row = headline.iloc[0]
        for label, value in [
            ("Best Beta sure F1", row.get("f1", np.nan)),
            ("Precision", row.get("precision", np.nan)),
            ("Recall", row.get("recall", np.nan)),
            ("Support", row.get("support", np.nan)),
        ]:
            card_html += f"<div class='card'><strong>{label}</strong><br><span style='font-size:26px'>{value:.4f}</span></div>" if isinstance(value, float) else f"<div class='card'><strong>{label}</strong><br><span style='font-size:26px'>{value}</span></div>"
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Overnight Weight And ML Experiment Summary</title><style>{css}</style></head>
<body><main>
<h1>Overnight Weight Optimisation And ML Experiment Summary</h1>
<p class="note"><strong>Status:</strong> This is a misc-folder model-development report. Strict validation and optimistic Beta-guided results are separated.</p>
<div class="cards">{card_html}</div>
<h2>Method</h2>
<p>The experiment reuses cached m9_pbm daily physical features, refreshes labels from final Alpha/Beta datasets, and compares larger physical-score weight grids with RF, XGB, and sklearn MLP classifiers. The primary strict metric is Beta LOSO sure-only day F1.</p>
<h2>Best Overall Rows</h2>
{html_table(summary.sort_values('f1', ascending=False) if not summary.empty else summary, ['stage','regime','model_family','model_variant','feature_set','subset','summary_scope','precision','recall','f1','support','positive_support'], 25)}
<h2>Strict Physical Weight Results</h2>
{html_table(best_beta_sure(strict_weight), ['regime','model_family','model_variant','feature_set','subset','summary_scope','precision','recall','f1','support','positive_support'], 20)}
<h2>Strict ML Results</h2>
{html_table(best_beta_sure(strict_ml), ['regime','model_family','model_variant','feature_set','subset','summary_scope','precision','recall','f1','support','positive_support'], 30)}
<h2>Time Split Results</h2>
{html_table(time_split.loc[time_split.get('summary_scope', pd.Series(dtype=str)).eq('pooled')].sort_values('f1', ascending=False) if not time_split.empty else time_split, ['regime','model_family','model_variant','feature_set','subset','precision','recall','f1','support','positive_support'], 30)}
<h2>Optimistic Beta-Guided Upper Bound</h2>
<p class="small">These rows are not validation. Beta sure labels are used for model and threshold selection.</p>
{html_table(best_beta_sure(optimistic), ['regime','model_family','model_variant','feature_set','subset','summary_scope','precision','recall','f1','support','positive_support'], 30)}
<h2>Run Manifest</h2>
{html_table(manifest, list(manifest.columns) if not manifest.empty else [], 50)}
<h2>Stage Progress</h2>
{html_table(progress.tail(30) if not progress.empty else progress, ['stage_number','stage_name','status','elapsed_seconds','best_metric_note','error'], 30)}
<h2>Key Insights Placeholder</h2>
<p>Use this HTML after the full overnight run to compare strict LOSO, time-split, and optimistic results. Promote only leakage-safe strict results into journal claims.</p>
</main></body></html>"""
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote HTML summary to {rel(out_path)}")
    return out_path


def run_smoke_checks(ctx: RunContext) -> None:
    cache = load_or_build_refreshed_cache(ctx)
    if not set(FEATURE_COLUMNS).issubset(cache.columns):
        raise AssertionError("Feature cache missing physical feature columns.")
    if cache.loc[cache["dataset"].eq("beta")].shape[0] != 2928:
        print("Warning: Beta day count is not 2928 after refresh.")
    output_paths = [
        ctx.out_dir / "00_run_manifest.csv",
        ctx.out_dir / "01_refreshed_daily_feature_cache.csv",
        ctx.out_dir / "02_weight_grid_candidates.csv",
        ctx.out_dir / "03_strict_weight_fold_results.csv",
        ctx.out_dir / "04_strict_weight_summary.csv",
        ctx.out_dir / "05_ml_config_grid.csv",
        ctx.out_dir / "06_strict_ml_fold_results.csv",
        ctx.out_dir / "07_strict_ml_summary.csv",
        ctx.out_dir / "08_time_split_results.csv",
        ctx.out_dir / "09_optimistic_beta_guided_results.csv",
        ctx.out_dir / "10_model_selection_summary.csv",
        ctx.out_dir / "11_prediction_audit.csv",
        ctx.out_dir / "12_failure_log.csv",
        ctx.out_dir / "14_stage_research_journal.md",
        ctx.out_dir / "15_stage_progress.csv",
    ]
    ensure_failure_log(ctx.out_dir / "12_failure_log.csv")
    existing = [path for path in output_paths if path.exists()]
    print(f"Smoke existing output files: {len(existing)}/{len(output_paths)}")


def build_context(args: argparse.Namespace) -> RunContext:
    out_dir = OUT_ROOT / args.mode
    if args.mode == "smoke":
        beta_folds = ["beta_A"]
        weight_limit = args.max_weight_configs or SMOKE_WEIGHT_CONFIGS
        ml_limit = args.max_ml_configs_per_family or SMOKE_ML_CONFIGS_PER_FAMILY
        feature_sets = ["phys9"]
    else:
        beta_folds = None
        weight_limit = args.max_weight_configs or FULL_WEIGHT_CONFIGS
        ml_limit = args.max_ml_configs_per_family
        feature_sets = ["phys9", "phys9_calendar"]
    out_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        mode=args.mode,
        out_dir=out_dir,
        force=args.force,
        weight_limit=weight_limit,
        ml_limit_per_family=ml_limit,
        feature_sets=feature_sets,
        beta_folds=beta_folds,
    )


def run_numbered_stage(ctx: RunContext, stage_key: str, action) -> None:
    number, name = STAGE_DEFS[stage_key]
    started_at = now_stamp()
    started_time = time.time()
    if stage_is_complete(ctx, stage_key):
        note = stage_metric_note(ctx, stage_key)
        print(f"Stage {number}/{STAGE_COUNT} skipped: {name} - output already exists. {note}")
        record_stage_event(
            ctx,
            stage_key=stage_key,
            status="skipped",
            started_at=started_at,
            started_time=started_time,
        )
        return
    print(f"Stage {number}/{STAGE_COUNT} started: {name}")
    try:
        action()
    except Exception as exc:
        error = repr(exc)
        print(f"Stage {number}/{STAGE_COUNT} failed: {name} - {error}")
        record_stage_event(
            ctx,
            stage_key=stage_key,
            status="failed",
            started_at=started_at,
            started_time=started_time,
            error=error,
        )
        raise
    note = stage_metric_note(ctx, stage_key)
    print(f"Stage {number}/{STAGE_COUNT} complete: {name}. {note}")
    record_stage_event(
        ctx,
        stage_key=stage_key,
        status="complete",
        started_at=started_at,
        started_time=started_time,
    )


def run_stage(ctx: RunContext, stage: str) -> None:
    if stage == "cache":
        def action() -> None:
            started = time.time()
            cache = load_or_build_refreshed_cache(ctx)
            append_manifest(ctx, "cache", started, {"rows": len(cache), "features": len(FEATURE_COLUMNS)})

        run_numbered_stage(ctx, "cache", action)
    elif stage == "weight_grid":
        def action() -> None:
            started = time.time()
            grid = build_weight_grid(ctx)
            append_manifest(ctx, "weight_grid", started, {"rows": len(grid)})

        run_numbered_stage(ctx, "weight_grid", action)
    elif stage == "weights":
        run_stage(ctx, "weight_grid")
        run_stage(ctx, "strict_weights")
    elif stage == "strict_weights":
        run_numbered_stage(ctx, "strict_weights", lambda: run_strict_weight_stage(ctx))
    elif stage == "ml_grid":
        def action() -> None:
            started = time.time()
            grid = build_ml_config_grid(ctx)
            counts = grid["model_family"].value_counts().to_dict() if "model_family" in grid.columns else {}
            append_manifest(ctx, "ml_grid", started, {"rows": len(grid), "counts": json.dumps(counts, sort_keys=True)})

        run_numbered_stage(ctx, "ml_grid", action)
    elif stage == "ml":
        run_stage(ctx, "ml_grid")
        run_stage(ctx, "strict_ml")
    elif stage == "strict_ml":
        run_numbered_stage(ctx, "strict_ml", lambda: run_strict_ml_stage(ctx))
    elif stage == "timesplit":
        run_numbered_stage(ctx, "timesplit", lambda: run_time_split_stage(ctx))
    elif stage == "optimistic":
        run_numbered_stage(ctx, "optimistic", lambda: run_optimistic_stage(ctx))
    elif stage == "selection":
        run_numbered_stage(ctx, "selection", lambda: run_model_selection_summary(ctx))
    elif stage == "html":
        def action() -> None:
            generate_html_summary(ctx)
            run_smoke_checks(ctx)

        run_numbered_stage(ctx, "final", action)
    elif stage == "checks":
        def action() -> None:
            run_smoke_checks(ctx)

        run_numbered_stage(ctx, "final", action)
    elif stage == "all":
        for item in ["cache", "weights", "ml", "timesplit", "optimistic", "selection", "html"]:
            run_stage(ctx, item)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight weight optimisation and ML day-level experiments.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "cache",
            "weight_grid",
            "weights",
            "strict_weights",
            "ml_grid",
            "ml",
            "strict_ml",
            "timesplit",
            "optimistic",
            "selection",
            "html",
            "checks",
        ],
        default="all",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite stage outputs instead of skipping existing files.")
    parser.add_argument("--max-weight-configs", type=int, default=None)
    parser.add_argument("--max-ml-configs-per-family", type=int, default=None)
    args = parser.parse_args()
    ctx = build_context(args)
    ensure_failure_log(ctx.out_dir / "12_failure_log.csv")
    started = time.time()
    failures: list[dict[str, Any]] = []
    try:
        run_stage(ctx, args.stage)
    except Exception as exc:
        failures.append({"timestamp": now_stamp(), "mode": ctx.mode, "stage": args.stage, "error": repr(exc)})
        failure_path = ctx.out_dir / "12_failure_log.csv"
        prior = read_csv_if_exists(failure_path)
        write_csv(failure_path, pd.concat([prior, pd.DataFrame(failures)], ignore_index=True))
        raise
    finally:
        write_json(
            ctx.out_dir / "00_latest_run.json",
            {
                "mode": ctx.mode,
                "stage": args.stage,
                "elapsed_seconds": round(time.time() - started, 3),
                "output_dir": rel(ctx.out_dir),
                "force": ctx.force,
            },
        )


if __name__ == "__main__":
    main()

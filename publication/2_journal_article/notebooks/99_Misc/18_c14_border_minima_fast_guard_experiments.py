from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

SLOTS_PER_DAY = 96
DAYTIME_START = 24
DAYTIME_END = 72
NEIGHBOUR_RADIUS_SLOTS = 4
RAW_BORDER_TOLERANCE_MW = 0.50
SCALED_BORDER_TOLERANCE_FRACTION = 0.10
SCALED_BORDER_TOLERANCE_FLOOR_MW = 0.25
OUTPUT_FOLDER = "C18_border_minima_fast_guard_experiments"

BASE_WEIGHTS = {
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
VARIANTS = [
    {
        "variant": "BG0_baseline_selected_window",
        "experiment": "baseline",
        "border_rule": "none",
        "soft_penalty_weight": 0.0,
    },
    {
        "variant": "BG1_soft_penalty_0p5",
        "experiment": "soft_penalty",
        "border_rule": "soft_scaled_badness",
        "soft_penalty_weight": 0.5,
    },
    {
        "variant": "BG2_soft_penalty_1p0",
        "experiment": "soft_penalty",
        "border_rule": "soft_scaled_badness",
        "soft_penalty_weight": 1.0,
    },
    {
        "variant": "BG3_soft_penalty_1p5",
        "experiment": "soft_penalty",
        "border_rule": "soft_scaled_badness",
        "soft_penalty_weight": 1.5,
    },
    {
        "variant": "BG4_hard_raw_0p5MW",
        "experiment": "hard_guard",
        "border_rule": "both_edges_within_0.5MW_of_local_min",
        "soft_penalty_weight": 0.0,
    },
    {
        "variant": "BG5_hard_scaled_0p10site",
        "experiment": "hard_guard",
        "border_rule": "both_edges_within_max_0.25MW_or_10pct_site_net_scale",
        "soft_penalty_weight": 0.0,
    },
]


def find_repo_root() -> Path:
    start = Path(__file__).resolve()
    marker = Path("publication/2_journal_article/dataset/final/dataset_beta.parquet")
    for candidate in [start.parent, *start.parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"Could not find repo root containing {marker}")


ROOT = find_repo_root()
JOURNAL = ROOT / "publication/2_journal_article"
MISC = JOURNAL / "notebooks/99_Misc"
FINAL_DATASET_DIR = JOURNAL / "dataset/final"
EXP_OUT = MISC / "outputs/260707_physical_score_loso_experiments"
C1_CACHE = EXP_OUT / "C1_cached_daily_features/02_c1_daily_feature_cache.csv"
OUT = EXP_OUT / OUTPUT_FOLDER


def load_exp13_module():
    path = MISC / "13_260707_physical_score_loso_experiments.py"
    spec = importlib.util.spec_from_file_location("exp13_physical_score", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helper script from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP13 = load_exp13_module()


def date_key(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.strftime("%Y-%m-%d")


def safe_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.fillna(False).astype(str).str.strip().str.lower().isin({"true", "t", "1", "yes", "y"})


def fill_series(values: np.ndarray, default: float = 0.0) -> np.ndarray:
    return (
        pd.Series(values, dtype="float64")
        .interpolate(limit_direction="both")
        .fillna(default)
        .to_numpy(dtype=float)
    )


def load_net_arrays_and_scales(dataset: str) -> tuple[dict[tuple[str, str], np.ndarray], pd.DataFrame]:
    df = pd.read_parquet(
        FINAL_DATASET_DIR / f"dataset_{dataset}.parquet",
        columns=["substation_id", "date", "timestamp", "net_load_MW"],
    )
    df["substation_id"] = df["substation_id"].astype(str)
    df["date"] = date_key(df["date"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    df["slot"] = df["timestamp"].dt.hour * 4 + df["timestamp"].dt.minute // 15
    arrays: dict[tuple[str, str], np.ndarray] = {}
    scale_rows: list[dict[str, object]] = []
    for (site, date), group in df.groupby(["substation_id", "date"], sort=True):
        group = group.sort_values("slot").drop_duplicates("slot", keep="last")
        net = np.full(SLOTS_PER_DAY, np.nan, dtype=float)
        valid = group.loc[group["slot"].between(0, SLOTS_PER_DAY - 1)]
        slots = valid["slot"].to_numpy(dtype=int)
        net[slots] = valid["net_load_MW"].to_numpy(dtype=float)
        net = fill_series(net, 0.0)
        arrays[(site, date)] = net
        scale_rows.append(
            {
                "dataset": dataset,
                "substation_id": site,
                "date": date,
                "net_abs_day_p95": float(np.nanpercentile(np.abs(net[DAYTIME_START : DAYTIME_END + 1]), 95)),
            }
        )
    scale = pd.DataFrame(scale_rows)
    scale["site_net_scale"] = scale.groupby("substation_id")["net_abs_day_p95"].transform("median").clip(lower=EXP13.EPS)
    return arrays, scale


def local_min_delta(net: np.ndarray, slot: float) -> float:
    if pd.isna(slot):
        return np.nan
    slot_i = int(slot)
    local = net[max(0, slot_i - NEIGHBOUR_RADIUS_SLOTS) : min(SLOTS_PER_DAY, slot_i + NEIGHBOUR_RADIUS_SLOTS + 1)]
    return float(net[slot_i] - np.nanmin(local))


def compute_base_score(frame: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for col, weight in BASE_WEIGHTS.items():
        if weight:
            score = score + weight * pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return score


def build_daily_feature_table() -> pd.DataFrame:
    daily = pd.read_csv(C1_CACHE)
    daily["date"] = date_key(daily["date"])
    daily["true_day"] = safe_bool(daily["true_day"])
    daily["confidence"] = daily["confidence"].fillna("missing").astype(str).str.strip().str.lower()
    daily["base_score"] = compute_base_score(daily)
    daily = daily.rename(
        columns={
            "v03_selected_left_slot": "selected_left_slot",
            "v03_selected_right_slot": "selected_right_slot",
            "v03_selected_duration_h": "selected_duration_h",
        }
    )

    parts: list[pd.DataFrame] = []
    for dataset in ["alpha", "beta"]:
        arrays, scale = load_net_arrays_and_scales(dataset)
        subset = daily.loc[daily["dataset"].eq(dataset)].copy()
        subset = subset.merge(
            scale[["dataset", "substation_id", "date", "site_net_scale"]],
            on=["dataset", "substation_id", "date"],
            how="left",
            validate="one_to_one",
        )
        left_deltas: list[float] = []
        right_deltas: list[float] = []
        for row in subset.itertuples(index=False):
            net = arrays[(row.substation_id, row.date)]
            left_deltas.append(local_min_delta(net, row.selected_left_slot))
            right_deltas.append(local_min_delta(net, row.selected_right_slot))
        subset["selected_left_delta_MW"] = left_deltas
        subset["selected_right_delta_MW"] = right_deltas
        subset["scaled_border_tolerance_MW"] = np.maximum(
            SCALED_BORDER_TOLERANCE_FLOOR_MW,
            SCALED_BORDER_TOLERANCE_FRACTION * subset["site_net_scale"].astype(float),
        )
        subset["border_badness_scaled"] = EXP13.clip01(
            (
                subset["selected_left_delta_MW"] / subset["scaled_border_tolerance_MW"]
                + subset["selected_right_delta_MW"] / subset["scaled_border_tolerance_MW"]
            )
            / 2
        )
        subset["both_edges_near_min_raw_0p5MW"] = (
            subset["selected_left_delta_MW"].le(RAW_BORDER_TOLERANCE_MW)
            & subset["selected_right_delta_MW"].le(RAW_BORDER_TOLERANCE_MW)
        )
        subset["both_edges_near_min_scaled"] = (
            subset["selected_left_delta_MW"].le(subset["scaled_border_tolerance_MW"])
            & subset["selected_right_delta_MW"].le(subset["scaled_border_tolerance_MW"])
        )
        parts.append(subset)
    return pd.concat(parts, ignore_index=True)


def apply_variant(frame: pd.DataFrame, variant: dict[str, object]) -> pd.DataFrame:
    out = frame.copy()
    out["variant"] = variant["variant"]
    out["experiment"] = variant["experiment"]
    out["border_rule"] = variant["border_rule"]
    if variant["border_rule"] == "none":
        out["score"] = out["base_score"]
        out["guard_pass"] = True
    elif variant["border_rule"] == "soft_scaled_badness":
        out["score"] = out["base_score"] - float(variant["soft_penalty_weight"]) * out["border_badness_scaled"]
        out["guard_pass"] = True
    elif variant["border_rule"] == "both_edges_within_0.5MW_of_local_min":
        out["score"] = out["base_score"]
        out["guard_pass"] = out["both_edges_near_min_raw_0p5MW"].astype(bool)
    elif variant["border_rule"] == "both_edges_within_max_0.25MW_or_10pct_site_net_scale":
        out["score"] = out["base_score"]
        out["guard_pass"] = out["both_edges_near_min_scaled"].astype(bool)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return out


def select_threshold(train: pd.DataFrame, *, dataset_balanced: bool) -> tuple[float, dict[str, object]]:
    data = train.copy()
    data["_score_for_threshold"] = data["score"].where(data["guard_pass"].astype(bool))
    if data["_score_for_threshold"].dropna().empty:
        return np.inf, {"weighted_macro_f1": 0.0, "pooled_f1": 0.0, "threshold": np.inf}
    threshold, selected, _ = EXP13.select_threshold_weighted_macro_site(
        data,
        "_score_for_threshold",
        dataset_balanced=dataset_balanced,
    )
    return threshold, selected


def evaluate(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alpha = selected.loc[selected["dataset"].eq("alpha")].copy()
    beta = selected.loc[selected["dataset"].eq("beta")].copy()
    beta_sites = sorted(beta["substation_id"].unique().tolist())
    threshold_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    for variant in [variant["variant"] for variant in VARIANTS]:
        alpha_v = alpha.loc[alpha["variant"].eq(variant)].copy()
        beta_v = beta.loc[beta["variant"].eq(variant)].copy()
        for regime in ["R1_beta_loso", "R2_beta_loso_plus_alpha"]:
            for heldout_site in beta_sites:
                beta_train = beta_v.loc[
                    beta_v["confidence"].eq("sure") & ~beta_v["substation_id"].eq(heldout_site)
                ].copy()
                if regime == "R1_beta_loso":
                    train = beta_train
                    dataset_balanced = False
                    training_subset = "other_7_beta_sites_sure_only"
                else:
                    train = pd.concat([alpha_v, beta_train], ignore_index=True)
                    dataset_balanced = True
                    training_subset = "all_alpha_plus_other_7_beta_sites_sure_only"
                threshold, info = select_threshold(train, dataset_balanced=dataset_balanced)
                eval_frame = beta_v.loc[beta_v["substation_id"].eq(heldout_site)].copy()
                eval_frame["regime"] = regime
                eval_frame["heldout_site"] = heldout_site
                eval_frame["threshold"] = threshold
                eval_frame["pred_day"] = eval_frame["guard_pass"].astype(bool) & eval_frame["score"].ge(threshold)
                prediction_parts.append(eval_frame)
                threshold_rows.append(
                    {
                        "regime": regime,
                        "variant": variant,
                        "heldout_site": heldout_site,
                        "training_subset": training_subset,
                        "dataset_balanced_threshold_selection": dataset_balanced,
                        "training_rows": len(train),
                        "training_positive_support": int(train["true_day"].sum()),
                        "threshold": threshold,
                        **{f"selected_{key}": value for key, value in info.items() if key != "threshold"},
                    }
                )
        threshold, info = select_threshold(alpha_v, dataset_balanced=False)
        eval_frame = beta_v.copy()
        eval_frame["regime"] = "R3_alpha_only_to_beta"
        eval_frame["heldout_site"] = "all_beta"
        eval_frame["threshold"] = threshold
        eval_frame["pred_day"] = eval_frame["guard_pass"].astype(bool) & eval_frame["score"].ge(threshold)
        prediction_parts.append(eval_frame)
        threshold_rows.append(
            {
                "regime": "R3_alpha_only_to_beta",
                "variant": variant,
                "heldout_site": "all_beta",
                "training_subset": "all_alpha",
                "dataset_balanced_threshold_selection": False,
                "training_rows": len(alpha_v),
                "training_positive_support": int(alpha_v["true_day"].sum()),
                "threshold": threshold,
                **{f"selected_{key}": value for key, value in info.items() if key != "threshold"},
            }
        )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metric_rows: list[dict[str, object]] = []
    for (regime, variant), group in predictions.groupby(["regime", "variant"], sort=True):
        for subset, frame in [
            ("beta_all", group),
            ("beta_sure_only", group.loc[group["confidence"].eq("sure")].copy()),
        ]:
            rows = EXP13.metric_rows_for_subset(
                frame,
                variant=variant,
                dataset="beta",
                subset=subset,
                pred_col="pred_day",
                threshold=float("nan"),
            )
            for row in rows:
                row["regime"] = regime
            metric_rows.extend(rows)
    metrics = pd.DataFrame(metric_rows)
    thresholds = pd.DataFrame(threshold_rows)
    variant_defs = pd.DataFrame(VARIANTS)

    def lookup(subset: str, scope: str, metric: str) -> pd.Series:
        return (
            metrics.loc[
                metrics["subset"].eq(subset) & metrics["summary_scope"].eq(scope),
                ["regime", "variant", metric],
            ]
            .set_index(["regime", "variant"])[metric]
        )

    ranking = thresholds[["regime", "variant"]].drop_duplicates().reset_index(drop=True)
    ranking = ranking.merge(variant_defs, on="variant", how="left")
    for subset, prefix in [("beta_all", "beta_all"), ("beta_sure_only", "beta_sure")]:
        for scope, scope_prefix in [
            ("pooled", "pooled"),
            ("macro_site_average", "site_avg"),
            ("positive_site_macro_average", "positive_site_avg"),
        ]:
            for metric in ["precision", "recall", "f1"]:
                series = lookup(subset, scope, metric)
                ranking[f"{prefix}_{scope_prefix}_{metric}"] = [
                    series.get((row.regime, row.variant), np.nan) for row in ranking.itertuples(index=False)
                ]
    ranking = ranking.sort_values(
        ["beta_sure_pooled_f1", "beta_sure_positive_site_avg_f1", "beta_all_pooled_f1"],
        ascending=[False, False, False],
    )
    return thresholds, metrics, ranking, predictions


def add_interval_predictions(beta_intervals: pd.DataFrame, pred_days: pd.DataFrame) -> pd.DataFrame:
    frame = pred_days[[
        "substation_id",
        "date",
        "pred_day",
        "selected_left_slot",
        "selected_right_slot",
        "score",
        "threshold",
    ]].copy()
    intervals = beta_intervals.merge(frame, on=["substation_id", "date"], how="inner", validate="many_to_one")
    intervals["pred_interval"] = (
        intervals["pred_day"].astype(bool)
        & intervals["selected_left_slot"].notna()
        & intervals["selected_right_slot"].notna()
        & intervals["slot"].ge(intervals["selected_left_slot"])
        & intervals["slot"].le(intervals["selected_right_slot"])
    )
    return intervals


def multi_metric_summary(predictions: pd.DataFrame, ranking: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    beta_intervals = EXP13.load_c12_final_beta_intervals()
    top_keys = ranking.head(10)[["regime", "variant"]].drop_duplicates()
    top_predictions = predictions.merge(top_keys, on=["regime", "variant"], how="inner")
    rows: list[dict[str, object]] = []
    site_rows: list[dict[str, object]] = []
    for (regime, variant), group in top_predictions.groupby(["regime", "variant"], sort=True):
        intervals = add_interval_predictions(beta_intervals, group)
        EXP13.c12_append_metric_rows(
            rows,
            site_rows,
            frame=intervals,
            model_family="border_minima_fast_guard",
            model_variant=variant,
            regime=regime,
            prediction_source=Path(__file__).name,
            notes="Fast post-selection border-minimum guard/penalty on current selected C14 window.",
        )
    return pd.DataFrame(rows), pd.DataFrame(site_rows)


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    base_daily = build_daily_feature_table()
    selected = pd.concat([apply_variant(base_daily, variant) for variant in VARIANTS], ignore_index=True)
    thresholds, metrics, ranking, predictions = evaluate(selected)
    multi_metrics, multi_site_metrics = multi_metric_summary(predictions, ranking)
    border_summary = (
        selected.groupby(["dataset", "variant", "experiment", "border_rule"], as_index=False)
        .agg(
            rows=("date", "count"),
            guard_pass_rate=("guard_pass", "mean"),
            median_border_badness=("border_badness_scaled", "median"),
            median_left_delta_MW=("selected_left_delta_MW", "median"),
            median_right_delta_MW=("selected_right_delta_MW", "median"),
        )
        .sort_values(["dataset", "variant"])
    )
    manifest = pd.DataFrame(
        [
            {
                "script": Path(__file__).name,
                "output_folder": OUTPUT_FOLDER,
                "variants": len(VARIANTS),
                "daily_rows": len(base_daily),
                "selected_rows": len(selected),
                "prediction_rows": len(predictions),
                "elapsed_seconds": time.time() - started,
                "note": "Fast screening only: applies border guard/penalty to already-selected C14 windows; does not rescan all candidates.",
                "outputs": str(OUT.relative_to(ROOT)),
            }
        ]
    )
    manifest.to_csv(OUT / "01_c18_manifest.csv", index=False)
    pd.DataFrame(VARIANTS).to_csv(OUT / "02_c18_variant_definitions.csv", index=False)
    border_summary.to_csv(OUT / "03_c18_border_guard_summary.csv", index=False)
    selected.to_csv(OUT / "04_c18_daily_selected_windows_with_border_metrics.csv", index=False)
    thresholds.to_csv(OUT / "05_c18_threshold_selection.csv", index=False)
    metrics.to_csv(OUT / "06_c18_day_level_metrics.csv", index=False)
    ranking.to_csv(OUT / "07_c18_regime_variant_ranking.csv", index=False)
    predictions.to_csv(OUT / "08_c18_beta_prediction_audit.csv", index=False)
    multi_metrics.to_csv(OUT / "09_c18_top_multi_metric_summary.csv", index=False)
    multi_site_metrics.to_csv(OUT / "10_c18_top_site_multi_metric_summary.csv", index=False)

    print(f"Wrote fast border-minima guard experiments to {OUT.relative_to(ROOT)}")
    print("\nManifest")
    print(manifest.round(3).to_string(index=False))
    print("\nTop day-level results")
    show_cols = [
        "regime",
        "variant",
        "experiment",
        "border_rule",
        "beta_all_pooled_precision",
        "beta_all_pooled_recall",
        "beta_all_pooled_f1",
        "beta_sure_pooled_precision",
        "beta_sure_pooled_recall",
        "beta_sure_pooled_f1",
        "beta_sure_positive_site_avg_f1",
    ]
    print(ranking[show_cols].head(20).round(4).to_string(index=False))
    if not multi_metrics.empty:
        pooled = multi_metrics.loc[
            multi_metrics["summary_scope"].eq("pooled")
            & multi_metrics["subset"].eq("beta_sure_only")
        ].sort_values(["day_f1", "interval_f1", "energy_f1"], ascending=[False, False, False])
        print("\nTop multi-metric Beta sure-only results")
        print(
            pooled[["regime", "model_variant", "day_f1", "interval_f1", "energy_f1", "energy_iou"]]
            .round(4)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()

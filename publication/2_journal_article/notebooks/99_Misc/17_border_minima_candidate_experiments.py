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
OUTPUT_FOLDER = "C18_border_minima_candidate_experiments"

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
        "variant": "B0_no_border_baseline_rescan",
        "experiment": "baseline_rescan",
        "border_rule": "none",
        "soft_penalty_weight": 0.0,
    },
    {
        "variant": "B1_soft_border_penalty_0p5",
        "experiment": "soft_penalty",
        "border_rule": "soft_scaled_badness",
        "soft_penalty_weight": 0.5,
    },
    {
        "variant": "B2_soft_border_penalty_1p0",
        "experiment": "soft_penalty",
        "border_rule": "soft_scaled_badness",
        "soft_penalty_weight": 1.0,
    },
    {
        "variant": "B3_soft_border_penalty_1p5",
        "experiment": "soft_penalty",
        "border_rule": "soft_scaled_badness",
        "soft_penalty_weight": 1.5,
    },
    {
        "variant": "B4_hard_raw_0p5MW",
        "experiment": "hard_filter",
        "border_rule": "both_edges_within_0.5MW_of_local_min",
        "soft_penalty_weight": 0.0,
    },
    {
        "variant": "B5_hard_scaled_0p10site",
        "experiment": "hard_filter",
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


def load_dataset_day_arrays(dataset: str) -> tuple[pd.DataFrame, dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]]:
    columns = ["substation_id", "date", "timestamp", "net_load_MW", "solar_MW", "label_day"]
    if dataset == "beta":
        columns.append("confidence")
    frame = pd.read_parquet(FINAL_DATASET_DIR / f"dataset_{dataset}.parquet", columns=columns)
    frame["substation_id"] = frame["substation_id"].astype(str)
    frame["date"] = date_key(frame["date"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    frame["slot"] = frame["timestamp"].dt.hour * 4 + frame["timestamp"].dt.minute // 15
    frame["label_day"] = safe_bool(frame["label_day"])
    if dataset == "beta":
        frame["confidence"] = frame["confidence"].fillna("missing").astype(str).str.strip().str.lower()
    else:
        frame["confidence"] = "not_applicable"

    rows: list[dict[str, object]] = []
    arrays: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for (site, date), group in frame.groupby(["substation_id", "date"], sort=True):
        group = group.sort_values("slot").drop_duplicates("slot", keep="last")
        net = np.full(SLOTS_PER_DAY, np.nan, dtype=float)
        solar = np.full(SLOTS_PER_DAY, np.nan, dtype=float)
        valid = group.loc[group["slot"].between(0, SLOTS_PER_DAY - 1)].copy()
        slots = valid["slot"].to_numpy(dtype=int)
        net[slots] = valid["net_load_MW"].to_numpy(dtype=float)
        solar[slots] = valid["solar_MW"].to_numpy(dtype=float)
        net = fill_series(net, 0.0)
        solar = np.maximum(fill_series(solar, 0.0), 0.0)
        arrays[(site, date)] = (net, solar)
        daytime_net = net[DAYTIME_START : DAYTIME_END + 1]
        daytime_solar = solar[DAYTIME_START : DAYTIME_END + 1]
        rows.append(
            {
                "dataset": dataset,
                "substation_id": site,
                "date": date,
                "true_day": bool(group["label_day"].max()),
                "confidence": str(group["confidence"].iloc[0]),
                "solar_day_p95": float(np.nanpercentile(daytime_solar, 95)),
                "net_abs_day_p95": float(np.nanpercentile(np.abs(daytime_net), 95)),
            }
        )
    days = pd.DataFrame(rows)
    days["site_solar_scale"] = days.groupby("substation_id")["solar_day_p95"].transform("median").clip(lower=EXP13.EPS)
    days["site_net_scale"] = days.groupby("substation_id")["net_abs_day_p95"].transform("median").clip(lower=EXP13.EPS)
    return days, arrays


def local_min_delta(net: np.ndarray, slots: np.ndarray) -> np.ndarray:
    local_min = np.array(
        [
            np.nanmin(net[max(0, slot - NEIGHBOUR_RADIUS_SLOTS) : min(SLOTS_PER_DAY, slot + NEIGHBOUR_RADIUS_SLOTS + 1)])
            for slot in range(SLOTS_PER_DAY)
        ],
        dtype=float,
    )
    return net[slots] - local_min[slots]


def base_score(candidates: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=candidates.index)
    for col, weight in BASE_WEIGHTS.items():
        if weight:
            score = score + weight * pd.to_numeric(candidates[col], errors="coerce").fillna(0.0)
    return score


def variant_score_and_eligibility(candidates: pd.DataFrame, variant: dict[str, object]) -> tuple[pd.Series, pd.Series]:
    score = candidates["base_score"].copy()
    border_rule = str(variant["border_rule"])
    if border_rule == "none":
        return score, pd.Series(True, index=candidates.index)
    if border_rule == "soft_scaled_badness":
        penalty = float(variant["soft_penalty_weight"])
        return score - penalty * candidates["border_badness_scaled"], pd.Series(True, index=candidates.index)
    if border_rule == "both_edges_within_0.5MW_of_local_min":
        eligible = candidates["both_edges_near_min_raw_0p5MW"].astype(bool)
        return score.where(eligible), eligible
    if border_rule == "both_edges_within_max_0.25MW_or_10pct_site_net_scale":
        eligible = candidates["both_edges_near_min_scaled"].astype(bool)
        return score.where(eligible), eligible
    raise ValueError(f"Unsupported border rule: {border_rule}")


def scan_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    days, arrays = load_dataset_day_arrays(dataset)
    c1 = pd.read_csv(
        EXP_OUT / "C1_cached_daily_features/02_c1_daily_feature_cache.csv",
        usecols=["dataset", "substation_id", "date", "F1_bridge_improvement", "F2_roughness_improvement", "F3_slope_continuity_improvement"],
    )
    c1 = c1.loc[c1["dataset"].eq(dataset)].copy()
    c1["date"] = date_key(c1["date"])
    c1["core_score"] = (
        pd.to_numeric(c1["F1_bridge_improvement"], errors="coerce").fillna(0.0)
        + pd.to_numeric(c1["F2_roughness_improvement"], errors="coerce").fillna(0.0)
        + pd.to_numeric(c1["F3_slope_continuity_improvement"], errors="coerce").fillna(0.0)
    )
    site_core_median = c1.groupby("substation_id")["core_score"].median().to_dict()

    selected_rows: list[dict[str, object]] = []
    scan_rows: list[dict[str, object]] = []
    started = time.time()
    for idx, day in enumerate(days.itertuples(index=False), start=1):
        net, solar = arrays[(day.substation_id, day.date)]
        candidates = EXP13.candidate_base_features(net, solar, float(day.site_solar_scale))
        if candidates.empty:
            for variant in VARIANTS:
                selected_rows.append(
                    {
                        "dataset": dataset,
                        "substation_id": day.substation_id,
                        "date": day.date,
                        "true_day": day.true_day,
                        "confidence": day.confidence,
                        "variant": variant["variant"],
                        "experiment": variant["experiment"],
                        "border_rule": variant["border_rule"],
                        "score": np.nan,
                        "selected_left_slot": np.nan,
                        "selected_right_slot": np.nan,
                        "eligible_candidates": 0,
                        "candidate_count": 0,
                    }
                )
            continue

        site_median_core = float(site_core_median.get(day.substation_id, 0.0))
        candidates["F8_site_centered_core_score"] = EXP13.robust_bound(candidates["core_score"] - site_median_core)
        candidates["F9_site_rank_core_score"] = 0.0
        left_slots = candidates["left_slot"].to_numpy(dtype=int)
        right_slots = candidates["right_slot"].to_numpy(dtype=int)
        candidates["left_delta_to_local_min_MW"] = local_min_delta(net, left_slots)
        candidates["right_delta_to_local_min_MW"] = local_min_delta(net, right_slots)
        scaled_tol = max(SCALED_BORDER_TOLERANCE_FLOOR_MW, SCALED_BORDER_TOLERANCE_FRACTION * float(day.site_net_scale))
        candidates["scaled_border_tolerance_MW"] = scaled_tol
        candidates["border_badness_scaled"] = EXP13.clip01(
            (
                candidates["left_delta_to_local_min_MW"] / scaled_tol
                + candidates["right_delta_to_local_min_MW"] / scaled_tol
            )
            / 2
        )
        candidates["both_edges_near_min_raw_0p5MW"] = (
            candidates["left_delta_to_local_min_MW"].le(RAW_BORDER_TOLERANCE_MW)
            & candidates["right_delta_to_local_min_MW"].le(RAW_BORDER_TOLERANCE_MW)
        )
        candidates["both_edges_near_min_scaled"] = (
            candidates["left_delta_to_local_min_MW"].le(scaled_tol)
            & candidates["right_delta_to_local_min_MW"].le(scaled_tol)
        )
        candidates["base_score"] = base_score(candidates)

        for variant in VARIANTS:
            score, eligible = variant_score_and_eligibility(candidates, variant)
            finite = score.dropna()
            if finite.empty:
                selected_rows.append(
                    {
                        "dataset": dataset,
                        "substation_id": day.substation_id,
                        "date": day.date,
                        "true_day": day.true_day,
                        "confidence": day.confidence,
                        "variant": variant["variant"],
                        "experiment": variant["experiment"],
                        "border_rule": variant["border_rule"],
                        "score": np.nan,
                        "selected_left_slot": np.nan,
                        "selected_right_slot": np.nan,
                        "selected_duration_h": np.nan,
                        "selected_border_badness_scaled": np.nan,
                        "selected_left_delta_MW": np.nan,
                        "selected_right_delta_MW": np.nan,
                        "eligible_candidates": int(eligible.sum()),
                        "candidate_count": len(candidates),
                        "scaled_border_tolerance_MW": scaled_tol,
                    }
                )
                continue
            best_idx = finite.idxmax()
            best = candidates.loc[best_idx]
            selected_rows.append(
                {
                    "dataset": dataset,
                    "substation_id": day.substation_id,
                    "date": day.date,
                    "true_day": day.true_day,
                    "confidence": day.confidence,
                    "variant": variant["variant"],
                    "experiment": variant["experiment"],
                    "border_rule": variant["border_rule"],
                    "score": float(score.loc[best_idx]),
                    "selected_left_slot": int(best["left_slot"]),
                    "selected_right_slot": int(best["right_slot"]),
                    "selected_duration_h": float(best["duration_h"]),
                    "selected_border_badness_scaled": float(best["border_badness_scaled"]),
                    "selected_left_delta_MW": float(best["left_delta_to_local_min_MW"]),
                    "selected_right_delta_MW": float(best["right_delta_to_local_min_MW"]),
                    "eligible_candidates": int(eligible.sum()),
                    "candidate_count": len(candidates),
                    "scaled_border_tolerance_MW": scaled_tol,
                }
            )

        if idx % 1000 == 0:
            print(f"  scanned {dataset}: {idx:,}/{len(days):,} site-days")

    scan_rows.append(
        {
            "dataset": dataset,
            "site_days": len(days),
            "elapsed_seconds": time.time() - started,
            "mean_candidate_count": float(pd.DataFrame(selected_rows).query("variant == 'B0_no_border_baseline_rescan'")["candidate_count"].mean()),
        }
    )
    return pd.DataFrame(selected_rows), pd.DataFrame(scan_rows)


def select_threshold_with_missing(train: pd.DataFrame, score_col: str, *, dataset_balanced: bool) -> tuple[float, dict[str, object]]:
    finite_scores = np.sort(train[score_col].dropna().unique())
    if len(finite_scores) == 0:
        return np.inf, {"weighted_macro_f1": 0.0, "pooled_f1": 0.0, "threshold": np.inf}
    thresholds = np.r_[finite_scores[-1] + 1e-9, finite_scores[::-1]]
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        pred = train[score_col].notna() & train[score_col].ge(threshold)
        eval_frame = train.copy()
        eval_frame["_pred"] = pred
        metrics = EXP13.compute_metrics(eval_frame["true_day"], eval_frame["_pred"])
        site_rows = []
        weights = []
        for (dataset, site), group in eval_frame.groupby(["dataset", "substation_id"], sort=True):
            site_rows.append({"dataset": dataset, **EXP13.compute_metrics(group["true_day"], group["_pred"])})
        site_metrics = pd.DataFrame(site_rows)
        if dataset_balanced:
            dataset_values = sorted(site_metrics["dataset"].unique().tolist())
            for dataset in site_metrics["dataset"]:
                weights.append(1 / len(dataset_values) / int(site_metrics["dataset"].eq(dataset).sum()))
        else:
            weights = [1 / len(site_metrics)] * len(site_metrics)
        weights_arr = np.asarray(weights, dtype=float)
        rows.append(
            {
                "threshold": float(threshold),
                "weighted_macro_precision": float((site_metrics["precision"].to_numpy() * weights_arr).sum()),
                "weighted_macro_recall": float((site_metrics["recall"].to_numpy() * weights_arr).sum()),
                "weighted_macro_f1": float((site_metrics["f1"].to_numpy() * weights_arr).sum()),
                **{f"pooled_{key}": value for key, value in metrics.items()},
            }
        )
    sweep = pd.DataFrame(rows)
    best = sweep.sort_values(
        ["weighted_macro_f1", "pooled_f1", "weighted_macro_precision", "weighted_macro_recall", "threshold"],
        ascending=[False, False, False, False, False],
    ).iloc[0]
    return float(best["threshold"]), best.to_dict()


def metric_rows_for_predictions(frame: pd.DataFrame, *, variant: str, regime: str, subset: str) -> list[dict[str, object]]:
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
    return rows


def evaluate_regimes(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alpha = selected.loc[selected["dataset"].eq("alpha")].copy()
    beta = selected.loc[selected["dataset"].eq("beta")].copy()
    beta_sites = sorted(beta["substation_id"].unique().tolist())
    threshold_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []

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
                threshold, threshold_info = select_threshold_with_missing(
                    train,
                    "score",
                    dataset_balanced=dataset_balanced,
                )
                eval_frame = beta_v.loc[beta_v["substation_id"].eq(heldout_site)].copy()
                eval_frame["regime"] = regime
                eval_frame["heldout_site"] = heldout_site
                eval_frame["threshold"] = threshold
                eval_frame["pred_day"] = eval_frame["score"].notna() & eval_frame["score"].ge(threshold)
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
                        **{f"selected_{key}": value for key, value in threshold_info.items() if key != "threshold"},
                    }
                )
        threshold, threshold_info = select_threshold_with_missing(alpha_v, "score", dataset_balanced=False)
        eval_frame = beta_v.copy()
        eval_frame["regime"] = "R3_alpha_only_to_beta"
        eval_frame["heldout_site"] = "all_beta"
        eval_frame["threshold"] = threshold
        eval_frame["pred_day"] = eval_frame["score"].notna() & eval_frame["score"].ge(threshold)
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
                **{f"selected_{key}": value for key, value in threshold_info.items() if key != "threshold"},
            }
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    for (regime, variant), group in predictions.groupby(["regime", "variant"], sort=True):
        metric_rows.extend(metric_rows_for_predictions(group, variant=variant, regime=regime, subset="beta_all"))
        metric_rows.extend(
            metric_rows_for_predictions(
                group.loc[group["confidence"].eq("sure")].copy(),
                variant=variant,
                regime=regime,
                subset="beta_sure_only",
            )
        )

    metrics = pd.DataFrame(metric_rows)
    thresholds = pd.DataFrame(threshold_rows)

    def lookup(subset: str, scope: str, metric: str) -> pd.Series:
        return (
            metrics.loc[
                metrics["subset"].eq(subset) & metrics["summary_scope"].eq(scope),
                ["regime", "variant", metric],
            ]
            .set_index(["regime", "variant"])[metric]
        )

    ranking = thresholds[["regime", "variant"]].drop_duplicates().reset_index(drop=True)
    variant_defs = pd.DataFrame(VARIANTS)
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
            model_family="border_minima_candidate_experiment",
            model_variant=variant,
            regime=regime,
            prediction_source=Path(__file__).name,
            notes="Top-10 day-F1 border-minima candidate experiment; candidate window reselected before thresholding.",
        )
    return pd.DataFrame(rows), pd.DataFrame(site_rows)


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    selected_parts: list[pd.DataFrame] = []
    scan_parts: list[pd.DataFrame] = []
    for dataset in ["alpha", "beta"]:
        print(f"Scanning {dataset} candidates with border-minima metrics...")
        selected, scan = scan_dataset(dataset)
        selected_parts.append(selected)
        scan_parts.append(scan)
    selected = pd.concat(selected_parts, ignore_index=True)
    scan_summary = pd.concat(scan_parts, ignore_index=True)
    thresholds, metrics, ranking, predictions = evaluate_regimes(selected)
    multi_metrics, multi_site_metrics = multi_metric_summary(predictions, ranking)

    manifest = pd.DataFrame(
        [
            {
                "script": Path(__file__).name,
                "output_folder": OUTPUT_FOLDER,
                "variants": len(VARIANTS),
                "selected_daily_rows": len(selected),
                "prediction_rows": len(predictions),
                "elapsed_seconds": time.time() - started,
                "outputs": str(OUT.relative_to(ROOT)),
            }
        ]
    )
    manifest.to_csv(OUT / "01_c18_manifest.csv", index=False)
    pd.DataFrame(VARIANTS).to_csv(OUT / "02_c18_variant_definitions.csv", index=False)
    scan_summary.to_csv(OUT / "03_c18_candidate_scan_summary.csv", index=False)
    selected.to_csv(OUT / "04_c18_daily_selected_windows.csv", index=False)
    thresholds.to_csv(OUT / "05_c18_threshold_selection.csv", index=False)
    metrics.to_csv(OUT / "06_c18_day_level_metrics.csv", index=False)
    ranking.to_csv(OUT / "07_c18_regime_variant_ranking.csv", index=False)
    predictions.to_csv(OUT / "08_c18_beta_prediction_audit.csv", index=False)
    multi_metrics.to_csv(OUT / "09_c18_top_multi_metric_summary.csv", index=False)
    multi_site_metrics.to_csv(OUT / "10_c18_top_site_multi_metric_summary.csv", index=False)

    print(f"Wrote border-minima candidate experiments to {OUT.relative_to(ROOT)}")
    print("\nManifest")
    print(manifest.round(3).to_string(index=False))
    print("\nCandidate scan summary")
    print(scan_summary.round(3).to_string(index=False))
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

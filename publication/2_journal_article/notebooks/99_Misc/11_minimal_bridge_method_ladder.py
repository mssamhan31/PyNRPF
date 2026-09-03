from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_FOLDER_NAME = "11_minimal_bridge_method_ladder"
SITE_MEDIAN_WEIGHT = 0.425
SLOTS_PER_DAY = 96
DAYTIME_START = 24
DAYTIME_END = 72
CONTEXT_SLOTS = 4
EPS = 1e-6
SCALE_FLOOR = 1e-3


def find_repo_root() -> Path:
    start = Path(__file__).resolve()
    marker = Path("publication/2_journal_article/dataset/final/dataset_alpha.parquet")
    for candidate in [start.parent, *start.parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"Could not find repo root containing {marker}")


ROOT = find_repo_root()
JOURNAL = ROOT / "publication/2_journal_article"
MISC_DIR = JOURNAL / "notebooks/99_Misc"
SOURCE_SCORE_DIR = MISC_DIR / "outputs/08_m9_2_bridge_score_development/csv"
OUT = MISC_DIR / "outputs" / OUTPUT_FOLDER_NAME
FINAL_DATASET_DIR = JOURNAL / "dataset/final"
REVIEWER_B_PATH = JOURNAL / "dataset/oracle_data_creation/archive/2026-07-02_reviewer_B_final/reviewer_B.csv"
JOINED_CACHE_PATH = OUT / "03_joined_daily_scores.csv"
V03_CACHE_COLUMNS = [
    "dataset",
    "substation_id",
    "date",
    "E4_v03_three_feature_score",
    "E5_v03_without_slope_score",
    "v03_candidate_count",
    "v03_bridge_best",
    "v03_roughness_best",
    "v03_slope_continuity_best",
    "v03_selected_left_slot",
    "v03_selected_right_slot",
    "v03_selected_duration_h",
]
SCALED_CACHE_COLUMNS = [
    "dataset",
    "substation_id",
    "date",
    "E9_v03_site_solar_scaled_score",
    "E10_v03_site_combined_scaled_score",
]
E13_DURATION_TARGET_H = 2.5
E13_DURATION_PENALTY_WEIGHT = 0.5


def reset_output_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def date_key(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.strftime("%Y-%m-%d")


def as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).astype(float).ne(0)
    return values.fillna(False).astype(str).str.strip().str.lower().isin({"true", "t", "1", "yes", "y"})


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def metric_counts(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, int | float]:
    true = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    tp = int((true & pred).sum())
    fp = int((~true & pred).sum())
    fn = int((true & ~pred).sum())
    tn = int((~true & ~pred).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "support": int(len(true)),
        "positive_support": int(true.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_final_truth(dataset: str) -> pd.DataFrame:
    path = FINAL_DATASET_DIR / f"dataset_{dataset}.parquet"
    df = pd.read_parquet(path, columns=["substation_id", "date", "label_day"])
    df["substation_id"] = df["substation_id"].astype(str)
    df["date"] = date_key(df["date"])
    df["label_day"] = as_bool(df["label_day"])
    truth = (
        df.groupby(["substation_id", "date"], as_index=False)["label_day"]
        .max()
        .rename(columns={"label_day": "true_day"})
    )
    return truth


def load_beta_confidence() -> pd.DataFrame:
    conf = pd.read_csv(REVIEWER_B_PATH, usecols=["substation_id", "date", "confidence"])
    conf["substation_id"] = conf["substation_id"].astype(str).str.replace("^act_", "beta_", regex=True)
    conf["date"] = date_key(conf["date"])
    conf["confidence"] = conf["confidence"].fillna("missing").astype(str).str.strip().str.lower()
    return conf.drop_duplicates(["substation_id", "date"], keep="last")


def load_daily_scores(dataset: str) -> pd.DataFrame:
    path = SOURCE_SCORE_DIR / f"01_{dataset}_daily_bridge_scores.csv"
    needed = [
        "substation_id",
        "date",
        "bridge_ratio_p99",
        "full_tv_ratio_p99",
    ]
    scores = pd.read_csv(path, usecols=needed)
    scores["substation_id"] = scores["substation_id"].astype(str)
    scores["date"] = date_key(scores["date"])
    return scores


def fill_series(values: np.ndarray, default: float = 0.0) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    filled = series.interpolate(limit_direction="both").fillna(default)
    return filled.to_numpy(dtype=float)


def load_daily_arrays(dataset: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    path = FINAL_DATASET_DIR / f"dataset_{dataset}.parquet"
    df = pd.read_parquet(path, columns=["substation_id", "date", "timestamp", "net_load_MW", "solar_MW"])
    df["substation_id"] = df["substation_id"].astype(str)
    df["date"] = date_key(df["date"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.sort_values(["substation_id", "date", "timestamp"])

    keys = []
    net_rows = []
    solar_rows = []
    for (site, day), group in df.groupby(["substation_id", "date"], sort=True):
        group = group.copy()
        group["slot"] = group["timestamp"].dt.hour * 4 + (group["timestamp"].dt.minute // 15)
        group = group.loc[group["slot"].between(0, SLOTS_PER_DAY - 1)]
        group = group.drop_duplicates("slot", keep="last")
        net = np.full(SLOTS_PER_DAY, np.nan, dtype=float)
        solar = np.full(SLOTS_PER_DAY, np.nan, dtype=float)
        slots = group["slot"].to_numpy(dtype=int)
        net[slots] = group["net_load_MW"].to_numpy(dtype=float)
        solar[slots] = group["solar_MW"].to_numpy(dtype=float)
        keys.append({"substation_id": site, "date": day})
        net_rows.append(fill_series(net, 0.0))
        solar_rows.append(np.maximum(fill_series(solar, 0.0), 0.0))
    return pd.DataFrame(keys), np.vstack(net_rows), np.vstack(solar_rows)


def compute_site_scales(keys: pd.DataFrame, net: np.ndarray, solar: np.ndarray) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    daylight = slice(DAYTIME_START, DAYTIME_END + 1)
    daily_solar_p95 = np.nanpercentile(solar[:, daylight], 95, axis=1)
    daily_abs_net_p95 = np.nanpercentile(np.abs(net[:, daylight]), 95, axis=1)
    daily_combined = np.maximum(daily_solar_p95, daily_abs_net_p95)
    scale_frame = keys.copy()
    scale_frame["daily_solar_p95"] = daily_solar_p95
    scale_frame["daily_combined_p95"] = daily_combined

    site_scales = (
        scale_frame.groupby("substation_id", sort=True)
        .agg(
            site_solar_scale=("daily_solar_p95", "median"),
            site_combined_scale=("daily_combined_p95", "median"),
            site_days=("date", "count"),
        )
        .reset_index()
    )
    site_scales["site_solar_scale"] = site_scales["site_solar_scale"].clip(lower=SCALE_FLOOR)
    site_scales["site_combined_scale"] = site_scales["site_combined_scale"].clip(lower=SCALE_FLOOR)

    solar_scale = keys["substation_id"].map(site_scales.set_index("substation_id")["site_solar_scale"])
    combined_scale = keys["substation_id"].map(site_scales.set_index("substation_id")["site_combined_scale"])
    return solar_scale, combined_scale, site_scales


def build_candidate_cache() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for peak in range(DAYTIME_START, DAYTIME_END + 1):
        lefts = []
        rights = []
        for left in range(DAYTIME_START, DAYTIME_END):
            max_right = min(DAYTIME_END, left + 31)
            for right in range(left + 1, max_right + 1):
                if abs((left + right) / 2 - peak) <= 14:
                    lefts.append(left)
                    rights.append(right)
        cache[peak] = (np.asarray(lefts, dtype=np.int16), np.asarray(rights, dtype=np.int16))
    return cache


CANDIDATE_CACHE = build_candidate_cache()
SLOTS = np.arange(SLOTS_PER_DAY)
DIFF_T = np.arange(1, SLOTS_PER_DAY)


def segment_mean_from_cumsum(values: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    csum = np.r_[0.0, np.cumsum(values)]
    count = (right - left + 1).astype(float)
    return (csum[right + 1] - csum[left]) / count


def bridge_mse(values: np.ndarray, left: np.ndarray, right: np.ndarray, anchor_values: np.ndarray) -> np.ndarray:
    slots = np.arange(SLOTS_PER_DAY, dtype=float)
    pre = np.maximum(left - 1, 0)
    post = np.minimum(right + 1, SLOTS_PER_DAY - 1)
    slope = (anchor_values[post] - anchor_values[pre]) / np.maximum(post - pre, 1)
    intercept = anchor_values[pre] - slope * pre
    count = (right - left + 1).astype(float)
    csum = np.r_[0.0, np.cumsum(values)]
    csum2 = np.r_[0.0, np.cumsum(values * values)]
    cslot_values = np.r_[0.0, np.cumsum(slots * values)]
    sx = csum[right + 1] - csum[left]
    sx2 = csum2[right + 1] - csum2[left]
    stx = cslot_values[right + 1] - cslot_values[left]
    st = (left + right) * count / 2

    def sumsq(k: int) -> float:
        return k * (k + 1) * (2 * k + 1) / 6

    st2 = np.array([sumsq(int(r)) - sumsq(int(l - 1)) for l, r in zip(left, right)], dtype=float)
    sse = sx2 - 2 * intercept * sx - 2 * slope * stx + count * intercept * intercept + 2 * intercept * slope * st + slope * slope * st2
    return np.maximum(sse / count, 0)


def median_diffs(diff_matrix: np.ndarray, t_values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Median of candidate-specific first differences at 1-based t positions."""
    indices = np.clip(t_values - 1, 0, diff_matrix.shape[1] - 1)
    gathered = np.take_along_axis(diff_matrix, indices, axis=1)
    gathered = np.where(valid, gathered, np.nan)
    return np.nanmedian(gathered, axis=1)


def scan_v03_day(net: np.ndarray, solar: np.ndarray) -> dict[str, float | int]:
    """Compute fast v0.3-style daily max scores from one candidate scan."""
    peak = int(np.argmax(solar[DAYTIME_START : DAYTIME_END + 1])) + DAYTIME_START
    left, right = CANDIDATE_CACHE[peak]
    n_candidates = len(left)
    if n_candidates == 0:
        return {
            "E4_v03_three_feature_score": np.nan,
            "E5_v03_without_slope_score": np.nan,
            "v03_candidate_count": 0,
            "v03_bridge_best": np.nan,
            "v03_roughness_best": np.nan,
            "v03_slope_continuity_best": np.nan,
            "v03_selected_left_slot": np.nan,
            "v03_selected_right_slot": np.nan,
            "v03_selected_duration_h": np.nan,
        }

    up = solar + net
    um = solar - net
    bup = bridge_mse(up, left, right, up)
    bum = bridge_mse(um, left, right, up)
    bridge_improvement = (bup - bum) / (bup + bum + EPS)

    # Feature 2: roughness improvement over daylight after replacing the candidate segment.
    up_diff_abs = np.abs(np.diff(up))
    base_tv = up_diff_abs[DAYTIME_START:DAYTIME_END].sum()
    ctv_no = np.r_[0.0, np.cumsum(up_diff_abs)]
    internal_no = ctv_no[right] - ctv_no[left]
    um_diff_abs = np.abs(np.diff(um))
    ctv_um = np.r_[0.0, np.cumsum(um_diff_abs)]
    internal_corr = ctv_um[right] - ctv_um[left]
    left_jump_no = np.where(left > DAYTIME_START, np.abs(up[left] - up[left - 1]), 0)
    left_jump_corr = np.where(left > DAYTIME_START, np.abs(um[left] - up[left - 1]), 0)
    right_jump_no = np.where(right < DAYTIME_END, np.abs(up[right + 1] - up[right]), 0)
    right_jump_corr = np.where(right < DAYTIME_END, np.abs(up[right + 1] - um[right]), 0)
    corr_tv = base_tv - (internal_no + left_jump_no + right_jump_no) + (internal_corr + left_jump_corr + right_jump_corr)
    roughness_improvement = (base_tv - corr_tv) / (base_tv + corr_tv + EPS)

    # Feature 3: v0.3 slope-continuity improvement using median slopes over shoulders.
    inside = (SLOTS[None, :] >= left[:, None]) & (SLOTS[None, :] <= right[:, None])
    ucorr = np.where(inside, um[None, :], up[None, :])
    up_diff = np.diff(up)[None, :]
    ucorr_diff = np.diff(ucorr, axis=1)

    left_before_t = left[:, None] + np.array([-3, -2, -1])
    left_after_t = left[:, None] + np.array([1, 2, 3])
    right_before_t = right[:, None] + np.array([-2, -1, 0])
    right_after_t = right[:, None] + np.array([2, 3, 4])

    left_before_valid = np.ones_like(left_before_t, dtype=bool)
    left_after_valid = left_after_t <= right[:, None]
    right_before_valid = right_before_t >= (left[:, None] + 1)
    right_after_valid = np.ones_like(right_after_t, dtype=bool)

    up_diff_matrix = np.repeat(up_diff, n_candidates, axis=0)
    no_left_before = median_diffs(up_diff_matrix, left_before_t, left_before_valid)
    no_left_after = median_diffs(up_diff_matrix, left_after_t, left_after_valid)
    no_right_before = median_diffs(up_diff_matrix, right_before_t, right_before_valid)
    no_right_after = median_diffs(up_diff_matrix, right_after_t, right_after_valid)

    corr_left_before = median_diffs(ucorr_diff, left_before_t, left_before_valid)
    corr_left_after = median_diffs(ucorr_diff, left_after_t, left_after_valid)
    corr_right_before = median_diffs(ucorr_diff, right_before_t, right_before_valid)
    corr_right_after = median_diffs(ucorr_diff, right_after_t, right_after_valid)

    slope_no = np.abs(no_left_before - no_left_after) + np.abs(no_right_before - no_right_after)
    slope_corr = np.abs(corr_left_before - corr_left_after) + np.abs(corr_right_before - corr_right_after)
    slope_continuity_improvement = (slope_no - slope_corr) / (slope_no + slope_corr + EPS)

    e4_score_by_window = bridge_improvement + roughness_improvement + slope_continuity_improvement
    e5_score_by_window = bridge_improvement + roughness_improvement
    e4_idx = int(np.nanargmax(e4_score_by_window))
    return {
        "E4_v03_three_feature_score": float(np.nanmax(e4_score_by_window)),
        "E5_v03_without_slope_score": float(np.nanmax(e5_score_by_window)),
        "v03_candidate_count": int(n_candidates),
        "v03_bridge_best": float(bridge_improvement[e4_idx]),
        "v03_roughness_best": float(roughness_improvement[e4_idx]),
        "v03_slope_continuity_best": float(slope_continuity_improvement[e4_idx]),
        "v03_selected_left_slot": int(left[e4_idx]),
        "v03_selected_right_slot": int(right[e4_idx]),
        "v03_selected_duration_h": float((int(right[e4_idx]) - int(left[e4_idx]) + 1) * 0.25),
    }


def scan_v03_dataset(dataset: str) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    started = time.time()
    keys, net, solar = load_daily_arrays(dataset)
    rows = []
    for idx in range(len(keys)):
        rows.append(scan_v03_day(net[idx], solar[idx]))
    scores = pd.concat([keys.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    counts = scores["v03_candidate_count"]
    summary = {
        "dataset": dataset,
        "site_days": int(len(scores)),
        "elapsed_seconds": float(time.time() - started),
        "candidate_count_min": int(counts.min()),
        "candidate_count_mean": float(counts.mean()),
        "candidate_count_max": int(counts.max()),
        "notes": "Candidate-level v0.3 scan; no candidate rows written.",
    }
    return scores, summary


def scan_scaled_v03_dataset(dataset: str) -> tuple[pd.DataFrame, list[dict[str, float | int | str]]]:
    keys, net, solar = load_daily_arrays(dataset)
    solar_scale, combined_scale, site_scales = compute_site_scales(keys, net, solar)
    score_frame = keys.copy()
    summaries: list[dict[str, float | int | str]] = []

    for mode, scale_values, score_col in [
        ("site_solar_scale", solar_scale.to_numpy(dtype=float), "E9_v03_site_solar_scaled_score"),
        ("site_combined_scale", combined_scale.to_numpy(dtype=float), "E10_v03_site_combined_scaled_score"),
    ]:
        started = time.time()
        rows = []
        for idx in range(len(keys)):
            scale = max(float(scale_values[idx]), SCALE_FLOOR)
            result = scan_v03_day(net[idx] / scale, solar[idx] / scale)
            rows.append(result["E4_v03_three_feature_score"])
        score_frame[score_col] = rows
        scale_col = mode
        summaries.append(
            {
                "dataset": dataset,
                "scaling_mode": mode,
                "site_count": int(site_scales["substation_id"].nunique()),
                "scale_min": float(site_scales[scale_col].min()),
                "scale_median": float(site_scales[scale_col].median()),
                "scale_max": float(site_scales[scale_col].max()),
                "elapsed_seconds": float(time.time() - started),
                "cached_e4_e5_reused": True,
                "notes": f"Computed E4 score after dividing net load and solar by {mode}.",
            }
        )
    return score_frame, summaries


def load_v03_scores_from_cache() -> tuple[dict[str, pd.DataFrame], list[dict[str, float | int | str]]] | None:
    if not JOINED_CACHE_PATH.exists():
        return None
    try:
        cache = pd.read_csv(JOINED_CACHE_PATH, usecols=V03_CACHE_COLUMNS)
    except ValueError:
        return None

    cache["date"] = date_key(cache["date"])
    cache["substation_id"] = cache["substation_id"].astype(str)
    cache["dataset"] = cache["dataset"].astype(str)
    if cache[V03_CACHE_COLUMNS].isna().any().any():
        return None

    expected_counts = {"alpha": 10643, "beta": 2928}
    scores_by_dataset: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, float | int | str]] = []
    for dataset, expected_count in expected_counts.items():
        subset = cache.loc[cache["dataset"].eq(dataset)].copy()
        if len(subset) != expected_count:
            return None
        if subset.duplicated(["substation_id", "date"]).any():
            return None
        scores = subset.drop(columns=["dataset"]).reset_index(drop=True)
        scores_by_dataset[dataset] = scores
        summaries.append(
            {
                "dataset": dataset,
                "site_days": int(len(scores)),
                "elapsed_seconds": 0.0,
                "candidate_count_min": np.nan,
                "candidate_count_mean": np.nan,
                "candidate_count_max": np.nan,
                "notes": "Reused cached E4/E5 daily v0.3 scores and selected-window metadata from 03_joined_daily_scores.csv; candidate scan skipped.",
            }
        )
    return scores_by_dataset, summaries


def load_scaled_scores_from_cache() -> tuple[dict[str, pd.DataFrame], list[dict[str, float | int | str]]] | None:
    if not JOINED_CACHE_PATH.exists():
        return None
    try:
        cache = pd.read_csv(JOINED_CACHE_PATH, usecols=SCALED_CACHE_COLUMNS)
    except ValueError:
        return None

    cache["date"] = date_key(cache["date"])
    cache["substation_id"] = cache["substation_id"].astype(str)
    cache["dataset"] = cache["dataset"].astype(str)
    if cache[SCALED_CACHE_COLUMNS].isna().any().any():
        return None

    expected_counts = {"alpha": 10643, "beta": 2928}
    scores_by_dataset: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, float | int | str]] = []
    for dataset, expected_count in expected_counts.items():
        subset = cache.loc[cache["dataset"].eq(dataset)].copy()
        if len(subset) != expected_count:
            return None
        if subset.duplicated(["substation_id", "date"]).any():
            return None
        scores_by_dataset[dataset] = subset.drop(columns=["dataset"]).reset_index(drop=True)
        summaries.append(
            {
                "dataset": dataset,
                "scaling_mode": "cached_site_scaled_scores",
                "site_count": int(subset["substation_id"].nunique()),
                "scale_min": np.nan,
                "scale_median": np.nan,
                "scale_max": np.nan,
                "elapsed_seconds": 0.0,
                "cached_e4_e5_reused": True,
                "notes": "Reused cached E9/E10 scaled daily scores from 03_joined_daily_scores.csv; scaled candidate scans skipped.",
            }
        )
    return scores_by_dataset, summaries


def add_experiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    site_median = out.groupby("substation_id")["bridge_ratio_p99"].transform("median")
    out["E1_bridge_only_score"] = out["bridge_ratio_p99"]
    out["E2_bridge_plus_roughness_score"] = out["bridge_ratio_p99"] + out["full_tv_ratio_p99"]
    out["E3_bridge_plus_site_median_score"] = out["bridge_ratio_p99"] - SITE_MEDIAN_WEIGHT * site_median
    if "E4_v03_three_feature_score" in out.columns:
        v03_site_median = out.groupby("substation_id")["E4_v03_three_feature_score"].transform("median")
        out["E6_v03_three_feature_site_median_score"] = out["E4_v03_three_feature_score"] - SITE_MEDIAN_WEIGHT * v03_site_median
        out["E7_v03_three_feature_site_centered_score"] = out["E4_v03_three_feature_score"] - v03_site_median
        out["E8_v03_three_feature_site_rank_score"] = out.groupby("substation_id")["E4_v03_three_feature_score"].rank(
            method="average",
            pct=True,
        )
        out["E11_v03_duration_ge_1h_score"] = out["E4_v03_three_feature_score"].where(out["v03_selected_duration_h"] >= 1.0)
        out["E12_v03_duration_ge_1p5h_score"] = out["E4_v03_three_feature_score"].where(out["v03_selected_duration_h"] >= 1.5)
        duration_shortfall = (E13_DURATION_TARGET_H - out["v03_selected_duration_h"]).clip(lower=0)
        out["E13_v03_soft_duration_penalty_score"] = (
            out["E4_v03_three_feature_score"] - E13_DURATION_PENALTY_WEIGHT * duration_shortfall
        )
    return out


EXPERIMENTS = [
    {
        "experiment": "E1_bridge_only",
        "score_col": "E1_bridge_only_score",
        "score_formula": "bridge_ratio_p99",
        "score_feature_count": 1,
        "fixed_parameter_count": 0,
        "alpha_selected_parameter_count": 1,
        "notes": "Core one-feature bridge p99 score.",
    },
    {
        "experiment": "E2_bridge_plus_roughness",
        "score_col": "E2_bridge_plus_roughness_score",
        "score_formula": "bridge_ratio_p99 + full_tv_ratio_p99",
        "score_feature_count": 2,
        "fixed_parameter_count": 0,
        "alpha_selected_parameter_count": 1,
        "notes": "Fast v0.3 proxy using available total-variation roughness improvement.",
    },
    {
        "experiment": "E3_bridge_plus_site_median",
        "score_col": "E3_bridge_plus_site_median_score",
        "score_formula": f"bridge_ratio_p99 - {SITE_MEDIAN_WEIGHT} * within_site_median(bridge_ratio_p99)",
        "score_feature_count": 1,
        "fixed_parameter_count": 1,
        "alpha_selected_parameter_count": 1,
        "notes": "Tests the main site-level normalisation discovered in earlier bridge scans.",
    },
    {
        "experiment": "E4_v03_three_feature",
        "score_col": "E4_v03_three_feature_score",
        "score_formula": "max_window(bridge_improvement + roughness_improvement + slope_continuity_improvement)",
        "score_feature_count": 3,
        "fixed_parameter_count": 0,
        "alpha_selected_parameter_count": 1,
        "notes": "Fast v0.3-style candidate-window score with equal feature weights; bridge uses MSE and roughness uses daylight total variation for speed.",
    },
    {
        "experiment": "E5_v03_without_slope",
        "score_col": "E5_v03_without_slope_score",
        "score_formula": "max_window(bridge_improvement + roughness_improvement)",
        "score_feature_count": 2,
        "fixed_parameter_count": 0,
        "alpha_selected_parameter_count": 1,
        "notes": "Fast candidate-window v0.3 ablation without the slope-continuity feature.",
    },
    {
        "experiment": "E6_v03_three_feature_site_median",
        "score_col": "E6_v03_three_feature_site_median_score",
        "score_formula": f"E4_v03_three_feature_score - {SITE_MEDIAN_WEIGHT} * within_site_median(E4_v03_three_feature_score)",
        "score_feature_count": 3,
        "fixed_parameter_count": 1,
        "alpha_selected_parameter_count": 1,
        "notes": "Fast v0.3-style score with one fixed site-median normalisation weight.",
    },
    {
        "experiment": "E7_v03_three_feature_site_centered",
        "score_col": "E7_v03_three_feature_site_centered_score",
        "score_formula": "E4_v03_three_feature_score - within_site_median(E4_v03_three_feature_score)",
        "score_feature_count": 3,
        "fixed_parameter_count": 0,
        "alpha_selected_parameter_count": 1,
        "notes": "Simple site-relative E4 score using each site's unlabeled score median.",
    },
    {
        "experiment": "E8_v03_three_feature_site_rank",
        "score_col": "E8_v03_three_feature_site_rank_score",
        "score_formula": "within_site_percentile_rank(E4_v03_three_feature_score)",
        "score_feature_count": 3,
        "fixed_parameter_count": 0,
        "alpha_selected_parameter_count": 1,
        "notes": "Simple site-relative E4 score using each site's unlabeled percentile rank.",
    },
    {
        "experiment": "E9_v03_site_solar_scaled",
        "score_col": "E9_v03_site_solar_scaled_score",
        "score_formula": "E4_v03_three_feature_score computed on net/site_solar_scale and solar/site_solar_scale",
        "score_feature_count": 3,
        "fixed_parameter_count": 0,
        "alpha_selected_parameter_count": 1,
        "notes": "Input-level site scaling using median daytime solar p95; threshold selected on Alpha.",
    },
    {
        "experiment": "E10_v03_site_combined_scaled",
        "score_col": "E10_v03_site_combined_scaled_score",
        "score_formula": "E4_v03_three_feature_score computed on net/site_combined_scale and solar/site_combined_scale",
        "score_feature_count": 3,
        "fixed_parameter_count": 0,
        "alpha_selected_parameter_count": 1,
        "notes": "Input-level site scaling using median max(daytime solar p95, daytime abs net p95); threshold selected on Alpha.",
    },
    {
        "experiment": "E11_v03_duration_ge_1h",
        "score_col": "E11_v03_duration_ge_1h_score",
        "score_formula": "E4_v03_three_feature_score if selected_duration_h >= 1.0 else no prediction",
        "score_feature_count": 3,
        "fixed_parameter_count": 1,
        "alpha_selected_parameter_count": 1,
        "notes": "E4 with a minimal sustained-window guard; final threshold selected on Alpha.",
    },
    {
        "experiment": "E12_v03_duration_ge_1p5h",
        "score_col": "E12_v03_duration_ge_1p5h_score",
        "score_formula": "E4_v03_three_feature_score if selected_duration_h >= 1.5 else no prediction",
        "score_feature_count": 3,
        "fixed_parameter_count": 1,
        "alpha_selected_parameter_count": 1,
        "notes": "E4 with a moderate sustained-window guard; final threshold selected on Alpha.",
    },
    {
        "experiment": "E13_v03_soft_duration_penalty",
        "score_col": "E13_v03_soft_duration_penalty_score",
        "score_formula": f"E4_v03_three_feature_score - {E13_DURATION_PENALTY_WEIGHT} * max(0, {E13_DURATION_TARGET_H} - selected_duration_h)",
        "score_feature_count": 3,
        "fixed_parameter_count": 2,
        "alpha_selected_parameter_count": 1,
        "notes": "E4 with a fixed soft short-window penalty; final threshold selected on Alpha.",
    },
]


def select_threshold_on_alpha(alpha: pd.DataFrame, score_col: str) -> dict[str, int | float]:
    finite = np.isfinite(alpha[score_col].to_numpy(dtype=float))
    if not finite.any():
        raise ValueError(f"No finite Alpha scores for {score_col}")
    thresholds = np.unique(alpha.loc[finite, score_col].to_numpy(dtype=float))
    rows = []
    y_true = alpha["true_day"].to_numpy(dtype=bool)
    scores = alpha[score_col].to_numpy(dtype=float)
    for threshold in thresholds:
        counts = metric_counts(y_true, np.where(np.isfinite(scores), scores >= threshold, False))
        counts["threshold"] = float(threshold)
        rows.append(counts)
    sweep = pd.DataFrame(rows)
    sweep = sweep.sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    return sweep.iloc[0].to_dict()


def metric_row(
    experiment: dict[str, object],
    dataset: str,
    subset: str,
    summary_scope: str,
    substation_id: str,
    counts: dict[str, int | float],
    threshold: float,
) -> dict[str, object]:
    return {
        "experiment": experiment["experiment"],
        "dataset": dataset,
        "subset": subset,
        "summary_scope": summary_scope,
        "substation_id": substation_id,
        **counts,
        "threshold": threshold,
        "score_formula": experiment["score_formula"],
        "notes": experiment["notes"],
    }


def evaluate_subset(
    experiment: dict[str, object],
    df: pd.DataFrame,
    dataset: str,
    subset: str,
    threshold: float,
) -> list[dict[str, object]]:
    pred_col = f"{experiment['experiment']}_pred_day"
    rows = [metric_row(experiment, dataset, subset, "pooled", "", metric_counts(df["true_day"], df[pred_col]), threshold)]

    site_counts = []
    for site, group in df.groupby("substation_id", sort=True):
        counts = metric_counts(group["true_day"], group[pred_col])
        site_counts.append(counts)
        rows.append(metric_row(experiment, dataset, subset, "site", site, counts, threshold))

    site_df = pd.DataFrame(site_counts)
    macro_counts = {
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
    rows.append(metric_row(experiment, dataset, subset, "macro_site_average", "", macro_counts, threshold))
    return rows


def build_dataset_frame(dataset: str, truth: pd.DataFrame, v03_scores: pd.DataFrame, scaled_scores: pd.DataFrame) -> pd.DataFrame:
    scores = load_daily_scores(dataset)
    df = (
        scores.merge(v03_scores, on=["substation_id", "date"], how="inner")
        .merge(scaled_scores, on=["substation_id", "date"], how="inner")
        .merge(truth, on=["substation_id", "date"], how="inner")
    )
    if dataset == "alpha":
        df["confidence"] = "not_applicable"
    return add_experiment_scores(df)


def main() -> None:
    alpha_truth = load_final_truth("alpha")
    beta_truth = load_final_truth("beta").merge(load_beta_confidence(), on=["substation_id", "date"], how="left")
    beta_truth["confidence"] = beta_truth["confidence"].fillna("missing")

    cached = load_v03_scores_from_cache()
    if cached is not None:
        v03_scores_by_dataset, v03_summaries = cached
        alpha_v03_scores = v03_scores_by_dataset["alpha"]
        beta_v03_scores = v03_scores_by_dataset["beta"]
    else:
        alpha_v03_scores, alpha_v03_summary = scan_v03_dataset("alpha")
        beta_v03_scores, beta_v03_summary = scan_v03_dataset("beta")
        v03_summaries = [alpha_v03_summary, beta_v03_summary]

    cached_scaled = load_scaled_scores_from_cache()
    if cached_scaled is not None:
        scaled_scores_by_dataset, scaled_summaries = cached_scaled
        alpha_scaled_scores = scaled_scores_by_dataset["alpha"]
        beta_scaled_scores = scaled_scores_by_dataset["beta"]
        alpha_scale_summaries = [row for row in scaled_summaries if row["dataset"] == "alpha"]
        beta_scale_summaries = [row for row in scaled_summaries if row["dataset"] == "beta"]
    else:
        alpha_scaled_scores, alpha_scale_summaries = scan_scaled_v03_dataset("alpha")
        beta_scaled_scores, beta_scale_summaries = scan_scaled_v03_dataset("beta")

    reset_output_folder(OUT)

    alpha = build_dataset_frame("alpha", alpha_truth, alpha_v03_scores, alpha_scaled_scores)
    beta = build_dataset_frame("beta", beta_truth, beta_v03_scores, beta_scaled_scores)

    threshold_rows = []
    metric_rows = []
    complexity_rows = []

    for experiment in EXPERIMENTS:
        score_col = str(experiment["score_col"])
        selected = select_threshold_on_alpha(alpha, score_col)
        threshold = float(selected["threshold"])

        threshold_rows.append(
            {
                "experiment": experiment["experiment"],
                "score_col": score_col,
                "threshold": threshold,
                "alpha_support": int(selected["support"]),
                "alpha_positive_support": int(selected["positive_support"]),
                "alpha_tp": int(selected["tp"]),
                "alpha_fp": int(selected["fp"]),
                "alpha_fn": int(selected["fn"]),
                "alpha_tn": int(selected["tn"]),
                "alpha_precision": float(selected["precision"]),
                "alpha_recall": float(selected["recall"]),
                "alpha_f1": float(selected["f1"]),
                "threshold_selection": "max_alpha_pooled_day_f1_tie_precision_recall_threshold",
                "score_formula": experiment["score_formula"],
            }
        )

        alpha[f"{experiment['experiment']}_pred_day"] = alpha[score_col] >= threshold
        beta[f"{experiment['experiment']}_pred_day"] = beta[score_col] >= threshold

        metric_rows.extend(evaluate_subset(experiment, alpha, "alpha", "all_alpha", threshold))
        metric_rows.extend(evaluate_subset(experiment, beta, "beta", "all_beta", threshold))
        metric_rows.extend(
            evaluate_subset(experiment, beta.loc[beta["confidence"].eq("sure")].copy(), "beta", "beta_sure_only", threshold)
        )

        complexity_rows.append(
            {
                "experiment": experiment["experiment"],
                "score_formula": experiment["score_formula"],
                "score_feature_count": experiment["score_feature_count"],
                "fixed_parameter_count": experiment["fixed_parameter_count"],
                "alpha_selected_parameter_count": experiment["alpha_selected_parameter_count"],
                "total_parameter_count": int(experiment["fixed_parameter_count"]) + int(experiment["alpha_selected_parameter_count"]),
                "uses_xgb": False,
                "uses_logistic_regression": False,
                "uses_site_specific_threshold": False,
                "uses_rolling_context": False,
                "uses_seasonal_prior": False,
                "notes": experiment["notes"],
            }
        )

    joined = pd.concat(
        [
            alpha.assign(dataset="alpha", subset_membership="all_alpha"),
            beta.assign(dataset="beta", subset_membership=np.where(beta["confidence"].eq("sure"), "all_beta;beta_sure_only", "all_beta")),
        ],
        ignore_index=True,
    )

    threshold_selection = pd.DataFrame(threshold_rows)
    day_metrics = pd.DataFrame(metric_rows)
    complexity = pd.DataFrame(complexity_rows)
    v03_summary = pd.DataFrame(v03_summaries + alpha_scale_summaries + beta_scale_summaries)

    score_pred_cols = []
    for experiment in EXPERIMENTS:
        score_pred_cols.extend([str(experiment["score_col"]), f"{experiment['experiment']}_pred_day"])
    v03_metadata_cols = [
        "v03_candidate_count",
        "v03_bridge_best",
        "v03_roughness_best",
        "v03_slope_continuity_best",
        "v03_selected_left_slot",
        "v03_selected_right_slot",
        "v03_selected_duration_h",
    ]
    joined_cols = [
        "dataset",
        "substation_id",
        "date",
        "true_day",
        "confidence",
        "subset_membership",
        *v03_metadata_cols,
        *score_pred_cols,
    ]
    joined = joined[joined_cols].sort_values(["dataset", "substation_id", "date"])

    threshold_selection.to_csv(OUT / "01_threshold_selection.csv", index=False)
    day_metrics.to_csv(OUT / "02_day_level_metrics.csv", index=False)
    joined.to_csv(OUT / "03_joined_daily_scores.csv", index=False)
    complexity.to_csv(OUT / "04_method_complexity_summary.csv", index=False)
    v03_summary.to_csv(OUT / "05_v03_candidate_score_summary.csv", index=False)

    assert len(beta) == 2928, f"Expected 2,928 Beta site-days, got {len(beta):,}"
    assert int(beta["confidence"].eq("sure").sum()) == int(load_beta_confidence()["confidence"].eq("sure").sum())
    assert {p.suffix for p in OUT.iterdir() if p.is_file()} == {".csv"}
    assert len(threshold_selection) == len(EXPERIMENTS)
    assert set(threshold_selection["experiment"]) == {str(experiment["experiment"]) for experiment in EXPERIMENTS}
    assert set(day_metrics["summary_scope"]) == {"pooled", "macro_site_average", "site"}

    print(f"Wrote outputs to {OUT.relative_to(ROOT)}")
    print("\nThreshold selection")
    print(threshold_selection[["experiment", "threshold", "alpha_precision", "alpha_recall", "alpha_f1"]].round(4).to_string(index=False))
    print("\nPooled metrics")
    pooled = day_metrics.loc[day_metrics["summary_scope"].eq("pooled")]
    print(pooled[["experiment", "dataset", "subset", "support", "positive_support", "precision", "recall", "f1"]].round(4).to_string(index=False))
    print("\nFocused E4/E11-E13 pooled comparison")
    focused = pooled.loc[
        pooled["experiment"].isin(
            [
                "E4_v03_three_feature",
                "E11_v03_duration_ge_1h",
                "E12_v03_duration_ge_1p5h",
                "E13_v03_soft_duration_penalty",
            ]
        )
        & (
            ((pooled["dataset"] == "alpha") & (pooled["subset"] == "all_alpha"))
            | ((pooled["dataset"] == "beta") & (pooled["subset"] == "beta_sure_only"))
        )
    ]
    print(focused[["experiment", "dataset", "subset", "support", "positive_support", "precision", "recall", "f1"]].round(4).to_string(index=False))
    print("\nv0.3 candidate scan summary")
    print(v03_summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

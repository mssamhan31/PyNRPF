from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_FOLDER_NAME = "260707_physical_score_loso_experiments"
SLOTS_PER_DAY = 96
DAYTIME_START = 24
DAYTIME_END = 72
SOLAR_PEAK_RADIUS_SLOTS = 14
MAX_DURATION_SLOTS = 32
EPS = 1e-9

SMOKE_ALPHA_SITE = "alpha_F"
SMOKE_BETA_SITE = "beta_F"
SMOKE_POSITIVE_DAYS = 20
SMOKE_NEGATIVE_DAYS = 20
FEATURE_COLUMNS = [
    "F1_bridge_improvement",
    "F2_roughness_improvement",
    "F3_slope_continuity_improvement",
    "F4_duration_plausibility",
    "F5_n_height_ratio",
    "F6_solar_strength_ratio",
    "F7_solar_peak_alignment",
    "F8_site_centered_core_score",
    "F9_site_rank_core_score",
]
ONE_FEATURE_VARIANTS = [
    ("S1_only_bridge", "F1_bridge_improvement"),
    ("S2_only_roughness", "F2_roughness_improvement"),
    ("S3_only_slope", "F3_slope_continuity_improvement"),
    ("S4_only_duration", "F4_duration_plausibility"),
    ("S5_only_n_height", "F5_n_height_ratio"),
    ("S6_only_solar_strength", "F6_solar_strength_ratio"),
    ("S7_only_peak_alignment", "F7_solar_peak_alignment"),
    ("S8_only_site_centered", "F8_site_centered_core_score"),
    ("S9_only_site_rank", "F9_site_rank_core_score"),
]
MANUAL_VARIANTS = [
    ("M0_all_equal", {}),
    ("M1_drop_bridge", {"F1_bridge_improvement": 0.0}),
    ("M2_drop_roughness", {"F2_roughness_improvement": 0.0}),
    ("M3_drop_slope", {"F3_slope_continuity_improvement": 0.0}),
    ("M4_drop_duration", {"F4_duration_plausibility": 0.0}),
    ("M5_drop_n_height", {"F5_n_height_ratio": 0.0}),
    ("M6_drop_solar_strength", {"F6_solar_strength_ratio": 0.0}),
    ("M7_drop_peak_alignment", {"F7_solar_peak_alignment": 0.0}),
    ("M8_drop_site_centered", {"F8_site_centered_core_score": 0.0}),
    ("M9_drop_site_rank", {"F9_site_rank_core_score": 0.0}),
]
C4_VARIANTS = [
    ("M0_all_equal", {}),
    ("M9_drop_site_rank", {"F9_site_rank_core_score": 0.0}),
    ("M8_drop_site_centered", {"F8_site_centered_core_score": 0.0}),
]
LOGISTIC_VARIANT = "M10_logistic_all9"
PALETTE = {
    "orange": "#eb932c",
    "dark_blue": "#22303d",
    "grey": "#2F4D67",
    "light_grey": "#5C7D99",
    "light_white": "#ebe3e3",
}


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
FINAL_DATASET_DIR = JOURNAL / "dataset/final"
REVIEWER_B_PATH = JOURNAL / "dataset/oracle_data_creation/archive/2026-07-02_reviewer_B_final/reviewer_B.csv"
OUT_ROOT = MISC_DIR / "outputs" / OUTPUT_FOLDER_NAME
BRIDGE_LADDER_CACHE = MISC_DIR / "outputs/11_minimal_bridge_method_ladder/03_joined_daily_scores.csv"


def date_key(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.strftime("%Y-%m-%d")


def fill_series(values: np.ndarray, default: float = 0.0) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    return series.interpolate(limit_direction="both").fillna(default).to_numpy(dtype=float)


def clip01(values: np.ndarray | float) -> np.ndarray | float:
    return np.clip(values, 0.0, 1.0)


def robust_bound(values: np.ndarray | float) -> np.ndarray | float:
    return np.clip(values, -3.0, 3.0) / 3.0


def safe_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.fillna(False).astype(str).str.strip().str.lower().isin({"true", "t", "1", "yes", "y"})


def load_beta_confidence() -> pd.DataFrame:
    conf = pd.read_csv(REVIEWER_B_PATH, usecols=["substation_id", "date", "confidence"])
    conf["substation_id"] = conf["substation_id"].astype(str).str.replace("act_", "beta_", regex=False)
    conf["date"] = date_key(conf["date"])
    conf["confidence"] = conf["confidence"].fillna("missing").astype(str).str.strip().str.lower()
    return conf


def select_evenly(items: list[str], n: int) -> list[str]:
    if len(items) <= n:
        return items
    positions = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[int(pos)] for pos in positions]


def choose_smoke_dates(day_truth: pd.DataFrame, n_pos: int, n_neg: int) -> list[str]:
    positive = day_truth.loc[day_truth["true_day"], "date"].sort_values().tolist()
    negative = day_truth.loc[~day_truth["true_day"], "date"].sort_values().tolist()
    selected = select_evenly(positive, n_pos) + select_evenly(negative, n_neg)
    return sorted(dict.fromkeys(selected))


def load_site_days(dataset: str, site: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    path = FINAL_DATASET_DIR / f"dataset_{dataset}.parquet"
    columns = ["substation_id", "date", "timestamp", "net_load_MW", "solar_MW", "label_day"]
    if dataset == "beta":
        columns.append("confidence")
    df = pd.read_parquet(
        path,
        columns=columns,
    )
    df["substation_id"] = df["substation_id"].astype(str)
    df = df.loc[df["substation_id"].eq(site)].copy()
    if df.empty:
        raise ValueError(f"No rows found for {dataset=} {site=}")
    df["date"] = date_key(df["date"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    df["label_day"] = safe_bool(df["label_day"])
    if dataset == "beta":
        df["confidence"] = df["confidence"].fillna("missing").astype(str).str.strip().str.lower()
    df = df.sort_values(["date", "timestamp"])

    day_truth = (
        df.groupby(["substation_id", "date"], as_index=False)
        .agg(true_day=("label_day", "max"))
        .sort_values("date")
    )
    selected_dates = choose_smoke_dates(day_truth, SMOKE_POSITIVE_DAYS, SMOKE_NEGATIVE_DAYS)
    df = df.loc[df["date"].isin(selected_dates)].copy()

    keys: list[dict[str, object]] = []
    net_rows: list[np.ndarray] = []
    solar_rows: list[np.ndarray] = []
    for (substation_id, day), group in df.groupby(["substation_id", "date"], sort=True):
        group = group.copy()
        group["slot"] = group["timestamp"].dt.hour * 4 + group["timestamp"].dt.minute // 15
        group = group.loc[group["slot"].between(0, SLOTS_PER_DAY - 1)]
        group = group.drop_duplicates("slot", keep="last")
        net = np.full(SLOTS_PER_DAY, np.nan, dtype=float)
        solar = np.full(SLOTS_PER_DAY, np.nan, dtype=float)
        slots = group["slot"].to_numpy(dtype=int)
        net[slots] = group["net_load_MW"].to_numpy(dtype=float)
        solar[slots] = group["solar_MW"].to_numpy(dtype=float)
        keys.append(
            {
                "dataset": dataset,
                "substation_id": substation_id,
                "date": day,
                "true_day": bool(group["label_day"].max()),
                "n_missing_net": int(np.isnan(net).sum()),
                "n_missing_solar": int(np.isnan(solar).sum()),
                "confidence": str(group["confidence"].iloc[0]) if dataset == "beta" else "not_applicable",
            }
        )
        net_rows.append(fill_series(net, 0.0))
        solar_rows.append(np.maximum(fill_series(solar, 0.0), 0.0))

    keys_df = pd.DataFrame(keys)
    return keys_df, np.vstack(net_rows), np.vstack(solar_rows)


def build_candidate_cache() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for peak in range(DAYTIME_START, DAYTIME_END + 1):
        lefts: list[int] = []
        rights: list[int] = []
        for left in range(DAYTIME_START, DAYTIME_END):
            max_right = min(DAYTIME_END, left + MAX_DURATION_SLOTS - 1)
            for right in range(left + 1, max_right + 1):
                if abs((left + right) / 2 - peak) <= SOLAR_PEAK_RADIUS_SLOTS:
                    lefts.append(left)
                    rights.append(right)
        cache[peak] = (np.asarray(lefts, dtype=np.int16), np.asarray(rights, dtype=np.int16))
    return cache


CANDIDATE_CACHE = build_candidate_cache()
SLOTS = np.arange(SLOTS_PER_DAY)


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
    return np.maximum(sse / count, 0.0)


def median_diffs(diff_matrix: np.ndarray, t_values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    indices = np.clip(t_values - 1, 0, diff_matrix.shape[1] - 1)
    gathered = np.take_along_axis(diff_matrix, indices, axis=1)
    gathered = np.where(valid, gathered, np.nan)
    return np.nanmedian(gathered, axis=1)


def candidate_base_features(net: np.ndarray, solar: np.ndarray, site_solar_scale: float) -> pd.DataFrame:
    peak = int(np.argmax(solar[DAYTIME_START : DAYTIME_END + 1])) + DAYTIME_START
    left, right = CANDIDATE_CACHE[peak]
    n_candidates = len(left)
    if n_candidates == 0:
        return pd.DataFrame()

    u_no = solar + net
    u_corr = solar - net
    bridge_no = bridge_mse(u_no, left, right, u_no)
    bridge_corr = bridge_mse(u_corr, left, right, u_no)
    f1 = (bridge_no - bridge_corr) / (bridge_no + bridge_corr + EPS)

    u_no_diff_abs = np.abs(np.diff(u_no))
    base_tv = u_no_diff_abs[DAYTIME_START:DAYTIME_END].sum()
    ctv_no = np.r_[0.0, np.cumsum(u_no_diff_abs)]
    internal_no = ctv_no[right] - ctv_no[left]
    u_corr_diff_abs = np.abs(np.diff(u_corr))
    ctv_corr = np.r_[0.0, np.cumsum(u_corr_diff_abs)]
    internal_corr = ctv_corr[right] - ctv_corr[left]
    left_jump_no = np.where(left > DAYTIME_START, np.abs(u_no[left] - u_no[left - 1]), 0.0)
    left_jump_corr = np.where(left > DAYTIME_START, np.abs(u_corr[left] - u_no[left - 1]), 0.0)
    right_jump_no = np.where(right < DAYTIME_END, np.abs(u_no[right + 1] - u_no[right]), 0.0)
    right_jump_corr = np.where(right < DAYTIME_END, np.abs(u_no[right + 1] - u_corr[right]), 0.0)
    corr_tv = base_tv - (internal_no + left_jump_no + right_jump_no) + (internal_corr + left_jump_corr + right_jump_corr)
    f2 = (base_tv - corr_tv) / (base_tv + corr_tv + EPS)

    inside = (SLOTS[None, :] >= left[:, None]) & (SLOTS[None, :] <= right[:, None])
    u_mix = np.where(inside, u_corr[None, :], u_no[None, :])
    u_no_diff = np.diff(u_no)[None, :]
    u_mix_diff = np.diff(u_mix, axis=1)
    left_before_t = left[:, None] + np.array([-3, -2, -1])
    left_after_t = left[:, None] + np.array([1, 2, 3])
    right_before_t = right[:, None] + np.array([-2, -1, 0])
    right_after_t = right[:, None] + np.array([2, 3, 4])
    left_before_valid = np.ones_like(left_before_t, dtype=bool)
    left_after_valid = left_after_t <= right[:, None]
    right_before_valid = right_before_t >= left[:, None] + 1
    right_after_valid = np.ones_like(right_after_t, dtype=bool)
    u_no_diff_matrix = np.repeat(u_no_diff, n_candidates, axis=0)
    no_left_before = median_diffs(u_no_diff_matrix, left_before_t, left_before_valid)
    no_left_after = median_diffs(u_no_diff_matrix, left_after_t, left_after_valid)
    no_right_before = median_diffs(u_no_diff_matrix, right_before_t, right_before_valid)
    no_right_after = median_diffs(u_no_diff_matrix, right_after_t, right_after_valid)
    corr_left_before = median_diffs(u_mix_diff, left_before_t, left_before_valid)
    corr_left_after = median_diffs(u_mix_diff, left_after_t, left_after_valid)
    corr_right_before = median_diffs(u_mix_diff, right_before_t, right_before_valid)
    corr_right_after = median_diffs(u_mix_diff, right_after_t, right_after_valid)
    slope_no = np.abs(no_left_before - no_left_after) + np.abs(no_right_before - no_right_after)
    slope_corr = np.abs(corr_left_before - corr_left_after) + np.abs(corr_right_before - corr_right_after)
    f3 = (slope_no - slope_corr) / (slope_no + slope_corr + EPS)

    duration_h = (right - left + 1) * 0.25
    f4 = clip01(duration_h / 1.5)
    day_net_scale = max(float(np.nanpercentile(np.abs(net[DAYTIME_START : DAYTIME_END + 1]), 95)), float(np.nanpercentile(solar[DAYTIME_START : DAYTIME_END + 1], 95)), EPS)
    net_peak_inside = np.array([np.nanmax(net[int(l) : int(r) + 1]) for l, r in zip(left, right)])
    net_edge = np.maximum(net[left], net[right])
    f5 = clip01((net_peak_inside - net_edge) / day_net_scale)
    solar_p95_inside = np.array([np.nanpercentile(solar[int(l) : int(r) + 1], 95) for l, r in zip(left, right)])
    f6 = clip01(solar_p95_inside / max(site_solar_scale, EPS))
    midpoint = (left + right) / 2
    f7 = clip01(1 - np.abs(midpoint - peak) / SOLAR_PEAK_RADIUS_SLOTS)

    out = pd.DataFrame(
        {
            "left_slot": left.astype(int),
            "right_slot": right.astype(int),
            "duration_h": duration_h,
            "solar_peak_slot": peak,
            "F1_bridge_improvement": f1,
            "F2_roughness_improvement": f2,
            "F3_slope_continuity_improvement": f3,
            "F4_duration_plausibility": f4,
            "F5_n_height_ratio": f5,
            "F6_solar_strength_ratio": f6,
            "F7_solar_peak_alignment": f7,
        }
    )
    out["core_score"] = out["F1_bridge_improvement"] + out["F2_roughness_improvement"] + out["F3_slope_continuity_improvement"]
    return out


def slot_to_time(slot: float | int | None) -> str:
    if pd.isna(slot):
        return ""
    slot = int(slot)
    return f"{slot // 4:02d}:{(slot % 4) * 15:02d}"


def compute_site_solar_scale(solar: np.ndarray) -> float:
    daily = np.nanpercentile(solar[:, DAYTIME_START : DAYTIME_END + 1], 95, axis=1)
    return max(float(np.nanmedian(daily)), EPS)


def load_final_day_truth(dataset: str) -> pd.DataFrame:
    columns = ["substation_id", "date", "label_day"]
    if dataset == "beta":
        columns.append("confidence")
    df = pd.read_parquet(FINAL_DATASET_DIR / f"dataset_{dataset}.parquet", columns=columns)
    df["substation_id"] = df["substation_id"].astype(str)
    df["date"] = date_key(df["date"])
    df["label_day"] = safe_bool(df["label_day"])
    agg_spec: dict[str, tuple[str, str]] = {"true_day": ("label_day", "max")}
    if dataset == "beta":
        df["confidence"] = df["confidence"].fillna("missing").astype(str).str.strip().str.lower()
        agg_spec["confidence"] = ("confidence", "first")
    out = df.groupby(["substation_id", "date"], as_index=False).agg(**agg_spec)
    out.insert(0, "dataset", dataset)
    if "confidence" not in out.columns:
        out["confidence"] = "not_applicable"
    return out


def load_refreshed_truth() -> pd.DataFrame:
    return pd.concat([load_final_day_truth("alpha"), load_final_day_truth("beta")], ignore_index=True)


def load_bridge_ladder_cache() -> pd.DataFrame:
    if not BRIDGE_LADDER_CACHE.exists():
        raise FileNotFoundError(f"Missing bridge ladder cache: {BRIDGE_LADDER_CACHE}")
    cache = pd.read_csv(BRIDGE_LADDER_CACHE)
    required = {
        "dataset",
        "substation_id",
        "date",
        "v03_candidate_count",
        "v03_bridge_best",
        "v03_roughness_best",
        "v03_slope_continuity_best",
        "v03_selected_left_slot",
        "v03_selected_right_slot",
        "v03_selected_duration_h",
    }
    missing = required.difference(cache.columns)
    if missing:
        raise ValueError(f"Bridge ladder cache is missing required columns: {sorted(missing)}")
    cache["date"] = date_key(cache["date"])
    return cache.drop(columns=[col for col in ["true_day", "confidence"] if col in cache.columns])


def attach_refreshed_truth(cache: pd.DataFrame) -> pd.DataFrame:
    truth = load_refreshed_truth()
    merged = cache.merge(truth, on=["dataset", "substation_id", "date"], how="inner", validate="one_to_one")
    expected_rows = len(truth)
    if len(merged) != expected_rows:
        missing = truth.merge(
            merged[["dataset", "substation_id", "date"]],
            on=["dataset", "substation_id", "date"],
            how="left",
            indicator=True,
        ).query("_merge == 'left_only'")
        raise ValueError(
            f"Bridge cache/truth row mismatch: merged {len(merged)} rows, expected {expected_rows}. "
            f"First missing rows: {missing.head(5).to_dict(orient='records')}"
        )
    return merged


def compute_selected_window_components(dataset: str, daily_scores: pd.DataFrame) -> pd.DataFrame:
    selected = daily_scores.loc[daily_scores["dataset"].eq(dataset)].copy()
    selected = selected.set_index(["substation_id", "date"], drop=False)
    df = pd.read_parquet(
        FINAL_DATASET_DIR / f"dataset_{dataset}.parquet",
        columns=["substation_id", "date", "timestamp", "net_load_MW", "solar_MW"],
    )
    df["substation_id"] = df["substation_id"].astype(str)
    df["date"] = date_key(df["date"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    df["slot"] = df["timestamp"].dt.hour * 4 + df["timestamp"].dt.minute // 15
    df = df.loc[df["slot"].between(0, SLOTS_PER_DAY - 1)].copy()
    df = df.sort_values(["substation_id", "date", "timestamp"]).drop_duplicates(
        ["substation_id", "date", "slot"],
        keep="last",
    )

    rows: list[dict[str, object]] = []
    for (site, day), group in df.groupby(["substation_id", "date"], sort=True):
        if (site, day) not in selected.index:
            continue
        score_row = selected.loc[(site, day)]
        net = np.full(SLOTS_PER_DAY, np.nan, dtype=float)
        solar = np.full(SLOTS_PER_DAY, np.nan, dtype=float)
        slots = group["slot"].to_numpy(dtype=int)
        net[slots] = group["net_load_MW"].to_numpy(dtype=float)
        solar[slots] = group["solar_MW"].to_numpy(dtype=float)
        n_missing_net = int(np.isnan(net).sum())
        n_missing_solar = int(np.isnan(solar).sum())
        net = fill_series(net, 0.0)
        solar = np.maximum(fill_series(solar, 0.0), 0.0)
        daytime_net = net[DAYTIME_START : DAYTIME_END + 1]
        daytime_solar = solar[DAYTIME_START : DAYTIME_END + 1]
        solar_day_p95 = float(np.nanpercentile(daytime_solar, 95))
        net_day_abs_p95 = float(np.nanpercentile(np.abs(daytime_net), 95))
        day_net_scale = max(net_day_abs_p95, solar_day_p95, EPS)
        solar_peak_slot = int(np.nanargmax(daytime_solar)) + DAYTIME_START

        left = score_row.get("v03_selected_left_slot")
        right = score_row.get("v03_selected_right_slot")
        has_window = not pd.isna(left) and not pd.isna(right)
        if has_window:
            left_i = int(left)
            right_i = int(right)
            window_net = net[left_i : right_i + 1]
            window_solar = solar[left_i : right_i + 1]
            net_peak_inside = float(np.nanmax(window_net))
            net_edge = float(max(net[left_i], net[right_i]))
            n_height_raw = net_peak_inside - net_edge
            solar_p95_inside = float(np.nanpercentile(window_solar, 95))
            midpoint = (left_i + right_i) / 2.0
            f5 = float(clip01(n_height_raw / day_net_scale))
            f7 = float(clip01(1 - abs(midpoint - solar_peak_slot) / SOLAR_PEAK_RADIUS_SLOTS))
        else:
            left_i = np.nan
            right_i = np.nan
            solar_p95_inside = 0.0
            n_height_raw = 0.0
            f5 = 0.0
            f7 = 0.0

        rows.append(
            {
                "dataset": dataset,
                "substation_id": site,
                "date": day,
                "selected_left_slot": left_i,
                "selected_right_slot": right_i,
                "solar_day_p95": solar_day_p95,
                "net_day_abs_p95": net_day_abs_p95,
                "day_net_scale": day_net_scale,
                "solar_peak_slot": solar_peak_slot,
                "solar_p95_inside_selected": solar_p95_inside,
                "n_height_raw_selected": n_height_raw,
                "F5_n_height_ratio": f5,
                "F7_solar_peak_alignment": f7,
                "n_missing_net": n_missing_net,
                "n_missing_solar": n_missing_solar,
            }
        )
    out = pd.DataFrame(rows)
    out["site_solar_scale"] = out.groupby(["dataset", "substation_id"])["solar_day_p95"].transform("median")
    out["F6_solar_strength_ratio"] = clip01(
        out["solar_p95_inside_selected"] / out["site_solar_scale"].clip(lower=EPS)
    )
    return out


def build_c1_daily_feature_cache() -> pd.DataFrame:
    cache = attach_refreshed_truth(load_bridge_ladder_cache())
    feature_parts = [compute_selected_window_components(dataset, cache) for dataset in ["alpha", "beta"]]
    selected_features = pd.concat(feature_parts, ignore_index=True)
    daily = cache.merge(
        selected_features,
        on=["dataset", "substation_id", "date"],
        how="left",
        validate="one_to_one",
    )
    daily["F1_bridge_improvement"] = daily["v03_bridge_best"].fillna(0.0)
    daily["F2_roughness_improvement"] = daily["v03_roughness_best"].fillna(0.0)
    daily["F3_slope_continuity_improvement"] = daily["v03_slope_continuity_best"].fillna(0.0)
    daily["F4_duration_plausibility"] = clip01(daily["v03_selected_duration_h"].fillna(0.0) / 1.5)
    daily["core_score"] = (
        daily["F1_bridge_improvement"]
        + daily["F2_roughness_improvement"]
        + daily["F3_slope_continuity_improvement"]
    )
    daily["site_median_daily_core_score"] = daily.groupby(["dataset", "substation_id"])["core_score"].transform(
        "median"
    )
    daily["site_core_rank_pct"] = daily.groupby(["dataset", "substation_id"])["core_score"].rank(pct=True)
    daily["F8_site_centered_core_score"] = robust_bound(
        daily["core_score"] - daily["site_median_daily_core_score"]
    )
    daily["F9_site_rank_core_score"] = 2 * daily["site_core_rank_pct"] - 1
    feature_cols = [col for col in daily.columns if col.startswith("F")]
    daily["M0_all_equal_score"] = daily[feature_cols].sum(axis=1)
    daily["selected_start_time"] = daily["v03_selected_left_slot"].map(slot_to_time)
    daily["selected_end_time"] = daily["v03_selected_right_slot"].map(slot_to_time)
    return daily


def build_smoke_features(keys: pd.DataFrame, net: np.ndarray, solar: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    site_solar_scale = compute_site_solar_scale(solar)
    day_candidates: list[pd.DataFrame] = []
    day_rows: list[dict[str, object]] = []
    for idx, key in keys.reset_index(drop=True).iterrows():
        candidates = candidate_base_features(net[idx], solar[idx], site_solar_scale)
        candidates.insert(0, "dataset", key["dataset"])
        candidates.insert(1, "substation_id", key["substation_id"])
        candidates.insert(2, "date", key["date"])
        candidates.insert(3, "candidate_id", np.arange(len(candidates), dtype=int))
        day_candidates.append(candidates)
        day_rows.append(
            {
                **key.to_dict(),
                "site_solar_scale": site_solar_scale,
                "candidate_count": int(len(candidates)),
                "core_score_day": float(candidates["core_score"].max()),
            }
        )

    candidate_frame = pd.concat(day_candidates, ignore_index=True)
    day_frame = pd.DataFrame(day_rows)
    day_frame["site_median_daily_core_score"] = day_frame.groupby(["dataset", "substation_id"])["core_score_day"].transform("median")
    day_frame["site_core_rank_pct"] = day_frame.groupby(["dataset", "substation_id"])["core_score_day"].rank(pct=True)
    candidate_frame = candidate_frame.merge(
        day_frame[["dataset", "substation_id", "date", "site_median_daily_core_score", "site_core_rank_pct"]],
        on=["dataset", "substation_id", "date"],
        how="left",
    )
    candidate_frame["F8_site_centered_core_score"] = robust_bound(candidate_frame["core_score"] - candidate_frame["site_median_daily_core_score"])
    candidate_frame["F9_site_rank_core_score"] = 2 * candidate_frame["site_core_rank_pct"] - 1
    feature_cols = [col for col in candidate_frame.columns if col.startswith("F")]
    candidate_frame["M0_all_equal_score"] = candidate_frame[feature_cols].sum(axis=1)

    best_idx = candidate_frame.groupby(["dataset", "substation_id", "date"])["M0_all_equal_score"].idxmax()
    selected = candidate_frame.loc[best_idx].copy()
    selected = selected.rename(
        columns={
            "left_slot": "selected_left_slot",
            "right_slot": "selected_right_slot",
            "duration_h": "selected_duration_h",
            "M0_all_equal_score": "score",
        }
    )
    day_frame = day_frame.merge(
        selected[
            [
                "dataset",
                "substation_id",
                "date",
                "candidate_id",
                "selected_left_slot",
                "selected_right_slot",
                "selected_duration_h",
                "solar_peak_slot",
                "score",
                *feature_cols,
            ]
        ],
        on=["dataset", "substation_id", "date"],
        how="left",
    )
    day_frame["selected_start_time"] = day_frame["selected_left_slot"].map(slot_to_time)
    day_frame["selected_end_time"] = day_frame["selected_right_slot"].map(slot_to_time)
    return day_frame, candidate_frame


def compute_metrics(true_values: pd.Series, pred_values: pd.Series) -> dict[str, float | int]:
    true = true_values.astype(bool).to_numpy()
    pred = pred_values.astype(bool).to_numpy()
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
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def select_threshold(train: pd.DataFrame) -> tuple[float, dict[str, float | int]]:
    scores = sorted(train["score"].dropna().unique().tolist())
    thresholds = [min(scores) - 1e-9, *scores, max(scores) + 1e-9]
    rows = []
    for threshold in thresholds:
        pred = train["score"] >= threshold
        metrics = compute_metrics(train["true_day"], pred)
        rows.append({"threshold": float(threshold), **metrics})
    sweep = pd.DataFrame(rows)
    best = sweep.sort_values(["f1", "precision", "recall", "threshold"], ascending=[False, False, False, False]).iloc[0]
    return float(best["threshold"]), best.to_dict()


def variant_score(frame: pd.DataFrame, zero_weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for col in FEATURE_COLUMNS:
        weight = zero_weights.get(col, 1.0)
        if weight:
            score = score + weight * frame[col].fillna(0.0)
    return score


def site_macro_metrics(frame: pd.DataFrame, pred_col: str) -> dict[str, float | int]:
    site_rows = [compute_metrics(group["true_day"], group[pred_col]) for _, group in frame.groupby("substation_id")]
    site_metrics = pd.DataFrame(site_rows)
    return {
        "support": int(site_metrics["support"].sum()),
        "positive_support": int(site_metrics["positive_support"].sum()),
        "tp": int(site_metrics["tp"].sum()),
        "fp": int(site_metrics["fp"].sum()),
        "fn": int(site_metrics["fn"].sum()),
        "tn": int(site_metrics["tn"].sum()),
        "precision": float(site_metrics["precision"].mean()),
        "recall": float(site_metrics["recall"].mean()),
        "f1": float(site_metrics["f1"].mean()),
    }


def site_metric_frame(frame: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for site, site_frame in frame.groupby("substation_id", sort=True):
        rows.append({"substation_id": site, **compute_metrics(site_frame["true_day"], site_frame[pred_col])})
    return pd.DataFrame(rows)


def select_threshold_macro_site(train: pd.DataFrame, score_col: str) -> tuple[float, dict[str, object], pd.DataFrame]:
    data = train[["substation_id", "true_day", score_col]].dropna().copy()
    sites = sorted(data["substation_id"].unique().tolist())
    site_index = {site: idx for idx, site in enumerate(sites)}
    site_codes = data["substation_id"].map(site_index).to_numpy(dtype=int)
    true = data["true_day"].astype(bool).to_numpy()
    scores_asc = np.sort(data[score_col].unique())
    score_codes_asc = np.searchsorted(scores_asc, data[score_col].to_numpy(dtype=float))
    score_codes_desc = len(scores_asc) - 1 - score_codes_asc

    tp_inc = np.zeros((len(scores_asc), len(sites)), dtype=float)
    fp_inc = np.zeros_like(tp_inc)
    np.add.at(tp_inc, (score_codes_desc[true], site_codes[true]), 1)
    np.add.at(fp_inc, (score_codes_desc[~true], site_codes[~true]), 1)
    tp_cum = np.cumsum(tp_inc, axis=0)
    fp_cum = np.cumsum(fp_inc, axis=0)

    site_support = data.groupby("substation_id").size().reindex(sites).to_numpy(dtype=float)
    site_pos = data.groupby("substation_id")["true_day"].sum().reindex(sites).to_numpy(dtype=float)
    site_neg = site_support - site_pos

    zero_tp = np.zeros((1, len(sites)), dtype=float)
    zero_fp = np.zeros_like(zero_tp)
    tp_all = np.vstack([zero_tp, tp_cum])
    fp_all = np.vstack([zero_fp, fp_cum])
    thresholds = np.r_[scores_asc[-1] + 1e-9, scores_asc[::-1]]

    fn_all = site_pos[None, :] - tp_all
    tn_all = site_neg[None, :] - fp_all
    site_precision = np.divide(tp_all, tp_all + fp_all, out=np.zeros_like(tp_all), where=(tp_all + fp_all) > 0)
    site_recall = np.divide(tp_all, site_pos[None, :], out=np.zeros_like(tp_all), where=site_pos[None, :] > 0)
    site_f1 = np.divide(
        2 * site_precision * site_recall,
        site_precision + site_recall,
        out=np.zeros_like(site_precision),
        where=(site_precision + site_recall) > 0,
    )

    tp = tp_all.sum(axis=1)
    fp = fp_all.sum(axis=1)
    fn = fn_all.sum(axis=1)
    tn = tn_all.sum(axis=1)
    pooled_precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    pooled_recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    pooled_f1 = np.divide(
        2 * pooled_precision * pooled_recall,
        pooled_precision + pooled_recall,
        out=np.zeros_like(pooled_precision),
        where=(pooled_precision + pooled_recall) > 0,
    )
    rows = {
        "threshold": thresholds.astype(float),
        "selection_metric": "alpha_macro_site_f1",
        "macro_precision": site_precision.mean(axis=1),
        "macro_recall": site_recall.mean(axis=1),
        "macro_f1": site_f1.mean(axis=1),
        "pooled_precision": pooled_precision,
        "pooled_recall": pooled_recall,
        "pooled_f1": pooled_f1,
        "tp": tp.astype(int),
        "fp": fp.astype(int),
        "fn": fn.astype(int),
        "tn": tn.astype(int),
    }
    sweep = pd.DataFrame(rows)
    best = sweep.sort_values(
        ["macro_f1", "pooled_f1", "macro_precision", "macro_recall", "threshold"],
        ascending=[False, False, False, False, False],
    ).iloc[0]
    return float(best["threshold"]), best.to_dict(), sweep


def select_threshold_weighted_macro_site(
    train: pd.DataFrame,
    score_col: str,
    *,
    dataset_balanced: bool,
) -> tuple[float, dict[str, object], pd.DataFrame]:
    data = train[["dataset", "substation_id", "true_day", score_col]].dropna().copy()
    data["tuning_group"] = data["dataset"].astype(str) + "|" + data["substation_id"].astype(str)
    groups = sorted(data["tuning_group"].unique().tolist())
    group_index = {group: idx for idx, group in enumerate(groups)}
    group_codes = data["tuning_group"].map(group_index).to_numpy(dtype=int)
    true = data["true_day"].astype(bool).to_numpy()
    scores_asc = np.sort(data[score_col].unique())
    score_codes_asc = np.searchsorted(scores_asc, data[score_col].to_numpy(dtype=float))
    score_codes_desc = len(scores_asc) - 1 - score_codes_asc

    tp_inc = np.zeros((len(scores_asc), len(groups)), dtype=float)
    fp_inc = np.zeros_like(tp_inc)
    np.add.at(tp_inc, (score_codes_desc[true], group_codes[true]), 1)
    np.add.at(fp_inc, (score_codes_desc[~true], group_codes[~true]), 1)
    tp_cum = np.cumsum(tp_inc, axis=0)
    fp_cum = np.cumsum(fp_inc, axis=0)

    group_frame = data[["tuning_group", "dataset"]].drop_duplicates().set_index("tuning_group").loc[groups]
    if dataset_balanced:
        dataset_counts = group_frame["dataset"].value_counts().to_dict()
        n_datasets = len(dataset_counts)
        weights = np.array(
            [1.0 / n_datasets / dataset_counts[group_frame.loc[group, "dataset"]] for group in groups],
            dtype=float,
        )
    else:
        weights = np.full(len(groups), 1.0 / len(groups), dtype=float)

    group_support = data.groupby("tuning_group").size().reindex(groups).to_numpy(dtype=float)
    group_pos = data.groupby("tuning_group")["true_day"].sum().reindex(groups).to_numpy(dtype=float)
    group_neg = group_support - group_pos
    tp_all = np.vstack([np.zeros((1, len(groups)), dtype=float), tp_cum])
    fp_all = np.vstack([np.zeros((1, len(groups)), dtype=float), fp_cum])
    thresholds = np.r_[scores_asc[-1] + 1e-9, scores_asc[::-1]]

    fn_all = group_pos[None, :] - tp_all
    group_precision = np.divide(tp_all, tp_all + fp_all, out=np.zeros_like(tp_all), where=(tp_all + fp_all) > 0)
    group_recall = np.divide(tp_all, group_pos[None, :], out=np.zeros_like(tp_all), where=group_pos[None, :] > 0)
    group_f1 = np.divide(
        2 * group_precision * group_recall,
        group_precision + group_recall,
        out=np.zeros_like(group_precision),
        where=(group_precision + group_recall) > 0,
    )

    tp = tp_all.sum(axis=1)
    fp = fp_all.sum(axis=1)
    fn = fn_all.sum(axis=1)
    tn = (group_neg[None, :] - fp_all).sum(axis=1)
    pooled_precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    pooled_recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    pooled_f1 = np.divide(
        2 * pooled_precision * pooled_recall,
        pooled_precision + pooled_recall,
        out=np.zeros_like(pooled_precision),
        where=(pooled_precision + pooled_recall) > 0,
    )
    weighted_precision = group_precision @ weights
    weighted_recall = group_recall @ weights
    weighted_f1 = group_f1 @ weights
    sweep = pd.DataFrame(
        {
            "threshold": thresholds.astype(float),
            "selection_metric": "dataset_balanced_macro_site_f1" if dataset_balanced else "macro_site_f1",
            "macro_precision": group_precision.mean(axis=1),
            "macro_recall": group_recall.mean(axis=1),
            "macro_f1": group_f1.mean(axis=1),
            "weighted_macro_precision": weighted_precision,
            "weighted_macro_recall": weighted_recall,
            "weighted_macro_f1": weighted_f1,
            "pooled_precision": pooled_precision,
            "pooled_recall": pooled_recall,
            "pooled_f1": pooled_f1,
            "tp": tp.astype(int),
            "fp": fp.astype(int),
            "fn": fn.astype(int),
            "tn": tn.astype(int),
        }
    )
    best = sweep.sort_values(
        ["weighted_macro_f1", "pooled_f1", "weighted_macro_precision", "weighted_macro_recall", "threshold"],
        ascending=[False, False, False, False, False],
    ).iloc[0]
    return float(best["threshold"]), best.to_dict(), sweep


def metric_rows_for_subset(
    frame: pd.DataFrame,
    *,
    variant: str,
    dataset: str,
    subset: str,
    pred_col: str,
    threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pooled = compute_metrics(frame["true_day"], frame[pred_col])
    rows.append(
        {
            "variant": variant,
            "dataset": dataset,
            "subset": subset,
            "summary_scope": "pooled",
            "substation_id": "",
            "threshold": threshold,
            **pooled,
        }
    )
    macro = site_macro_metrics(frame, pred_col)
    rows.append(
        {
            "variant": variant,
            "dataset": dataset,
            "subset": subset,
            "summary_scope": "macro_site_average",
            "substation_id": "",
            "threshold": threshold,
            **macro,
        }
    )
    site_metrics = site_metric_frame(frame, pred_col)
    positive_site_metrics = site_metrics.loc[site_metrics["positive_support"] > 0].copy()
    if not positive_site_metrics.empty:
        rows.append(
            {
                "variant": variant,
                "dataset": dataset,
                "subset": subset,
                "summary_scope": "positive_site_macro_average",
                "substation_id": "",
                "threshold": threshold,
                "support": int(positive_site_metrics["support"].sum()),
                "positive_support": int(positive_site_metrics["positive_support"].sum()),
                "tp": int(positive_site_metrics["tp"].sum()),
                "fp": int(positive_site_metrics["fp"].sum()),
                "fn": int(positive_site_metrics["fn"].sum()),
                "tn": int(positive_site_metrics["tn"].sum()),
                "precision": float(positive_site_metrics["precision"].mean()),
                "recall": float(positive_site_metrics["recall"].mean()),
                "f1": float(positive_site_metrics["f1"].mean()),
            }
        )
    for row in site_metrics.to_dict(orient="records"):
        rows.append(
            {
                "variant": variant,
                "dataset": dataset,
                "subset": subset,
                "summary_scope": "site",
                "substation_id": row.pop("substation_id"),
                "threshold": threshold,
                **row,
            }
        )
    return rows


def load_c1_daily_feature_cache() -> pd.DataFrame:
    path = OUT_ROOT / "C1_cached_daily_features/02_c1_daily_feature_cache.csv"
    if not path.exists():
        print("C1 daily feature cache is missing; building it now.")
        run_c1_cached_daily_features()
    daily = pd.read_csv(path)
    missing = [col for col in FEATURE_COLUMNS if col not in daily.columns]
    if missing:
        raise ValueError(f"C1 daily feature cache is missing feature columns: {missing}")
    daily["true_day"] = safe_bool(daily["true_day"])
    daily["date"] = date_key(daily["date"])
    daily["confidence"] = daily["confidence"].fillna("missing").astype(str).str.strip().str.lower()
    return daily


def logistic_sample_weights(train: pd.DataFrame, *, dataset_balanced: bool) -> np.ndarray:
    frame = train[["dataset", "substation_id", "true_day"]].copy()
    frame["group"] = frame["dataset"].astype(str) + "|" + frame["substation_id"].astype(str)
    weights = np.zeros(len(frame), dtype=float)
    if dataset_balanced:
        dataset_values = sorted(frame["dataset"].unique().tolist())
        dataset_weight = {dataset: 1.0 / len(dataset_values) for dataset in dataset_values}
        for dataset, dataset_frame in frame.groupby("dataset", sort=True):
            groups = dataset_frame["group"].unique().tolist()
            group_weight = dataset_weight[dataset] / len(groups)
            for group in groups:
                idx = frame.index[frame["group"].eq(group)].to_numpy()
                weights[idx] = group_weight / len(idx)
    else:
        groups = frame["group"].unique().tolist()
        for group in groups:
            idx = frame.index[frame["group"].eq(group)].to_numpy()
            weights[idx] = (1.0 / len(groups)) / len(idx)

    y = frame["true_day"].astype(bool).to_numpy()
    pos = max(int(y.sum()), 1)
    neg = max(int((~y).sum()), 1)
    class_weights = np.where(y, len(y) / (2 * pos), len(y) / (2 * neg))
    weights = weights * class_weights
    return weights / weights.mean()


def fit_logistic_model(train: pd.DataFrame, *, dataset_balanced: bool):
    from sklearn.linear_model import LogisticRegression

    x = train[FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=float)
    y = train["true_day"].astype(bool).to_numpy(dtype=int)
    weights = logistic_sample_weights(train.reset_index(drop=True), dataset_balanced=dataset_balanced)
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=9,
    )
    model.fit(x, y, sample_weight=weights)
    return model


def predict_logistic(model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame[FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=float))[:, 1]


def run_c0_smoke() -> Path:
    started = time.time()
    out = OUT_ROOT / "C0_smoke_feature_cache"
    out.mkdir(parents=True, exist_ok=True)

    alpha_keys, alpha_net, alpha_solar = load_site_days("alpha", SMOKE_ALPHA_SITE)
    beta_keys, beta_net, beta_solar = load_site_days("beta", SMOKE_BETA_SITE)
    alpha_days, alpha_candidates = build_smoke_features(alpha_keys, alpha_net, alpha_solar)
    beta_days, beta_candidates = build_smoke_features(beta_keys, beta_net, beta_solar)
    days = pd.concat([alpha_days, beta_days], ignore_index=True)
    candidates = pd.concat([alpha_candidates, beta_candidates], ignore_index=True)

    threshold, threshold_metrics = select_threshold(alpha_days)
    days["threshold"] = threshold
    days["pred_day"] = days["score"] >= threshold
    days["confidence_score"] = (days["score"] - days["threshold"]).abs()
    days["predicted_start_time"] = np.where(days["pred_day"], days["selected_start_time"], "")
    days["predicted_end_time"] = np.where(days["pred_day"], days["selected_end_time"], "")

    metrics_rows: list[dict[str, object]] = []
    for dataset, subset in [("alpha", "smoke_alpha"), ("beta", "smoke_beta_all"), ("beta", "smoke_beta_sure_only")]:
        frame = days.loc[days["dataset"].eq(dataset)].copy()
        if subset == "smoke_beta_sure_only":
            frame = frame.loc[frame["confidence"].eq("sure")].copy()
        metrics_rows.append({"dataset": dataset, "subset": subset, **compute_metrics(frame["true_day"], frame["pred_day"])})
    metrics = pd.DataFrame(metrics_rows)

    feature_cols = [col for col in days.columns if col.startswith("F")]
    feature_summary = (
        days.groupby(["dataset", "substation_id"], as_index=False)
        .agg(
            site_days=("date", "count"),
            rpf_days=("true_day", "sum"),
            mean_candidate_count=("candidate_count", "mean"),
            min_candidate_count=("candidate_count", "min"),
            max_candidate_count=("candidate_count", "max"),
            mean_score=("score", "mean"),
            min_score=("score", "min"),
            max_score=("score", "max"),
            missing_net_total=("n_missing_net", "sum"),
            missing_solar_total=("n_missing_solar", "sum"),
        )
    )
    component_summary = (
        days.melt(
            id_vars=["dataset", "substation_id", "date", "true_day"],
            value_vars=feature_cols,
            var_name="component",
            value_name="value",
        )
        .groupby(["dataset", "substation_id", "component"], as_index=False)
        .agg(mean=("value", "mean"), min=("value", "min"), max=("value", "max"))
    )
    threshold_frame = pd.DataFrame(
        [
            {
                "chunk": "C0_smoke_feature_cache",
                "variant": "M0_all_equal",
                "threshold_source": f"{SMOKE_ALPHA_SITE}_smoke_alpha",
                "threshold": threshold,
                **{f"alpha_select_{key}": value for key, value in threshold_metrics.items() if key != "threshold"},
            }
        ]
    )
    manifest = pd.DataFrame(
        [
            {
                "chunk": "C0_smoke_feature_cache",
                "alpha_site": SMOKE_ALPHA_SITE,
                "beta_site": SMOKE_BETA_SITE,
                "alpha_site_days": len(alpha_days),
                "beta_site_days": len(beta_days),
                "variant": "M0_all_equal",
                "elapsed_seconds": time.time() - started,
                "outputs": str(out.relative_to(ROOT)),
            }
        ]
    )

    manifest.to_csv(out / "01_smoke_manifest.csv", index=False)
    feature_summary.to_csv(out / "02_smoke_feature_summary.csv", index=False)
    metrics.to_csv(out / "03_smoke_day_metrics.csv", index=False)
    days.to_csv(out / "04_smoke_selected_windows_audit.csv", index=False)
    component_summary.to_csv(out / "05_smoke_component_summary.csv", index=False)
    threshold_frame.to_csv(out / "06_smoke_threshold_selection.csv", index=False)
    candidates.head(2000).to_csv(out / "07_smoke_candidate_rows_sample.csv", index=False)

    print(f"Wrote C0 smoke outputs to {out.relative_to(ROOT)}")
    print("\nManifest")
    print(manifest.round(3).to_string(index=False))
    print("\nFeature summary")
    print(feature_summary.round(4).to_string(index=False))
    print("\nThreshold")
    print(threshold_frame.round(4).to_string(index=False))
    print("\nDay metrics")
    print(metrics.round(4).to_string(index=False))
    return out


def run_c1_cached_daily_features() -> Path:
    started = time.time()
    out = OUT_ROOT / "C1_cached_daily_features"
    out.mkdir(parents=True, exist_ok=True)

    daily = build_c1_daily_feature_cache()
    feature_cols = [col for col in daily.columns if col.startswith("F")]
    identity_cols = [
        "dataset",
        "substation_id",
        "date",
        "true_day",
        "confidence",
        "v03_candidate_count",
        "v03_selected_left_slot",
        "v03_selected_right_slot",
        "v03_selected_duration_h",
        "selected_start_time",
        "selected_end_time",
        "M0_all_equal_score",
        "n_missing_net",
        "n_missing_solar",
    ]
    keep_cols = [col for col in identity_cols if col in daily.columns] + feature_cols
    daily_cache = daily[keep_cols].copy()

    dataset_summary = (
        daily.groupby("dataset", as_index=False)
        .agg(
            site_days=("date", "count"),
            sites=("substation_id", "nunique"),
            rpf_days=("true_day", "sum"),
            mean_candidate_count=("v03_candidate_count", "mean"),
            median_candidate_count=("v03_candidate_count", "median"),
            min_candidate_count=("v03_candidate_count", "min"),
            max_candidate_count=("v03_candidate_count", "max"),
            mean_m0_score=("M0_all_equal_score", "mean"),
            median_m0_score=("M0_all_equal_score", "median"),
            missing_net_total=("n_missing_net", "sum"),
            missing_solar_total=("n_missing_solar", "sum"),
        )
    )
    beta_confidence_summary = (
        daily.loc[daily["dataset"].eq("beta")]
        .groupby("confidence", as_index=False)
        .agg(site_days=("date", "count"), rpf_days=("true_day", "sum"))
        .sort_values("confidence")
    )
    feature_summary = (
        daily.melt(
            id_vars=["dataset", "substation_id", "date", "true_day"],
            value_vars=feature_cols,
            var_name="component",
            value_name="value",
        )
        .groupby(["dataset", "component"], as_index=False)
        .agg(
            non_null=("value", "count"),
            missing=("value", lambda x: int(x.isna().sum())),
            mean=("value", "mean"),
            median=("value", "median"),
            min=("value", "min"),
            max=("value", "max"),
        )
    )
    missing_feature_summary = pd.DataFrame(
        [
            {
                "component": col,
                "missing_count": int(daily[col].isna().sum()),
                "missing_pct": float(daily[col].isna().mean() * 100),
            }
            for col in feature_cols
        ]
    )
    candidate_count_summary = (
        daily.groupby(["dataset", "substation_id"], as_index=False)
        .agg(
            site_days=("date", "count"),
            rpf_days=("true_day", "sum"),
            mean_candidate_count=("v03_candidate_count", "mean"),
            median_candidate_count=("v03_candidate_count", "median"),
            min_candidate_count=("v03_candidate_count", "min"),
            max_candidate_count=("v03_candidate_count", "max"),
        )
    )
    manifest = pd.DataFrame(
        [
            {
                "chunk": "C1_cached_daily_features",
                "used_bridge_ladder_cache": True,
                "bridge_ladder_cache": str(BRIDGE_LADDER_CACHE.relative_to(ROOT)),
                "daily_rows": len(daily),
                "alpha_rows": int(daily["dataset"].eq("alpha").sum()),
                "beta_rows": int(daily["dataset"].eq("beta").sum()),
                "feature_count": len(feature_cols),
                "elapsed_seconds": time.time() - started,
                "outputs": str(out.relative_to(ROOT)),
            }
        ]
    )

    manifest.to_csv(out / "01_c1_manifest.csv", index=False)
    daily_cache.to_csv(out / "02_c1_daily_feature_cache.csv", index=False)
    feature_summary.to_csv(out / "03_c1_feature_summary.csv", index=False)
    missing_feature_summary.to_csv(out / "04_c1_missing_feature_summary.csv", index=False)
    candidate_count_summary.to_csv(out / "05_c1_candidate_count_summary.csv", index=False)
    dataset_summary.to_csv(out / "06_c1_dataset_summary.csv", index=False)
    beta_confidence_summary.to_csv(out / "07_c1_beta_confidence_summary.csv", index=False)

    print(f"Wrote C1 cached daily feature outputs to {out.relative_to(ROOT)}")
    print("\nManifest")
    print(manifest.round(3).to_string(index=False))
    print("\nDataset summary")
    print(dataset_summary.round(4).to_string(index=False))
    print("\nBeta confidence summary")
    print(beta_confidence_summary.to_string(index=False))
    print("\nMissing feature summary")
    print(missing_feature_summary.round(4).to_string(index=False))
    print("\nCandidate count summary by site")
    print(candidate_count_summary.round(3).to_string(index=False))
    return out


def run_c2_manual_ablation_ladder() -> Path:
    started = time.time()
    out = OUT_ROOT / "C2_manual_ablation_ladder"
    out.mkdir(parents=True, exist_ok=True)
    daily = load_c1_daily_feature_cache()

    threshold_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    sweep_rows: list[pd.DataFrame] = []
    audit = daily[
        [
            "dataset",
            "substation_id",
            "date",
            "true_day",
            "confidence",
            "v03_candidate_count",
            "v03_selected_left_slot",
            "v03_selected_right_slot",
            "v03_selected_duration_h",
            *FEATURE_COLUMNS,
        ]
    ].copy()

    for variant, zero_weights in MANUAL_VARIANTS:
        score_col = f"{variant}_score"
        pred_col = f"{variant}_pred_day"
        daily[score_col] = variant_score(daily, zero_weights)
        threshold, selected, sweep = select_threshold_macro_site(
            daily.loc[daily["dataset"].eq("alpha")].copy(),
            score_col,
        )
        daily[pred_col] = daily[score_col] >= threshold
        audit[score_col] = daily[score_col]
        audit[pred_col] = daily[pred_col]
        threshold_rows.append(
            {
                "chunk": "C2_manual_ablation_ladder",
                "variant": variant,
                "zero_weighted_components": ";".join(zero_weights.keys()) if zero_weights else "",
                "threshold": threshold,
                **{f"selected_{key}": value for key, value in selected.items() if key != "threshold"},
            }
        )
        sweep.insert(0, "variant", variant)
        sweep_rows.append(sweep)

        alpha = daily.loc[daily["dataset"].eq("alpha")].copy()
        beta = daily.loc[daily["dataset"].eq("beta")].copy()
        beta_sure = beta.loc[beta["confidence"].eq("sure")].copy()
        metric_rows.extend(
            metric_rows_for_subset(
                alpha,
                variant=variant,
                dataset="alpha",
                subset="all_alpha",
                pred_col=pred_col,
                threshold=threshold,
            )
        )
        metric_rows.extend(
            metric_rows_for_subset(
                beta,
                variant=variant,
                dataset="beta",
                subset="all_beta",
                pred_col=pred_col,
                threshold=threshold,
            )
        )
        metric_rows.extend(
            metric_rows_for_subset(
                beta_sure,
                variant=variant,
                dataset="beta",
                subset="beta_sure_only",
                pred_col=pred_col,
                threshold=threshold,
            )
        )

    thresholds = pd.DataFrame(threshold_rows)
    metrics = pd.DataFrame(metric_rows)
    sweeps = pd.concat(sweep_rows, ignore_index=True)

    def lookup(dataset: str, subset: str, scope: str, metric: str) -> pd.Series:
        return (
            metrics.loc[
                metrics["dataset"].eq(dataset)
                & metrics["subset"].eq(subset)
                & metrics["summary_scope"].eq(scope),
                ["variant", metric],
            ]
            .set_index("variant")[metric]
        )

    ranking = pd.DataFrame({"variant": [variant for variant, _ in MANUAL_VARIANTS]})
    for dataset, subset, prefix in [
        ("alpha", "all_alpha", "alpha"),
        ("beta", "all_beta", "beta_all"),
        ("beta", "beta_sure_only", "beta_sure"),
    ]:
        for scope, scope_prefix in [("pooled", "pooled"), ("macro_site_average", "site_avg")]:
            for metric in ["precision", "recall", "f1"]:
                ranking[f"{prefix}_{scope_prefix}_{metric}"] = ranking["variant"].map(
                    lookup(dataset, subset, scope, metric)
                )
    ranking = ranking.merge(thresholds[["variant", "threshold"]], on="variant", how="left")
    ranking = ranking.sort_values(
        ["beta_sure_pooled_f1", "beta_all_pooled_f1", "alpha_pooled_f1"],
        ascending=[False, False, False],
    )
    base = ranking.loc[ranking["variant"].eq("M0_all_equal")].iloc[0]
    ablation_effects = ranking.copy()
    for col in [
        "alpha_pooled_f1",
        "beta_all_pooled_f1",
        "beta_sure_pooled_f1",
        "alpha_site_avg_f1",
        "beta_all_site_avg_f1",
        "beta_sure_site_avg_f1",
    ]:
        ablation_effects[f"delta_vs_M0_{col}"] = ablation_effects[col] - float(base[col])

    manifest = pd.DataFrame(
        [
            {
                "chunk": "C2_manual_ablation_ladder",
                "source_cache": str((OUT_ROOT / "C1_cached_daily_features/02_c1_daily_feature_cache.csv").relative_to(ROOT)),
                "variants": len(MANUAL_VARIANTS),
                "daily_rows": len(daily),
                "alpha_rows": int(daily["dataset"].eq("alpha").sum()),
                "beta_rows": int(daily["dataset"].eq("beta").sum()),
                "beta_sure_rows": int(daily["dataset"].eq("beta").mul(daily["confidence"].eq("sure")).sum()),
                "threshold_selection": "alpha_macro_site_f1",
                "elapsed_seconds": time.time() - started,
                "outputs": str(out.relative_to(ROOT)),
            }
        ]
    )

    manifest.to_csv(out / "01_c2_manifest.csv", index=False)
    thresholds.to_csv(out / "02_c2_threshold_selection.csv", index=False)
    metrics.to_csv(out / "03_c2_day_level_metrics.csv", index=False)
    ranking.to_csv(out / "04_c2_variant_ranking.csv", index=False)
    audit.to_csv(out / "05_c2_daily_prediction_audit.csv", index=False)
    ablation_effects.to_csv(out / "06_c2_ablation_effects.csv", index=False)
    sweeps.to_csv(out / "07_c2_threshold_sweeps.csv", index=False)

    print(f"Wrote C2 manual ablation outputs to {out.relative_to(ROOT)}")
    print("\nManifest")
    print(manifest.round(3).to_string(index=False))
    print("\nVariant ranking")
    show_cols = [
        "variant",
        "threshold",
        "alpha_pooled_f1",
        "beta_all_pooled_precision",
        "beta_all_pooled_recall",
        "beta_all_pooled_f1",
        "beta_sure_pooled_precision",
        "beta_sure_pooled_recall",
        "beta_sure_pooled_f1",
        "beta_sure_site_avg_f1",
    ]
    print(ranking[show_cols].round(4).to_string(index=False))
    print("\nAblation delta vs M0")
    delta_cols = [
        "variant",
        "delta_vs_M0_beta_all_pooled_f1",
        "delta_vs_M0_beta_sure_pooled_f1",
        "delta_vs_M0_alpha_pooled_f1",
    ]
    print(ablation_effects[delta_cols].round(4).to_string(index=False))
    return out


def run_c3_beta_loso_manual_ablation() -> Path:
    started = time.time()
    out = OUT_ROOT / "C3_beta_loso_manual_ablation"
    out.mkdir(parents=True, exist_ok=True)
    daily = load_c1_daily_feature_cache()
    beta = daily.loc[daily["dataset"].eq("beta")].copy()
    sites = sorted(beta["substation_id"].unique().tolist())

    threshold_rows: list[dict[str, object]] = []
    fold_metric_rows: list[dict[str, object]] = []
    aggregate_metric_rows: list[dict[str, object]] = []
    prediction_audit_rows: list[pd.DataFrame] = []

    for variant, zero_weights in MANUAL_VARIANTS:
        score_col = f"{variant}_score"
        pred_col = f"{variant}_pred_day"
        beta[score_col] = variant_score(beta, zero_weights)

        for heldout_site in sites:
            train = beta.loc[
                beta["confidence"].eq("sure") & ~beta["substation_id"].eq(heldout_site)
            ].copy()
            threshold, selected, _ = select_threshold_macro_site(train, score_col)
            eval_all = beta.loc[beta["substation_id"].eq(heldout_site)].copy()
            eval_all[pred_col] = eval_all[score_col] >= threshold
            eval_all["heldout_site"] = heldout_site
            eval_all["threshold"] = threshold
            eval_all["variant"] = variant
            prediction_audit_rows.append(
                eval_all[
                    [
                        "variant",
                        "heldout_site",
                        "dataset",
                        "substation_id",
                        "date",
                        "confidence",
                        "true_day",
                        score_col,
                        pred_col,
                        "threshold",
                        "v03_selected_left_slot",
                        "v03_selected_right_slot",
                        "v03_selected_duration_h",
                    ]
                ].rename(columns={score_col: "score", pred_col: "pred_day"})
            )
            threshold_rows.append(
                {
                    "chunk": "C3_beta_loso_manual_ablation",
                    "regime": "R1_beta_loso",
                    "variant": variant,
                    "heldout_site": heldout_site,
                    "training_sites": ";".join(site for site in sites if site != heldout_site),
                    "training_subset": "other_beta_sites_sure_only",
                    "training_rows": len(train),
                    "training_positive_support": int(train["true_day"].sum()),
                    "zero_weighted_components": ";".join(zero_weights.keys()) if zero_weights else "",
                    "threshold": threshold,
                    **{f"selected_{key}": value for key, value in selected.items() if key != "threshold"},
                }
            )
            for subset, frame in [
                ("heldout_all", eval_all),
                ("heldout_sure_only", eval_all.loc[eval_all["confidence"].eq("sure")].copy()),
            ]:
                fold_metric_rows.append(
                    {
                        "regime": "R1_beta_loso",
                        "variant": variant,
                        "dataset": "beta",
                        "subset": subset,
                        "summary_scope": "heldout_site",
                        "substation_id": heldout_site,
                        "threshold": threshold,
                        **compute_metrics(frame["true_day"], frame[pred_col]),
                    }
                )

    predictions = pd.concat(prediction_audit_rows, ignore_index=True)
    for variant in [variant for variant, _ in MANUAL_VARIANTS]:
        variant_predictions = predictions.loc[predictions["variant"].eq(variant)].copy()
        for subset, frame in [
            ("beta_loso_all", variant_predictions),
            ("beta_loso_sure_only", variant_predictions.loc[variant_predictions["confidence"].eq("sure")].copy()),
        ]:
            aggregate_metric_rows.extend(
                metric_rows_for_subset(
                    frame,
                    variant=variant,
                    dataset="beta",
                    subset=subset,
                    pred_col="pred_day",
                    threshold=float("nan"),
                )
            )

    thresholds = pd.DataFrame(threshold_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    aggregate_metrics = pd.DataFrame(aggregate_metric_rows)

    def lookup(subset: str, scope: str, metric: str) -> pd.Series:
        return (
            aggregate_metrics.loc[
                aggregate_metrics["subset"].eq(subset)
                & aggregate_metrics["summary_scope"].eq(scope),
                ["variant", metric],
            ]
            .set_index("variant")[metric]
        )

    ranking = pd.DataFrame({"variant": [variant for variant, _ in MANUAL_VARIANTS]})
    for subset, prefix in [("beta_loso_all", "beta_all"), ("beta_loso_sure_only", "beta_sure")]:
        for scope, scope_prefix in [("pooled", "pooled"), ("macro_site_average", "site_avg")]:
            for metric in ["precision", "recall", "f1"]:
                ranking[f"{prefix}_{scope_prefix}_{metric}"] = ranking["variant"].map(
                    lookup(subset, scope, metric)
                )
    ranking = ranking.sort_values(
        ["beta_sure_pooled_f1", "beta_sure_site_avg_f1", "beta_all_pooled_f1"],
        ascending=[False, False, False],
    )
    base = ranking.loc[ranking["variant"].eq("M0_all_equal")].iloc[0]
    ablation_effects = ranking.copy()
    for col in [
        "beta_all_pooled_f1",
        "beta_sure_pooled_f1",
        "beta_all_site_avg_f1",
        "beta_sure_site_avg_f1",
    ]:
        ablation_effects[f"delta_vs_M0_{col}"] = ablation_effects[col] - float(base[col])

    manifest = pd.DataFrame(
        [
            {
                "chunk": "C3_beta_loso_manual_ablation",
                "regime": "R1_beta_loso",
                "source_cache": str((OUT_ROOT / "C1_cached_daily_features/02_c1_daily_feature_cache.csv").relative_to(ROOT)),
                "variants": len(MANUAL_VARIANTS),
                "folds": len(sites),
                "beta_rows": len(beta),
                "beta_sure_rows": int(beta["confidence"].eq("sure").sum()),
                "threshold_selection": "other_7_beta_sites_sure_only_macro_site_f1",
                "elapsed_seconds": time.time() - started,
                "outputs": str(out.relative_to(ROOT)),
            }
        ]
    )

    manifest.to_csv(out / "01_c3_manifest.csv", index=False)
    thresholds.to_csv(out / "02_c3_threshold_selection.csv", index=False)
    aggregate_metrics.to_csv(out / "03_c3_day_level_metrics.csv", index=False)
    fold_metrics.to_csv(out / "04_c3_fold_metrics.csv", index=False)
    ranking.to_csv(out / "05_c3_variant_ranking.csv", index=False)
    predictions.to_csv(out / "06_c3_daily_prediction_audit.csv", index=False)
    ablation_effects.to_csv(out / "07_c3_ablation_effects.csv", index=False)

    print(f"Wrote C3 Beta LOSO outputs to {out.relative_to(ROOT)}")
    print("\nManifest")
    print(manifest.round(3).to_string(index=False))
    print("\nVariant ranking")
    show_cols = [
        "variant",
        "beta_all_pooled_precision",
        "beta_all_pooled_recall",
        "beta_all_pooled_f1",
        "beta_sure_pooled_precision",
        "beta_sure_pooled_recall",
        "beta_sure_pooled_f1",
        "beta_sure_site_avg_f1",
    ]
    print(ranking[show_cols].round(4).to_string(index=False))
    print("\nAblation delta vs M0")
    delta_cols = [
        "variant",
        "delta_vs_M0_beta_all_pooled_f1",
        "delta_vs_M0_beta_sure_pooled_f1",
        "delta_vs_M0_beta_sure_site_avg_f1",
    ]
    print(ablation_effects[delta_cols].round(4).to_string(index=False))
    return out


def run_c4_compact_regime_comparison() -> Path:
    started = time.time()
    out = OUT_ROOT / "C4_compact_regime_comparison"
    out.mkdir(parents=True, exist_ok=True)
    daily = load_c1_daily_feature_cache()
    alpha = daily.loc[daily["dataset"].eq("alpha")].copy()
    beta = daily.loc[daily["dataset"].eq("beta")].copy()
    beta_sites = sorted(beta["substation_id"].unique().tolist())

    threshold_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []

    for variant, zero_weights in C4_VARIANTS:
        score_col = f"{variant}_score"
        alpha[score_col] = variant_score(alpha, zero_weights)
        beta[score_col] = variant_score(beta, zero_weights)

        for regime in ["R1_beta_loso", "R2_beta_loso_plus_alpha"]:
            for heldout_site in beta_sites:
                beta_train = beta.loc[
                    beta["confidence"].eq("sure") & ~beta["substation_id"].eq(heldout_site)
                ].copy()
                if regime == "R1_beta_loso":
                    train = beta_train
                    dataset_balanced = False
                    training_subset = "other_7_beta_sites_sure_only"
                else:
                    train = pd.concat([alpha, beta_train], ignore_index=True)
                    dataset_balanced = True
                    training_subset = "all_alpha_plus_other_7_beta_sites_sure_only"

                threshold, selected, _ = select_threshold_weighted_macro_site(
                    train,
                    score_col,
                    dataset_balanced=dataset_balanced,
                )
                eval_frame = beta.loc[beta["substation_id"].eq(heldout_site)].copy()
                eval_frame["regime"] = regime
                eval_frame["variant"] = variant
                eval_frame["heldout_site"] = heldout_site
                eval_frame["threshold"] = threshold
                eval_frame["score"] = eval_frame[score_col]
                eval_frame["pred_day"] = eval_frame["score"] >= threshold
                prediction_parts.append(
                    eval_frame[
                        [
                            "regime",
                            "variant",
                            "heldout_site",
                            "dataset",
                            "substation_id",
                            "date",
                            "confidence",
                            "true_day",
                            "score",
                            "pred_day",
                            "threshold",
                        ]
                    ]
                )
                threshold_rows.append(
                    {
                        "chunk": "C4_compact_regime_comparison",
                        "regime": regime,
                        "variant": variant,
                        "heldout_site": heldout_site,
                        "training_subset": training_subset,
                        "dataset_balanced_threshold_selection": dataset_balanced,
                        "training_rows": len(train),
                        "training_positive_support": int(train["true_day"].sum()),
                        "zero_weighted_components": ";".join(zero_weights.keys()) if zero_weights else "",
                        "threshold": threshold,
                        **{f"selected_{key}": value for key, value in selected.items() if key != "threshold"},
                    }
                )

        threshold, selected, _ = select_threshold_weighted_macro_site(
            alpha,
            score_col,
            dataset_balanced=False,
        )
        eval_frame = beta.copy()
        eval_frame["regime"] = "R3_alpha_only_to_beta"
        eval_frame["variant"] = variant
        eval_frame["heldout_site"] = "all_beta"
        eval_frame["threshold"] = threshold
        eval_frame["score"] = eval_frame[score_col]
        eval_frame["pred_day"] = eval_frame["score"] >= threshold
        prediction_parts.append(
            eval_frame[
                [
                    "regime",
                    "variant",
                    "heldout_site",
                    "dataset",
                    "substation_id",
                    "date",
                    "confidence",
                    "true_day",
                    "score",
                    "pred_day",
                    "threshold",
                ]
            ]
        )
        threshold_rows.append(
            {
                "chunk": "C4_compact_regime_comparison",
                "regime": "R3_alpha_only_to_beta",
                "variant": variant,
                "heldout_site": "all_beta",
                "training_subset": "all_alpha",
                "dataset_balanced_threshold_selection": False,
                "training_rows": len(alpha),
                "training_positive_support": int(alpha["true_day"].sum()),
                "zero_weighted_components": ";".join(zero_weights.keys()) if zero_weights else "",
                "threshold": threshold,
                **{f"selected_{key}": value for key, value in selected.items() if key != "threshold"},
            }
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metric_rows: list[dict[str, object]] = []
    for (regime, variant), frame in predictions.groupby(["regime", "variant"], sort=True):
        for subset, subset_frame in [
            ("beta_all", frame),
            ("beta_sure_only", frame.loc[frame["confidence"].eq("sure")].copy()),
        ]:
            rows = metric_rows_for_subset(
                subset_frame,
                variant=variant,
                dataset="beta",
                subset=subset,
                pred_col="pred_day",
                threshold=float("nan"),
            )
            for row in rows:
                row["regime"] = regime
            metric_rows.extend(rows)

    thresholds = pd.DataFrame(threshold_rows)
    metrics = pd.DataFrame(metric_rows)

    def lookup(subset: str, scope: str, metric: str) -> pd.Series:
        return (
            metrics.loc[
                metrics["subset"].eq(subset)
                & metrics["summary_scope"].eq(scope),
                ["regime", "variant", metric],
            ]
            .set_index(["regime", "variant"])[metric]
        )

    ranking = thresholds[["regime", "variant"]].drop_duplicates().reset_index(drop=True)
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

    manifest = pd.DataFrame(
        [
            {
                "chunk": "C4_compact_regime_comparison",
                "source_cache": str((OUT_ROOT / "C1_cached_daily_features/02_c1_daily_feature_cache.csv").relative_to(ROOT)),
                "variants": len(C4_VARIANTS),
                "regimes": "R1_beta_loso;R2_beta_loso_plus_alpha;R3_alpha_only_to_beta",
                "beta_rows": len(beta),
                "beta_sure_rows": int(beta["confidence"].eq("sure").sum()),
                "threshold_selection": "R1/R3 macro-site F1; R2 dataset-balanced macro-site F1",
                "elapsed_seconds": time.time() - started,
                "outputs": str(out.relative_to(ROOT)),
            }
        ]
    )

    manifest.to_csv(out / "01_c4_manifest.csv", index=False)
    thresholds.to_csv(out / "02_c4_threshold_selection.csv", index=False)
    metrics.to_csv(out / "03_c4_day_level_metrics.csv", index=False)
    ranking.to_csv(out / "04_c4_regime_variant_ranking.csv", index=False)
    predictions.to_csv(out / "05_c4_daily_prediction_audit.csv", index=False)

    print(f"Wrote C4 compact regime comparison outputs to {out.relative_to(ROOT)}")
    print("\nManifest")
    print(manifest.round(3).to_string(index=False))
    print("\nRegime/variant ranking")
    show_cols = [
        "regime",
        "variant",
        "beta_all_pooled_precision",
        "beta_all_pooled_recall",
        "beta_all_pooled_f1",
        "beta_sure_pooled_precision",
        "beta_sure_pooled_recall",
        "beta_sure_pooled_f1",
        "beta_sure_positive_site_avg_f1",
    ]
    print(ranking[show_cols].round(4).to_string(index=False))
    return out


def run_c5_logistic_check() -> Path:
    started = time.time()
    out = OUT_ROOT / "C5_logistic_check"
    out.mkdir(parents=True, exist_ok=True)
    daily = load_c1_daily_feature_cache()
    alpha = daily.loc[daily["dataset"].eq("alpha")].copy()
    beta = daily.loc[daily["dataset"].eq("beta")].copy()
    beta_sites = sorted(beta["substation_id"].unique().tolist())

    threshold_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []

    for regime in ["R1_beta_loso", "R2_beta_loso_plus_alpha"]:
        for heldout_site in beta_sites:
            beta_train = beta.loc[
                beta["confidence"].eq("sure") & ~beta["substation_id"].eq(heldout_site)
            ].copy()
            if regime == "R1_beta_loso":
                train = beta_train
                dataset_balanced = False
                training_subset = "other_7_beta_sites_sure_only"
            else:
                train = pd.concat([alpha, beta_train], ignore_index=True)
                dataset_balanced = True
                training_subset = "all_alpha_plus_other_7_beta_sites_sure_only"

            model = fit_logistic_model(train, dataset_balanced=dataset_balanced)
            train = train.copy()
            train["score"] = predict_logistic(model, train)
            threshold, selected, _ = select_threshold_weighted_macro_site(
                train,
                "score",
                dataset_balanced=dataset_balanced,
            )

            eval_frame = beta.loc[beta["substation_id"].eq(heldout_site)].copy()
            eval_frame["regime"] = regime
            eval_frame["variant"] = LOGISTIC_VARIANT
            eval_frame["heldout_site"] = heldout_site
            eval_frame["threshold"] = threshold
            eval_frame["score"] = predict_logistic(model, eval_frame)
            eval_frame["pred_day"] = eval_frame["score"] >= threshold
            prediction_parts.append(
                eval_frame[
                    [
                        "regime",
                        "variant",
                        "heldout_site",
                        "dataset",
                        "substation_id",
                        "date",
                        "confidence",
                        "true_day",
                        "score",
                        "pred_day",
                        "threshold",
                    ]
                ]
            )
            threshold_rows.append(
                {
                    "chunk": "C5_logistic_check",
                    "regime": regime,
                    "variant": LOGISTIC_VARIANT,
                    "heldout_site": heldout_site,
                    "training_subset": training_subset,
                    "dataset_balanced_threshold_selection": dataset_balanced,
                    "training_rows": len(train),
                    "training_positive_support": int(train["true_day"].sum()),
                    "threshold": threshold,
                    **{f"selected_{key}": value for key, value in selected.items() if key != "threshold"},
                }
            )
            coefficient_rows.append(
                {
                    "regime": regime,
                    "variant": LOGISTIC_VARIANT,
                    "heldout_site": heldout_site,
                    "intercept": float(model.intercept_[0]),
                    **{f"coef_{col}": float(coef) for col, coef in zip(FEATURE_COLUMNS, model.coef_[0])},
                }
            )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metric_rows: list[dict[str, object]] = []
    for regime, frame in predictions.groupby("regime", sort=True):
        for subset, subset_frame in [
            ("beta_all", frame),
            ("beta_sure_only", frame.loc[frame["confidence"].eq("sure")].copy()),
        ]:
            rows = metric_rows_for_subset(
                subset_frame,
                variant=LOGISTIC_VARIANT,
                dataset="beta",
                subset=subset,
                pred_col="pred_day",
                threshold=float("nan"),
            )
            for row in rows:
                row["regime"] = regime
            metric_rows.extend(rows)

    thresholds = pd.DataFrame(threshold_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    metrics = pd.DataFrame(metric_rows)

    def lookup(subset: str, scope: str, metric: str) -> pd.Series:
        return (
            metrics.loc[
                metrics["subset"].eq(subset)
                & metrics["summary_scope"].eq(scope),
                ["regime", "variant", metric],
            ]
            .set_index(["regime", "variant"])[metric]
        )

    ranking = thresholds[["regime", "variant"]].drop_duplicates().reset_index(drop=True)
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
    coefficient_summary = (
        coefficients.melt(
            id_vars=["regime", "variant", "heldout_site"],
            value_vars=[f"coef_{col}" for col in FEATURE_COLUMNS],
            var_name="component",
            value_name="coefficient",
        )
        .assign(component=lambda df: df["component"].str.replace("coef_", "", regex=False))
        .groupby(["regime", "component"], as_index=False)
        .agg(mean=("coefficient", "mean"), median=("coefficient", "median"), min=("coefficient", "min"), max=("coefficient", "max"))
    )

    manifest = pd.DataFrame(
        [
            {
                "chunk": "C5_logistic_check",
                "source_cache": str((OUT_ROOT / "C1_cached_daily_features/02_c1_daily_feature_cache.csv").relative_to(ROOT)),
                "variant": LOGISTIC_VARIANT,
                "regimes": "R1_beta_loso;R2_beta_loso_plus_alpha",
                "folds_per_regime": len(beta_sites),
                "beta_rows": len(beta),
                "beta_sure_rows": int(beta["confidence"].eq("sure").sum()),
                "elapsed_seconds": time.time() - started,
                "outputs": str(out.relative_to(ROOT)),
            }
        ]
    )

    manifest.to_csv(out / "01_c5_manifest.csv", index=False)
    thresholds.to_csv(out / "02_c5_threshold_selection.csv", index=False)
    metrics.to_csv(out / "03_c5_day_level_metrics.csv", index=False)
    ranking.to_csv(out / "04_c5_regime_ranking.csv", index=False)
    predictions.to_csv(out / "05_c5_daily_prediction_audit.csv", index=False)
    coefficients.to_csv(out / "06_c5_logistic_coefficients_by_fold.csv", index=False)
    coefficient_summary.to_csv(out / "07_c5_logistic_coefficient_summary.csv", index=False)

    print(f"Wrote C5 logistic outputs to {out.relative_to(ROOT)}")
    print("\nManifest")
    print(manifest.round(3).to_string(index=False))
    print("\nRegime ranking")
    show_cols = [
        "regime",
        "variant",
        "beta_all_pooled_precision",
        "beta_all_pooled_recall",
        "beta_all_pooled_f1",
        "beta_sure_pooled_precision",
        "beta_sure_pooled_recall",
        "beta_sure_pooled_f1",
        "beta_sure_positive_site_avg_f1",
    ]
    print(ranking[show_cols].round(4).to_string(index=False))
    print("\nCoefficient summary")
    print(coefficient_summary.round(4).to_string(index=False))
    return out


def load_c6_prediction_audits() -> pd.DataFrame:
    c4_path = OUT_ROOT / "C4_compact_regime_comparison/05_c4_daily_prediction_audit.csv"
    c5_path = OUT_ROOT / "C5_logistic_check/05_c5_daily_prediction_audit.csv"
    if not c4_path.exists():
        print("C4 prediction audit is missing; running C4 first.")
        run_c4_compact_regime_comparison()
    if not c5_path.exists():
        print("C5 prediction audit is missing; running C5 first.")
        run_c5_logistic_check()

    c4 = pd.read_csv(c4_path)
    c4["source_chunk"] = "C4_compact_regime_comparison"
    c5 = pd.read_csv(c5_path)
    c5["source_chunk"] = "C5_logistic_check"
    predictions = pd.concat([c4, c5], ignore_index=True)
    predictions["date"] = date_key(predictions["date"])
    predictions["true_day"] = safe_bool(predictions["true_day"])
    predictions["pred_day"] = safe_bool(predictions["pred_day"])
    predictions["confidence"] = predictions["confidence"].fillna("missing").astype(str).str.strip().str.lower()
    predictions["score"] = pd.to_numeric(predictions["score"], errors="coerce")
    predictions["threshold"] = pd.to_numeric(predictions["threshold"], errors="coerce")
    predictions["confidence_score"] = (predictions["score"] - predictions["threshold"]).abs()
    return predictions


def coverage_summary_rows(
    frame: pd.DataFrame,
    *,
    regime: str,
    variant: str,
    subset: str,
    coverage_pct: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], pd.DataFrame]:
    n_select = max(1, int(np.ceil(len(frame) * coverage_pct / 100)))
    selected = (
        frame.sort_values(["confidence_score", "score", "substation_id", "date"], ascending=[False, False, True, True])
        .head(n_select)
        .copy()
    )
    selected["coverage_pct"] = coverage_pct
    rows: list[dict[str, object]] = []
    site_rows: list[dict[str, object]] = []

    pooled = compute_metrics(selected["true_day"], selected["pred_day"])
    rows.append(
        {
            "regime": regime,
            "variant": variant,
            "subset": subset,
            "coverage_pct": coverage_pct,
            "summary_scope": "pooled",
            "selected_rows": len(selected),
            "available_rows": len(frame),
            "actual_coverage_pct": len(selected) / len(frame) * 100,
            **pooled,
        }
    )
    site_metrics = site_metric_frame(selected, "pred_day")
    if not site_metrics.empty:
        rows.append(
            {
                "regime": regime,
                "variant": variant,
                "subset": subset,
                "coverage_pct": coverage_pct,
                "summary_scope": "macro_site_average",
                "selected_rows": len(selected),
                "available_rows": len(frame),
                "actual_coverage_pct": len(selected) / len(frame) * 100,
                "support": int(site_metrics["support"].sum()),
                "positive_support": int(site_metrics["positive_support"].sum()),
                "tp": int(site_metrics["tp"].sum()),
                "fp": int(site_metrics["fp"].sum()),
                "fn": int(site_metrics["fn"].sum()),
                "tn": int(site_metrics["tn"].sum()),
                "precision": float(site_metrics["precision"].mean()),
                "recall": float(site_metrics["recall"].mean()),
                "f1": float(site_metrics["f1"].mean()),
            }
        )
        positive_site_metrics = site_metrics.loc[site_metrics["positive_support"] > 0].copy()
        if not positive_site_metrics.empty:
            rows.append(
                {
                    "regime": regime,
                    "variant": variant,
                    "subset": subset,
                    "coverage_pct": coverage_pct,
                    "summary_scope": "positive_site_macro_average",
                    "selected_rows": len(selected),
                    "available_rows": len(frame),
                    "actual_coverage_pct": len(selected) / len(frame) * 100,
                    "support": int(positive_site_metrics["support"].sum()),
                    "positive_support": int(positive_site_metrics["positive_support"].sum()),
                    "tp": int(positive_site_metrics["tp"].sum()),
                    "fp": int(positive_site_metrics["fp"].sum()),
                    "fn": int(positive_site_metrics["fn"].sum()),
                    "tn": int(positive_site_metrics["tn"].sum()),
                    "precision": float(positive_site_metrics["precision"].mean()),
                    "recall": float(positive_site_metrics["recall"].mean()),
                    "f1": float(positive_site_metrics["f1"].mean()),
                }
            )
        for row in site_metrics.to_dict(orient="records"):
            site_rows.append(
                {
                    "regime": regime,
                    "variant": variant,
                    "subset": subset,
                    "coverage_pct": coverage_pct,
                    "substation_id": row.pop("substation_id"),
                    "selected_rows": len(selected),
                    "available_rows": len(frame),
                    "actual_coverage_pct": len(selected) / len(frame) * 100,
                    **row,
                }
            )
    return rows, site_rows, selected


def run_c6_confidence_coverage() -> Path:
    started = time.time()
    out = OUT_ROOT / "C6_confidence_coverage"
    out.mkdir(parents=True, exist_ok=True)
    predictions = load_c6_prediction_audits()
    coverage_levels = [50, 60, 70, 80, 90, 100]

    summary_rows: list[dict[str, object]] = []
    site_rows: list[dict[str, object]] = []
    selected_parts: list[pd.DataFrame] = []
    for (regime, variant), group in predictions.groupby(["regime", "variant"], sort=True):
        for subset, frame in [
            ("beta_all", group),
            ("beta_sure_only", group.loc[group["confidence"].eq("sure")].copy()),
        ]:
            if frame.empty:
                continue
            for coverage_pct in coverage_levels:
                rows, sites, selected = coverage_summary_rows(
                    frame,
                    regime=regime,
                    variant=variant,
                    subset=subset,
                    coverage_pct=coverage_pct,
                )
                summary_rows.extend(rows)
                site_rows.extend(sites)
                selected_parts.append(selected.assign(regime=regime, variant=variant, subset=subset))

    coverage = pd.DataFrame(summary_rows)
    site_coverage = pd.DataFrame(site_rows)
    selected_audit = pd.concat(selected_parts, ignore_index=True)
    best_pooled = (
        coverage.loc[coverage["summary_scope"].eq("pooled")]
        .sort_values(["subset", "coverage_pct", "f1", "precision"], ascending=[True, True, False, False])
        .groupby(["subset", "coverage_pct"], as_index=False)
        .head(5)
    )
    manifest = pd.DataFrame(
        [
            {
                "chunk": "C6_confidence_coverage",
                "source_chunks": "C4_compact_regime_comparison;C5_logistic_check",
                "coverage_levels": ";".join(map(str, coverage_levels)),
                "model_regime_variants": predictions.groupby(["regime", "variant"]).ngroups,
                "prediction_rows": len(predictions),
                "elapsed_seconds": time.time() - started,
                "outputs": str(out.relative_to(ROOT)),
            }
        ]
    )

    manifest.to_csv(out / "01_c6_manifest.csv", index=False)
    coverage.to_csv(out / "02_c6_confidence_coverage_metrics.csv", index=False)
    site_coverage.to_csv(out / "03_c6_site_confidence_coverage_metrics.csv", index=False)
    best_pooled.to_csv(out / "04_c6_best_pooled_by_coverage.csv", index=False)
    selected_audit.to_csv(out / "05_c6_selected_prediction_audit.csv", index=False)

    print(f"Wrote C6 confidence coverage outputs to {out.relative_to(ROOT)}")
    print("\nManifest")
    print(manifest.round(3).to_string(index=False))
    print("\nBest pooled beta_sure_only rows")
    show = best_pooled.loc[best_pooled["subset"].eq("beta_sure_only")].copy()
    show_cols = [
        "coverage_pct",
        "regime",
        "variant",
        "summary_scope",
        "selected_rows",
        "positive_support",
        "precision",
        "recall",
        "f1",
    ]
    print(show[show_cols].round(4).to_string(index=False))
    return out


def run_c7_confidence_triage_diagnostics() -> Path:
    started = time.time()
    out = OUT_ROOT / "C7_confidence_triage_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    predictions = load_c6_prediction_audits()
    focus_pairs = {
        ("R1_beta_loso", "M9_drop_site_rank"),
        ("R2_beta_loso_plus_alpha", "M9_drop_site_rank"),
        ("R1_beta_loso", LOGISTIC_VARIANT),
        ("R2_beta_loso_plus_alpha", LOGISTIC_VARIANT),
    }
    predictions = predictions.loc[
        predictions[["regime", "variant"]].apply(tuple, axis=1).isin(focus_pairs)
    ].copy()
    coverage_levels = [50, 60, 70, 80, 90, 100]
    rows: list[dict[str, object]] = []
    selected_rows: list[pd.DataFrame] = []

    for (regime, variant), group in predictions.groupby(["regime", "variant"], sort=True):
        for subset, frame in [
            ("beta_all", group),
            ("beta_sure_only", group.loc[group["confidence"].eq("sure")].copy()),
        ]:
            if frame.empty:
                continue
            total_true_rpf = int(frame["true_day"].sum())
            total_pred_positive = int(frame["pred_day"].sum())
            total_errors = int((frame["true_day"] != frame["pred_day"]).sum())
            for coverage_pct in coverage_levels:
                n_select = max(1, int(np.ceil(len(frame) * coverage_pct / 100)))
                selected = (
                    frame.sort_values(
                        ["confidence_score", "score", "substation_id", "date"],
                        ascending=[False, False, True, True],
                    )
                    .head(n_select)
                    .copy()
                )
                manual = frame.drop(index=selected.index).copy()
                selected["coverage_pct"] = coverage_pct
                selected["subset"] = subset
                selected_rows.append(selected)

                auto_metrics = compute_metrics(selected["true_day"], selected["pred_day"])
                manual_metrics = compute_metrics(manual["true_day"], manual["pred_day"]) if len(manual) else {
                    "support": 0,
                    "positive_support": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "tn": 0,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                }
                auto_pred_positive = int(selected["pred_day"].sum())
                auto_pred_negative = int((~selected["pred_day"]).sum())
                auto_true_rpf = int(selected["true_day"].sum())
                manual_true_rpf = int(manual["true_day"].sum()) if len(manual) else 0
                auto_errors = int((selected["true_day"] != selected["pred_day"]).sum())
                manual_errors = int((manual["true_day"] != manual["pred_day"]).sum()) if len(manual) else 0
                auto_npv = auto_metrics["tn"] / (auto_metrics["tn"] + auto_metrics["fn"]) if auto_metrics["tn"] + auto_metrics["fn"] else np.nan
                auto_error_rate = auto_errors / len(selected) if len(selected) else np.nan
                manual_error_capture = manual_errors / total_errors if total_errors else np.nan
                rows.append(
                    {
                        "regime": regime,
                        "variant": variant,
                        "subset": subset,
                        "coverage_pct": coverage_pct,
                        "available_rows": len(frame),
                        "auto_rows": len(selected),
                        "manual_rows": len(manual),
                        "actual_auto_coverage_pct": len(selected) / len(frame) * 100,
                        "manual_review_pct": len(manual) / len(frame) * 100,
                        "total_true_rpf": total_true_rpf,
                        "total_pred_positive": total_pred_positive,
                        "total_errors": total_errors,
                        "auto_pred_positive": auto_pred_positive,
                        "auto_pred_negative": auto_pred_negative,
                        "auto_true_rpf": auto_true_rpf,
                        "manual_true_rpf": manual_true_rpf,
                        "auto_true_rpf_coverage_pct": auto_true_rpf / total_true_rpf * 100 if total_true_rpf else np.nan,
                        "manual_true_rpf_remaining_pct": manual_true_rpf / total_true_rpf * 100 if total_true_rpf else np.nan,
                        "auto_tp": auto_metrics["tp"],
                        "auto_fp": auto_metrics["fp"],
                        "auto_fn": auto_metrics["fn"],
                        "auto_tn": auto_metrics["tn"],
                        "auto_precision": auto_metrics["precision"],
                        "auto_recall": auto_metrics["recall"],
                        "auto_f1": auto_metrics["f1"],
                        "auto_npv": auto_npv,
                        "auto_errors": auto_errors,
                        "auto_error_rate": auto_error_rate,
                        "manual_tp": manual_metrics["tp"],
                        "manual_fp": manual_metrics["fp"],
                        "manual_fn": manual_metrics["fn"],
                        "manual_tn": manual_metrics["tn"],
                        "manual_errors": manual_errors,
                        "manual_error_capture_pct": manual_error_capture * 100 if not pd.isna(manual_error_capture) else np.nan,
                    }
                )

    triage = pd.DataFrame(rows)
    selected_audit = pd.concat(selected_rows, ignore_index=True)
    best = (
        triage.loc[triage["subset"].eq("beta_sure_only")]
        .sort_values(["coverage_pct", "auto_f1", "auto_error_rate"], ascending=[True, False, True])
        .groupby("coverage_pct", as_index=False)
        .head(4)
    )
    manifest = pd.DataFrame(
        [
            {
                "chunk": "C7_confidence_triage_diagnostics",
                "source_chunks": "C4_compact_regime_comparison;C5_logistic_check",
                "coverage_levels": ";".join(map(str, coverage_levels)),
                "focus_pairs": ";".join(f"{regime}/{variant}" for regime, variant in sorted(focus_pairs)),
                "prediction_rows": len(predictions),
                "elapsed_seconds": time.time() - started,
                "outputs": str(out.relative_to(ROOT)),
            }
        ]
    )
    manifest.to_csv(out / "01_c7_manifest.csv", index=False)
    triage.to_csv(out / "02_c7_triage_diagnostics.csv", index=False)
    best.to_csv(out / "03_c7_best_beta_sure_triage_rows.csv", index=False)
    selected_audit.to_csv(out / "04_c7_auto_accepted_prediction_audit.csv", index=False)

    print(f"Wrote C7 confidence triage outputs to {out.relative_to(ROOT)}")
    print("\nManifest")
    print(manifest.round(3).to_string(index=False))
    print("\nBest beta_sure_only triage rows")
    show_cols = [
        "coverage_pct",
        "regime",
        "variant",
        "auto_rows",
        "manual_rows",
        "auto_pred_positive",
        "auto_pred_negative",
        "auto_true_rpf",
        "manual_true_rpf",
        "auto_precision",
        "auto_recall",
        "auto_f1",
        "auto_npv",
        "auto_errors",
        "manual_error_capture_pct",
    ]
    print(best[show_cols].round(4).to_string(index=False))
    return out


def apply_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.edgecolor": PALETTE["dark_blue"],
            "axes.labelcolor": PALETTE["dark_blue"],
            "xtick.color": PALETTE["dark_blue"],
            "ytick.color": PALETTE["dark_blue"],
            "text.color": PALETTE["dark_blue"],
            "axes.titleweight": "bold",
            "axes.axisbelow": True,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def save_bar_labels(ax, fmt: str = "{:.3f}", dy: float = 0.01) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if pd.isna(height):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            color=PALETTE["dark_blue"],
        )


def run_c8_key_result_figures() -> Path:
    started = time.time()
    out = OUT_ROOT / "C8_key_result_figures"
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    apply_plot_style()
    c3_metrics_path = OUT_ROOT / "C3_beta_loso_manual_ablation/03_c3_day_level_metrics.csv"
    c4_ranking_path = OUT_ROOT / "C4_compact_regime_comparison/04_c4_regime_variant_ranking.csv"
    c6_coverage_path = OUT_ROOT / "C6_confidence_coverage/02_c6_confidence_coverage_metrics.csv"
    c7_triage_path = OUT_ROOT / "C7_confidence_triage_diagnostics/02_c7_triage_diagnostics.csv"
    for path in [c3_metrics_path, c4_ranking_path, c6_coverage_path, c7_triage_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required input is missing: {path}")

    c3_metrics = pd.read_csv(c3_metrics_path)
    c4_ranking = pd.read_csv(c4_ranking_path)
    c6_coverage = pd.read_csv(c6_coverage_path)
    c7_triage = pd.read_csv(c7_triage_path)
    figure_rows: list[dict[str, object]] = []

    # Figure 1: top model/regime comparison.
    compare = c4_ranking.copy()
    compare["label"] = compare["regime"].str.replace("_", " ", regex=False) + "\n" + compare["variant"].str.replace("_", " ", regex=False)
    compare = compare.sort_values("beta_sure_pooled_f1", ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars = ax.bar(range(len(compare)), compare["beta_sure_pooled_f1"], color=PALETTE["orange"], width=0.7)
    ax.set_xticks(range(len(compare)))
    ax.set_xticklabels(compare["label"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Beta sure-only pooled F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Top Regime And Variant Comparison")
    ax.grid(axis="y", color=PALETTE["light_white"], linewidth=0.8)
    save_bar_labels(ax)
    fig.tight_layout()
    path = out / "fig01_top_model_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figure_rows.append({"figure": path.name, "description": "Top C4 regime/variant comparison by Beta sure-only pooled F1"})

    # Figure 2: individual Beta site precision/recall/F1 for current best model.
    best_site = c3_metrics.loc[
        c3_metrics["variant"].eq("M9_drop_site_rank")
        & c3_metrics["subset"].eq("beta_loso_sure_only")
        & c3_metrics["summary_scope"].eq("site")
    ].copy()
    best_site = best_site.sort_values("f1", ascending=False)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(best_site))
    width = 0.24
    ax.bar(x - width, best_site["precision"], width, label="Precision", color=PALETTE["dark_blue"])
    ax.bar(x, best_site["recall"], width, label="Recall", color=PALETTE["orange"])
    ax.bar(x + width, best_site["f1"], width, label="F1", color=PALETTE["light_grey"])
    ax.set_xticks(x)
    ax.set_xticklabels(best_site["substation_id"], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("R1 M9 Beta Sure-Only Site Performance")
    ax.grid(axis="y", color=PALETTE["light_white"], linewidth=0.8)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    for idx, value in enumerate(best_site["f1"]):
        ax.text(idx + width, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = out / "fig02_beta_site_precision_recall_f1.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figure_rows.append({"figure": path.name, "description": "Per-site Beta sure-only precision/recall/F1 for R1 M9_drop_site_rank"})

    # Figure 3: confidence coverage F1 curves.
    focus = c6_coverage.loc[
        c6_coverage["subset"].eq("beta_sure_only")
        & c6_coverage["summary_scope"].eq("pooled")
        & c6_coverage[["regime", "variant"]].apply(tuple, axis=1).isin(
            {
                ("R1_beta_loso", "M9_drop_site_rank"),
                ("R2_beta_loso_plus_alpha", "M9_drop_site_rank"),
                ("R1_beta_loso", LOGISTIC_VARIANT),
            }
        )
    ].copy()
    labels = {
        ("R1_beta_loso", "M9_drop_site_rank"): "R1 M9",
        ("R2_beta_loso_plus_alpha", "M9_drop_site_rank"): "R2 M9",
        ("R1_beta_loso", LOGISTIC_VARIANT): "R1 logistic",
    }
    colors = [PALETTE["orange"], PALETTE["dark_blue"], PALETTE["light_grey"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for color, ((regime, variant), group) in zip(colors, focus.groupby(["regime", "variant"], sort=False)):
        group = group.sort_values("coverage_pct")
        ax.plot(group["coverage_pct"], group["f1"], marker="o", linewidth=2.2, color=color, label=labels[(regime, variant)])
    ax.set_xlabel("Auto-accepted days (%)")
    ax.set_ylabel("Beta sure-only F1")
    ax.set_ylim(0.75, 1.02)
    ax.set_title("Confidence Coverage Curve")
    ax.grid(axis="both", color=PALETTE["light_white"], linewidth=0.8)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    path = out / "fig03_confidence_coverage_f1.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figure_rows.append({"figure": path.name, "description": "F1 vs auto-accepted coverage for best models"})

    # Figure 4: manual review burden and auto errors for best model.
    triage = c7_triage.loc[
        c7_triage["subset"].eq("beta_sure_only")
        & c7_triage["regime"].eq("R1_beta_loso")
        & c7_triage["variant"].eq("M9_drop_site_rank")
    ].sort_values("coverage_pct")
    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax1.plot(triage["coverage_pct"], triage["manual_rows"], marker="o", color=PALETTE["dark_blue"], linewidth=2.2, label="Manual rows left")
    ax1.plot(triage["coverage_pct"], triage["manual_true_rpf"], marker="o", color=PALETTE["light_grey"], linewidth=2.2, label="Manual RPF days left")
    ax1.set_xlabel("Auto-accepted days (%)")
    ax1.set_ylabel("Days left for manual review")
    ax1.grid(axis="both", color=PALETTE["light_white"], linewidth=0.8)
    ax2 = ax1.twinx()
    ax2.bar(triage["coverage_pct"], triage["auto_errors"], width=4.5, alpha=0.35, color=PALETTE["orange"], label="Auto errors")
    ax2.set_ylabel("Auto-accepted errors")
    ax1.set_title("Manual Review Burden vs Auto Errors")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")
    fig.tight_layout()
    path = out / "fig04_manual_review_burden.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figure_rows.append({"figure": path.name, "description": "Manual-review burden and auto errors for R1 M9 confidence triage"})

    # Figure 5: C3 ablation deltas vs M0.
    ranking = pd.read_csv(OUT_ROOT / "C3_beta_loso_manual_ablation/07_c3_ablation_effects.csv")
    delta = ranking.loc[~ranking["variant"].eq("M0_all_equal")].copy()
    delta = delta.sort_values("delta_vs_M0_beta_sure_pooled_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    values = delta["delta_vs_M0_beta_sure_pooled_f1"]
    colors_delta = np.where(values >= 0, PALETTE["orange"], PALETTE["grey"])
    ax.barh(delta["variant"].str.replace("_", " ", regex=False), values, color=colors_delta)
    ax.axvline(0, color=PALETTE["dark_blue"], linewidth=1.0)
    ax.set_xlabel("Delta Beta sure-only F1 vs M0")
    ax.set_title("Ablation Effect Under Beta LOSO")
    ax.grid(axis="x", color=PALETTE["light_white"], linewidth=0.8)
    fig.tight_layout()
    path = out / "fig05_ablation_delta_vs_m0.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figure_rows.append({"figure": path.name, "description": "C3 ablation delta in Beta sure-only pooled F1 vs M0"})

    manifest = pd.DataFrame(
        [
            {
                "chunk": "C8_key_result_figures",
                "figures": len(figure_rows),
                "elapsed_seconds": time.time() - started,
                "outputs": str(out.relative_to(ROOT)),
            }
        ]
    )
    figures = pd.DataFrame(figure_rows)
    manifest.to_csv(out / "01_c8_manifest.csv", index=False)
    figures.to_csv(out / "02_c8_figure_index.csv", index=False)
    print(f"Wrote C8 figures to {out.relative_to(ROOT)}")
    print("\nFigure index")
    print(figures.to_string(index=False))
    return out


def load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return pd.read_csv(path)


def save_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    *,
    title: str,
    cbar_label: str,
    figsize: tuple[float, float],
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "journal_heat",
        [PALETTE["light_white"], PALETTE["light_grey"], PALETTE["orange"]],
    )
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if pd.isna(value):
                continue
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=PALETTE["dark_blue"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_c9_all_result_visualisations() -> Path:
    started = time.time()
    out = OUT_ROOT / "C9_all_result_visualisations"
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    apply_plot_style()
    c2_metrics = load_required_csv(OUT_ROOT / "C2_manual_ablation_ladder/03_c2_day_level_metrics.csv")
    c3_metrics = load_required_csv(OUT_ROOT / "C3_beta_loso_manual_ablation/03_c3_day_level_metrics.csv")
    c3_effects = load_required_csv(OUT_ROOT / "C3_beta_loso_manual_ablation/07_c3_ablation_effects.csv")
    c4_metrics = load_required_csv(OUT_ROOT / "C4_compact_regime_comparison/03_c4_day_level_metrics.csv")
    c4_ranking = load_required_csv(OUT_ROOT / "C4_compact_regime_comparison/04_c4_regime_variant_ranking.csv")
    c5_metrics = load_required_csv(OUT_ROOT / "C5_logistic_check/03_c5_day_level_metrics.csv")
    c5_coef = load_required_csv(OUT_ROOT / "C5_logistic_check/07_c5_logistic_coefficient_summary.csv")
    c6_coverage = load_required_csv(OUT_ROOT / "C6_confidence_coverage/02_c6_confidence_coverage_metrics.csv")
    c7_triage = load_required_csv(OUT_ROOT / "C7_confidence_triage_diagnostics/02_c7_triage_diagnostics.csv")

    figure_rows: list[dict[str, object]] = []
    summary_tables: list[dict[str, object]] = []

    # Combined ranking table from C4/C5.
    c4_rank_for_table = c4_ranking.copy()
    c5_rank_for_table = load_required_csv(OUT_ROOT / "C5_logistic_check/04_c5_regime_ranking.csv")
    combined_rank = pd.concat([c4_rank_for_table, c5_rank_for_table], ignore_index=True, sort=False)
    combined_rank = combined_rank.sort_values(
        ["beta_sure_pooled_f1", "beta_sure_positive_site_avg_f1", "beta_all_pooled_f1"],
        ascending=[False, False, False],
    )
    combined_rank.to_csv(out / "01_c9_combined_regime_variant_ranking.csv", index=False)
    summary_tables.append({"table": "01_c9_combined_regime_variant_ranking.csv", "description": "C4/C5 combined ranking"})

    # Fig01: all variant/regime F1 heatmap.
    heat = combined_rank.copy()
    heat["regime_variant"] = heat["regime"].str.replace("_", " ", regex=False) + "\n" + heat["variant"].str.replace("_", " ", regex=False)
    heat_matrix = heat.set_index("regime_variant")[
        [
            "beta_all_pooled_f1",
            "beta_sure_pooled_f1",
            "beta_sure_positive_site_avg_f1",
        ]
    ]
    heat_matrix.columns = ["Beta all\npooled F1", "Beta sure\npooled F1", "Beta sure\npositive-site F1"]
    path = out / "fig01_all_regime_variant_f1_heatmap.png"
    save_heatmap(heat_matrix, path, title="All Regime/Variant F1 Summary", cbar_label="F1", figsize=(6.8, 5.4))
    figure_rows.append({"figure": path.name, "description": "F1 heatmap for all C4/C5 regime variants"})

    # Fig02: per-site F1 heatmap for C3 all manual variants.
    site = c3_metrics.loc[
        c3_metrics["subset"].eq("beta_loso_sure_only") & c3_metrics["summary_scope"].eq("site")
    ].copy()
    site_matrix = site.pivot(index="substation_id", columns="variant", values="f1")
    variant_order = (
        site.groupby("variant")["f1"].mean().sort_values(ascending=False).index.tolist()
    )
    site_matrix = site_matrix[variant_order]
    path = out / "fig02_beta_site_f1_heatmap_all_manual_variants.png"
    save_heatmap(site_matrix, path, title="Beta Sure-Only Site F1 Across Manual Variants", cbar_label="F1", figsize=(9.0, 4.8))
    figure_rows.append({"figure": path.name, "description": "Per-site F1 heatmap for C3 manual variants"})

    # Fig03: precision/recall scatter for all C4/C5 regime variants.
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    markers = {"R1_beta_loso": "o", "R2_beta_loso_plus_alpha": "s", "R3_alpha_only_to_beta": "^"}
    colors = {
        "M9_drop_site_rank": PALETTE["orange"],
        "M8_drop_site_centered": PALETTE["dark_blue"],
        "M0_all_equal": PALETTE["light_grey"],
        LOGISTIC_VARIANT: PALETTE["grey"],
    }
    for _, row in combined_rank.iterrows():
        ax.scatter(
            row["beta_sure_pooled_recall"],
            row["beta_sure_pooled_precision"],
            s=85,
            marker=markers.get(row["regime"], "o"),
            color=colors.get(row["variant"], PALETTE["orange"]),
            edgecolor=PALETTE["dark_blue"],
            linewidth=0.5,
        )
        ax.text(
            row["beta_sure_pooled_recall"] + 0.004,
            row["beta_sure_pooled_precision"] + 0.004,
            row["variant"].replace("M10_logistic_all9", "logit").replace("M9_drop_site_rank", "M9").replace("M8_drop_site_centered", "M8").replace("M0_all_equal", "M0"),
            fontsize=7,
        )
    ax.set_xlabel("Beta sure-only recall")
    ax.set_ylabel("Beta sure-only precision")
    ax.set_xlim(0.75, 1.01)
    ax.set_ylim(0.60, 1.01)
    ax.set_title("Precision/Recall Tradeoff Across Regime Variants")
    ax.grid(axis="both", color=PALETTE["light_white"], linewidth=0.8)
    fig.tight_layout()
    path = out / "fig03_precision_recall_scatter_all_regime_variants.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figure_rows.append({"figure": path.name, "description": "Precision/recall scatter for all C4/C5 regime variants"})

    # Fig04: C3 ablation impact on P/R/F1.
    delta = c3_effects.loc[~c3_effects["variant"].eq("M0_all_equal")].copy()
    delta = delta.sort_values("delta_vs_M0_beta_sure_pooled_f1", ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.8), sharey=True)
    metric_specs = [
        ("delta_vs_M0_beta_sure_pooled_f1", "Delta F1"),
        ("delta_vs_M0_beta_sure_pooled_f1", "Delta F1"),
        ("delta_vs_M0_beta_all_pooled_f1", "Delta all-days F1"),
    ]
    # Build precision/recall deltas from C3 metrics.
    c3_summary = c3_metrics.loc[
        c3_metrics["subset"].eq("beta_loso_sure_only") & c3_metrics["summary_scope"].eq("pooled")
    ].set_index("variant")
    base = c3_summary.loc["M0_all_equal"]
    delta["delta_precision"] = delta["variant"].map(c3_summary["precision"] - base["precision"])
    delta["delta_recall"] = delta["variant"].map(c3_summary["recall"] - base["recall"])
    plot_cols = [("delta_precision", "Delta precision"), ("delta_recall", "Delta recall"), ("delta_vs_M0_beta_sure_pooled_f1", "Delta F1")]
    for ax, (col, title) in zip(axes, plot_cols):
        values = delta[col]
        ax.barh(delta["variant"].str.replace("_", " ", regex=False), values, color=np.where(values >= 0, PALETTE["orange"], PALETTE["grey"]))
        ax.axvline(0, color=PALETTE["dark_blue"], linewidth=1.0)
        ax.set_title(title)
        ax.grid(axis="x", color=PALETTE["light_white"], linewidth=0.8)
    fig.suptitle("C3 Ablation Impact vs M0, Beta Sure-Only")
    fig.tight_layout()
    path = out / "fig04_ablation_precision_recall_f1_delta.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figure_rows.append({"figure": path.name, "description": "Ablation delta precision/recall/F1 versus M0"})

    # Fig05: threshold stability by heldout site and variant.
    threshold = load_required_csv(OUT_ROOT / "C3_beta_loso_manual_ablation/02_c3_threshold_selection.csv")
    threshold = threshold.loc[threshold["variant"].isin(["M0_all_equal", "M8_drop_site_centered", "M9_drop_site_rank"])].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for color, (variant, group) in zip([PALETTE["orange"], PALETTE["dark_blue"], PALETTE["light_grey"]], threshold.groupby("variant", sort=True)):
        group = group.sort_values("heldout_site")
        ax.plot(group["heldout_site"], group["threshold"], marker="o", linewidth=2, color=color, label=variant.replace("_", " "))
    ax.set_ylabel("Selected threshold")
    ax.set_title("Beta LOSO Threshold Stability")
    ax.grid(axis="y", color=PALETTE["light_white"], linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out / "fig05_threshold_by_heldout_site.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figure_rows.append({"figure": path.name, "description": "Selected threshold by held-out Beta site"})

    # Fig06: confidence coverage curves for all C4/C5 rows.
    cov = c6_coverage.loc[
        c6_coverage["subset"].eq("beta_sure_only") & c6_coverage["summary_scope"].eq("pooled")
    ].copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for (regime, variant), group in cov.groupby(["regime", "variant"], sort=True):
        group = group.sort_values("coverage_pct")
        label = regime.replace("_", " ") + " / " + variant.replace("_", " ")
        linestyle = "--" if "logistic" in variant else "-"
        ax.plot(group["coverage_pct"], group["f1"], marker="o", linewidth=1.8, linestyle=linestyle, label=label)
    ax.set_xlabel("Auto-accepted days (%)")
    ax.set_ylabel("Beta sure-only F1")
    ax.set_ylim(0.70, 1.02)
    ax.set_title("Confidence Coverage Curves For All Compared Models")
    ax.grid(axis="both", color=PALETTE["light_white"], linewidth=0.8)
    ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    path = out / "fig06_confidence_coverage_all_models.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figure_rows.append({"figure": path.name, "description": "Confidence coverage curves for all C4/C5 models"})

    # Fig07: logistic coefficient heatmap.
    coef_matrix = c5_coef.pivot(index="component", columns="regime", values="mean")
    path = out / "fig07_logistic_coefficient_heatmap.png"
    save_heatmap(coef_matrix, path, title="Mean Logistic Coefficients By Regime", cbar_label="Coefficient", figsize=(6.8, 4.8), vmin=float(coef_matrix.min().min()), vmax=float(coef_matrix.max().max()))
    figure_rows.append({"figure": path.name, "description": "Mean logistic coefficients by regime"})

    # Fig08: triage auto errors and manual rows by coverage.
    triage = c7_triage.loc[
        c7_triage["subset"].eq("beta_sure_only")
        & c7_triage["regime"].isin(["R1_beta_loso", "R2_beta_loso_plus_alpha"])
        & c7_triage["variant"].eq("M9_drop_site_rank")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for color, (regime, group) in zip([PALETTE["orange"], PALETTE["dark_blue"]], triage.groupby("regime", sort=True)):
        group = group.sort_values("coverage_pct")
        label = regime.replace("_", " ")
        axes[0].plot(group["coverage_pct"], group["manual_rows"], marker="o", color=color, linewidth=2, label=label)
        axes[1].plot(group["coverage_pct"], group["auto_errors"], marker="o", color=color, linewidth=2, label=label)
    axes[0].set_title("Manual rows left")
    axes[1].set_title("Auto-accepted errors")
    for ax in axes:
        ax.set_xlabel("Auto-accepted days (%)")
        ax.grid(axis="both", color=PALETTE["light_white"], linewidth=0.8)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Triage Burden Across Best M9 Regimes")
    fig.tight_layout()
    path = out / "fig08_triage_burden_all_best_m9.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figure_rows.append({"figure": path.name, "description": "Manual rows and auto errors for R1/R2 M9 triage"})

    manifest = pd.DataFrame(
        [
            {
                "chunk": "C9_all_result_visualisations",
                "figures": len(figure_rows),
                "tables": len(summary_tables),
                "elapsed_seconds": time.time() - started,
                "outputs": str(out.relative_to(ROOT)),
            }
        ]
    )
    figures = pd.DataFrame(figure_rows)
    tables = pd.DataFrame(summary_tables)
    manifest.to_csv(out / "00_c9_manifest.csv", index=False)
    figures.to_csv(out / "02_c9_figure_index.csv", index=False)
    tables.to_csv(out / "03_c9_table_index.csv", index=False)
    print(f"Wrote C9 all-result visualisations to {out.relative_to(ROOT)}")
    print("\nFigure index")
    print(figures.to_string(index=False))
    return out


def interval_metrics(true_values: pd.Series, pred_values: pd.Series) -> dict[str, float | int]:
    return compute_metrics(true_values.astype(bool), pred_values.astype(bool))


def contiguous_window_from_slots(slots: pd.Series) -> tuple[float, float]:
    values = sorted(slots.dropna().astype(int).unique().tolist())
    if not values:
        return np.nan, np.nan
    return float(min(values)), float(max(values))


def window_iou_from_slots(true_slots: set[int], pred_slots: set[int]) -> float:
    union = true_slots | pred_slots
    if not union:
        return np.nan
    return len(true_slots & pred_slots) / len(union)


def correction_energy_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    manual = float(frame["manual_correction_MWh"].sum())
    pred = float(frame["predicted_correction_MWh"].sum())
    overlap = float(frame["overlap_correction_MWh"].sum())
    union = float(frame["union_correction_MWh"].sum())
    precision = overlap / pred if pred else 0.0
    recall = overlap / manual if manual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = overlap / union if union else np.nan
    return {
        "manual_correction_MWh": manual,
        "predicted_correction_MWh": pred,
        "overlap_correction_MWh": overlap,
        "union_correction_MWh": union,
        "energy_precision": precision,
        "energy_recall": recall,
        "energy_f1": f1,
        "energy_iou": iou,
    }


def add_correction_energy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["correction_magnitude_MWh"] = (
        2 * pd.to_numeric(out["net_load_MW"], errors="coerce").clip(lower=0) * 0.25
    ).fillna(0.0)
    out["manual_correction_MWh"] = np.where(out["label_interval"], out["correction_magnitude_MWh"], 0.0)
    out["predicted_correction_MWh"] = np.where(out["pred_interval"], out["correction_magnitude_MWh"], 0.0)
    out["overlap_correction_MWh"] = np.where(
        out["label_interval"] & out["pred_interval"],
        out["correction_magnitude_MWh"],
        0.0,
    )
    out["union_correction_MWh"] = np.where(
        out["label_interval"] | out["pred_interval"],
        out["correction_magnitude_MWh"],
        0.0,
    )
    return out


def finite_median(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.median()) if len(values) else np.nan


def finite_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def run_c10_window_interval_evaluation() -> Path:
    started = time.time()
    out = OUT_ROOT / "C10_window_interval_evaluation"
    out.mkdir(parents=True, exist_ok=True)
    prediction_path = OUT_ROOT / "C3_beta_loso_manual_ablation/06_c3_daily_prediction_audit.csv"
    if not prediction_path.exists():
        print("C3 prediction audit is missing; running C3 first.")
        run_c3_beta_loso_manual_ablation()
    pred_days = pd.read_csv(prediction_path)
    pred_days = pred_days.loc[
        pred_days["variant"].eq("M9_drop_site_rank")
        & pred_days["dataset"].eq("beta")
        & pred_days["heldout_site"].eq(pred_days["substation_id"])
    ].copy()
    pred_days["date"] = date_key(pred_days["date"])
    pred_days["pred_day"] = safe_bool(pred_days["pred_day"])
    pred_days["true_day"] = safe_bool(pred_days["true_day"])
    pred_days["confidence"] = pred_days["confidence"].fillna("missing").astype(str).str.strip().str.lower()
    pred_days["v03_selected_left_slot"] = pd.to_numeric(pred_days["v03_selected_left_slot"], errors="coerce")
    pred_days["v03_selected_right_slot"] = pd.to_numeric(pred_days["v03_selected_right_slot"], errors="coerce")

    beta = pd.read_parquet(
        FINAL_DATASET_DIR / "dataset_beta.parquet",
        columns=["substation_id", "date", "timestamp", "net_load_MW", "label_interval", "label_day", "confidence"],
    )
    beta["substation_id"] = beta["substation_id"].astype(str)
    beta["date"] = date_key(beta["date"])
    beta["timestamp"] = pd.to_datetime(beta["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    beta["slot"] = beta["timestamp"].dt.hour * 4 + beta["timestamp"].dt.minute // 15
    beta["label_interval"] = safe_bool(beta["label_interval"])
    beta["label_day"] = safe_bool(beta["label_day"])
    beta["confidence"] = beta["confidence"].fillna("missing").astype(str).str.strip().str.lower()

    intervals = beta.merge(
        pred_days[
            [
                "substation_id",
                "date",
                "pred_day",
                "v03_selected_left_slot",
                "v03_selected_right_slot",
                "score",
                "threshold",
            ]
        ],
        on=["substation_id", "date"],
        how="inner",
        validate="many_to_one",
    )
    has_pred_window = (
        intervals["pred_day"]
        & intervals["v03_selected_left_slot"].notna()
        & intervals["v03_selected_right_slot"].notna()
    )
    intervals["pred_interval"] = (
        has_pred_window
        & intervals["slot"].ge(intervals["v03_selected_left_slot"])
        & intervals["slot"].le(intervals["v03_selected_right_slot"])
    )
    intervals["is_daytime"] = intervals["slot"].between(DAYTIME_START, DAYTIME_END)
    intervals["correction_magnitude_MWh"] = (2 * pd.to_numeric(intervals["net_load_MW"], errors="coerce").clip(lower=0) * 0.25).fillna(0.0)
    intervals["manual_correction_MWh"] = np.where(
        intervals["label_interval"],
        intervals["correction_magnitude_MWh"],
        0.0,
    )
    intervals["predicted_correction_MWh"] = np.where(
        intervals["pred_interval"],
        intervals["correction_magnitude_MWh"],
        0.0,
    )
    intervals["overlap_correction_MWh"] = np.where(
        intervals["label_interval"] & intervals["pred_interval"],
        intervals["correction_magnitude_MWh"],
        0.0,
    )
    intervals["union_correction_MWh"] = np.where(
        intervals["label_interval"] | intervals["pred_interval"],
        intervals["correction_magnitude_MWh"],
        0.0,
    )

    metric_rows: list[dict[str, object]] = []
    for subset, frame in [
        ("beta_all_all_intervals", intervals),
        ("beta_all_daytime", intervals.loc[intervals["is_daytime"]].copy()),
        ("beta_sure_all_intervals", intervals.loc[intervals["confidence"].eq("sure")].copy()),
        ("beta_sure_daytime", intervals.loc[intervals["confidence"].eq("sure") & intervals["is_daytime"]].copy()),
    ]:
        metric_rows.append(
            {
                "model": "R1_beta_loso/M9_drop_site_rank",
                "dataset": "beta",
                "subset": subset,
                "summary_scope": "pooled",
                "substation_id": "",
                **interval_metrics(frame["label_interval"], frame["pred_interval"]),
            }
        )
        for site, site_frame in frame.groupby("substation_id", sort=True):
            metric_rows.append(
                {
                    "model": "R1_beta_loso/M9_drop_site_rank",
                    "dataset": "beta",
                    "subset": subset,
                    "summary_scope": "site",
                    "substation_id": site,
                    **interval_metrics(site_frame["label_interval"], site_frame["pred_interval"]),
                }
            )
    interval_metric_frame = pd.DataFrame(metric_rows)

    energy_rows: list[dict[str, object]] = []
    for subset, frame in [
        ("beta_all_all_intervals", intervals),
        ("beta_all_daytime", intervals.loc[intervals["is_daytime"]].copy()),
        ("beta_sure_all_intervals", intervals.loc[intervals["confidence"].eq("sure")].copy()),
        ("beta_sure_daytime", intervals.loc[intervals["confidence"].eq("sure") & intervals["is_daytime"]].copy()),
    ]:
        energy_rows.append(
            {
                "model": "R1_beta_loso/M9_drop_site_rank",
                "dataset": "beta",
                "subset": subset,
                "summary_scope": "pooled",
                "substation_id": "",
                **correction_energy_metrics(frame),
            }
        )
        for site, site_frame in frame.groupby("substation_id", sort=True):
            energy_rows.append(
                {
                    "model": "R1_beta_loso/M9_drop_site_rank",
                    "dataset": "beta",
                    "subset": subset,
                    "summary_scope": "site",
                    "substation_id": site,
                    **correction_energy_metrics(site_frame),
                }
            )
    energy_metric_frame = pd.DataFrame(energy_rows)

    audit_rows: list[dict[str, object]] = []
    for (site, day), group in intervals.groupby(["substation_id", "date"], sort=True):
        true_slots = set(group.loc[group["label_interval"], "slot"].astype(int).tolist())
        pred_slots = set(group.loc[group["pred_interval"], "slot"].astype(int).tolist())
        true_start, true_end = contiguous_window_from_slots(group.loc[group["label_interval"], "slot"])
        pred_start, pred_end = contiguous_window_from_slots(group.loc[group["pred_interval"], "slot"])
        iou = window_iou_from_slots(true_slots, pred_slots)
        true_day = bool(group["label_day"].max())
        pred_day = bool(group["pred_day"].max())
        if true_day and pred_day:
            day_group = "TP_day"
        elif (not true_day) and pred_day:
            day_group = "FP_day"
        elif true_day and (not pred_day):
            day_group = "FN_day"
        else:
            day_group = "TN_day"
        audit_rows.append(
            {
                "substation_id": site,
                "date": day,
                "confidence": str(group["confidence"].iloc[0]),
                "true_day": true_day,
                "pred_day": pred_day,
                "day_group": day_group,
                "true_start_slot": true_start,
                "true_end_slot": true_end,
                "pred_start_slot": pred_start,
                "pred_end_slot": pred_end,
                "true_start_time": slot_to_time(true_start),
                "true_end_time": slot_to_time(true_end),
                "pred_start_time": slot_to_time(pred_start),
                "pred_end_time": slot_to_time(pred_end),
                "true_interval_count": len(true_slots),
                "pred_interval_count": len(pred_slots),
                "intersection_intervals": len(true_slots & pred_slots),
                "union_intervals": len(true_slots | pred_slots),
                "iou": iou,
                "iou_with_fp_fn_zero": 0.0 if pd.isna(iou) and (true_day or pred_day) else iou,
                "start_error_minutes": (pred_start - true_start) * 15 if true_day and pred_day and not pd.isna(true_start) and not pd.isna(pred_start) else np.nan,
                "end_error_minutes": (pred_end - true_end) * 15 if true_day and pred_day and not pd.isna(true_end) and not pd.isna(pred_end) else np.nan,
                "abs_start_error_minutes": abs((pred_start - true_start) * 15) if true_day and pred_day and not pd.isna(true_start) and not pd.isna(pred_start) else np.nan,
                "abs_end_error_minutes": abs((pred_end - true_end) * 15) if true_day and pred_day and not pd.isna(true_end) and not pd.isna(pred_end) else np.nan,
                "score": float(group["score"].iloc[0]),
                "threshold": float(group["threshold"].iloc[0]),
                "manual_correction_MWh": float(group["manual_correction_MWh"].sum()),
                "predicted_correction_MWh": float(group["predicted_correction_MWh"].sum()),
                "overlap_correction_MWh": float(group["overlap_correction_MWh"].sum()),
                "union_correction_MWh": float(group["union_correction_MWh"].sum()),
                "energy_iou": (
                    float(group["overlap_correction_MWh"].sum() / group["union_correction_MWh"].sum())
                    if group["union_correction_MWh"].sum()
                    else np.nan
                ),
            }
        )
    audit = pd.DataFrame(audit_rows)

    window_rows: list[dict[str, object]] = []
    for subset, frame in [
        ("beta_all", audit),
        ("beta_sure_only", audit.loc[audit["confidence"].eq("sure")].copy()),
    ]:
        for scope, scope_frame in [
            ("event_days_truth_or_pred", frame.loc[frame["true_day"] | frame["pred_day"]].copy()),
            ("tp_days_only", frame.loc[frame["day_group"].eq("TP_day")].copy()),
        ]:
            if scope_frame.empty:
                continue
            iou_values = scope_frame["iou_with_fp_fn_zero"] if scope == "event_days_truth_or_pred" else scope_frame["iou"]
            window_rows.append(
                {
                    "model": "R1_beta_loso/M9_drop_site_rank",
                    "dataset": "beta",
                    "subset": subset,
                    "scope": scope,
                    "days": len(scope_frame),
                    "mean_iou": float(iou_values.mean()),
                    "median_iou": float(iou_values.median()),
                    "iou_ge_0p50_rate": float((iou_values >= 0.50).mean()),
                    "iou_ge_0p70_rate": float((iou_values >= 0.70).mean()),
                    "median_abs_start_error_minutes": finite_median(scope_frame["abs_start_error_minutes"]),
                    "median_abs_end_error_minutes": finite_median(scope_frame["abs_end_error_minutes"]),
                    "mean_abs_start_error_minutes": finite_mean(scope_frame["abs_start_error_minutes"]),
                    "mean_abs_end_error_minutes": finite_mean(scope_frame["abs_end_error_minutes"]),
                }
            )
        for site, site_frame in frame.groupby("substation_id", sort=True):
            event = site_frame.loc[site_frame["true_day"] | site_frame["pred_day"]].copy()
            if event.empty:
                continue
            iou_values = event["iou_with_fp_fn_zero"]
            window_rows.append(
                {
                    "model": "R1_beta_loso/M9_drop_site_rank",
                    "dataset": "beta",
                    "subset": subset,
                    "scope": "site_event_days_truth_or_pred",
                    "substation_id": site,
                    "days": len(event),
                    "mean_iou": float(iou_values.mean()),
                    "median_iou": float(iou_values.median()),
                    "iou_ge_0p50_rate": float((iou_values >= 0.50).mean()),
                    "iou_ge_0p70_rate": float((iou_values >= 0.70).mean()),
                    "median_abs_start_error_minutes": finite_median(event["abs_start_error_minutes"]),
                    "median_abs_end_error_minutes": finite_median(event["abs_end_error_minutes"]),
                    "mean_abs_start_error_minutes": finite_mean(event["abs_start_error_minutes"]),
                    "mean_abs_end_error_minutes": finite_mean(event["abs_end_error_minutes"]),
                }
            )
    window_metrics = pd.DataFrame(window_rows)

    manifest = pd.DataFrame(
        [
            {
                "chunk": "C10_window_interval_evaluation",
                "model": "R1_beta_loso/M9_drop_site_rank",
                "prediction_source": str(prediction_path.relative_to(ROOT)),
                "interval_rows": len(intervals),
                "site_days": len(audit),
                "elapsed_seconds": time.time() - started,
                "outputs": str(out.relative_to(ROOT)),
            }
        ]
    )

    manifest.to_csv(out / "01_c10_manifest.csv", index=False)
    interval_metric_frame.to_csv(out / "02_c10_interval_metrics.csv", index=False)
    window_metrics.to_csv(out / "03_c10_window_iou_metrics.csv", index=False)
    audit.to_csv(out / "04_c10_window_day_audit.csv", index=False)
    energy_metric_frame.to_csv(out / "05_c10_correction_energy_metrics.csv", index=False)

    print(f"Wrote C10 window/interval evaluation to {out.relative_to(ROOT)}")
    print("\nManifest")
    print(manifest.round(3).to_string(index=False))
    print("\nInterval metrics")
    pooled = interval_metric_frame.loc[interval_metric_frame["summary_scope"].eq("pooled")]
    print(pooled[["subset", "support", "positive_support", "precision", "recall", "f1", "tp", "fp", "fn"]].round(4).to_string(index=False))
    print("\nWindow IoU metrics")
    show_cols = ["subset", "scope", "days", "mean_iou", "median_iou", "iou_ge_0p50_rate", "iou_ge_0p70_rate", "median_abs_start_error_minutes", "median_abs_end_error_minutes"]
    print(window_metrics.loc[window_metrics["scope"].isin(["event_days_truth_or_pred", "tp_days_only"]), show_cols].round(4).to_string(index=False))
    print("\nCorrection-energy metrics")
    pooled_energy = energy_metric_frame.loc[energy_metric_frame["summary_scope"].eq("pooled")]
    energy_cols = [
        "subset",
        "manual_correction_MWh",
        "predicted_correction_MWh",
        "overlap_correction_MWh",
        "energy_precision",
        "energy_recall",
        "energy_f1",
        "energy_iou",
    ]
    print(pooled_energy[energy_cols].round(4).to_string(index=False))
    return out


def load_c12_final_beta_intervals() -> pd.DataFrame:
    beta = pd.read_parquet(
        FINAL_DATASET_DIR / "dataset_beta.parquet",
        columns=["substation_id", "date", "timestamp", "net_load_MW", "label_interval", "label_day", "confidence"],
    )
    beta["substation_id"] = beta["substation_id"].astype(str)
    beta["date"] = date_key(beta["date"])
    beta["timestamp"] = pd.to_datetime(beta["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    beta["slot"] = beta["timestamp"].dt.hour * 4 + beta["timestamp"].dt.minute // 15
    beta["label_interval"] = safe_bool(beta["label_interval"])
    beta["label_day"] = safe_bool(beta["label_day"])
    beta["confidence"] = beta["confidence"].fillna("missing").astype(str).str.strip().str.lower()
    return beta.sort_values(["substation_id", "date", "slot"]).reset_index(drop=True)


def prefix_metric_dict(prefix: str, metrics: dict[str, float | int]) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def c12_metric_bundle(intervals: pd.DataFrame) -> dict[str, float | int]:
    if intervals.empty:
        return {
            "day_support": 0,
            "day_positive_support": 0,
            "day_tp": 0,
            "day_fp": 0,
            "day_fn": 0,
            "day_tn": 0,
            "day_precision": 0.0,
            "day_recall": 0.0,
            "day_f1": 0.0,
            "interval_support": 0,
            "interval_positive_support": 0,
            "interval_tp": 0,
            "interval_fp": 0,
            "interval_fn": 0,
            "interval_tn": 0,
            "interval_precision": 0.0,
            "interval_recall": 0.0,
            "interval_f1": 0.0,
            "manual_correction_MWh": 0.0,
            "predicted_correction_MWh": 0.0,
            "overlap_correction_MWh": 0.0,
            "union_correction_MWh": 0.0,
            "energy_precision": 0.0,
            "energy_recall": 0.0,
            "energy_f1": 0.0,
            "energy_iou": np.nan,
        }

    day = (
        intervals.groupby(["substation_id", "date"], as_index=False)
        .agg(true_day=("label_day", "max"), pred_day=("pred_interval", "max"))
        .sort_values(["substation_id", "date"])
    )
    interval_metrics_row = interval_metrics(intervals["label_interval"], intervals["pred_interval"])
    energy_frame = add_correction_energy_columns(intervals)
    return {
        **prefix_metric_dict("day", compute_metrics(day["true_day"], day["pred_day"])),
        **prefix_metric_dict("interval", interval_metrics_row),
        **correction_energy_metrics(energy_frame),
    }


def c12_append_metric_rows(
    rows: list[dict[str, object]],
    site_rows: list[dict[str, object]],
    *,
    frame: pd.DataFrame,
    model_family: str,
    model_variant: str,
    regime: str,
    prediction_source: str,
    notes: str,
) -> None:
    for subset, subset_frame in [
        ("beta_all", frame),
        ("beta_sure_only", frame.loc[frame["confidence"].eq("sure")].copy()),
    ]:
        pooled = c12_metric_bundle(subset_frame)
        rows.append(
            {
                "model_family": model_family,
                "model_variant": model_variant,
                "regime": regime,
                "dataset": "beta",
                "subset": subset,
                "summary_scope": "pooled",
                "substation_id": "",
                "prediction_source": prediction_source,
                "notes": notes,
                **pooled,
            }
        )

        current_site_rows: list[dict[str, object]] = []
        for site, site_frame in subset_frame.groupby("substation_id", sort=True):
            metrics = c12_metric_bundle(site_frame)
            row = {
                "model_family": model_family,
                "model_variant": model_variant,
                "regime": regime,
                "dataset": "beta",
                "subset": subset,
                "summary_scope": "site",
                "substation_id": site,
                "prediction_source": prediction_source,
                "notes": notes,
                **metrics,
            }
            site_rows.append(row)
            current_site_rows.append(row)

        if current_site_rows:
            site_metrics = pd.DataFrame(current_site_rows)
            count_cols = [
                "day_support",
                "day_positive_support",
                "day_tp",
                "day_fp",
                "day_fn",
                "day_tn",
                "interval_support",
                "interval_positive_support",
                "interval_tp",
                "interval_fp",
                "interval_fn",
                "interval_tn",
            ]
            mean_cols = [
                "day_precision",
                "day_recall",
                "day_f1",
                "interval_precision",
                "interval_recall",
                "interval_f1",
                "manual_correction_MWh",
                "predicted_correction_MWh",
                "overlap_correction_MWh",
                "union_correction_MWh",
                "energy_precision",
                "energy_recall",
                "energy_f1",
                "energy_iou",
            ]
            macro: dict[str, object] = {}
            for col in count_cols:
                macro[col] = int(site_metrics[col].sum())
            for col in mean_cols:
                macro[col] = float(site_metrics[col].mean())
            rows.append(
                {
                    "model_family": model_family,
                    "model_variant": model_variant,
                    "regime": regime,
                    "dataset": "beta",
                    "subset": subset,
                    "summary_scope": "macro_site_average",
                    "substation_id": "",
                    "prediction_source": prediction_source,
                    "notes": "Unweighted average of per-site metrics.",
                    **macro,
                }
            )


def c12_interval_frame_from_physical_predictions(beta: pd.DataFrame, pred_days: pd.DataFrame) -> pd.DataFrame:
    c1_path = OUT_ROOT / "C1_cached_daily_features/02_c1_daily_feature_cache.csv"
    if not c1_path.exists():
        print("C1 selected-window cache is missing; running C1 first.")
        run_c1_cached_daily_features()
    selected_windows = pd.read_csv(
        c1_path,
        usecols=["dataset", "substation_id", "date", "v03_selected_left_slot", "v03_selected_right_slot"],
    )
    selected_windows = selected_windows.loc[selected_windows["dataset"].eq("beta")].copy()
    selected_windows["substation_id"] = selected_windows["substation_id"].astype(str)
    selected_windows["date"] = date_key(selected_windows["date"])
    selected_windows["v03_selected_left_slot"] = pd.to_numeric(
        selected_windows["v03_selected_left_slot"], errors="coerce"
    )
    selected_windows["v03_selected_right_slot"] = pd.to_numeric(
        selected_windows["v03_selected_right_slot"], errors="coerce"
    )

    pred_days = pred_days.copy()
    pred_days["substation_id"] = pred_days["substation_id"].astype(str)
    pred_days["date"] = date_key(pred_days["date"])
    pred_days["pred_day"] = safe_bool(pred_days["pred_day"])
    pred_days = pred_days.merge(
        selected_windows[["substation_id", "date", "v03_selected_left_slot", "v03_selected_right_slot"]],
        on=["substation_id", "date"],
        how="left",
        validate="many_to_one",
    )
    intervals = beta.merge(
        pred_days[["substation_id", "date", "pred_day", "v03_selected_left_slot", "v03_selected_right_slot"]],
        on=["substation_id", "date"],
        how="inner",
        validate="many_to_one",
    )
    has_pred_window = (
        intervals["pred_day"]
        & intervals["v03_selected_left_slot"].notna()
        & intervals["v03_selected_right_slot"].notna()
    )
    intervals["pred_interval"] = (
        has_pred_window
        & intervals["slot"].ge(intervals["v03_selected_left_slot"])
        & intervals["slot"].le(intervals["v03_selected_right_slot"])
    )
    return intervals


def c12_interval_frame_from_notebook2_prediction(beta: pd.DataFrame, prediction_path: Path) -> pd.DataFrame:
    pred = pd.read_csv(
        prediction_path,
        usecols=["substation_id", "date", "timestamp", "pred_interval"],
    )
    pred["substation_id"] = pred["substation_id"].astype(str)
    pred["date"] = date_key(pred["date"])
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    pred["slot"] = pred["timestamp"].dt.hour * 4 + pred["timestamp"].dt.minute // 15
    pred["pred_interval"] = safe_bool(pred["pred_interval"])
    pred = pred.drop(columns=["timestamp"]).drop_duplicates(["substation_id", "date", "slot"], keep="last")
    intervals = beta.merge(
        pred,
        on=["substation_id", "date", "slot"],
        how="inner",
        validate="one_to_one",
    )
    if len(intervals) != len(beta):
        raise ValueError(
            f"Prediction file {prediction_path.name} joined {len(intervals):,} rows, expected {len(beta):,}."
        )
    return intervals


def run_c12_multi_model_metric_summary() -> Path:
    started = time.time()
    out = OUT_ROOT / "C12_multi_model_metric_summary"
    out.mkdir(parents=True, exist_ok=True)

    beta = load_c12_final_beta_intervals()
    metric_rows: list[dict[str, object]] = []
    site_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []

    try:
        physical_predictions = load_c6_prediction_audits()
        physical_predictions = physical_predictions.loc[physical_predictions["dataset"].eq("beta")].copy()
        for (regime, variant), group in physical_predictions.groupby(["regime", "variant"], sort=True):
            source_chunk = str(group["source_chunk"].iloc[0]) if "source_chunk" in group.columns else "unknown"
            intervals = c12_interval_frame_from_physical_predictions(beta, group)
            model_family = "logistic_physical_score" if variant == LOGISTIC_VARIANT else "physical_score"
            c12_append_metric_rows(
                metric_rows,
                site_rows,
                frame=intervals,
                model_family=model_family,
                model_variant=variant,
                regime=regime,
                prediction_source=source_chunk,
                notes="Daily physical-score prediction expanded to selected v0.3 candidate window.",
            )
            source_rows.append(
                {
                    "model_family": model_family,
                    "model_variant": variant,
                    "regime": regime,
                    "source": source_chunk,
                    "status": "included",
                    "rows": len(group),
                    "notes": "Expanded cached daily predictions into interval flags.",
                }
            )
    except Exception as exc:  # pragma: no cover - explicit audit path for exploratory chunk
        source_rows.append(
            {
                "model_family": "physical_score",
                "model_variant": "C4_C5_predictions",
                "regime": "multiple",
                "source": "C4/C5 prediction audits",
                "status": "skipped",
                "rows": 0,
                "notes": str(exc),
            }
        )

    notebook2_sources = [
        (
            "m8_xgb",
            "m8_xgb",
            JOURNAL / "outputs/intermediate/02_correction_validation/22_correction_predictions_beta_transfer_m8_xgb.csv",
        ),
        (
            "m7_dtr",
            "m7_dtr",
            JOURNAL / "outputs/intermediate/02_correction_validation/23_correction_predictions_beta_transfer_m7_dtr.csv",
        ),
    ]
    for model_family, variant, prediction_path in notebook2_sources:
        if not prediction_path.exists():
            source_rows.append(
                {
                    "model_family": model_family,
                    "model_variant": variant,
                    "regime": "notebook2_alpha_to_beta_transfer",
                    "source": str(prediction_path.relative_to(ROOT)),
                    "status": "skipped",
                    "rows": 0,
                    "notes": "Prediction file missing.",
                }
            )
            continue
        try:
            intervals = c12_interval_frame_from_notebook2_prediction(beta, prediction_path)
            c12_append_metric_rows(
                metric_rows,
                site_rows,
                frame=intervals,
                model_family=model_family,
                model_variant=variant,
                regime="notebook2_alpha_to_beta_transfer",
                prediction_source=str(prediction_path.relative_to(ROOT)),
                notes="Re-scored stored Notebook 2 interval predictions against final Beta labels.",
            )
            source_rows.append(
                {
                    "model_family": model_family,
                    "model_variant": variant,
                    "regime": "notebook2_alpha_to_beta_transfer",
                    "source": str(prediction_path.relative_to(ROOT)),
                    "status": "included",
                    "rows": len(intervals),
                    "notes": "Joined by site/date/15-minute slot; embedded stale labels ignored.",
                }
            )
        except Exception as exc:  # pragma: no cover - explicit audit path for exploratory chunk
            source_rows.append(
                {
                    "model_family": model_family,
                    "model_variant": variant,
                    "regime": "notebook2_alpha_to_beta_transfer",
                    "source": str(prediction_path.relative_to(ROOT)),
                    "status": "skipped",
                    "rows": 0,
                    "notes": str(exc),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    site_metrics = pd.DataFrame(site_rows)
    source_manifest = pd.DataFrame(source_rows)
    sort_cols = ["subset", "summary_scope", "day_f1", "interval_f1", "energy_f1"]
    if not metrics.empty:
        metrics = metrics.sort_values(sort_cols, ascending=[True, True, False, False, False]).reset_index(drop=True)
    if not site_metrics.empty:
        site_metrics = site_metrics.sort_values(
            ["subset", "model_family", "regime", "model_variant", "substation_id"]
        ).reset_index(drop=True)

    manifest = pd.DataFrame(
        [
            {
                "chunk": "C12_multi_model_metric_summary",
                "dataset": "beta",
                "truth_source": str((FINAL_DATASET_DIR / "dataset_beta.parquet").relative_to(ROOT)),
                "interval_rows": len(beta),
                "site_days": int(beta.groupby(["substation_id", "date"]).ngroups),
                "sure_site_days": int(
                    beta[["substation_id", "date", "confidence"]]
                    .drop_duplicates(["substation_id", "date"])["confidence"]
                    .eq("sure")
                    .sum()
                ),
                "included_sources": int(source_manifest["status"].eq("included").sum()),
                "skipped_sources": int(source_manifest["status"].eq("skipped").sum()),
                "elapsed_seconds": time.time() - started,
                "outputs": str(out.relative_to(ROOT)),
            }
        ]
    )

    manifest.to_csv(out / "01_c12_manifest.csv", index=False)
    metrics.to_csv(out / "02_c12_multi_model_metric_summary.csv", index=False)
    site_metrics.to_csv(out / "03_c12_site_metric_summary.csv", index=False)
    source_manifest.to_csv(out / "04_c12_prediction_source_manifest.csv", index=False)

    print(f"Wrote C12 multi-model metric summary to {out.relative_to(ROOT)}")
    print("\nManifest")
    print(manifest.round(3).to_string(index=False))
    print("\nIncluded/skipped prediction sources")
    print(source_manifest[["model_family", "model_variant", "regime", "status", "rows", "notes"]].to_string(index=False))
    if not metrics.empty:
        headline = metrics.loc[
            metrics["summary_scope"].eq("pooled")
            & metrics["subset"].isin(["beta_all", "beta_sure_only"])
        ].copy()
        headline = headline.sort_values(["subset", "day_f1"], ascending=[True, False])
        show_cols = [
            "subset",
            "model_family",
            "regime",
            "model_variant",
            "day_precision",
            "day_recall",
            "day_f1",
            "interval_f1",
            "energy_f1",
            "energy_iou",
        ]
        print("\nPooled headline metrics")
        print(headline[show_cols].round(4).to_string(index=False))
    return out


def run_c13_single_feature_ladder() -> Path:
    started = time.time()
    out = OUT_ROOT / "C13_single_feature_ladder"
    out.mkdir(parents=True, exist_ok=True)

    daily = load_c1_daily_feature_cache()
    alpha = daily.loc[daily["dataset"].eq("alpha")].copy()
    beta = daily.loc[daily["dataset"].eq("beta")].copy()
    beta_sites = sorted(beta["substation_id"].unique().tolist())

    threshold_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []

    for variant, component in ONE_FEATURE_VARIANTS:
        score_col = f"{variant}_score"
        alpha[score_col] = pd.to_numeric(alpha[component], errors="coerce").fillna(0.0)
        beta[score_col] = pd.to_numeric(beta[component], errors="coerce").fillna(0.0)

        for regime in ["R1_beta_loso", "R2_beta_loso_plus_alpha"]:
            for heldout_site in beta_sites:
                beta_train = beta.loc[
                    beta["confidence"].eq("sure") & ~beta["substation_id"].eq(heldout_site)
                ].copy()
                if regime == "R1_beta_loso":
                    train = beta_train
                    dataset_balanced = False
                    training_subset = "other_7_beta_sites_sure_only"
                else:
                    train = pd.concat([alpha, beta_train], ignore_index=True)
                    dataset_balanced = True
                    training_subset = "all_alpha_plus_other_7_beta_sites_sure_only"

                threshold, selected, _ = select_threshold_weighted_macro_site(
                    train,
                    score_col,
                    dataset_balanced=dataset_balanced,
                )
                eval_frame = beta.loc[beta["substation_id"].eq(heldout_site)].copy()
                eval_frame["regime"] = regime
                eval_frame["variant"] = variant
                eval_frame["component"] = component
                eval_frame["heldout_site"] = heldout_site
                eval_frame["threshold"] = threshold
                eval_frame["score"] = eval_frame[score_col]
                eval_frame["pred_day"] = eval_frame["score"] >= threshold
                prediction_parts.append(
                    eval_frame[
                        [
                            "regime",
                            "variant",
                            "component",
                            "heldout_site",
                            "dataset",
                            "substation_id",
                            "date",
                            "confidence",
                            "true_day",
                            "score",
                            "pred_day",
                            "threshold",
                        ]
                    ]
                )
                threshold_rows.append(
                    {
                        "chunk": "C13_single_feature_ladder",
                        "regime": regime,
                        "variant": variant,
                        "component": component,
                        "heldout_site": heldout_site,
                        "training_subset": training_subset,
                        "dataset_balanced_threshold_selection": dataset_balanced,
                        "training_rows": len(train),
                        "training_positive_support": int(train["true_day"].sum()),
                        "threshold": threshold,
                        **{f"selected_{key}": value for key, value in selected.items() if key != "threshold"},
                    }
                )

        threshold, selected, _ = select_threshold_weighted_macro_site(
            alpha,
            score_col,
            dataset_balanced=False,
        )
        eval_frame = beta.copy()
        eval_frame["regime"] = "R3_alpha_only_to_beta"
        eval_frame["variant"] = variant
        eval_frame["component"] = component
        eval_frame["heldout_site"] = "all_beta"
        eval_frame["threshold"] = threshold
        eval_frame["score"] = eval_frame[score_col]
        eval_frame["pred_day"] = eval_frame["score"] >= threshold
        prediction_parts.append(
            eval_frame[
                [
                    "regime",
                    "variant",
                    "component",
                    "heldout_site",
                    "dataset",
                    "substation_id",
                    "date",
                    "confidence",
                    "true_day",
                    "score",
                    "pred_day",
                    "threshold",
                ]
            ]
        )
        threshold_rows.append(
            {
                "chunk": "C13_single_feature_ladder",
                "regime": "R3_alpha_only_to_beta",
                "variant": variant,
                "component": component,
                "heldout_site": "all_beta",
                "training_subset": "all_alpha",
                "dataset_balanced_threshold_selection": False,
                "training_rows": len(alpha),
                "training_positive_support": int(alpha["true_day"].sum()),
                "threshold": threshold,
                **{f"selected_{key}": value for key, value in selected.items() if key != "threshold"},
            }
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metric_rows: list[dict[str, object]] = []
    for (regime, variant), frame in predictions.groupby(["regime", "variant"], sort=True):
        component = str(frame["component"].iloc[0])
        for subset, subset_frame in [
            ("beta_all", frame),
            ("beta_sure_only", frame.loc[frame["confidence"].eq("sure")].copy()),
        ]:
            rows = metric_rows_for_subset(
                subset_frame,
                variant=variant,
                dataset="beta",
                subset=subset,
                pred_col="pred_day",
                threshold=float("nan"),
            )
            for row in rows:
                row["regime"] = regime
                row["component"] = component
            metric_rows.extend(rows)

    thresholds = pd.DataFrame(threshold_rows)
    metrics = pd.DataFrame(metric_rows)

    def lookup(subset: str, scope: str, metric: str) -> pd.Series:
        return (
            metrics.loc[
                metrics["subset"].eq(subset)
                & metrics["summary_scope"].eq(scope),
                ["regime", "variant", metric],
            ]
            .set_index(["regime", "variant"])[metric]
        )

    ranking = thresholds[["regime", "variant", "component"]].drop_duplicates().reset_index(drop=True)
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

    beta_intervals = load_c12_final_beta_intervals()
    multi_metric_rows: list[dict[str, object]] = []
    multi_site_rows: list[dict[str, object]] = []
    for (regime, variant), frame in predictions.groupby(["regime", "variant"], sort=True):
        component = str(frame["component"].iloc[0])
        intervals = c12_interval_frame_from_physical_predictions(beta_intervals, frame)
        metric_start = len(multi_metric_rows)
        site_start = len(multi_site_rows)
        c12_append_metric_rows(
            multi_metric_rows,
            multi_site_rows,
            frame=intervals,
            model_family="single_feature_physical_score",
            model_variant=variant,
            regime=regime,
            prediction_source="C13_single_feature_ladder",
            notes=f"One-feature score using {component}; expanded to selected v0.3 candidate window.",
        )
        for row in multi_metric_rows[metric_start:]:
            row["component"] = component
        for row in multi_site_rows[site_start:]:
            row["component"] = component

    multi_metrics = pd.DataFrame(multi_metric_rows)
    multi_site_metrics = pd.DataFrame(multi_site_rows)
    if not multi_metrics.empty:
        multi_metrics = multi_metrics.sort_values(
            ["subset", "summary_scope", "day_f1", "interval_f1", "energy_f1"],
            ascending=[True, True, False, False, False],
        ).reset_index(drop=True)
    if not multi_site_metrics.empty:
        multi_site_metrics = multi_site_metrics.sort_values(
            ["subset", "regime", "model_variant", "substation_id"]
        ).reset_index(drop=True)

    manifest = pd.DataFrame(
        [
            {
                "chunk": "C13_single_feature_ladder",
                "source_cache": str((OUT_ROOT / "C1_cached_daily_features/02_c1_daily_feature_cache.csv").relative_to(ROOT)),
                "variants": len(ONE_FEATURE_VARIANTS),
                "regimes": "R1_beta_loso;R2_beta_loso_plus_alpha;R3_alpha_only_to_beta",
                "beta_rows": len(beta),
                "beta_sure_rows": int(beta["confidence"].eq("sure").sum()),
                "threshold_selection": "R1/R3 macro-site F1; R2 dataset-balanced macro-site F1",
                "interval_energy_confirmation": True,
                "elapsed_seconds": time.time() - started,
                "outputs": str(out.relative_to(ROOT)),
            }
        ]
    )

    manifest.to_csv(out / "01_c13_manifest.csv", index=False)
    thresholds.to_csv(out / "02_c13_threshold_selection.csv", index=False)
    metrics.to_csv(out / "03_c13_day_level_metrics.csv", index=False)
    ranking.to_csv(out / "04_c13_single_feature_ranking.csv", index=False)
    predictions.to_csv(out / "05_c13_daily_prediction_audit.csv", index=False)
    multi_metrics.to_csv(out / "06_c13_multi_metric_summary.csv", index=False)
    multi_site_metrics.to_csv(out / "07_c13_site_multi_metric_summary.csv", index=False)

    print(f"Wrote C13 single-feature outputs to {out.relative_to(ROOT)}")
    print("\nManifest")
    print(manifest.round(3).to_string(index=False))
    print("\nSingle-feature day-level ranking")
    show_cols = [
        "regime",
        "variant",
        "component",
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
        ].copy()
        pooled = pooled.sort_values(["day_f1", "interval_f1", "energy_f1"], ascending=[False, False, False])
        print("\nSingle-feature pooled Beta sure-only day/interval/energy metrics")
        print(
            pooled[
                [
                    "regime",
                    "model_variant",
                    "component",
                    "day_f1",
                    "interval_f1",
                    "energy_f1",
                    "energy_iou",
                ]
            ]
            .head(20)
            .round(4)
            .to_string(index=False)
        )
    return out


def c14_weighted_score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for col in FEATURE_COLUMNS:
        score = score + weights.get(col, 0.0) * pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return score


def run_c14_small_weight_grid() -> Path:
    started = time.time()
    out = OUT_ROOT / "C14_small_weight_grid"
    out.mkdir(parents=True, exist_ok=True)

    daily = load_c1_daily_feature_cache()
    alpha = daily.loc[daily["dataset"].eq("alpha")].copy()
    beta = daily.loc[daily["dataset"].eq("beta")].copy()
    beta_sites = sorted(beta["substation_id"].unique().tolist())

    bridge_weights = [1.0, 1.5, 2.0]
    roughness_weights = [0.5, 1.0, 1.5, 2.0]
    site_centered_weights = [0.0, 0.5, 1.0]
    grid_rows: list[dict[str, object]] = []
    grid_definitions: list[dict[str, object]] = []
    for bridge_weight in bridge_weights:
        for roughness_weight in roughness_weights:
            for site_centered_weight in site_centered_weights:
                variant = (
                    f"G_b{str(bridge_weight).replace('.', 'p')}"
                    f"_r{str(roughness_weight).replace('.', 'p')}"
                    f"_sc{str(site_centered_weight).replace('.', 'p')}"
                )
                weights = {
                    "F1_bridge_improvement": bridge_weight,
                    "F2_roughness_improvement": roughness_weight,
                    "F3_slope_continuity_improvement": 1.0,
                    "F4_duration_plausibility": 1.0,
                    "F5_n_height_ratio": 1.0,
                    "F6_solar_strength_ratio": 1.0,
                    "F7_solar_peak_alignment": 1.0,
                    "F8_site_centered_core_score": site_centered_weight,
                    "F9_site_rank_core_score": 0.0,
                }
                grid_rows.append({"variant": variant, "weights": weights})
                grid_definitions.append({"variant": variant, **{f"weight_{key}": value for key, value in weights.items()}})

    threshold_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    for grid in grid_rows:
        variant = str(grid["variant"])
        weights = grid["weights"]
        score_col = f"{variant}_score"
        alpha[score_col] = c14_weighted_score(alpha, weights)
        beta[score_col] = c14_weighted_score(beta, weights)

        for regime in ["R1_beta_loso", "R2_beta_loso_plus_alpha"]:
            for heldout_site in beta_sites:
                beta_train = beta.loc[
                    beta["confidence"].eq("sure") & ~beta["substation_id"].eq(heldout_site)
                ].copy()
                if regime == "R1_beta_loso":
                    train = beta_train
                    dataset_balanced = False
                    training_subset = "other_7_beta_sites_sure_only"
                else:
                    train = pd.concat([alpha, beta_train], ignore_index=True)
                    dataset_balanced = True
                    training_subset = "all_alpha_plus_other_7_beta_sites_sure_only"

                threshold, selected, _ = select_threshold_weighted_macro_site(
                    train,
                    score_col,
                    dataset_balanced=dataset_balanced,
                )
                eval_frame = beta.loc[beta["substation_id"].eq(heldout_site)].copy()
                eval_frame["regime"] = regime
                eval_frame["variant"] = variant
                eval_frame["heldout_site"] = heldout_site
                eval_frame["threshold"] = threshold
                eval_frame["score"] = eval_frame[score_col]
                eval_frame["pred_day"] = eval_frame["score"] >= threshold
                prediction_parts.append(
                    eval_frame[
                        [
                            "regime",
                            "variant",
                            "heldout_site",
                            "dataset",
                            "substation_id",
                            "date",
                            "confidence",
                            "true_day",
                            "score",
                            "pred_day",
                            "threshold",
                        ]
                    ]
                )
                threshold_rows.append(
                    {
                        "chunk": "C14_small_weight_grid",
                        "regime": regime,
                        "variant": variant,
                        "heldout_site": heldout_site,
                        "training_subset": training_subset,
                        "dataset_balanced_threshold_selection": dataset_balanced,
                        "training_rows": len(train),
                        "training_positive_support": int(train["true_day"].sum()),
                        "threshold": threshold,
                        **{f"selected_{key}": value for key, value in selected.items() if key != "threshold"},
                    }
                )

        threshold, selected, _ = select_threshold_weighted_macro_site(
            alpha,
            score_col,
            dataset_balanced=False,
        )
        eval_frame = beta.copy()
        eval_frame["regime"] = "R3_alpha_only_to_beta"
        eval_frame["variant"] = variant
        eval_frame["heldout_site"] = "all_beta"
        eval_frame["threshold"] = threshold
        eval_frame["score"] = eval_frame[score_col]
        eval_frame["pred_day"] = eval_frame["score"] >= threshold
        prediction_parts.append(
            eval_frame[
                [
                    "regime",
                    "variant",
                    "heldout_site",
                    "dataset",
                    "substation_id",
                    "date",
                    "confidence",
                    "true_day",
                    "score",
                    "pred_day",
                    "threshold",
                ]
            ]
        )
        threshold_rows.append(
            {
                "chunk": "C14_small_weight_grid",
                "regime": "R3_alpha_only_to_beta",
                "variant": variant,
                "heldout_site": "all_beta",
                "training_subset": "all_alpha",
                "dataset_balanced_threshold_selection": False,
                "training_rows": len(alpha),
                "training_positive_support": int(alpha["true_day"].sum()),
                "threshold": threshold,
                **{f"selected_{key}": value for key, value in selected.items() if key != "threshold"},
            }
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metric_rows: list[dict[str, object]] = []
    for (regime, variant), frame in predictions.groupby(["regime", "variant"], sort=True):
        for subset, subset_frame in [
            ("beta_all", frame),
            ("beta_sure_only", frame.loc[frame["confidence"].eq("sure")].copy()),
        ]:
            rows = metric_rows_for_subset(
                subset_frame,
                variant=variant,
                dataset="beta",
                subset=subset,
                pred_col="pred_day",
                threshold=float("nan"),
            )
            for row in rows:
                row["regime"] = regime
            metric_rows.extend(rows)

    grid_definitions_frame = pd.DataFrame(grid_definitions)
    thresholds = pd.DataFrame(threshold_rows).merge(grid_definitions_frame, on="variant", how="left")
    metrics = pd.DataFrame(metric_rows).merge(grid_definitions_frame, on="variant", how="left")

    def lookup(subset: str, scope: str, metric: str) -> pd.Series:
        return (
            metrics.loc[
                metrics["subset"].eq(subset)
                & metrics["summary_scope"].eq(scope),
                ["regime", "variant", metric],
            ]
            .set_index(["regime", "variant"])[metric]
        )

    ranking = thresholds[["regime", "variant"]].drop_duplicates().reset_index(drop=True)
    ranking = ranking.merge(grid_definitions_frame, on="variant", how="left")
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

    top_keys = ranking.head(10)[["regime", "variant"]].drop_duplicates()
    top_predictions = predictions.merge(top_keys, on=["regime", "variant"], how="inner")
    beta_intervals = load_c12_final_beta_intervals()
    multi_metric_rows: list[dict[str, object]] = []
    multi_site_rows: list[dict[str, object]] = []
    for (regime, variant), frame in top_predictions.groupby(["regime", "variant"], sort=True):
        intervals = c12_interval_frame_from_physical_predictions(beta_intervals, frame)
        metric_start = len(multi_metric_rows)
        site_start = len(multi_site_rows)
        c12_append_metric_rows(
            multi_metric_rows,
            multi_site_rows,
            frame=intervals,
            model_family="small_grid_physical_score",
            model_variant=variant,
            regime=regime,
            prediction_source="C14_small_weight_grid",
            notes="Top-10 day-F1 grid candidate expanded to selected v0.3 candidate window.",
        )
        weights = grid_definitions_frame.loc[grid_definitions_frame["variant"].eq(variant)].iloc[0].to_dict()
        for row in multi_metric_rows[metric_start:]:
            row.update({key: value for key, value in weights.items() if key.startswith("weight_")})
        for row in multi_site_rows[site_start:]:
            row.update({key: value for key, value in weights.items() if key.startswith("weight_")})

    multi_metrics = pd.DataFrame(multi_metric_rows)
    multi_site_metrics = pd.DataFrame(multi_site_rows)
    if not multi_metrics.empty:
        multi_metrics = multi_metrics.sort_values(
            ["subset", "summary_scope", "day_f1", "interval_f1", "energy_f1"],
            ascending=[True, True, False, False, False],
        ).reset_index(drop=True)
    if not multi_site_metrics.empty:
        multi_site_metrics = multi_site_metrics.sort_values(
            ["subset", "regime", "model_variant", "substation_id"]
        ).reset_index(drop=True)

    manifest = pd.DataFrame(
        [
            {
                "chunk": "C14_small_weight_grid",
                "source_cache": str((OUT_ROOT / "C1_cached_daily_features/02_c1_daily_feature_cache.csv").relative_to(ROOT)),
                "weight_grid_count": len(grid_definitions_frame),
                "regime_variant_runs": int(ranking[["regime", "variant"]].drop_duplicates().shape[0]),
                "top_interval_energy_models": int(top_keys.shape[0]),
                "beta_rows": len(beta),
                "beta_sure_rows": int(beta["confidence"].eq("sure").sum()),
                "threshold_selection": "R1/R3 macro-site F1; R2 dataset-balanced macro-site F1",
                "interval_energy_confirmation": "top_10_day_f1_only",
                "elapsed_seconds": time.time() - started,
                "outputs": str(out.relative_to(ROOT)),
            }
        ]
    )

    manifest.to_csv(out / "01_c14_manifest.csv", index=False)
    grid_definitions_frame.to_csv(out / "02_c14_grid_definition.csv", index=False)
    thresholds.to_csv(out / "03_c14_threshold_selection.csv", index=False)
    metrics.to_csv(out / "04_c14_day_level_metrics.csv", index=False)
    ranking.to_csv(out / "05_c14_grid_ranking.csv", index=False)
    predictions.to_csv(out / "06_c14_daily_prediction_audit.csv", index=False)
    multi_metrics.to_csv(out / "07_c14_top_multi_metric_summary.csv", index=False)
    multi_site_metrics.to_csv(out / "08_c14_top_site_multi_metric_summary.csv", index=False)

    print(f"Wrote C14 small weight-grid outputs to {out.relative_to(ROOT)}")
    print("\nManifest")
    print(manifest.round(3).to_string(index=False))
    print("\nTop day-level grid results")
    show_cols = [
        "regime",
        "variant",
        "weight_F1_bridge_improvement",
        "weight_F2_roughness_improvement",
        "weight_F8_site_centered_core_score",
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
        ].copy()
        pooled = pooled.sort_values(["day_f1", "interval_f1", "energy_f1"], ascending=[False, False, False])
        print("\nTop-grid pooled Beta sure-only day/interval/energy metrics")
        print(
            pooled[
                [
                    "regime",
                    "model_variant",
                    "day_f1",
                    "interval_f1",
                    "energy_f1",
                    "energy_iou",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )
    return out


def run_c15_feature_distribution_summary() -> Path:
    started = time.time()
    out = OUT_ROOT / "C15_feature_distribution_summary"
    out.mkdir(parents=True, exist_ok=True)

    daily = load_c1_daily_feature_cache()
    daily["truth_group"] = np.where(daily["true_day"].astype(bool), "RPF day", "non-RPF day")
    daily["confidence_subset"] = np.where(
        daily["dataset"].eq("beta"),
        np.where(daily["confidence"].eq("sure"), "beta_sure", "beta_unsure"),
        "alpha_not_applicable",
    )

    summary_frames: list[pd.DataFrame] = []
    group_sets = [
        ("dataset_truth", ["dataset", "truth_group"]),
        ("dataset_truth_confidence", ["dataset", "confidence_subset", "truth_group"]),
        ("site_truth", ["dataset", "substation_id", "truth_group"]),
    ]
    for group_name, group_cols in group_sets:
        melted = daily.melt(
            id_vars=group_cols,
            value_vars=FEATURE_COLUMNS,
            var_name="feature",
            value_name="value",
        )
        grouped = melted.groupby([*group_cols, "feature"], dropna=False)["value"]
        summary = grouped.agg(
            count="count",
            missing=lambda x: int(x.isna().sum()),
            min="min",
            p05=lambda x: float(x.quantile(0.05)),
            p25=lambda x: float(x.quantile(0.25)),
            mean="mean",
            median="median",
            p75=lambda x: float(x.quantile(0.75)),
            p95=lambda x: float(x.quantile(0.95)),
            max="max",
            std="std",
        ).reset_index()
        summary.insert(0, "grouping", group_name)
        summary_frames.append(summary)
    distribution_summary = pd.concat(summary_frames, ignore_index=True)

    separation_rows: list[dict[str, object]] = []
    separation_specs = [
        ("alpha_all", daily.loc[daily["dataset"].eq("alpha")].copy()),
        ("beta_all", daily.loc[daily["dataset"].eq("beta")].copy()),
        ("beta_sure_only", daily.loc[daily["dataset"].eq("beta") & daily["confidence"].eq("sure")].copy()),
    ]
    for subset, frame in separation_specs:
        for feature in FEATURE_COLUMNS:
            pos = pd.to_numeric(frame.loc[frame["true_day"].astype(bool), feature], errors="coerce").dropna()
            neg = pd.to_numeric(frame.loc[~frame["true_day"].astype(bool), feature], errors="coerce").dropna()
            pos_median = float(pos.median()) if len(pos) else np.nan
            neg_median = float(neg.median()) if len(neg) else np.nan
            pos_mean = float(pos.mean()) if len(pos) else np.nan
            neg_mean = float(neg.mean()) if len(neg) else np.nan
            pooled_std = float(pd.concat([pos, neg]).std()) if len(pos) + len(neg) > 1 else np.nan
            separation_rows.append(
                {
                    "subset": subset,
                    "feature": feature,
                    "positive_days": int(len(pos)),
                    "negative_days": int(len(neg)),
                    "rpf_median": pos_median,
                    "non_rpf_median": neg_median,
                    "median_difference": pos_median - neg_median,
                    "rpf_mean": pos_mean,
                    "non_rpf_mean": neg_mean,
                    "mean_difference": pos_mean - neg_mean,
                    "pooled_std": pooled_std,
                    "standardised_median_difference": (
                        (pos_median - neg_median) / pooled_std if pooled_std and not np.isnan(pooled_std) else np.nan
                    ),
                }
            )
    separation = pd.DataFrame(separation_rows).sort_values(
        ["subset", "standardised_median_difference"],
        ascending=[True, False],
    )

    feature_correlation_rows: list[dict[str, object]] = []
    for subset, frame in separation_specs:
        corr_frame = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        corr = corr_frame.corr(method="spearman")
        for left in FEATURE_COLUMNS:
            for right in FEATURE_COLUMNS:
                if left >= right:
                    continue
                feature_correlation_rows.append(
                    {
                        "subset": subset,
                        "feature_1": left,
                        "feature_2": right,
                        "spearman_corr": float(corr.loc[left, right]),
                    }
                )
    correlations = pd.DataFrame(feature_correlation_rows).sort_values(
        ["subset", "spearman_corr"],
        ascending=[True, False],
    )

    manifest = pd.DataFrame(
        [
            {
                "chunk": "C15_feature_distribution_summary",
                "source_cache": str((OUT_ROOT / "C1_cached_daily_features/02_c1_daily_feature_cache.csv").relative_to(ROOT)),
                "rows": len(daily),
                "features": len(FEATURE_COLUMNS),
                "alpha_rows": int(daily["dataset"].eq("alpha").sum()),
                "beta_rows": int(daily["dataset"].eq("beta").sum()),
                "beta_sure_rows": int(daily["dataset"].eq("beta").mul(daily["confidence"].eq("sure")).sum()),
                "elapsed_seconds": time.time() - started,
                "outputs": str(out.relative_to(ROOT)),
            }
        ]
    )

    manifest.to_csv(out / "01_c15_manifest.csv", index=False)
    distribution_summary.to_csv(out / "02_c15_feature_distribution_summary.csv", index=False)
    separation.to_csv(out / "03_c15_feature_separation_summary.csv", index=False)
    correlations.to_csv(out / "04_c15_feature_spearman_correlations.csv", index=False)

    print(f"Wrote C15 feature distribution outputs to {out.relative_to(ROOT)}")
    print("\nManifest")
    print(manifest.round(3).to_string(index=False))
    print("\nMedian separation by feature")
    show = separation.loc[separation["subset"].isin(["alpha_all", "beta_all", "beta_sure_only"])].copy()
    show_cols = [
        "subset",
        "feature",
        "rpf_median",
        "non_rpf_median",
        "median_difference",
        "standardised_median_difference",
    ]
    print(show[show_cols].round(4).to_string(index=False))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast physical-score LOSO experiment chunks.")
    parser.add_argument(
        "--chunk",
        default="C0",
        choices=["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C12", "C13", "C14", "C15"],
        help="Experiment chunk to run.",
    )
    args = parser.parse_args()
    if args.chunk == "C0":
        run_c0_smoke()
    elif args.chunk == "C1":
        run_c1_cached_daily_features()
    elif args.chunk == "C2":
        run_c2_manual_ablation_ladder()
    elif args.chunk == "C3":
        run_c3_beta_loso_manual_ablation()
    elif args.chunk == "C4":
        run_c4_compact_regime_comparison()
    elif args.chunk == "C5":
        run_c5_logistic_check()
    elif args.chunk == "C6":
        run_c6_confidence_coverage()
    elif args.chunk == "C7":
        run_c7_confidence_triage_diagnostics()
    elif args.chunk == "C8":
        run_c8_key_result_figures()
    elif args.chunk == "C9":
        run_c9_all_result_visualisations()
    elif args.chunk == "C10":
        run_c10_window_interval_evaluation()
    elif args.chunk == "C12":
        run_c12_multi_model_metric_summary()
    elif args.chunk == "C13":
        run_c13_single_feature_ladder()
    elif args.chunk == "C14":
        run_c14_small_weight_grid()
    elif args.chunk == "C15":
        run_c15_feature_distribution_summary()


if __name__ == "__main__":
    main()

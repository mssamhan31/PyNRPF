from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_NAME = "m9.2_physics"
RANDOM_SEED = 9

WINDOW_START_HOUR = 6
WINDOW_END_HOUR = 18
MIN_DURATION_MINUTES = 30
MAX_DURATION_MINUTES = 8 * 60
TOP_SOLAR_PEAKS = 3
SOLAR_PEAK_MIN_FRAC = 0.25
SOLAR_PEAK_MIN_SEPARATION_MINUTES = 90
MAX_TRAIN_NEGATIVES_PER_DAY = 40
FEATURE_VERSION = "full_loso_lean_v1_counterfactual_2026_06_25"
FAST_FULL_FEATURES = True

EXPECTED_COLUMNS = [
    "substation_id",
    "date",
    "timestamp",
    "net_load_MW",
    "solar_MW",
    "label_interval",
    "label_day",
]

METADATA_COLUMNS = {
    "dataset",
    "substation_id",
    "date",
    "candidate_id",
    "candidate_start",
    "candidate_end",
    "true_start",
    "true_end",
    "true_label_day",
    "iou_with_true",
    "relevance",
    "start_error_minutes",
    "end_error_minutes",
}


def resolve_repo_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "publication" / "2_journal_article" / "dataset" / "final").exists():
            return candidate
        if candidate.name == "2_journal_article" and (candidate / "dataset" / "final").exists():
            return candidate.parent.parent
    raise FileNotFoundError("Could not resolve PyNRPF repo root.")


REPO_ROOT = resolve_repo_root()
ARTICLE_ROOT = REPO_ROOT / "publication" / "2_journal_article"
MISC_DIR = ARTICLE_ROOT / "notebooks" / "99_Misc"
OUTPUT_ROOT = MISC_DIR / "outputs" / "07_m9_2_physics_counterfactual_ranker"
CSV_DIR = OUTPUT_ROOT / "csv"
CACHE_DIR = OUTPUT_ROOT / "cache" / "full_loso"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
for folder in [CSV_DIR, CACHE_DIR, MANIFEST_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


@dataclass
class DayRecord:
    dataset: str
    substation_id: str
    date: str
    ts: np.ndarray
    minute_of_day: np.ndarray
    net: np.ndarray
    solar: np.ndarray
    labels: np.ndarray
    true_start_idx: int | None
    true_end_idx: int | None
    daily_solar_peak_idx: int | None
    top_solar_peak_indices: list[int]

    @property
    def has_rpf(self) -> bool:
        return self.true_start_idx is not None and self.true_end_idx is not None


def naive_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None)


def load_final_dataset(name: str, site: str | None = None) -> pd.DataFrame:
    path = ARTICLE_ROOT / "dataset" / "final" / f"dataset_{name}.parquet"
    df = pd.read_parquet(path, columns=EXPECTED_COLUMNS)
    df["timestamp"] = naive_timestamp(df["timestamp"])
    df["date"] = df["date"].astype(str)
    df["substation_id"] = df["substation_id"].astype(str)
    df["label_interval"] = df["label_interval"].astype(bool)
    df["label_day"] = df["label_day"].astype(bool)
    if site is not None:
        df = df.loc[df["substation_id"].eq(site)].copy()
    return df.sort_values(["substation_id", "date", "timestamp"]).reset_index(drop=True)


def list_sites(name: str) -> list[str]:
    df = pd.read_parquet(ARTICLE_ROOT / "dataset" / "final" / f"dataset_{name}.parquet", columns=["substation_id"])
    return sorted(df["substation_id"].astype(str).unique().tolist())


def finite(value: float, default: float = 0.0) -> float:
    return float(value) if np.isfinite(value) else default


def nan_stat(values: np.ndarray, fn, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return default
    return finite(fn(values), default)


def roughness(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    return finite(np.nansum(np.abs(np.diff(values))))


def curvature(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return 0.0
    return finite(np.nansum(np.abs(np.diff(values, n=2))))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.nanstd(a[mask]) == 0 or np.nanstd(b[mask]) == 0:
        return 0.0
    return finite(np.corrcoef(a[mask], b[mask])[0, 1])


def shape_score(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < 4:
        return 0.0
    y = values[mask]
    span = np.nanmax(y) - np.nanmin(y)
    if not np.isfinite(span) or span <= 1e-9:
        return 0.0
    y_norm = (y - np.nanmin(y)) / span
    x = np.linspace(-1.0, 1.0, len(y_norm))
    bell = 1.0 - x**2
    bell = (bell - bell.min()) / (bell.max() - bell.min())
    return safe_corr(y_norm, bell)


def prefix(values: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.asarray(values, dtype=float))]


def prefix_sum(pref: np.ndarray, start: int, end_exclusive: int) -> float:
    start = max(0, int(start))
    end_exclusive = max(start, min(int(end_exclusive), len(pref) - 1))
    return finite(pref[end_exclusive] - pref[start])


def select_top_solar_peaks(ts: np.ndarray, solar: np.ndarray, minute_of_day: np.ndarray) -> tuple[int | None, list[int]]:
    daytime = np.flatnonzero(
        (minute_of_day >= WINDOW_START_HOUR * 60) & (minute_of_day <= WINDOW_END_HOUR * 60)
    )
    if len(daytime) == 0:
        return None, []
    vals = solar[daytime]
    if np.all(np.isnan(vals)):
        return None, []
    daily_peak_idx = int(daytime[int(np.nanargmax(vals))])
    daily_peak_value = float(solar[daily_peak_idx])
    if not np.isfinite(daily_peak_value) or daily_peak_value <= 0:
        return daily_peak_idx, [daily_peak_idx]

    threshold = daily_peak_value * SOLAR_PEAK_MIN_FRAC
    local_candidates: list[int] = []
    for idx in daytime:
        val = solar[idx]
        if not np.isfinite(val) or val < threshold:
            continue
        left = solar[idx - 1] if idx > 0 else -np.inf
        right = solar[idx + 1] if idx + 1 < len(solar) else -np.inf
        if val >= left and val >= right:
            local_candidates.append(int(idx))
    if daily_peak_idx not in local_candidates:
        local_candidates.append(daily_peak_idx)

    local_candidates = sorted(set(local_candidates), key=lambda i: (-float(solar[i]), i))
    selected = [daily_peak_idx]
    min_sep_steps = SOLAR_PEAK_MIN_SEPARATION_MINUTES / 15
    for idx in local_candidates:
        if idx == daily_peak_idx:
            continue
        if all(abs(idx - prior) >= min_sep_steps for prior in selected):
            selected.append(idx)
        if len(selected) >= TOP_SOLAR_PEAKS:
            break
    return daily_peak_idx, sorted(set(selected))


def build_day_records(df: pd.DataFrame, dataset: str) -> dict[tuple[str, str], DayRecord]:
    records: dict[tuple[str, str], DayRecord] = {}
    for (site, date), group in df.groupby(["substation_id", "date"], sort=True):
        g = group.sort_values("timestamp")
        ts = g["timestamp"].to_numpy()
        stamp = pd.Series(pd.to_datetime(ts))
        minute_of_day = (stamp.dt.hour.to_numpy() * 60 + stamp.dt.minute.to_numpy()).astype(int)
        net = g["net_load_MW"].astype(float).to_numpy()
        solar = g["solar_MW"].astype(float).to_numpy()
        labels = g["label_interval"].astype(bool).to_numpy()
        label_idx = np.flatnonzero(labels)
        true_start_idx = int(label_idx[0]) if len(label_idx) else None
        true_end_idx = int(label_idx[-1]) if len(label_idx) else None
        daily_peak_idx, top_peaks = select_top_solar_peaks(ts, solar, minute_of_day)
        records[(str(site), str(date))] = DayRecord(
            dataset=dataset,
            substation_id=str(site),
            date=str(date),
            ts=ts,
            minute_of_day=minute_of_day,
            net=net,
            solar=solar,
            labels=labels,
            true_start_idx=true_start_idx,
            true_end_idx=true_end_idx,
            daily_solar_peak_idx=daily_peak_idx,
            top_solar_peak_indices=top_peaks,
        )
    return records


def duration_minutes_from_indices(start_idx: int, end_idx: int) -> float:
    return float((end_idx - start_idx + 1) * 15)


def dense_bounds(rec: DayRecord) -> list[tuple[int, int, int]]:
    daytime = np.flatnonzero(
        (rec.minute_of_day >= WINDOW_START_HOUR * 60) & (rec.minute_of_day <= WINDOW_END_HOUR * 60)
    )
    if len(daytime) == 0 or not rec.top_solar_peak_indices:
        return []
    seen: set[tuple[int, int]] = set()
    rows: list[tuple[int, int, int]] = []
    for peak_idx in rec.top_solar_peak_indices:
        starts = daytime[daytime <= peak_idx]
        ends = daytime[daytime >= peak_idx]
        for s_idx in starts:
            for e_idx in ends:
                duration = duration_minutes_from_indices(int(s_idx), int(e_idx))
                if duration < MIN_DURATION_MINUTES or duration > MAX_DURATION_MINUTES:
                    continue
                key = (int(s_idx), int(e_idx))
                if key in seen:
                    continue
                seen.add(key)
                rows.append((int(s_idx), int(e_idx), int(peak_idx)))
    return rows


def iou_for_bounds(rec: DayRecord, s: int, e: int) -> float:
    if not rec.has_rpf:
        return 0.0
    pred = np.zeros(len(rec.labels), dtype=bool)
    pred[s : e + 1] = True
    union = np.logical_or(pred, rec.labels).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(pred, rec.labels).sum() / union)


def relevance_from_iou(iou: float) -> int:
    if iou >= 0.85:
        return 4
    if iou >= 0.70:
        return 3
    if iou >= 0.50:
        return 2
    if iou >= 0.25:
        return 1
    return 0


def index_timestamp(rec: DayRecord, idx: int | None):
    if idx is None:
        return pd.NaT
    return pd.Timestamp(rec.ts[int(idx)])


def bridge_residual(values: np.ndarray, s: int, e: int, alt_inside: np.ndarray | None = None) -> float:
    left_idx = max(s - 1, 0)
    right_idx = min(e + 1, len(values) - 1)
    left_val = values[left_idx]
    right_val = values[right_idx]
    seg = values[s : e + 1] if alt_inside is None else alt_inside
    if len(seg) == 0 or not np.isfinite(left_val) or not np.isfinite(right_val):
        return 0.0
    line = np.linspace(left_val, right_val, len(seg))
    return finite(np.nanmean(np.abs(seg - line)))


def boundary_jump(values: np.ndarray, s: int, e: int, alt_inside: np.ndarray | None = None) -> float:
    start_val = values[s] if alt_inside is None else alt_inside[0]
    end_val = values[e] if alt_inside is None else alt_inside[-1]
    jump = 0.0
    if s > 0 and np.isfinite(start_val) and np.isfinite(values[s - 1]):
        jump += abs(start_val - values[s - 1])
    if e + 1 < len(values) and np.isfinite(end_val) and np.isfinite(values[e + 1]):
        jump += abs(values[e + 1] - end_val)
    return finite(jump)


def roughness_delta_for_window(u_empty: np.ndarray, alt: np.ndarray, s: int, e: int, pref_empty: np.ndarray, pref_alt: np.ndarray) -> float:
    original = prefix_sum(pref_empty, s, e)
    new = prefix_sum(pref_alt, s, e)
    if s > 0:
        original += abs(u_empty[s] - u_empty[s - 1])
        new += abs(alt[s] - u_empty[s - 1])
    if e + 1 < len(u_empty):
        original += abs(u_empty[e + 1] - u_empty[e])
        new += abs(u_empty[e + 1] - alt[e])
    return finite(original - new)


def build_features_for_record(rec: DayRecord) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    date_ts = pd.Timestamp(rec.date)
    day_solar_peak = nan_stat(rec.solar, np.nanmax)
    day_solar_p95 = nan_stat(rec.solar, lambda x: np.nanpercentile(x, 95))
    day_net_p05 = nan_stat(rec.net, lambda x: np.nanpercentile(x, 5))
    day_net_p95 = nan_stat(rec.net, lambda x: np.nanpercentile(x, 95))
    base_meta = {
        "dataset": rec.dataset,
        "substation_id": rec.substation_id,
        "date": rec.date,
        "true_start": index_timestamp(rec, rec.true_start_idx),
        "true_end": index_timestamp(rec, rec.true_end_idx),
        "true_label_day": rec.has_rpf,
        "n_rows": len(rec.ts),
        "day_solar_peak": day_solar_peak,
        "day_solar_p95": day_solar_p95,
        "day_net_p05": day_net_p05,
        "day_net_p95": day_net_p95,
        "day_missing_net": int(np.isnan(rec.net).sum()),
        "day_missing_solar": int(np.isnan(rec.solar).sum()),
    }

    def null_row() -> dict[str, Any]:
        return {
            **base_meta,
            "candidate_id": 0,
            "is_null": True,
            "candidate_start": pd.NaT,
            "candidate_end": pd.NaT,
            "duration_minutes": 0.0,
            "start_hour": 0.0,
            "end_hour": 0.0,
            "mid_hour": 0.0,
            "month": int(date_ts.month),
            "weekday": int(date_ts.weekday()),
            "is_weekend": int(date_ts.weekday() >= 5),
            "contains_daily_solar_peak": 0,
            "distance_to_daily_solar_peak_minutes": 999.0,
            "iou_with_true": 0.0,
            "relevance": 4 if not rec.has_rpf else 0,
            "start_error_minutes": np.nan,
            "end_error_minutes": np.nan,
            "candidate_solar_peak": 0.0,
            "candidate_net_peak": 0.0,
            "candidate_net_p05": 0.0,
            "candidate_net_p95": 0.0,
            "candidate_solar_p95": 0.0,
            "candidate_pseudoload_std": 0.0,
            "candidate_pseudoload_range": 0.0,
            "candidate_pseudoload_roughness": 0.0,
            "counterfactual_roughness_delta": 0.0,
            "counterfactual_curvature_delta": 0.0,
            "bridge_residual_delta": 0.0,
            "boundary_jump_delta": 0.0,
            "solar_net_corr": 0.0,
            "derivative_same_sign_fraction": 0.0,
            "derivative_product_mean": 0.0,
            "ramp_up_same_sign": 0.0,
            "ramp_down_same_sign": 0.0,
            "solar_bell_score": 0.0,
            "net_n_shape_score": 0.0,
            "negative_reconstructed_fraction": 0.0,
        }

    rows.append(null_row())
    u_empty = rec.solar + rec.net
    alt_full = rec.solar - rec.net
    pref_empty = prefix(np.abs(np.diff(u_empty)))
    pref_alt = prefix(np.abs(np.diff(alt_full)))

    cid = 1
    for s, e, peak_idx in dense_bounds(rec):
        solar_seg = rec.solar[s : e + 1]
        net_seg = rec.net[s : e + 1]
        alt_seg = alt_full[s : e + 1]
        pseudo = alt_seg
        dsolar = np.diff(solar_seg)
        dnet = np.diff(net_seg)
        valid_deriv = np.isfinite(dsolar) & np.isfinite(dnet)
        same_fraction = derivative_product = ramp_up = ramp_down = 0.0
        if valid_deriv.any():
            same = np.sign(dsolar[valid_deriv]) == np.sign(dnet[valid_deriv])
            same_fraction = float(np.mean(same))
            derivative_product = finite(np.nanmean(dsolar[valid_deriv] * dnet[valid_deriv]))
            ramp_up_mask = dsolar[valid_deriv] > 0
            ramp_down_mask = dsolar[valid_deriv] < 0
            ramp_up = float(np.mean(same[ramp_up_mask])) if ramp_up_mask.any() else 0.0
            ramp_down = float(np.mean(same[ramp_down_mask])) if ramp_down_mask.any() else 0.0

        iou = iou_for_bounds(rec, s, e)
        relevance = relevance_from_iou(iou) if rec.has_rpf else 0
        start_error = abs((s - rec.true_start_idx) * 15.0) if rec.has_rpf else np.nan
        end_error = abs((e - rec.true_end_idx) * 15.0) if rec.has_rpf else np.nan
        peak_distance = 999.0
        if rec.daily_solar_peak_idx is not None:
            peak_distance = 0.0 if s <= rec.daily_solar_peak_idx <= e else min(abs(s - rec.daily_solar_peak_idx), abs(e - rec.daily_solar_peak_idx)) * 15.0

        if FAST_FULL_FEATURES:
            candidate_net_p05 = nan_stat(net_seg, np.nanmin)
            candidate_net_p95 = nan_stat(net_seg, np.nanmax)
            candidate_solar_p95 = nan_stat(solar_seg, np.nanmax)
            counterfactual_curvature_delta = 0.0
            solar_net_corr = 0.0
            solar_bell_score = 0.0
            net_n_shape_score = 0.0
        else:
            candidate_net_p05 = nan_stat(net_seg, lambda x: np.nanpercentile(x, 5))
            candidate_net_p95 = nan_stat(net_seg, lambda x: np.nanpercentile(x, 95))
            candidate_solar_p95 = nan_stat(solar_seg, lambda x: np.nanpercentile(x, 95))
            counterfactual_curvature_delta = curvature(u_empty[s : e + 1]) - curvature(alt_seg)
            solar_net_corr = safe_corr(solar_seg, net_seg)
            solar_bell_score = shape_score(solar_seg)
            net_n_shape_score = shape_score(net_seg)

        rows.append({
            **base_meta,
            "candidate_id": cid,
            "is_null": False,
            "candidate_start": rec.ts[s],
            "candidate_end": rec.ts[e],
            "duration_minutes": duration_minutes_from_indices(s, e),
            "start_hour": rec.minute_of_day[s] / 60.0,
            "end_hour": rec.minute_of_day[e] / 60.0,
            "mid_hour": (rec.minute_of_day[s] + rec.minute_of_day[e]) / 120.0,
            "month": int(date_ts.month),
            "weekday": int(date_ts.weekday()),
            "is_weekend": int(date_ts.weekday() >= 5),
            "contains_daily_solar_peak": int(rec.daily_solar_peak_idx is not None and s <= rec.daily_solar_peak_idx <= e),
            "distance_to_daily_solar_peak_minutes": peak_distance,
            "iou_with_true": iou,
            "relevance": relevance,
            "start_error_minutes": start_error,
            "end_error_minutes": end_error,
            "candidate_solar_peak": nan_stat(solar_seg, np.nanmax),
            "candidate_net_peak": nan_stat(net_seg, np.nanmax),
            "candidate_net_p05": candidate_net_p05,
            "candidate_net_p95": candidate_net_p95,
            "candidate_solar_p95": candidate_solar_p95,
            "candidate_pseudoload_std": nan_stat(pseudo, np.nanstd),
            "candidate_pseudoload_range": nan_stat(pseudo, lambda x: np.nanmax(x) - np.nanmin(x)),
            "candidate_pseudoload_roughness": roughness(pseudo),
            "counterfactual_roughness_delta": roughness_delta_for_window(u_empty, alt_full, s, e, pref_empty, pref_alt),
            "counterfactual_curvature_delta": counterfactual_curvature_delta,
            "bridge_residual_delta": bridge_residual(u_empty, s, e) - bridge_residual(u_empty, s, e, alt_seg),
            "boundary_jump_delta": boundary_jump(u_empty, s, e) - boundary_jump(u_empty, s, e, alt_seg),
            "solar_net_corr": solar_net_corr,
            "derivative_same_sign_fraction": finite(same_fraction),
            "derivative_product_mean": finite(derivative_product),
            "ramp_up_same_sign": finite(ramp_up),
            "ramp_down_same_sign": finite(ramp_down),
            "solar_bell_score": solar_bell_score,
            "net_n_shape_score": net_n_shape_score,
            "negative_reconstructed_fraction": float(np.mean(alt_seg < 0)) if len(alt_seg) else 0.0,
        })
        cid += 1
    return rows


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [
        col for col in features.columns
        if col not in METADATA_COLUMNS and (pd.api.types.is_numeric_dtype(features[col]) or pd.api.types.is_bool_dtype(features[col]))
    ]


def feature_path(dataset: str, site: str) -> Path:
    return CACHE_DIR / f"{dataset}_{site}_features.parquet"


def sample_path(dataset: str, site: str) -> Path:
    return CACHE_DIR / f"{dataset}_{site}_train_sample.parquet"


def sample_training_rows(features: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in features.groupby(["substation_id", "date"], sort=False):
        keep = group.loc[group["is_null"].astype(bool) | (group["relevance"] > 0)].copy()
        neg = group.loc[(~group["is_null"].astype(bool)) & (group["relevance"] == 0)].copy()
        if len(neg):
            neg = neg.assign(
                hard_score=(
                    neg["counterfactual_roughness_delta"].rank(method="first", ascending=False)
                    + neg["candidate_solar_peak"].rank(method="first", ascending=False)
                    + neg["derivative_same_sign_fraction"].rank(method="first", ascending=False)
                )
            ).sort_values("hard_score")
            keep = pd.concat([keep, neg.head(MAX_TRAIN_NEGATIVES_PER_DAY).drop(columns="hard_score")], ignore_index=True)
        parts.append(keep)
    return pd.concat(parts, ignore_index=True) if parts else features.head(0).copy()


def build_site_features(dataset: str, site: str, force: bool = False) -> tuple[Path, Path]:
    fpath = feature_path(dataset, site)
    spath = sample_path(dataset, site)
    if fpath.exists() and spath.exists() and not force:
        print(f"[cache] {dataset} {site}: {fpath.name}")
        return fpath, spath
    t0 = time.time()
    df = load_final_dataset(dataset, site)
    records = build_day_records(df, dataset)
    rows: list[dict[str, Any]] = []
    for rec in records.values():
        rows.extend(build_features_for_record(rec))
    features = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    numeric = features.select_dtypes(include=[np.number]).columns
    features[numeric] = features[numeric].fillna(0.0)
    features.to_parquet(fpath, index=False)
    sample = sample_training_rows(features)
    sample.to_parquet(spath, index=False)
    print(
        f"[built] {dataset} {site}: {len(features):,} rows, {len(sample):,} train-sample rows "
        f"in {time.time() - t0:.1f}s"
    )
    return fpath, spath


def load_many(paths: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def fit_ranker(train: pd.DataFrame, feature_cols: list[str]):
    train = train.sort_values(["substation_id", "date", "candidate_id"]).reset_index(drop=True)
    y = train["relevance"].astype(float)
    group = train.groupby(["substation_id", "date"], sort=False).size().to_numpy()
    try:
        model = xgb.XGBRanker(
            objective="rank:pairwise",
            eval_metric="ndcg",
            n_estimators=90,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.90,
            colsample_bytree=0.90,
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=4,
        )
        model.fit(train[feature_cols].astype(float), y, group=group, verbose=False)
        return model, "xgb_ranker"
    except Exception as exc:
        print(f"[warn] XGBRanker failed; using classifier fallback: {exc}")
        y_bin = (train["relevance"] >= 3).astype(int)
        weights = np.where(train["relevance"] >= 3, 8.0, 1.0 + train["relevance"].astype(float))
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=90,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.90,
            colsample_bytree=0.90,
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=4,
        )
        model.fit(train[feature_cols].astype(float), y_bin, sample_weight=weights, verbose=False)
        return model, "xgb_classifier_fallback"


def score_features(model, model_kind: str, features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = features[[
        "dataset", "substation_id", "date", "candidate_id", "is_null", "candidate_start", "candidate_end",
        "true_label_day", "iou_with_true", "relevance"
    ]].copy()
    xmat = features[feature_cols].astype(float)
    if model_kind == "xgb_classifier_fallback":
        out["score"] = model.predict_proba(xmat)[:, 1]
    else:
        out["score"] = model.predict(xmat)
    return out


def raw_day_margins(scored: pd.DataFrame, fold: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (site, date), group in scored.groupby(["substation_id", "date"], sort=True):
        null = group.loc[group["is_null"].astype(bool)]
        non = group.loc[~group["is_null"].astype(bool)]
        if len(null) != 1:
            raise AssertionError(f"{site} {date} expected exactly one null candidate, got {len(null)}")
        null_row = null.iloc[0]
        if non.empty:
            rows.append({
                "fold": fold,
                "substation_id": site,
                "date": date,
                "true_label_day": bool(null_row["true_label_day"]),
                "best_candidate_id": np.nan,
                "pred_start": pd.NaT,
                "pred_end": pd.NaT,
                "best_score": np.nan,
                "null_score": float(null_row["score"]),
                "score_margin": -np.inf,
                "selected_iou": 0.0,
                "selected_relevance": 0,
            })
            continue
        best = non.loc[non["score"].idxmax()]
        rows.append({
            "fold": fold,
            "substation_id": site,
            "date": date,
            "true_label_day": bool(best["true_label_day"]),
            "best_candidate_id": int(best["candidate_id"]),
            "pred_start": best["candidate_start"],
            "pred_end": best["candidate_end"],
            "best_score": float(best["score"]),
            "null_score": float(null_row["score"]),
            "score_margin": float(best["score"] - null_row["score"]),
            "selected_iou": float(best["iou_with_true"]),
            "selected_relevance": int(best["relevance"]),
        })
    return pd.DataFrame(rows)


def apply_threshold(raw: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = raw.copy()
    out["pred_label_day"] = out["score_margin"] >= threshold
    out.loc[~out["pred_label_day"], ["pred_start", "pred_end"]] = pd.NaT
    return out


def binary_metrics(y_true, y_pred) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.logical_and(y_true, y_pred).sum())
    fp = int(np.logical_and(~y_true, y_pred).sum())
    fn = int(np.logical_and(y_true, ~y_pred).sum())
    tn = int(np.logical_and(~y_true, ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "support": int(len(y_true)),
        "positive_support": int(y_true.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def day_metrics(decoded: pd.DataFrame) -> dict[str, Any]:
    return binary_metrics(decoded["true_label_day"], decoded["pred_label_day"])


def interval_metrics(decoded: pd.DataFrame, dataset: str, site: str | None = None) -> dict[str, Any]:
    df = load_final_dataset(dataset, site)
    merged = df.merge(decoded[["substation_id", "date", "pred_label_day", "pred_start", "pred_end"]], on=["substation_id", "date"], how="left")
    starts = pd.to_datetime(merged["pred_start"])
    ends = pd.to_datetime(merged["pred_end"])
    pred = merged["pred_label_day"].fillna(False).astype(bool) & (merged["timestamp"] >= starts) & (merged["timestamp"] <= ends)
    return binary_metrics(merged["label_interval"], pred)


def select_margin_from_alpha(raw_alpha: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    finite_margins = raw_alpha["score_margin"].replace([np.inf, -np.inf], np.nan).dropna()
    if finite_margins.empty:
        thresholds = np.array([0.0])
    else:
        thresholds = np.unique(np.r_[np.linspace(np.nanpercentile(finite_margins, 1), np.nanpercentile(finite_margins, 99), 121), 0.0])
    rows = []
    for threshold in thresholds:
        met = day_metrics(apply_threshold(raw_alpha, float(threshold)))
        met["margin_threshold"] = float(threshold)
        rows.append(met)
    sweep = pd.DataFrame(rows).sort_values(["f1", "precision", "recall"], ascending=[False, False, False]).reset_index(drop=True)
    return float(sweep.iloc[0]["margin_threshold"]), sweep


def ensure_all_features(force: bool = False):
    for dataset in ["alpha", "beta"]:
        for site in list_sites(dataset):
            build_site_features(dataset, site, force=force)


def run_loso(force_features: bool = False):
    t0 = time.time()
    alpha_sites = list_sites("alpha")
    beta_sites = list_sites("beta")
    print(f"Alpha sites: {alpha_sites}")
    print(f"Beta sites: {beta_sites}")
    ensure_all_features(force=force_features)

    feature_cols = feature_columns(pd.read_parquet(sample_path("alpha", alpha_sites[0])))
    alpha_raw_parts: list[pd.DataFrame] = []
    fold_meta: list[dict[str, Any]] = []

    for heldout in alpha_sites:
        fold_t0 = time.time()
        train_paths = [sample_path("alpha", site) for site in alpha_sites if site != heldout]
        train = load_many(train_paths)
        model, model_kind = fit_ranker(train, feature_cols)
        heldout_features = pd.read_parquet(feature_path("alpha", heldout))
        scored = score_features(model, model_kind, heldout_features, feature_cols)
        raw = raw_day_margins(scored, fold=f"loso_{heldout}")
        alpha_raw_parts.append(raw)
        fold_meta.append({
            "heldout_site": heldout,
            "model_kind": model_kind,
            "train_rows": int(len(train)),
            "heldout_candidate_rows": int(len(heldout_features)),
            "elapsed_seconds": time.time() - fold_t0,
        })
        print(f"[loso] {heldout}: train {len(train):,}, score {len(heldout_features):,}, {time.time() - fold_t0:.1f}s")

    alpha_raw = pd.concat(alpha_raw_parts, ignore_index=True)
    alpha_raw.to_csv(CSV_DIR / "full_loso_01_alpha_raw_margins.csv", index=False)
    selected_threshold, sweep = select_margin_from_alpha(alpha_raw)
    sweep.to_csv(CSV_DIR / "full_loso_02_alpha_margin_sweep.csv", index=False)
    alpha_decoded = apply_threshold(alpha_raw, selected_threshold)
    alpha_decoded.to_csv(CSV_DIR / "full_loso_03_alpha_decoded_days.csv", index=False)

    alpha_site_rows = []
    for site, group in alpha_decoded.groupby("substation_id", sort=True):
        dm = day_metrics(group)
        dm.update({"dataset": "alpha_loso", "substation_id": site, "level": "day", "model": MODEL_NAME})
        im = interval_metrics(group, "alpha", site)
        im.update({"dataset": "alpha_loso", "substation_id": site, "level": "interval", "model": MODEL_NAME})
        alpha_site_rows.extend([dm, im])
    alpha_site_metrics = pd.DataFrame(alpha_site_rows)
    alpha_site_metrics.to_csv(CSV_DIR / "full_loso_04_alpha_site_metrics.csv", index=False)
    alpha_overall_rows = []
    dm = day_metrics(alpha_decoded)
    dm.update({"dataset": "alpha_loso", "level": "day", "model": MODEL_NAME})
    im = interval_metrics(alpha_decoded, "alpha", None)
    im.update({"dataset": "alpha_loso", "level": "interval", "model": MODEL_NAME})
    alpha_overall_rows.extend([dm, im])
    alpha_overall = pd.DataFrame(alpha_overall_rows)
    alpha_overall.to_csv(CSV_DIR / "full_loso_05_alpha_overall_metrics.csv", index=False)

    print(f"[threshold] selected Alpha LOSO margin: {selected_threshold:.6f}")
    print("[alpha overall]")
    print(alpha_overall[["level", "support", "positive_support", "precision", "recall", "f1"]].round(4).to_string(index=False))

    final_train = load_many([sample_path("alpha", site) for site in alpha_sites])
    final_model, final_model_kind = fit_ranker(final_train, feature_cols)
    beta_raw_parts: list[pd.DataFrame] = []
    for site in beta_sites:
        site_t0 = time.time()
        features = pd.read_parquet(feature_path("beta", site))
        scored = score_features(final_model, final_model_kind, features, feature_cols)
        raw = raw_day_margins(scored, fold="alpha_all_to_beta")
        beta_raw_parts.append(raw)
        print(f"[beta] {site}: score {len(features):,}, {time.time() - site_t0:.1f}s")
    beta_raw = pd.concat(beta_raw_parts, ignore_index=True)
    beta_raw.to_csv(CSV_DIR / "full_loso_06_beta_raw_margins.csv", index=False)
    beta_decoded = apply_threshold(beta_raw, selected_threshold)
    beta_decoded.to_csv(CSV_DIR / "full_loso_07_beta_decoded_days.csv", index=False)

    beta_site_rows = []
    for site, group in beta_decoded.groupby("substation_id", sort=True):
        dm = day_metrics(group)
        dm.update({"dataset": "beta_transfer", "substation_id": site, "level": "day", "model": MODEL_NAME})
        im = interval_metrics(group, "beta", site)
        im.update({"dataset": "beta_transfer", "substation_id": site, "level": "interval", "model": MODEL_NAME})
        beta_site_rows.extend([dm, im])
    beta_site_metrics = pd.DataFrame(beta_site_rows)
    beta_site_metrics.to_csv(CSV_DIR / "full_loso_08_beta_site_metrics.csv", index=False)
    beta_overall_rows = []
    dm = day_metrics(beta_decoded)
    dm.update({"dataset": "beta_transfer", "level": "day", "model": MODEL_NAME})
    im = interval_metrics(beta_decoded, "beta", None)
    im.update({"dataset": "beta_transfer", "level": "interval", "model": MODEL_NAME})
    beta_overall_rows.extend([dm, im])
    beta_overall = pd.DataFrame(beta_overall_rows)
    beta_overall.to_csv(CSV_DIR / "full_loso_09_beta_overall_metrics.csv", index=False)

    print("[beta overall]")
    print(beta_overall[["level", "support", "positive_support", "precision", "recall", "f1"]].round(4).to_string(index=False))

    manifest = {
        "model_name": MODEL_NAME,
        "run_type": "full_alpha_loso_beta_transfer",
        "feature_version": FEATURE_VERSION,
        "publication_ready": False,
        "exploratory_warning": "m9.2_physics remains exploratory; Alpha threshold selected from Alpha LOSO raw margins, Beta labels used only for evaluation.",
        "selected_alpha_margin_threshold": selected_threshold,
        "alpha_sites": alpha_sites,
        "beta_sites": beta_sites,
        "folds": fold_meta,
        "final_model_kind": final_model_kind,
        "final_train_rows": int(len(final_train)),
        "elapsed_seconds": time.time() - t0,
        "outputs": {
            "alpha_site_metrics": str(CSV_DIR / "full_loso_04_alpha_site_metrics.csv"),
            "alpha_overall_metrics": str(CSV_DIR / "full_loso_05_alpha_overall_metrics.csv"),
            "beta_site_metrics": str(CSV_DIR / "full_loso_08_beta_site_metrics.csv"),
            "beta_overall_metrics": str(CSV_DIR / "full_loso_09_beta_overall_metrics.csv"),
        },
    }
    manifest_path = MANIFEST_DIR / "full_loso_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[done] full LOSO run in {(time.time() - t0) / 60:.1f} min")
    print(f"[manifest] {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Run full m9.2_physics Alpha LOSO and Beta transfer.")
    parser.add_argument("--build-features-only", action="store_true", help="Only build per-site full feature caches.")
    parser.add_argument("--dataset", choices=["alpha", "beta"], help="Dataset for feature-only build.")
    parser.add_argument("--site", help="Site for feature-only build.")
    parser.add_argument("--force-features", action="store_true", help="Rebuild feature caches even if present.")
    args = parser.parse_args()

    if args.build_features_only:
        if args.dataset and args.site:
            build_site_features(args.dataset, args.site, force=args.force_features)
        else:
            ensure_all_features(force=args.force_features)
        return
    run_loso(force_features=args.force_features)


if __name__ == "__main__":
    main()

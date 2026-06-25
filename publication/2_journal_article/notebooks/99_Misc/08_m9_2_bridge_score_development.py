from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_NAME = "m9.2_bridge_score"

ARTICLE_ROOT = Path(__file__).resolve().parents[2]
MISC_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = MISC_DIR / "outputs" / "08_m9_2_bridge_score_development"
CSV_DIR = OUTPUT_ROOT / "csv"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
for folder in [CSV_DIR, MANIFEST_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

EXPECTED_COLUMNS = [
    "substation_id",
    "date",
    "timestamp",
    "net_load_MW",
    "solar_MW",
    "label_interval",
    "label_day",
]

# CGPT Pro strict-pre-Oct defaults from targeted_trainonly_fast.csv.
STRICT_VARIANT = {
    "variant": "v2_alpha_strict",
    "feature": "bridge_ratio_p99",
    "site_median_weight": 0.425,
    "site_rank_weight": 0.0,
    "season_weight": 0.025,
    "rolling_window": 7,
    "rolling_weight": 0.25,
    "threshold": 0.51892,
    "threshold_source": "cgpt_pro_alpha_pre_oct_strict",
}

# CGPT Pro best development row from targeted_small8.csv.
DEV_VARIANT = {
    "variant": "v2_dev_best",
    "feature": "bridge_ratio_p99",
    "site_median_weight": 0.425,
    "site_rank_weight": 0.075,
    "season_weight": 0.050,
    "rolling_window": 5,
    "rolling_weight": 0.10,
    "threshold": 0.55859,
    "threshold_source": "cgpt_pro_beta_guided_development",
}

VARIANTS = [STRICT_VARIANT, DEV_VARIANT]
SLOTS_PER_DAY = 96
DAYTIME_START = 24  # 06:00
DAYTIME_END = 72  # 18:00, inclusive candidate right edge can reach 18:00


def naive_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None)


def smooth(x: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return x.copy()
    return np.convolve(x, np.ones(window) / window, mode="same")


def fill_series(values: np.ndarray, default: float = 0.0) -> np.ndarray:
    s = pd.Series(values, dtype="float64")
    s = s.interpolate(limit_direction="both")
    return s.fillna(default).to_numpy(dtype=float)


def metric_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
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


def load_final_dataset(name: str) -> pd.DataFrame:
    path = ARTICLE_ROOT / "dataset" / "final" / f"dataset_{name}.parquet"
    df = pd.read_parquet(path, columns=EXPECTED_COLUMNS)
    df["timestamp"] = naive_timestamp(df["timestamp"])
    df["date"] = df["date"].astype(str)
    df["substation_id"] = df["substation_id"].astype(str)
    df["label_interval"] = df["label_interval"].astype(bool)
    df["label_day"] = df["label_day"].astype(bool)
    return df.sort_values(["substation_id", "date", "timestamp"]).reset_index(drop=True)


def build_daily_arrays(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    keys = []
    net_rows = []
    solar_rows = []
    label_rows = []
    observed_rows = []
    missing_net_rows = []
    missing_solar_rows = []

    for (site, date), group in df.groupby(["substation_id", "date"], sort=True):
        net = np.full(SLOTS_PER_DAY, np.nan, dtype=float)
        solar = np.full(SLOTS_PER_DAY, np.nan, dtype=float)
        labels = np.zeros(SLOTS_PER_DAY, dtype=bool)
        observed = np.zeros(SLOTS_PER_DAY, dtype=bool)

        stamps = pd.to_datetime(group["timestamp"])
        slots = (stamps.dt.hour.to_numpy() * 4 + (stamps.dt.minute.to_numpy() // 15)).astype(int)
        valid = (slots >= 0) & (slots < SLOTS_PER_DAY)
        net[slots[valid]] = group["net_load_MW"].astype(float).to_numpy()[valid]
        solar[slots[valid]] = group["solar_MW"].astype(float).to_numpy()[valid]
        labels[slots[valid]] = group["label_interval"].astype(bool).to_numpy()[valid]
        observed[slots[valid]] = True

        keys.append(
            {
                "substation_id": site,
                "date": date,
                "label_day": bool(labels.any()),
                "n_observed": int(valid.sum()),
                "missing_net_count": int(np.isnan(net).sum()),
                "missing_solar_count": int(np.isnan(solar).sum()),
            }
        )
        missing_net_rows.append(np.isnan(net))
        missing_solar_rows.append(np.isnan(solar))
        net_rows.append(fill_series(net))
        solar_rows.append(np.maximum(fill_series(solar), 0.0))
        label_rows.append(labels)
        observed_rows.append(observed)

    key_df = pd.DataFrame(keys)
    arrays = {
        "net": np.vstack(net_rows).astype(np.float32),
        "solar": np.vstack(solar_rows).astype(np.float32),
        "labels": np.vstack(label_rows).astype(bool),
        "observed": np.vstack(observed_rows).astype(bool),
        "missing_net": np.vstack(missing_net_rows).astype(bool),
        "missing_solar": np.vstack(missing_solar_rows).astype(bool),
    }
    return key_df, arrays


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


def seg_stats(x: np.ndarray, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cs = np.r_[0.0, np.cumsum(x)]
    cs2 = np.r_[0.0, np.cumsum(x * x)]
    count = (right - left + 1).astype(float)
    total = cs[right + 1] - cs[left]
    mean = total / count
    var = np.maximum((cs2[right + 1] - cs2[left]) / count - mean * mean, 0)
    dx = np.abs(np.diff(x, prepend=x[0]))
    ctv = np.r_[0.0, np.cumsum(dx)]
    tv = ctv[right + 1] - ctv[left + 1]
    return mean, np.sqrt(var), tv


def seg_corr(x: np.ndarray, y: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    count = (right - left + 1).astype(float)
    csx = np.r_[0.0, np.cumsum(x)]
    csy = np.r_[0.0, np.cumsum(y)]
    csx2 = np.r_[0.0, np.cumsum(x * x)]
    csy2 = np.r_[0.0, np.cumsum(y * y)]
    csxy = np.r_[0.0, np.cumsum(x * y)]
    mx = (csx[right + 1] - csx[left]) / count
    my = (csy[right + 1] - csy[left]) / count
    cov = (csxy[right + 1] - csxy[left]) / count - mx * my
    vx = np.maximum((csx2[right + 1] - csx2[left]) / count - mx * mx, 0)
    vy = np.maximum((csy2[right + 1] - csy2[left]) / count - my * my, 0)
    corr = cov / np.sqrt(vx * vy + 1e-10)
    corr[count < 2] = 0
    return np.clip(corr, -1, 1)


def bridge_mse(x: np.ndarray, left: np.ndarray, right: np.ndarray, up: np.ndarray) -> np.ndarray:
    t = np.arange(SLOTS_PER_DAY, dtype=float)
    pre = np.maximum(left - 1, 0)
    post = np.minimum(right + 1, SLOTS_PER_DAY - 1)
    b = (up[post] - up[pre]) / np.maximum(post - pre, 1)
    a = up[pre] - b * pre
    count = (right - left + 1).astype(float)
    cs = np.r_[0.0, np.cumsum(x)]
    cs2 = np.r_[0.0, np.cumsum(x * x)]
    cstx = np.r_[0.0, np.cumsum(t * x)]
    sx = cs[right + 1] - cs[left]
    sx2 = cs2[right + 1] - cs2[left]
    stx = cstx[right + 1] - cstx[left]
    st = (left + right) * count / 2

    def sumsq(k):
        return k * (k + 1) * (2 * k + 1) / 6

    st2 = np.array([sumsq(int(r)) - sumsq(int(l - 1)) for l, r in zip(left, right)], dtype=float)
    sse = sx2 - 2 * a * sx - 2 * b * stx + count * a * a + 2 * a * b * st + b * b * st2
    return np.maximum(sse / count, 0)


def true_bounds(labels: np.ndarray) -> tuple[int | None, int | None]:
    idx = np.flatnonzero(labels)
    if len(idx) == 0:
        return None, None
    return int(idx[0]), int(idx[-1])


def candidate_recall_stats(labels: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, float | bool]:
    start, end = true_bounds(labels)
    if start is None or end is None or len(left) == 0:
        return {"best_iou": 0.0, "has_iou50": False, "has_iou70": False, "boundary30": False}
    inter = np.maximum(0, np.minimum(right, end) - np.maximum(left, start) + 1)
    union = (right - left + 1) + (end - start + 1) - inter
    iou = inter / np.maximum(union, 1)
    best_iou = float(iou.max()) if len(iou) else 0.0
    boundary30 = bool(np.any((np.abs(left - start) <= 2) & (np.abs(right - end) <= 2)))
    return {
        "best_iou": best_iou,
        "has_iou50": bool(np.any(iou >= 0.50)),
        "has_iou70": bool(np.any(iou >= 0.70)),
        "boundary30": boundary30,
    }


def scan_day(net_raw: np.ndarray, solar_raw: np.ndarray, labels: np.ndarray) -> dict[str, float | int | bool]:
    net = smooth(np.nan_to_num(net_raw, nan=0.0), 3)
    solar = smooth(np.maximum(np.nan_to_num(solar_raw, nan=0.0), 0), 3)
    peak = int(np.argmax(solar[DAYTIME_START : DAYTIME_END + 1])) + DAYTIME_START
    left, right = CANDIDATE_CACHE[peak]
    duration = (right - left + 1).astype(float)
    mid = (left + right) / 2

    net_mean, net_std, net_tv = seg_stats(net, left, right)
    solar_mean, solar_std, solar_tv = seg_stats(solar, left, right)
    up = solar + net
    um = solar - net
    um_mean, um_std, um_tv = seg_stats(um, left, right)
    up_mean, up_std, up_tv = seg_stats(up, left, right)

    bup = bridge_mse(up, left, right, up)
    bum = bridge_mse(um, left, right, up)
    bridge_improve = bup - bum
    bridge_ratio = bridge_improve / (bup + bum + 1e-6)

    tv0edge = np.abs(np.diff(up))
    base_tv = tv0edge[DAYTIME_START:DAYTIME_END].sum()
    ctv0 = np.r_[0.0, np.cumsum(tv0edge)]
    base_internal = ctv0[right] - ctv0[left]
    dum = np.abs(np.diff(um))
    ctvm = np.r_[0.0, np.cumsum(dum)]
    corr_internal = ctvm[right] - ctvm[left]
    lb = np.where(left > DAYTIME_START, np.abs(up[left] - up[left - 1]), 0)
    lcorr = np.where(left > DAYTIME_START, np.abs(um[left] - up[left - 1]), 0)
    rb = np.where(right < DAYTIME_END, np.abs(up[right + 1] - up[right]), 0)
    rcorr = np.where(right < DAYTIME_END, np.abs(up[right + 1] - um[right]), 0)
    corr_tv = base_tv - (base_internal + lb + rb) + (corr_internal + lcorr + rcorr)
    full_tv_improve = base_tv - corr_tv
    full_tv_ratio = full_tv_improve / (base_tv + corr_tv + 1e-6)

    # A compact version of the combined heuristic from the CGPT Pro scan.
    corr_ns = seg_corr(net, solar, left, right)
    dnet = np.diff(net, prepend=net[0])
    dsolar = np.diff(solar, prepend=solar[0])
    corr_d = seg_corr(dnet, dsolar, np.minimum(left + 1, right), right)
    same_sign = np.sign(dnet) == np.sign(dsolar)
    csame = np.r_[0.0, np.cumsum(same_sign.astype(float))]
    same = (csame[right + 1] - csame[np.minimum(left + 1, SLOTS_PER_DAY)]) / np.maximum(right - left, 1)
    slots = np.arange(SLOTS_PER_DAY)
    mask = (slots[None, :] >= left[:, None]) & (slots[None, :] <= right[:, None])
    net_max = np.where(mask, net[None, :], -np.inf).max(axis=1)
    boundary_mean = 0.5 * (net[left] + net[right])
    prominence = net_max - boundary_mean
    negfrac = seg_stats((um < 0).astype(float), left, right)[0]
    combined = 0.65 * full_tv_ratio + 0.65 * bridge_ratio + 0.25 * prominence - 0.20 * boundary_mean - 0.30 * negfrac - 0.012 * np.abs(mid - peak)

    best_tv_idx = int(np.argmax(full_tv_improve))
    best_bridge_idx = int(np.argmax(bridge_ratio))
    recall = candidate_recall_stats(labels, left, right)
    return {
        "solar_peak_slot": peak,
        "candidate_count": int(len(left)),
        "bridge_ratio_max": float(np.max(bridge_ratio)),
        "bridge_ratio_p99": float(np.quantile(bridge_ratio, 0.99)),
        "bridge_ratio_p95": float(np.quantile(bridge_ratio, 0.95)),
        "bridge_improve_p99": float(np.quantile(bridge_improve, 0.99)),
        "full_tv_ratio_p99": float(np.quantile(full_tv_ratio, 0.99)),
        "full_tv_improve_p99": float(np.quantile(full_tv_improve, 0.99)),
        "combined_p99": float(np.quantile(combined, 0.99)),
        "best_tv_left": int(left[best_tv_idx]),
        "best_tv_right": int(right[best_tv_idx]),
        "best_tv_improve": float(full_tv_improve[best_tv_idx]),
        "best_bridge_left": int(left[best_bridge_idx]),
        "best_bridge_right": int(right[best_bridge_idx]),
        "best_bridge_ratio": float(bridge_ratio[best_bridge_idx]),
        **recall,
    }


def alpha_season_prior() -> np.ndarray:
    alpha = load_final_dataset("alpha")
    day = alpha.groupby(["substation_id", "date"], sort=True)["label_day"].max().reset_index()
    day["date_ts"] = pd.to_datetime(day["date"])
    mask = day["date_ts"] < pd.Timestamp("2023-10-01")
    y = day["label_day"].astype(bool).to_numpy()
    doy = day["date_ts"].dt.dayofyear.to_numpy()
    n = 366
    cnt = np.bincount(doy[mask], minlength=n + 1)[1:].astype(float)
    pos = np.bincount(doy[mask], weights=y[mask].astype(float), minlength=n + 1)[1:]
    pad = 15
    kernel = np.ones(31)
    num = np.convolve(np.r_[pos[-pad:], pos, pos[:pad]], kernel, "valid")[:n]
    den = np.convolve(np.r_[cnt[-pad:], cnt, cnt[:pad]], kernel, "valid")[:n]
    prior_prob = np.divide(num, den, out=np.full(n, y[mask].mean()), where=den > 0)
    z = np.log((prior_prob + 0.02) / (1 - prior_prob + 0.02))
    z -= np.average(z, weights=np.maximum(cnt, 1))
    return z


def site_median(keys: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    out = np.empty(len(score), dtype=float)
    sites = keys["substation_id"].to_numpy()
    for site in np.unique(sites):
        mask = sites == site
        out[mask] = np.median(score[mask])
    return out


def site_rank(keys: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    out = np.empty(len(score), dtype=float)
    sites = keys["substation_id"].to_numpy()
    for site in np.unique(sites):
        idx = np.where(sites == site)[0]
        order = np.argsort(score[idx], kind="mergesort")
        ranks = np.empty(len(idx), dtype=float)
        ranks[order] = (np.arange(len(idx)) + 0.5) / len(idx)
        out[idx] = ranks - 0.5
    return out


def centered_rolling(keys: pd.DataFrame, score: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return score.copy()
    out = np.empty(len(score), dtype=float)
    temp = pd.DataFrame(
        {
            "substation_id": keys["substation_id"].to_numpy(),
            "date": pd.to_datetime(keys["date"]),
            "score": score,
            "idx": np.arange(len(score)),
        }
    )
    for _, group in temp.groupby("substation_id", sort=False):
        group = group.sort_values("date")
        rolled = pd.Series(group["score"].to_numpy()).rolling(window, center=True, min_periods=1).mean().to_numpy()
        out[group["idx"].to_numpy()] = rolled
    return out


def apply_variant(keys: pd.DataFrame, scores: pd.DataFrame, prior: np.ndarray, variant: dict) -> pd.Series:
    raw = scores[variant["feature"]].to_numpy(dtype=float)
    med = site_median(keys, raw)
    rank = site_rank(keys, raw)
    doy = pd.to_datetime(keys["date"]).dt.dayofyear.to_numpy()
    z0 = raw - variant["site_median_weight"] * med
    z1 = z0 + variant["site_rank_weight"] * rank + variant["season_weight"] * prior[doy - 1]
    rolled = centered_rolling(keys, z1, int(variant["rolling_window"]))
    final = (z1 + variant["rolling_weight"] * rolled) / (1 + variant["rolling_weight"])
    return pd.Series(final, index=keys.index)


def interval_predictions(keys: pd.DataFrame, scores: pd.DataFrame, pred_day: np.ndarray) -> np.ndarray:
    pred = np.zeros((len(keys), SLOTS_PER_DAY), dtype=bool)
    for i, is_day in enumerate(pred_day):
        if not is_day:
            continue
        left = int(scores.iloc[i]["best_tv_left"])
        right = int(scores.iloc[i]["best_tv_right"])
        pred[i, left : right + 1] = True
    return pred


def scan_dataset(dataset_name: str, keys: pd.DataFrame, arrays: dict[str, np.ndarray], started: float) -> pd.DataFrame:
    scan_rows = []
    for i in range(len(keys)):
        scan_rows.append(scan_day(arrays["net"][i], arrays["solar"][i], arrays["labels"][i]))
        if i and i % 500 == 0:
            print(f"scanned {dataset_name} day {i:,}/{len(keys):,} in {time.time() - started:.1f}s", flush=True)
    return pd.DataFrame(scan_rows)


def beta_threshold_sweep(y_true: np.ndarray, score: np.ndarray, variant: str) -> pd.DataFrame:
    finite = np.isfinite(score)
    thresholds = np.unique(np.r_[np.quantile(score[finite], np.linspace(0.01, 0.99, 199)), score[finite]])
    rows = []
    for threshold in thresholds:
        met = metric_counts(y_true, score >= threshold)
        met.update({"variant": variant, "threshold": float(threshold), "diagnostic": "beta_label_sweep_not_publication_ready"})
        rows.append(met)
    return pd.DataFrame(rows).sort_values(["f1", "precision", "recall"], ascending=[False, False, False]).reset_index(drop=True)


def evaluate_dataset(
    dataset_name: str,
    keys: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    scores: pd.DataFrame,
    prior: np.ndarray,
    include_threshold_sweep: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = pd.concat([keys.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    metrics_rows = []
    site_metrics_rows = []
    sweep_rows = []

    for variant in VARIANTS:
        score_col = f"score_{variant['variant']}"
        pred_col = f"pred_day_{variant['variant']}"
        out[score_col] = apply_variant(keys, scores, prior, variant)
        out[pred_col] = out[score_col] >= variant["threshold"]
        if include_threshold_sweep:
            sweep_rows.append(beta_threshold_sweep(out["label_day"].to_numpy(), out[score_col].to_numpy(), variant["variant"]))

        day_metrics = metric_counts(out["label_day"].to_numpy(), out[pred_col].to_numpy())
        day_metrics.update(
            {
                "dataset": dataset_name,
                "model": MODEL_NAME,
                "variant": variant["variant"],
                "level": "day",
                "threshold": variant["threshold"],
                "threshold_source": variant["threshold_source"],
            }
        )
        metrics_rows.append(day_metrics)

        pred_interval = interval_predictions(keys, scores, out[pred_col].to_numpy())
        observed = arrays["observed"].reshape(-1)
        interval_metrics = metric_counts(arrays["labels"].reshape(-1)[observed], pred_interval.reshape(-1)[observed])
        interval_metrics.update(
            {
                "dataset": dataset_name,
                "model": MODEL_NAME,
                "variant": variant["variant"],
                "level": "interval",
                "threshold": variant["threshold"],
                "threshold_source": variant["threshold_source"],
            }
        )
        metrics_rows.append(interval_metrics)

        for site, group in out.groupby("substation_id", sort=True):
            idx = group.index.to_numpy()
            sm = metric_counts(group["label_day"].to_numpy(), group[pred_col].to_numpy())
            sm.update({"dataset": dataset_name, "model": MODEL_NAME, "variant": variant["variant"], "substation_id": site, "level": "day"})
            site_metrics_rows.append(sm)
            observed_site = arrays["observed"][idx].reshape(-1)
            im = metric_counts(arrays["labels"][idx].reshape(-1)[observed_site], pred_interval[idx].reshape(-1)[observed_site])
            im.update({"dataset": dataset_name, "model": MODEL_NAME, "variant": variant["variant"], "substation_id": site, "level": "interval"})
            site_metrics_rows.append(im)

    metrics = pd.DataFrame(metrics_rows)
    site_metrics = pd.DataFrame(site_metrics_rows)
    threshold_sweep = pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame()
    return out, metrics, site_metrics, threshold_sweep


def main() -> None:
    started = time.time()
    alpha = load_final_dataset("alpha")
    beta = load_final_dataset("beta")
    alpha_keys, alpha_arrays = build_daily_arrays(alpha)
    beta_keys, beta_arrays = build_daily_arrays(beta)
    prior = alpha_season_prior()

    alpha_scores = scan_dataset("alpha", alpha_keys, alpha_arrays, started)
    beta_scores = scan_dataset("beta", beta_keys, beta_arrays, started)

    alpha_out, alpha_metrics, alpha_site_metrics, _ = evaluate_dataset("alpha", alpha_keys, alpha_arrays, alpha_scores, prior, include_threshold_sweep=False)
    beta_out, beta_metrics, beta_site_metrics, threshold_sweep = evaluate_dataset("beta", beta_keys, beta_arrays, beta_scores, prior, include_threshold_sweep=True)

    candidate_recall = []
    rpf = beta_out.loc[beta_out["label_day"]].copy()
    for name, col in [("iou50", "has_iou50"), ("iou70", "has_iou70"), ("boundary30", "boundary30")]:
        candidate_recall.append({"criterion": name, "support": int(len(rpf)), "recall": float(rpf[col].mean()) if len(rpf) else 0.0})
    for site, group in rpf.groupby("substation_id", sort=True):
        for name, col in [("iou50", "has_iou50"), ("iou70", "has_iou70"), ("boundary30", "boundary30")]:
            candidate_recall.append(
                {
                    "criterion": name,
                    "substation_id": site,
                    "support": int(len(group)),
                    "recall": float(group[col].mean()) if len(group) else 0.0,
                }
            )
    candidate_recall = pd.DataFrame(candidate_recall)

    alpha_out.to_csv(CSV_DIR / "01_alpha_daily_bridge_scores.csv", index=False)
    beta_out.to_csv(CSV_DIR / "01_beta_daily_bridge_scores.csv", index=False)
    alpha_metrics.to_csv(CSV_DIR / "02_alpha_metrics.csv", index=False)
    beta_metrics.to_csv(CSV_DIR / "02_beta_metrics.csv", index=False)
    alpha_site_metrics.to_csv(CSV_DIR / "03_alpha_site_metrics.csv", index=False)
    beta_site_metrics.to_csv(CSV_DIR / "03_beta_site_metrics.csv", index=False)
    candidate_recall.to_csv(CSV_DIR / "04_beta_candidate_recall.csv", index=False)
    threshold_sweep.to_csv(CSV_DIR / "05_beta_threshold_sweep_diagnostic.csv", index=False)
    all_metrics = pd.concat([alpha_metrics, beta_metrics], ignore_index=True)
    all_site_metrics = pd.concat([alpha_site_metrics, beta_site_metrics], ignore_index=True)
    all_metrics.to_csv(CSV_DIR / "06_alpha_beta_metrics.csv", index=False)
    all_site_metrics.to_csv(CSV_DIR / "07_alpha_beta_site_metrics.csv", index=False)

    manifest = {
        "model_name": MODEL_NAME,
        "run_type": "alpha_beta_daily_score_scan",
        "publication_ready": False,
        "warning": "This is a fast misc reproduction of CGPT Pro bridge-score ideas. Thresholds/weights are imported from CGPT Pro artifacts; this is exploratory.",
        "variants": VARIANTS,
        "rows": {
            "alpha_site_days": int(len(alpha_keys)),
            "alpha_rows": int(len(alpha)),
            "beta_site_days": int(len(beta_keys)),
            "beta_rows": int(len(beta)),
        },
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "alpha_daily_scores": str(CSV_DIR / "01_alpha_daily_bridge_scores.csv"),
            "beta_daily_scores": str(CSV_DIR / "01_beta_daily_bridge_scores.csv"),
            "alpha_metrics": str(CSV_DIR / "02_alpha_metrics.csv"),
            "beta_metrics": str(CSV_DIR / "02_beta_metrics.csv"),
            "alpha_site_metrics": str(CSV_DIR / "03_alpha_site_metrics.csv"),
            "beta_site_metrics": str(CSV_DIR / "03_beta_site_metrics.csv"),
            "alpha_beta_metrics": str(CSV_DIR / "06_alpha_beta_metrics.csv"),
            "alpha_beta_site_metrics": str(CSV_DIR / "07_alpha_beta_site_metrics.csv"),
            "candidate_recall": str(CSV_DIR / "04_beta_candidate_recall.csv"),
            "threshold_sweep": str(CSV_DIR / "05_beta_threshold_sweep_diagnostic.csv"),
        },
    }
    (MANIFEST_DIR / "full_beta_daily_score_scan_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nAlpha/Beta metrics")
    print(all_metrics[["dataset", "variant", "level", "support", "positive_support", "precision", "recall", "f1"]].round(4).to_string(index=False))
    print("\nCandidate recall")
    print(candidate_recall.loc[candidate_recall["substation_id"].isna() if "substation_id" in candidate_recall.columns else slice(None)].round(4).to_string(index=False))
    print("\nBest Beta threshold sweep rows (diagnostic only)")
    print(threshold_sweep.groupby("variant").head(1)[["variant", "threshold", "precision", "recall", "f1", "tp", "fp", "fn"]].round(4).to_string(index=False))
    print(f"\nDone in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()

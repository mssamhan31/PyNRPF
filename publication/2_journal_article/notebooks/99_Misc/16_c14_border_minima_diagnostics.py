from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SLOTS_PER_DAY = 96
DAYTIME_START = 24
DAYTIME_END = 72
BEST_REGIME = "R2_beta_loso_plus_alpha"
BEST_VARIANT = "G_b1p0_r1p5_sc0p5"
NEIGHBOUR_RADIUS_SLOTS = 4
INSIDE_DEPTH_SLOTS = 4
TOLERANCE_MW_GRID = [0.0, 0.10, 0.25, 0.50, 1.00]


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
EXP_OUT = MISC / "outputs/260707_physical_score_loso_experiments"
C14_AUDIT = EXP_OUT / "C14_small_weight_grid/06_c14_daily_prediction_audit.csv"
OUT = EXP_OUT / "C17_c14_border_minima_diagnostics"


def date_key(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.strftime("%Y-%m-%d")


def safe_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.fillna(False).astype(str).str.strip().str.lower().isin({"true", "t", "1", "yes", "y"})


def slot_to_time(slot: float | int | None) -> str:
    if pd.isna(slot):
        return ""
    slot = int(slot)
    return f"{slot // 4:02d}:{(slot % 4) * 15:02d}"


def fill_series(values: np.ndarray, default: float = 0.0) -> np.ndarray:
    return (
        pd.Series(values, dtype="float64")
        .interpolate(limit_direction="both")
        .fillna(default)
        .to_numpy(dtype=float)
    )


def load_beta_day_arrays() -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    beta = pd.read_parquet(
        JOURNAL / "dataset/final/dataset_beta.parquet",
        columns=[
            "substation_id",
            "date",
            "timestamp",
            "net_load_MW",
            "label_interval",
            "label_day",
            "confidence",
        ],
    )
    beta["substation_id"] = beta["substation_id"].astype(str)
    beta["date"] = date_key(beta["date"])
    beta["timestamp"] = pd.to_datetime(beta["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    beta["slot"] = beta["timestamp"].dt.hour * 4 + beta["timestamp"].dt.minute // 15
    beta["label_interval"] = safe_bool(beta["label_interval"])
    beta["label_day"] = safe_bool(beta["label_day"])
    beta["confidence"] = beta["confidence"].fillna("missing").astype(str).str.strip().str.lower()

    rows: list[dict[str, object]] = []
    arrays: dict[tuple[str, str], np.ndarray] = {}
    for (site, date), group in beta.groupby(["substation_id", "date"], sort=True):
        group = group.sort_values("slot")
        net = np.full(SLOTS_PER_DAY, np.nan, dtype=float)
        valid = group.loc[group["slot"].between(0, SLOTS_PER_DAY - 1)].drop_duplicates("slot", keep="last")
        slots = valid["slot"].to_numpy(dtype=int)
        net[slots] = valid["net_load_MW"].to_numpy(dtype=float)
        net = fill_series(net, 0.0)
        arrays[(site, date)] = net
        true_slots = sorted(group.loc[group["label_interval"], "slot"].astype(int).unique().tolist())
        rows.append(
            {
                "substation_id": site,
                "date": date,
                "confidence": str(group["confidence"].iloc[0]),
                "true_day": bool(group["label_day"].max()),
                "true_start_slot": float(min(true_slots)) if true_slots else np.nan,
                "true_end_slot": float(max(true_slots)) if true_slots else np.nan,
                "true_start_time": slot_to_time(min(true_slots)) if true_slots else "",
                "true_end_time": slot_to_time(max(true_slots)) if true_slots else "",
            }
        )
    return pd.DataFrame(rows), arrays


def load_c14_best_predictions() -> pd.DataFrame:
    if not C14_AUDIT.exists():
        raise FileNotFoundError(f"Run C14 first. Missing: {C14_AUDIT}")
    pred = pd.read_csv(C14_AUDIT)
    pred = pred.loc[pred["regime"].eq(BEST_REGIME) & pred["variant"].eq(BEST_VARIANT)].copy()
    pred["substation_id"] = pred["substation_id"].astype(str)
    pred["date"] = date_key(pred["date"])
    pred["pred_day"] = safe_bool(pred["pred_day"])
    pred["score"] = pd.to_numeric(pred["score"], errors="coerce")
    pred["threshold"] = pd.to_numeric(pred["threshold"], errors="coerce")

    c1 = pd.read_csv(
        EXP_OUT / "C1_cached_daily_features/02_c1_daily_feature_cache.csv",
        usecols=["dataset", "substation_id", "date", "v03_selected_left_slot", "v03_selected_right_slot"],
    )
    c1 = c1.loc[c1["dataset"].eq("beta")].copy()
    c1["substation_id"] = c1["substation_id"].astype(str)
    c1["date"] = date_key(c1["date"])
    c1["v03_selected_left_slot"] = pd.to_numeric(c1["v03_selected_left_slot"], errors="coerce")
    c1["v03_selected_right_slot"] = pd.to_numeric(c1["v03_selected_right_slot"], errors="coerce")
    pred = pred.merge(
        c1[["substation_id", "date", "v03_selected_left_slot", "v03_selected_right_slot"]],
        on=["substation_id", "date"],
        how="left",
        validate="one_to_one",
    )
    pred["pred_start_slot"] = np.where(pred["pred_day"], pred["v03_selected_left_slot"], np.nan)
    pred["pred_end_slot"] = np.where(pred["pred_day"], pred["v03_selected_right_slot"], np.nan)
    return pred[["substation_id", "date", "pred_day", "score", "threshold", "pred_start_slot", "pred_end_slot"]]


def border_metrics_for_window(
    net: np.ndarray,
    start_slot: float,
    end_slot: float,
    *,
    window_source: str,
) -> dict[str, object]:
    if pd.isna(start_slot) or pd.isna(end_slot):
        return {}
    start = int(start_slot)
    end = int(end_slot)
    if start < 0 or end >= len(net) or start > end:
        return {}

    left_neigh = net[max(0, start - NEIGHBOUR_RADIUS_SLOTS) : min(len(net), start + NEIGHBOUR_RADIUS_SLOTS + 1)]
    right_neigh = net[max(0, end - NEIGHBOUR_RADIUS_SLOTS) : min(len(net), end + NEIGHBOUR_RADIUS_SLOTS + 1)]
    left_inside = net[start : min(len(net), start + INSIDE_DEPTH_SLOTS + 1)]
    right_inside = net[max(0, end - INSIDE_DEPTH_SLOTS) : end + 1]

    left_value = float(net[start])
    right_value = float(net[end])
    left_neigh_min = float(np.nanmin(left_neigh))
    right_neigh_min = float(np.nanmin(right_neigh))
    left_inside_min = float(np.nanmin(left_inside))
    right_inside_min = float(np.nanmin(right_inside))
    left_inside_max = float(np.nanmax(left_inside))
    right_inside_max = float(np.nanmax(right_inside))
    window_values = net[start : end + 1]

    left_rank = float((np.sum(left_neigh < left_value) + 0.5 * np.sum(np.isclose(left_neigh, left_value))) / len(left_neigh))
    right_rank = float((np.sum(right_neigh < right_value) + 0.5 * np.sum(np.isclose(right_neigh, right_value))) / len(right_neigh))
    return {
        "window_source": window_source,
        "start_slot": start,
        "end_slot": end,
        "start_time": slot_to_time(start),
        "end_time": slot_to_time(end),
        "duration_h": (end - start + 1) * 0.25,
        "left_border_net_MW": left_value,
        "right_border_net_MW": right_value,
        "left_delta_to_neighbour_min_MW": left_value - left_neigh_min,
        "right_delta_to_neighbour_min_MW": right_value - right_neigh_min,
        "left_delta_to_inside_min_MW": left_value - left_inside_min,
        "right_delta_to_inside_min_MW": right_value - right_inside_min,
        "left_inside_rise_MW": left_inside_max - left_value,
        "right_inside_rise_MW": right_inside_max - right_value,
        "left_local_min_rank": left_rank,
        "right_local_min_rank": right_rank,
        "window_net_min_MW": float(np.nanmin(window_values)),
        "window_net_max_MW": float(np.nanmax(window_values)),
        "window_n_height_MW": float(np.nanmax(window_values) - max(left_value, right_value)),
    }


def summarise_border_rows(rows: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for (subset, window_source, day_group), group in rows.groupby(["subset", "window_source", "day_group"], dropna=False):
        row: dict[str, object] = {
            "subset": subset,
            "window_source": window_source,
            "day_group": day_group,
            "windows": len(group),
            "median_left_delta_MW": float(group["left_delta_to_neighbour_min_MW"].median()),
            "median_right_delta_MW": float(group["right_delta_to_neighbour_min_MW"].median()),
            "median_left_rank": float(group["left_local_min_rank"].median()),
            "median_right_rank": float(group["right_local_min_rank"].median()),
            "median_n_height_MW": float(group["window_n_height_MW"].median()),
        }
        for tolerance in TOLERANCE_MW_GRID:
            left_ok = group["left_delta_to_neighbour_min_MW"].le(tolerance)
            right_ok = group["right_delta_to_neighbour_min_MW"].le(tolerance)
            row[f"left_near_min_rate_tol_{tolerance:g}MW"] = float(left_ok.mean())
            row[f"right_near_min_rate_tol_{tolerance:g}MW"] = float(right_ok.mean())
            row[f"both_near_min_rate_tol_{tolerance:g}MW"] = float((left_ok & right_ok).mean())
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values(["subset", "window_source", "day_group"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    day_truth, arrays = load_beta_day_arrays()
    pred = load_c14_best_predictions()
    days = day_truth.merge(pred, on=["substation_id", "date"], how="left", validate="one_to_one")
    days["day_group"] = np.select(
        [
            days["true_day"] & days["pred_day"],
            ~days["true_day"] & days["pred_day"],
            days["true_day"] & ~days["pred_day"],
            ~days["true_day"] & ~days["pred_day"],
        ],
        ["TP_day", "FP_day", "FN_day", "TN_day"],
        default="unknown",
    )

    rows: list[dict[str, object]] = []
    for day in days.itertuples(index=False):
        net = arrays[(day.substation_id, day.date)]
        base = {
            "substation_id": day.substation_id,
            "date": day.date,
            "confidence": day.confidence,
            "true_day": bool(day.true_day),
            "pred_day": bool(day.pred_day),
            "day_group": day.day_group,
            "score": day.score,
            "threshold": day.threshold,
        }
        if bool(day.true_day):
            metrics = border_metrics_for_window(net, day.true_start_slot, day.true_end_slot, window_source="manual_true")
            if metrics:
                rows.append({**base, **metrics})
        if bool(day.pred_day):
            metrics = border_metrics_for_window(net, day.pred_start_slot, day.pred_end_slot, window_source="model_pred")
            if metrics:
                rows.append({**base, **metrics})

    border_rows = pd.DataFrame(rows)
    border_rows["subset"] = np.where(border_rows["confidence"].eq("sure"), "beta_sure_only", "beta_all")
    beta_all_copy = border_rows.copy()
    beta_all_copy["subset"] = "beta_all"
    border_rows = pd.concat(
        [beta_all_copy, border_rows.loc[border_rows["subset"].eq("beta_sure_only")].copy()],
        ignore_index=True,
    )
    summary = summarise_border_rows(border_rows)

    manifest = pd.DataFrame(
        [
            {
                "script": Path(__file__).name,
                "model": f"{BEST_REGIME}/{BEST_VARIANT}",
                "beta_site_days": len(days),
                "border_window_rows": len(border_rows),
                "neighbour_radius_slots": NEIGHBOUR_RADIUS_SLOTS,
                "neighbour_radius_minutes": NEIGHBOUR_RADIUS_SLOTS * 15,
                "inside_depth_slots": INSIDE_DEPTH_SLOTS,
                "inside_depth_minutes": INSIDE_DEPTH_SLOTS * 15,
                "outputs": str(OUT.relative_to(ROOT)),
            }
        ]
    )
    manifest.to_csv(OUT / "01_c17_manifest.csv", index=False)
    border_rows.to_csv(OUT / "02_c17_border_window_metrics.csv", index=False)
    summary.to_csv(OUT / "03_c17_border_minima_summary.csv", index=False)
    days.to_csv(OUT / "04_c17_c14_best_day_context.csv", index=False)

    print(f"Wrote border-minima diagnostics to {OUT.relative_to(ROOT)}")
    print("\nSummary")
    show_cols = [
        "subset",
        "window_source",
        "day_group",
        "windows",
        "median_left_delta_MW",
        "median_right_delta_MW",
        "both_near_min_rate_tol_0.5MW",
        "both_near_min_rate_tol_1MW",
        "median_n_height_MW",
    ]
    print(summary[show_cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

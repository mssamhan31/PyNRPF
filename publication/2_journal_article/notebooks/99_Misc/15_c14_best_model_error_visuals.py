from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


SLOTS_PER_DAY = 96
DAYTIME_START = 24
DAYTIME_END = 72
BEST_REGIME = "R2_beta_loso_plus_alpha"
BEST_VARIANT = "G_b1p0_r1p5_sc0p5"
MODEL_LABEL = f"{BEST_REGIME} / {BEST_VARIANT}"
WEIGHTS = {
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
C1_CACHE = EXP_OUT / "C1_cached_daily_features/02_c1_daily_feature_cache.csv"
C14_AUDIT = EXP_OUT / "C14_small_weight_grid/06_c14_daily_prediction_audit.csv"
OUT = EXP_OUT / "C16_c14_best_error_review_visuals"


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


def load_beta_intervals() -> pd.DataFrame:
    beta = pd.read_parquet(
        JOURNAL / "dataset/final/dataset_beta.parquet",
        columns=[
            "substation_id",
            "date",
            "timestamp",
            "net_load_MW",
            "solar_MW",
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
    return beta.sort_values(["substation_id", "date", "slot"]).reset_index(drop=True)


def true_windows_from_beta(beta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (site, date), group in beta.groupby(["substation_id", "date"], sort=True):
        true_slots = sorted(group.loc[group["label_interval"], "slot"].astype(int).unique().tolist())
        rows.append(
            {
                "substation_id": site,
                "date": date,
                "true_day": bool(group["label_day"].max()),
                "confidence": str(group["confidence"].iloc[0]),
                "true_start_slot": float(min(true_slots)) if true_slots else np.nan,
                "true_end_slot": float(max(true_slots)) if true_slots else np.nan,
                "true_start_time": slot_to_time(min(true_slots)) if true_slots else "",
                "true_end_time": slot_to_time(max(true_slots)) if true_slots else "",
                "true_slots": set(true_slots),
            }
        )
    return pd.DataFrame(rows)


def load_c14_best_daily() -> pd.DataFrame:
    if not C14_AUDIT.exists():
        raise FileNotFoundError(f"Run C14 first. Missing: {C14_AUDIT}")
    audit = pd.read_csv(C14_AUDIT)
    audit = audit.loc[audit["regime"].eq(BEST_REGIME) & audit["variant"].eq(BEST_VARIANT)].copy()
    audit["substation_id"] = audit["substation_id"].astype(str)
    audit["date"] = date_key(audit["date"])
    audit["true_day"] = safe_bool(audit["true_day"])
    audit["pred_day"] = safe_bool(audit["pred_day"])
    audit["confidence"] = audit["confidence"].fillna("missing").astype(str).str.strip().str.lower()
    audit["score"] = pd.to_numeric(audit["score"], errors="coerce")
    audit["threshold"] = pd.to_numeric(audit["threshold"], errors="coerce")
    audit["confidence_score"] = (audit["score"] - audit["threshold"]).abs()

    cache = pd.read_csv(C1_CACHE)
    cache = cache.loc[cache["dataset"].eq("beta")].copy()
    cache["substation_id"] = cache["substation_id"].astype(str)
    cache["date"] = date_key(cache["date"])
    keep_cols = [
        "substation_id",
        "date",
        "v03_selected_left_slot",
        "v03_selected_right_slot",
        "v03_selected_duration_h",
        *WEIGHTS.keys(),
    ]
    cache = cache[keep_cols].copy()
    for col in ["v03_selected_left_slot", "v03_selected_right_slot", "v03_selected_duration_h", *WEIGHTS.keys()]:
        cache[col] = pd.to_numeric(cache[col], errors="coerce")
    return audit.merge(cache, on=["substation_id", "date"], how="left", validate="one_to_one")


def window_iou(true_slots: set[int], pred_slots: set[int]) -> float:
    union = true_slots | pred_slots
    if not union:
        return np.nan
    return len(true_slots & pred_slots) / len(union)


def build_error_audit(beta: pd.DataFrame) -> pd.DataFrame:
    truth = true_windows_from_beta(beta)
    pred = load_c14_best_daily()
    audit = pred.merge(
        truth.drop(columns=["true_day", "confidence"]),
        on=["substation_id", "date"],
        how="left",
        validate="one_to_one",
    )
    audit["pred_start_slot"] = np.where(audit["v03_selected_left_slot"].notna(), audit["v03_selected_left_slot"], np.nan)
    audit["pred_end_slot"] = np.where(audit["v03_selected_right_slot"].notna(), audit["v03_selected_right_slot"], np.nan)
    audit["pred_start_time"] = audit["pred_start_slot"].map(slot_to_time)
    audit["pred_end_time"] = audit["pred_end_slot"].map(slot_to_time)
    audit["day_group"] = np.select(
        [
            audit["true_day"] & audit["pred_day"],
            ~audit["true_day"] & audit["pred_day"],
            audit["true_day"] & ~audit["pred_day"],
            ~audit["true_day"] & ~audit["pred_day"],
        ],
        ["TP_day", "FP_day", "FN_day", "TN_day"],
        default="unknown",
    )
    component_cols = []
    for col, weight in WEIGHTS.items():
        weighted_col = f"weighted_{col}"
        audit[weighted_col] = pd.to_numeric(audit[col], errors="coerce").fillna(0.0) * weight
        component_cols.append(weighted_col)
    audit["recomputed_score"] = audit[component_cols].sum(axis=1)

    iou_rows = []
    for row in audit.itertuples(index=False):
        true_slots = row.true_slots if isinstance(row.true_slots, set) else set()
        pred_slots = (
            set(range(int(row.pred_start_slot), int(row.pred_end_slot) + 1))
            if not pd.isna(row.pred_start_slot) and not pd.isna(row.pred_end_slot)
            else set()
        )
        effective_pred_slots = pred_slots if bool(row.pred_day) else set()
        pred_start = min(effective_pred_slots) if effective_pred_slots else np.nan
        pred_end = max(effective_pred_slots) if effective_pred_slots else np.nan
        true_start = min(true_slots) if true_slots else np.nan
        true_end = max(true_slots) if true_slots else np.nan
        iou_rows.append(
            {
                "iou": window_iou(true_slots, effective_pred_slots),
                "iou_with_fp_fn_zero": window_iou(true_slots, effective_pred_slots)
                if bool(row.true_day) and bool(row.pred_day)
                else 0.0,
                "start_error_minutes": (pred_start - true_start) * 15
                if not pd.isna(pred_start) and not pd.isna(true_start)
                else np.nan,
                "end_error_minutes": (pred_end - true_end) * 15
                if not pd.isna(pred_end) and not pd.isna(true_end)
                else np.nan,
            }
        )
    audit = pd.concat([audit.drop(columns=["true_slots"]), pd.DataFrame(iou_rows)], axis=1)
    return audit.sort_values(["substation_id", "date"]).reset_index(drop=True)


def add_window_shape(fig: go.Figure, row_idx: int, x0: str, x1: str, color: str, label: str, opacity: float) -> None:
    fig.add_vrect(
        x0=x0,
        x1=x1,
        fillcolor=color,
        opacity=opacity,
        line_width=0,
        row=row_idx,
        col=1,
        annotation_text=label,
        annotation_position="top left",
    )


def score_title(row: pd.Series) -> str:
    return (
        f"B={row['weighted_F1_bridge_improvement']:.2f}, "
        f"Rough={row['weighted_F2_roughness_improvement']:.2f}, "
        f"Slope={row['weighted_F3_slope_continuity_improvement']:.2f}, "
        f"Dur={row['weighted_F4_duration_plausibility']:.2f}, "
        f"N={row['weighted_F5_n_height_ratio']:.2f}, "
        f"Solar={row['weighted_F6_solar_strength_ratio']:.2f}, "
        f"Peak={row['weighted_F7_solar_peak_alignment']:.2f}, "
        f"SC={row['weighted_F8_site_centered_core_score']:.2f}"
    )


def build_gallery(name: str, examples: pd.DataFrame, beta: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    if examples.empty:
        out_path = OUT / f"{name}_0days.html"
        out_path.write_text("<html><body><h1>No examples</h1></body></html>", encoding="utf-8")
        return out_path, pd.DataFrame()

    examples = examples.reset_index(drop=True)
    rows = len(examples)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=min(0.03, 0.32 / max(rows, 1)),
        subplot_titles=[
            (
                f"{r.substation_id} {r.date} | {r.day_group} | "
                f"score={r.score:.3f}, thr={r.threshold:.3f}, margin={r.confidence_score:.3f}, "
                f"IoU={r.iou if not pd.isna(r.iou) else 0:.3f}<br>{score_title(pd.Series(r._asdict()))}"
            )
            for r in examples.itertuples(index=False)
        ],
    )
    index_rows: list[dict[str, object]] = []
    for plot_row, example in enumerate(examples.itertuples(index=False), start=1):
        day = beta.loc[
            beta["substation_id"].eq(example.substation_id)
            & beta["date"].eq(example.date)
            & beta["slot"].between(DAYTIME_START, DAYTIME_END)
        ].sort_values("slot")
        x = day["slot"].map(slot_to_time)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=day["net_load_MW"],
                mode="lines",
                name="Raw net load",
                line=dict(color="#22303d", width=2),
                showlegend=plot_row == 1,
            ),
            row=plot_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=day["solar_MW"],
                mode="lines",
                name="Solar",
                line=dict(color="#eb932c", width=2),
                showlegend=plot_row == 1,
            ),
            row=plot_row,
            col=1,
        )
        fig.add_hline(y=0, line=dict(color="#5C7D99", width=1, dash="dot"), row=plot_row, col=1)

        if example.true_start_time and example.true_end_time:
            add_window_shape(fig, plot_row, example.true_start_time, example.true_end_time, "#2F4D67", "manual", 0.18)
        if example.pred_start_time and example.pred_end_time:
            pred_label = "pred" if bool(example.pred_day) else "selected below threshold"
            add_window_shape(fig, plot_row, example.pred_start_time, example.pred_end_time, "#eb932c", pred_label, 0.20)

        index_rows.append(
            {
                "gallery": name,
                "substation_id": example.substation_id,
                "date": example.date,
                "day_group": example.day_group,
                "confidence": example.confidence,
                "score": example.score,
                "threshold": example.threshold,
                "confidence_score": example.confidence_score,
                "iou": example.iou,
                "true_window": f"{example.true_start_time}-{example.true_end_time}",
                "pred_window": f"{example.pred_start_time}-{example.pred_end_time}",
                "start_error_minutes": example.start_error_minutes,
                "end_error_minutes": example.end_error_minutes,
            }
        )

    fig.update_layout(
        title=f"{MODEL_LABEL}: {name.replace('_', ' ')} ({rows} days)",
        height=max(340 * rows, 560),
        width=980,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=55, r=30, t=95, b=40),
    )
    fig.update_xaxes(tickangle=0, tickmode="array", tickvals=["06:00", "09:00", "12:00", "15:00", "18:00"])
    fig.update_yaxes(title_text="MW")
    out_path = OUT / f"{name}_{rows}days.html"
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    return out_path, pd.DataFrame(index_rows)


def select_examples(audit: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sure = audit.loc[audit["confidence"].eq("sure")].copy()
    errors = sure.loc[sure["true_day"].ne(sure["pred_day"])].copy()
    fp = errors.loc[errors["day_group"].eq("FP_day")].sort_values("confidence_score", ascending=False).head(12)
    fn = errors.loc[errors["day_group"].eq("FN_day")].sort_values("confidence_score", ascending=False).head(12)
    return {
        "c14_best_fp_top_confidence": fp,
        "c14_best_fn_top_confidence": fn,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    beta = load_beta_intervals()
    audit = build_error_audit(beta)
    sure_errors = audit.loc[audit["confidence"].eq("sure") & audit["true_day"].ne(audit["pred_day"])].copy()
    site_counts = (
        sure_errors.groupby(["substation_id", "day_group"], as_index=False)
        .agg(days=("date", "count"), mean_score=("score", "mean"), mean_threshold=("threshold", "mean"))
        .sort_values(["substation_id", "day_group"])
    )
    selections = select_examples(audit)

    gallery_rows: list[dict[str, object]] = []
    index_parts: list[pd.DataFrame] = []
    for name, examples in selections.items():
        path, index = build_gallery(name, examples, beta)
        gallery_rows.append({"gallery": name, "n_days": len(examples), "html_file": path.name})
        if not index.empty:
            index["html_file"] = path.name
            index_parts.append(index)

    gallery_index = pd.DataFrame(gallery_rows)
    example_index = pd.concat(index_parts, ignore_index=True) if index_parts else pd.DataFrame()
    audit.to_csv(OUT / "01_c16_c14_best_day_audit.csv", index=False)
    sure_errors.to_csv(OUT / "02_c16_c14_best_sure_error_audit.csv", index=False)
    site_counts.to_csv(OUT / "03_c16_c14_best_sure_error_counts_by_site.csv", index=False)
    gallery_index.to_csv(OUT / "04_c16_gallery_index.csv", index=False)
    example_index.to_csv(OUT / "05_c16_gallery_example_index.csv", index=False)
    print(f"Wrote C14 best-model visual diagnostics to {OUT.relative_to(ROOT)}")
    print("\nSure-only FP/FN counts by site")
    print(site_counts.to_string(index=False))
    print("\nGalleries")
    print(gallery_index.to_string(index=False))


if __name__ == "__main__":
    main()

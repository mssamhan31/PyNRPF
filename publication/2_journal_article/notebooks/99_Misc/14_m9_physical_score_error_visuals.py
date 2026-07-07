from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


SLOTS_PER_DAY = 96
DAYTIME_START = 24
DAYTIME_END = 72
MODEL_LABEL = "R1_beta_loso / M9_drop_site_rank"


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
C10_AUDIT = EXP_OUT / "C10_window_interval_evaluation/04_c10_window_day_audit.csv"
OUT = EXP_OUT / "C11_m9_error_review_visuals"


def date_key(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.strftime("%Y-%m-%d")


def slot_to_time(slot: float | int | None) -> str:
    if pd.isna(slot):
        return ""
    slot = int(slot)
    return f"{slot // 4:02d}:{(slot % 4) * 15:02d}"


def time_labels() -> list[str]:
    return [slot_to_time(slot) for slot in range(SLOTS_PER_DAY)]


def load_beta_intervals() -> pd.DataFrame:
    beta = pd.read_parquet(
        JOURNAL / "dataset/final/dataset_beta.parquet",
        columns=["substation_id", "date", "timestamp", "net_load_MW", "solar_MW", "label_interval", "confidence"],
    )
    beta["substation_id"] = beta["substation_id"].astype(str)
    beta["date"] = date_key(beta["date"])
    beta["timestamp"] = pd.to_datetime(beta["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    beta["slot"] = beta["timestamp"].dt.hour * 4 + beta["timestamp"].dt.minute // 15
    beta["label_interval"] = beta["label_interval"].fillna(False).astype(bool)
    beta["confidence"] = beta["confidence"].fillna("missing").astype(str).str.strip().str.lower()
    return beta.loc[beta["slot"].between(DAYTIME_START, DAYTIME_END)].copy()


def load_audit() -> pd.DataFrame:
    if not C10_AUDIT.exists():
        raise FileNotFoundError(f"Run C10 first. Missing: {C10_AUDIT}")
    audit = pd.read_csv(C10_AUDIT)
    audit["date"] = date_key(audit["date"])
    audit["true_day"] = audit["true_day"].astype(bool)
    audit["pred_day"] = audit["pred_day"].astype(bool)
    audit["confidence"] = audit["confidence"].fillna("missing").astype(str).str.strip().str.lower()
    audit["confidence_score"] = (audit["score"] - audit["threshold"]).abs()
    return audit


def select_examples(audit: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sure = audit.loc[audit["confidence"].eq("sure")].copy()
    errors = sure.loc[sure["true_day"].ne(sure["pred_day"])].copy()
    fp = errors.loc[errors["day_group"].eq("FP_day")].sort_values("confidence_score", ascending=False).head(12)
    fn = errors.loc[errors["day_group"].eq("FN_day")].sort_values("confidence_score", ascending=False).head(12)

    top80 = sure.sort_values(["confidence_score", "score", "substation_id", "date"], ascending=[False, False, True, True]).head(
        int(np.ceil(len(sure) * 0.80))
    )
    auto80_errors = top80.loc[top80["true_day"].ne(top80["pred_day"])].sort_values(
        ["day_group", "substation_id", "date"]
    )
    low_conf_errors = errors.loc[~errors.index.isin(top80.index)].sort_values(
        ["confidence_score", "substation_id", "date"],
        ascending=[True, True, True],
    ).head(24)

    return {
        "fp_top_confidence": fp,
        "fn_top_confidence": fn,
        "auto80_errors": auto80_errors,
        "low_confidence_error_sample": low_conf_errors,
    }


def shape_for_window(row: pd.Series, start_col: str, end_col: str) -> tuple[str, str] | None:
    start = row.get(start_col)
    end = row.get(end_col)
    if pd.isna(start) or pd.isna(end):
        return None
    return slot_to_time(start), slot_to_time(end)


def add_window_shape(fig: go.Figure, row_idx: int, x0: str, x1: str, color: str, label: str) -> None:
    fig.add_vrect(
        x0=x0,
        x1=x1,
        fillcolor=color,
        opacity=0.18,
        line_width=0,
        row=row_idx,
        col=1,
        annotation_text=label,
        annotation_position="top left",
    )


def build_gallery(name: str, examples: pd.DataFrame, beta: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    if examples.empty:
        return OUT / f"{name}_0days.html", pd.DataFrame()

    rows = len(examples)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=min(0.025, 0.25 / max(rows, 1)),
        subplot_titles=[
            (
                f"{r.substation_id} {r.date} | {r.day_group} | "
                f"score={r.score:.3f}, thr={r.threshold:.3f}, IoU={r.iou if not pd.isna(r.iou) else 0:.3f}, "
                f"start_err={r.start_error_minutes if not pd.isna(r.start_error_minutes) else np.nan:.0f}m, "
                f"end_err={r.end_error_minutes if not pd.isna(r.end_error_minutes) else np.nan:.0f}m"
            )
            for r in examples.itertuples(index=False)
        ],
    )
    index_rows: list[dict[str, object]] = []
    for plot_row, example in enumerate(examples.itertuples(index=False), start=1):
        day = beta.loc[
            beta["substation_id"].eq(example.substation_id) & beta["date"].eq(example.date)
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

        true_window = shape_for_window(pd.Series(example._asdict()), "true_start_slot", "true_end_slot")
        pred_window = shape_for_window(pd.Series(example._asdict()), "pred_start_slot", "pred_end_slot")
        if true_window:
            add_window_shape(fig, plot_row, true_window[0], true_window[1], "#2F4D67", "manual")
        if pred_window:
            add_window_shape(fig, plot_row, pred_window[0], pred_window[1], "#eb932c", "pred")

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
        height=max(320 * rows, 520),
        width=980,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=80, b=40),
    )
    fig.update_xaxes(tickangle=0, nticks=7)
    fig.update_yaxes(title_text="MW")
    out_path = OUT / f"{name}_{rows}days.html"
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    return out_path, pd.DataFrame(index_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = load_audit()
    beta = load_beta_intervals()
    selections = select_examples(audit)
    index_parts: list[pd.DataFrame] = []
    gallery_rows: list[dict[str, object]] = []
    for name, examples in selections.items():
        path, index = build_gallery(name, examples, beta)
        gallery_rows.append(
            {
                "gallery": name,
                "n_days": len(examples),
                "html_file": path.name,
                "description": name.replace("_", " "),
            }
        )
        if not index.empty:
            index["html_file"] = path.name
            index_parts.append(index)

    gallery_index = pd.DataFrame(gallery_rows)
    example_index = pd.concat(index_parts, ignore_index=True) if index_parts else pd.DataFrame()
    gallery_index.to_csv(OUT / "01_c11_gallery_index.csv", index=False)
    example_index.to_csv(OUT / "02_c11_error_example_index.csv", index=False)
    print(f"Wrote visual error galleries to {OUT.relative_to(ROOT)}")
    print(gallery_index.to_string(index=False))


if __name__ == "__main__":
    main()

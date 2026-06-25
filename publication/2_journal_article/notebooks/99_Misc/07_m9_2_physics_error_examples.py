from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


ARTICLE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "07_m9_2_physics_counterfactual_ranker"
SOURCE_CSV_DIR = OUTPUT_ROOT / "csv"
EXAMPLE_ROOT = OUTPUT_ROOT / "error_examples"
CSV_DIR = EXAMPLE_ROOT / "csv"
HTML_DIR = EXAMPLE_ROOT / "html"
CSV_DIR.mkdir(parents=True, exist_ok=True)
HTML_DIR.mkdir(parents=True, exist_ok=True)

JCOL = {
    "orange": "#eb932c",
    "dark_blue": "#22303d",
    "grey": "#2F4D67",
    "light_grey": "#5C7D99",
}

EXAMPLES_PER_GROUP_SITE = 2


def naive_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None)


def read_decoded(dataset: str) -> pd.DataFrame:
    name = "full_loso_03_alpha_decoded_days.csv" if dataset == "alpha" else "full_loso_07_beta_decoded_days.csv"
    df = pd.read_csv(SOURCE_CSV_DIR / name)
    df["dataset"] = dataset
    df["true_label_day"] = df["true_label_day"].astype(bool)
    df["pred_label_day"] = df["pred_label_day"].astype(bool)
    df["pred_start"] = pd.to_datetime(df["pred_start"], errors="coerce")
    df["pred_end"] = pd.to_datetime(df["pred_end"], errors="coerce")
    df["confusion"] = "TN"
    df.loc[df["true_label_day"] & df["pred_label_day"], "confusion"] = "TP"
    df.loc[df["true_label_day"] & ~df["pred_label_day"], "confusion"] = "FN"
    df.loc[~df["true_label_day"] & df["pred_label_day"], "confusion"] = "FP"
    return df


def select_examples(decoded: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, site, confusion), group in decoded.groupby(["dataset", "substation_id", "confusion"], sort=True):
        if confusion not in {"FP", "FN"}:
            continue
        if confusion == "FP":
            selected = group.sort_values(["score_margin", "selected_iou", "date"], ascending=[False, False, True])
        else:
            selected = group.sort_values(["score_margin", "selected_iou", "date"], ascending=[False, False, True])
        rows.append(selected.head(EXAMPLES_PER_GROUP_SITE))
    if not rows:
        return decoded.head(0).copy()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["dataset", "substation_id", "confusion", "date"]).reset_index(drop=True)


def load_day(dataset: str, site: str, date: str) -> pd.DataFrame:
    path = ARTICLE_ROOT / "dataset" / "final" / f"dataset_{dataset}.parquet"
    df = pd.read_parquet(path)
    df = df.loc[df["substation_id"].astype(str).eq(site) & df["date"].astype(str).eq(str(date))].copy()
    df["timestamp"] = naive_timestamp(df["timestamp"])
    df = df.sort_values("timestamp")
    return df


def true_window(day: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    labelled = day.loc[day["label_interval"].astype(bool)]
    if labelled.empty:
        return None, None
    return labelled["timestamp"].iloc[0], labelled["timestamp"].iloc[-1]


def write_example(row: pd.Series) -> str:
    day = load_day(row["dataset"], row["substation_id"], row["date"])
    if day.empty:
        return ""
    true_start, true_end = true_window(day)
    day["u_empty"] = day["solar_MW"] + day["net_load_MW"]
    day["u_pred"] = day["u_empty"]
    if bool(row["pred_label_day"]) and pd.notna(row["pred_start"]) and pd.notna(row["pred_end"]):
        mask = (day["timestamp"] >= row["pred_start"]) & (day["timestamp"] <= row["pred_end"])
        day.loc[mask, "u_pred"] = day.loc[mask, "solar_MW"] - day.loc[mask, "net_load_MW"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=day["timestamp"], y=day["net_load_MW"], mode="lines", name="Raw net load", line=dict(color=JCOL["dark_blue"], width=2)))
    fig.add_trace(go.Scatter(x=day["timestamp"], y=day["solar_MW"], mode="lines", name="Solar", line=dict(color=JCOL["orange"], width=2)))
    fig.add_trace(go.Scatter(x=day["timestamp"], y=day["u_empty"], mode="lines", name="U_empty = solar + net", line=dict(color=JCOL["light_grey"], width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=day["timestamp"], y=day["u_pred"], mode="lines", name="Predicted U_W", line=dict(color=JCOL["grey"], width=2, dash="dash")))

    if true_start is not None:
        fig.add_vrect(x0=true_start, x1=true_end, fillcolor="rgba(92,125,153,0.22)", line_width=0, annotation_text="Manual", annotation_position="top left")
    if bool(row["pred_label_day"]) and pd.notna(row["pred_start"]) and pd.notna(row["pred_end"]):
        fig.add_vrect(x0=row["pred_start"], x1=row["pred_end"], fillcolor="rgba(235,147,44,0.22)", line_width=1, line_color=JCOL["orange"], annotation_text="m9.2", annotation_position="top right")

    title = (
        f"{row['dataset']} {row['substation_id']} {row['date']} {row['confusion']} | "
        f"margin={row['score_margin']:.3f}, best IoU={row['selected_iou']:.3f}"
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1000,
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Timestamp",
        yaxis_title="MW",
    )
    out_name = f"{row['dataset']}_{row['substation_id']}_{row['date']}_{row['confusion']}.html"
    out_path = HTML_DIR / out_name
    fig.write_html(out_path)
    return str(out_path.relative_to(EXAMPLE_ROOT))


def main() -> None:
    for old in HTML_DIR.glob("*.html"):
        old.unlink()

    decoded = pd.concat([read_decoded("alpha"), read_decoded("beta")], ignore_index=True)
    counts = (
        decoded.groupby(["dataset", "substation_id", "confusion"], as_index=False)
        .size()
        .pivot_table(index=["dataset", "substation_id"], columns="confusion", values="size", fill_value=0)
        .reset_index()
    )
    counts.to_csv(CSV_DIR / "01_error_counts_by_site.csv", index=False)

    examples = select_examples(decoded)
    html_files = []
    for _, row in examples.iterrows():
        html_files.append(write_example(row))
    examples["html_file"] = html_files
    examples.to_csv(CSV_DIR / "02_selected_fp_fn_examples.csv", index=False)
    print(f"Wrote {len(examples)} example rows")
    print(f"CSV: {CSV_DIR / '02_selected_fp_fn_examples.csv'}")
    print(f"HTML: {HTML_DIR}")


if __name__ == "__main__":
    main()

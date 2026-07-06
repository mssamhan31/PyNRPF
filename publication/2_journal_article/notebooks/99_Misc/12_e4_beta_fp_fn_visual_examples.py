from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


EXAMPLES_PER_GROUP = 9
MODEL_EXPERIMENT = "E4_v03_three_feature"
PLOT_START_SLOT = 24
PLOT_END_SLOT = 72
PLOT_WIDTH = 1000
PLOT_HEIGHT_PER_DAY = 450
EXAMPLE_MODE = os.environ.get("E4_EXAMPLE_MODE", "top_examples").strip().lower()
IS_SITE_GROUP_MODE = EXAMPLE_MODE in {"site_group_all", "by_site_group", "site_group"}
BETA_SUBSET = os.environ.get("E4_EXAMPLE_SUBSET", "sure_only" if IS_SITE_GROUP_MODE else "all").strip().lower()
IS_SURE_ONLY = BETA_SUBSET in {"sure", "sure_only", "beta_sure_only"}
if IS_SITE_GROUP_MODE:
    OUTPUT_FOLDER_NAME = (
        "12_e4_beta_fp_fn_visual_examples_by_site_group_sure_only"
        if IS_SURE_ONLY
        else "12_e4_beta_fp_fn_visual_examples_by_site_group"
    )
else:
    OUTPUT_FOLDER_NAME = (
        "12_e4_beta_fp_fn_visual_examples_sure_only"
        if IS_SURE_ONLY
        else "12_e4_beta_fp_fn_visual_examples"
    )


def find_repo_root() -> Path:
    start = Path(__file__).resolve()
    marker = Path("publication/2_journal_article/dataset/final/dataset_beta.parquet")
    for candidate in [start.parent, *start.parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"Could not find repo root containing {marker}")


ROOT = find_repo_root()
JOURNAL = ROOT / "publication/2_journal_article"
MISC_DIR = JOURNAL / "notebooks/99_Misc"
LADDER_SCRIPT = MISC_DIR / "11_minimal_bridge_method_ladder.py"
LADDER_OUTPUT = MISC_DIR / "outputs/11_minimal_bridge_method_ladder"
OUT = MISC_DIR / "outputs" / OUTPUT_FOLDER_NAME
REVIEWER_B_PATH = JOURNAL / "dataset/oracle_data_creation/archive/2026-07-02_reviewer_B_final/reviewer_B.csv"


def load_ladder_module():
    spec = importlib.util.spec_from_file_location("minimal_bridge_ladder", LADDER_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {LADDER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ladder = load_ladder_module()


def reset_output_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def date_key(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.strftime("%Y-%m-%d")


def load_beta_confidence() -> pd.DataFrame:
    confidence = pd.read_csv(REVIEWER_B_PATH, usecols=["substation_id", "date", "confidence"])
    confidence["substation_id"] = confidence["substation_id"].astype(str).str.replace("^act_", "beta_", regex=True)
    confidence["date"] = date_key(confidence["date"])
    confidence["confidence"] = confidence["confidence"].fillna("missing").astype(str).str.strip().str.lower()
    return confidence.drop_duplicates(["substation_id", "date"], keep="last")


def hhmm_from_slot(slot: int) -> str:
    slot = int(slot)
    hour = slot // 4
    minute = (slot % 4) * 15
    return f"{hour:02d}:{minute:02d}"


def selected_e4_window_components(net: np.ndarray, solar: np.ndarray) -> dict[str, float | int | str]:
    """Return selected E4 window and component scores for one site-day."""
    peak = int(np.argmax(solar[ladder.DAYTIME_START : ladder.DAYTIME_END + 1])) + ladder.DAYTIME_START
    left, right = ladder.CANDIDATE_CACHE[peak]
    up = solar + net
    um = solar - net

    bup = ladder.bridge_mse(up, left, right, up)
    bum = ladder.bridge_mse(um, left, right, up)
    bridge = (bup - bum) / (bup + bum + ladder.EPS)

    up_diff_abs = np.abs(np.diff(up))
    base_tv = up_diff_abs[ladder.DAYTIME_START : ladder.DAYTIME_END].sum()
    ctv_no = np.r_[0.0, np.cumsum(up_diff_abs)]
    internal_no = ctv_no[right] - ctv_no[left]
    um_diff_abs = np.abs(np.diff(um))
    ctv_um = np.r_[0.0, np.cumsum(um_diff_abs)]
    internal_corr = ctv_um[right] - ctv_um[left]
    left_jump_no = np.where(left > ladder.DAYTIME_START, np.abs(up[left] - up[left - 1]), 0)
    left_jump_corr = np.where(left > ladder.DAYTIME_START, np.abs(um[left] - up[left - 1]), 0)
    right_jump_no = np.where(right < ladder.DAYTIME_END, np.abs(up[right + 1] - up[right]), 0)
    right_jump_corr = np.where(right < ladder.DAYTIME_END, np.abs(up[right + 1] - um[right]), 0)
    corr_tv = base_tv - (internal_no + left_jump_no + right_jump_no) + (internal_corr + left_jump_corr + right_jump_corr)
    roughness = (base_tv - corr_tv) / (base_tv + corr_tv + ladder.EPS)

    inside = (ladder.SLOTS[None, :] >= left[:, None]) & (ladder.SLOTS[None, :] <= right[:, None])
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

    up_diff_matrix = np.repeat(up_diff, len(left), axis=0)
    no_left_before = ladder.median_diffs(up_diff_matrix, left_before_t, left_before_valid)
    no_left_after = ladder.median_diffs(up_diff_matrix, left_after_t, left_after_valid)
    no_right_before = ladder.median_diffs(up_diff_matrix, right_before_t, right_before_valid)
    no_right_after = ladder.median_diffs(up_diff_matrix, right_after_t, right_after_valid)
    corr_left_before = ladder.median_diffs(ucorr_diff, left_before_t, left_before_valid)
    corr_left_after = ladder.median_diffs(ucorr_diff, left_after_t, left_after_valid)
    corr_right_before = ladder.median_diffs(ucorr_diff, right_before_t, right_before_valid)
    corr_right_after = ladder.median_diffs(ucorr_diff, right_after_t, right_after_valid)

    slope_no = np.abs(no_left_before - no_left_after) + np.abs(no_right_before - no_right_after)
    slope_corr = np.abs(corr_left_before - corr_left_after) + np.abs(corr_right_before - corr_right_after)
    slope = (slope_no - slope_corr) / (slope_no + slope_corr + ladder.EPS)

    total = bridge + roughness + slope
    idx = int(np.nanargmax(total))
    return {
        "selected_left_slot": int(left[idx]),
        "selected_right_slot": int(right[idx]),
        "selected_start_time": hhmm_from_slot(int(left[idx])),
        "selected_end_time_exclusive": hhmm_from_slot(int(right[idx]) + 1),
        "solar_peak_slot": peak,
        "solar_peak_time": hhmm_from_slot(peak),
        "candidate_count": int(len(left)),
        "bridge_score": float(bridge[idx]),
        "roughness_score": float(roughness[idx]),
        "slope_continuity_score": float(slope[idx]),
        "total_score": float(total[idx]),
    }


def selected_e4_window_components_from_row(row: pd.Series, net: np.ndarray, solar: np.ndarray) -> dict[str, float | int | str]:
    required = [
        "v03_selected_left_slot",
        "v03_selected_right_slot",
        "v03_bridge_best",
        "v03_roughness_best",
        "v03_slope_continuity_best",
        "E4_v03_three_feature_score",
    ]
    if not all(name in row.index for name in required):
        return selected_e4_window_components(net, solar)
    if not all(pd.notna(row[name]) for name in required):
        return selected_e4_window_components(net, solar)

    left = int(row["v03_selected_left_slot"])
    right = int(row["v03_selected_right_slot"])
    solar_peak = int(np.argmax(solar[ladder.DAYTIME_START : ladder.DAYTIME_END + 1])) + ladder.DAYTIME_START
    return {
        "selected_left_slot": left,
        "selected_right_slot": right,
        "selected_start_time": hhmm_from_slot(left),
        "selected_end_time_exclusive": hhmm_from_slot(right + 1),
        "solar_peak_slot": solar_peak,
        "solar_peak_time": hhmm_from_slot(solar_peak),
        "candidate_count": int(row["v03_candidate_count"]) if "v03_candidate_count" in row.index and pd.notna(row["v03_candidate_count"]) else int(len(ladder.CANDIDATE_CACHE[solar_peak][0])),
        "bridge_score": float(row["v03_bridge_best"]),
        "roughness_score": float(row["v03_roughness_best"]),
        "slope_continuity_score": float(row["v03_slope_continuity_best"]),
        "total_score": float(row["E4_v03_three_feature_score"]),
    }


def load_beta_scored_days(threshold: float) -> pd.DataFrame:
    joined = pd.read_csv(LADDER_OUTPUT / "03_joined_daily_scores.csv")
    joined = joined.loc[joined["dataset"].eq("beta")].copy()
    joined["date"] = date_key(joined["date"])
    if "confidence" in joined.columns:
        joined["confidence"] = joined["confidence"].fillna("missing").astype(str).str.strip().str.lower()
    else:
        joined["confidence"] = "missing"
    if IS_SURE_ONLY:
        if joined["confidence"].eq("missing").all():
            confidence = load_beta_confidence()
            joined = joined.drop(columns=["confidence"]).merge(confidence, on=["substation_id", "date"], how="left")
            joined["confidence"] = joined["confidence"].fillna("missing").astype(str).str.strip().str.lower()
        joined = joined.loc[joined["confidence"].eq("sure")].copy()
    joined["true_day"] = bool_series(joined["true_day"])
    joined[f"{MODEL_EXPERIMENT}_pred_day"] = bool_series(joined[f"{MODEL_EXPERIMENT}_pred_day"])
    joined["score_margin"] = joined[f"{MODEL_EXPERIMENT}_score"] - threshold
    joined["beta_subset"] = "sure_only" if IS_SURE_ONLY else "all_days"
    return joined


def add_confusion_labels(scored: pd.DataFrame) -> pd.DataFrame:
    fp = scored.loc[(~scored["true_day"]) & scored[f"{MODEL_EXPERIMENT}_pred_day"]].copy()
    fn = scored.loc[scored["true_day"] & (~scored[f"{MODEL_EXPERIMENT}_pred_day"])].copy()
    fp["confusion_group"] = "FP"
    fn["confusion_group"] = "FN"
    fp["selection_reason"] = "false_positive"
    fn["selection_reason"] = "false_negative"
    return pd.concat([fp, fn], ignore_index=True)


def load_selected_examples(threshold: float) -> pd.DataFrame:
    mistakes = add_confusion_labels(load_beta_scored_days(threshold))
    fp = mistakes.loc[mistakes["confusion_group"].eq("FP")].copy()
    fn = mistakes.loc[mistakes["confusion_group"].eq("FN")].copy()
    fp = fp.sort_values("score_margin", ascending=False).head(EXAMPLES_PER_GROUP)
    fn = fn.sort_values("score_margin", ascending=True).head(EXAMPLES_PER_GROUP)
    fp["selection_reason"] = "largest_positive_margin_false_positive"
    fn["selection_reason"] = "most_negative_margin_false_negative"
    examples = pd.concat([fp, fn], ignore_index=True)
    examples["example_rank"] = examples.groupby("confusion_group").cumcount() + 1
    return examples


def load_all_mistakes(threshold: float) -> pd.DataFrame:
    examples = add_confusion_labels(load_beta_scored_days(threshold))
    examples = examples.sort_values(["substation_id", "confusion_group", "date"]).reset_index(drop=True)
    examples["example_rank"] = examples.groupby(["substation_id", "confusion_group"]).cumcount() + 1
    return examples


def load_beta_days(examples: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    df = pd.read_parquet(
        JOURNAL / "dataset/final/dataset_beta.parquet",
        columns=["substation_id", "date", "timestamp", "net_load_MW", "solar_MW", "label_interval"],
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    selected_keys = examples[["substation_id", "date"]].drop_duplicates()
    df = df.merge(selected_keys, on=["substation_id", "date"], how="inner")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    return {key: group.sort_values("timestamp").copy() for key, group in df.groupby(["substation_id", "date"], sort=False)}


def day_arrays(day: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    day = day.copy()
    day["slot"] = day["timestamp"].dt.hour * 4 + (day["timestamp"].dt.minute // 15)
    net = np.full(ladder.SLOTS_PER_DAY, np.nan)
    solar = np.full(ladder.SLOTS_PER_DAY, np.nan)
    label = np.zeros(ladder.SLOTS_PER_DAY, dtype=bool)
    timestamp = pd.Series(pd.NaT, index=np.arange(ladder.SLOTS_PER_DAY), dtype="datetime64[ns]")
    for _, row in day.drop_duplicates("slot", keep="last").iterrows():
        slot = int(row["slot"])
        if 0 <= slot < ladder.SLOTS_PER_DAY:
            net[slot] = row["net_load_MW"]
            solar[slot] = row["solar_MW"]
            label[slot] = bool(row["label_interval"])
            timestamp.iloc[slot] = row["timestamp"]
    base_date = pd.to_datetime(day["date"].iloc[0])
    fallback_times = pd.Series([base_date + pd.Timedelta(minutes=15 * i) for i in range(ladder.SLOTS_PER_DAY)])
    timestamp = timestamp.fillna(fallback_times)
    return ladder.fill_series(net, 0.0), np.maximum(ladder.fill_series(solar, 0.0), 0.0), label, timestamp


def add_window_shape(fig: go.Figure, row: int, x0, x1, color: str, label: str) -> None:
    fig.add_vrect(
        x0=x0,
        x1=x1,
        row=row,
        col=1,
        fillcolor=color,
        opacity=0.18,
        line_width=0,
        annotation_text=label,
        annotation_position="top left",
    )


def make_plot(
    examples: pd.DataFrame,
    day_map: dict[tuple[str, str], pd.DataFrame],
    threshold: float,
    html_path: Path,
    title_context: str,
) -> pd.DataFrame:
    prepared = []
    rows = []

    for _, ex in examples.iterrows():
        key = (ex["substation_id"], ex["date"])
        day = day_map[key]
        net, solar, label, timestamp = day_arrays(day)
        detail = selected_e4_window_components_from_row(ex, net, solar)
        prepared.append((ex, day, net, solar, label, timestamp, detail))

    fig = make_subplots(
        rows=len(prepared),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=min(0.018, 0.8 / max(len(prepared) - 1, 1)),
        subplot_titles=[
            (
                f"{ex.substation_id} {ex.date} | {ex.confusion_group} | "
                f"B={detail['bridge_score']:.3f}, Rough={detail['roughness_score']:.3f}, "
                f"Slope={detail['slope_continuity_score']:.3f}, Total={detail['total_score']:.3f}, "
                f"Thr={threshold:.3f} | window {detail['selected_start_time']}-{detail['selected_end_time_exclusive']}"
            )
            for ex, _, _, _, _, _, detail in prepared
        ],
    )

    for plot_row, (ex, _day, net, solar, label, timestamp, detail) in enumerate(prepared, start=1):
        left = int(detail["selected_left_slot"])
        right = int(detail["selected_right_slot"])
        corrected = net.copy()
        corrected[left : right + 1] = -corrected[left : right + 1]

        selected_x0 = timestamp.iloc[left]
        selected_x1 = timestamp.iloc[right] + pd.Timedelta(minutes=15)
        selected_label = "E4 selected candidate below threshold" if ex["confusion_group"] == "FN" else "E4 predicted RPF"
        add_window_shape(fig, plot_row, selected_x0, selected_x1, "#eb932c", selected_label)

        true_slots = np.flatnonzero(label)
        true_start = ""
        true_end = ""
        if len(true_slots):
            true_x0 = timestamp.iloc[int(true_slots[0])]
            true_x1 = timestamp.iloc[int(true_slots[-1])] + pd.Timedelta(minutes=15)
            true_start = hhmm_from_slot(int(true_slots[0]))
            true_end = hhmm_from_slot(int(true_slots[-1]) + 1)
            add_window_shape(fig, plot_row, true_x0, true_x1, "#5C7D99", "actual/manual RPF")

        showlegend = plot_row == 1
        plot_slice = slice(PLOT_START_SLOT, PLOT_END_SLOT + 1)
        hover_meta = np.column_stack(
            [
                np.repeat(float(detail["bridge_score"]), len(timestamp)),
                np.repeat(float(detail["roughness_score"]), len(timestamp)),
                np.repeat(float(detail["slope_continuity_score"]), len(timestamp)),
                np.repeat(float(detail["total_score"]), len(timestamp)),
                np.repeat(float(threshold), len(timestamp)),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=timestamp.iloc[plot_slice],
                y=net[plot_slice],
                mode="lines",
                name="Raw net load",
                line=dict(color="#22303d", width=1.6),
                customdata=hover_meta[plot_slice],
                hovertemplate=(
                    "Raw net load: %{y:.3f} MW<br>%{x|%H:%M}"
                    "<br>bridge=%{customdata[0]:.3f}"
                    "<br>roughness=%{customdata[1]:.3f}"
                    "<br>slope=%{customdata[2]:.3f}"
                    "<br>total=%{customdata[3]:.3f}"
                    "<br>threshold=%{customdata[4]:.3f}<extra></extra>"
                ),
                showlegend=showlegend,
            ),
            row=plot_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=timestamp.iloc[plot_slice],
                y=solar[plot_slice],
                mode="lines",
                name="Solar",
                line=dict(color="#eb932c", width=1.4),
                hovertemplate="Solar: %{y:.3f} MW<br>%{x|%H:%M}<extra></extra>",
                showlegend=showlegend,
            ),
            row=plot_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=timestamp.iloc[plot_slice],
                y=corrected[plot_slice],
                mode="lines",
                name="E4 sign-flipped net load",
                line=dict(color="#2F4D67", width=1.1, dash="dot"),
                hovertemplate="E4 sign-flipped net load: %{y:.3f} MW<br>%{x|%H:%M}<extra></extra>",
                showlegend=showlegend,
            ),
            row=plot_row,
            col=1,
        )
        fig.add_hline(y=0, row=plot_row, col=1, line_width=0.8, line_dash="dash", line_color="#5C7D99")
        fig.update_xaxes(range=[timestamp.iloc[PLOT_START_SLOT], timestamp.iloc[PLOT_END_SLOT]], row=plot_row, col=1)

        rows.append(
            {
                **ex.to_dict(),
                **detail,
                "threshold": threshold,
                "score_margin_recomputed": float(detail["total_score"] - threshold),
                "manual_start_time": true_start,
                "manual_end_time_exclusive": true_end,
            }
        )

    fig.update_layout(
        title=(
            f"E4 Beta FP/FN Examples ({title_context}): "
            "06:00-18:00 raw net load, solar, selected E4 window, and manual label window"
            "<br><sup>Orange shading = E4 selected/predicted window; blue-grey shading = actual/manual RPF window. "
            "Panel titles and raw-net-load hover show bridge/roughness/slope scores.</sup>"
        ),
        height=max(800, PLOT_HEIGHT_PER_DAY * len(examples)),
        width=PLOT_WIDTH,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        font=dict(family="Arial", size=11),
    )
    fig.update_yaxes(title_text="MW")
    fig.write_html(html_path, include_plotlyjs="cdn")
    return pd.DataFrame(rows)


def run_top_examples(threshold: float) -> None:
    examples = load_selected_examples(threshold)
    if examples.empty:
        raise RuntimeError("No FP/FN examples were found after applying the requested subset filter.")
    day_map = load_beta_days(examples)
    title_context = "sure-only days" if IS_SURE_ONLY else "all days"
    details = make_plot(examples, day_map, threshold, OUT / "e4_beta_fp_fn_examples.html", title_context)
    details.to_csv(OUT / "01_e4_beta_fp_fn_example_index.csv", index=False)
    print(f"Wrote {len(details)} examples to {OUT.relative_to(ROOT)}")
    print(f"Subset: {'sure_only' if IS_SURE_ONLY else 'all_days'}")
    print(
        details[
            [
                "confusion_group",
                "example_rank",
                "substation_id",
                "date",
                "bridge_score",
                "roughness_score",
                "slope_continuity_score",
                "total_score",
                "threshold",
                "selected_start_time",
                "selected_end_time_exclusive",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


def run_site_group_all(threshold: float) -> None:
    examples = load_all_mistakes(threshold)
    if examples.empty:
        raise RuntimeError("No FP/FN examples were found after applying the requested subset filter.")

    day_map = load_beta_days(examples)
    index_rows = []
    detail_frames = []
    subset_label = "sure_only" if IS_SURE_ONLY else "all_days"
    title_subset = "sure-only days" if IS_SURE_ONLY else "all days"

    for (site, group), group_examples in examples.groupby(["substation_id", "confusion_group"], sort=True):
        group_examples = group_examples.sort_values("date").reset_index(drop=True)
        n_days = len(group_examples)
        html_name = f"E4_{site}_{group}_{n_days}days.html"
        html_path = OUT / html_name
        first_date = str(group_examples["date"].min())
        last_date = str(group_examples["date"].max())
        title_context = (
            f"{title_subset}; {site} {group}; {n_days} days; "
            f"{first_date} to {last_date}; threshold={threshold:.3f}"
        )
        details = make_plot(group_examples, day_map, threshold, html_path, title_context)
        details["html_file"] = html_name
        detail_frames.append(details)
        index_rows.append(
            {
                "site": site,
                "confusion_group": group,
                "n_days": n_days,
                "first_date": first_date,
                "last_date": last_date,
                "html_file": html_name,
                "threshold": threshold,
                "subset": subset_label,
            }
        )

    index = pd.DataFrame(index_rows)
    all_details = pd.concat(detail_frames, ignore_index=True)
    index.to_csv(OUT / "01_e4_beta_fp_fn_by_site_group_index.csv", index=False)
    all_details.to_csv(OUT / "02_e4_beta_fp_fn_all_examples.csv", index=False)
    print(f"Wrote {len(index)} HTML files and {len(all_details)} examples to {OUT.relative_to(ROOT)}")
    print(f"Subset: {subset_label}")
    print(index.to_string(index=False))


def main() -> None:
    reset_output_folder(OUT)
    threshold_row = pd.read_csv(LADDER_OUTPUT / "01_threshold_selection.csv")
    threshold = float(threshold_row.loc[threshold_row["experiment"].eq(MODEL_EXPERIMENT), "threshold"].iloc[0])
    if IS_SITE_GROUP_MODE:
        run_site_group_all(threshold)
    else:
        run_top_examples(threshold)


if __name__ == "__main__":
    main()

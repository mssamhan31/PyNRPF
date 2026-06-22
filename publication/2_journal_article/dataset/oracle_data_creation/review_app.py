from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

APP_DIR = Path(__file__).resolve().parent
CORE_PATH = APP_DIR / "oracle_review_core.py"
REQUIRED_CORE_NAMES = [
    "ACTION_ACCEPT_OLD",
    "ACTION_MANUAL_WINDOW",
    "ACTION_NO_RPF",
    "default_review_window",
    "review_control_defaults",
    "apply_annotation_batch",
    "with_reviewed_preview_labels",
    "review_preview_summary",
]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

stale_core = sys.modules.get("oracle_review_core")
if stale_core is not None:
    stale_path = Path(getattr(stale_core, "__file__", "")).resolve()
    missing_core_names = [
        required_name
        for required_name in REQUIRED_CORE_NAMES
        if not hasattr(stale_core, required_name)
    ]
    if stale_path != CORE_PATH.resolve() or missing_core_names:
        del sys.modules["oracle_review_core"]

import oracle_review_core as core

loaded_core_path = Path(getattr(core, "__file__", "")).resolve()
if loaded_core_path != CORE_PATH.resolve():
    raise RuntimeError(f"Loaded oracle_review_core from {loaded_core_path}, expected {CORE_PATH}.")
for required_name in REQUIRED_CORE_NAMES:
    if not hasattr(core, required_name):
        raise RuntimeError(
            f"Loaded stale oracle_review_core from {loaded_core_path}; missing {required_name}."
        )


st.set_page_config(page_title="Oracle RPF Review", layout="wide")


ACTION_LABELS = {
    core.ACTION_ACCEPT_OLD: "Accept old labels",
    core.ACTION_MANUAL_WINDOW: "Manual RPF window",
    core.ACTION_NO_RPF: "No RPF",
}
FLAG_VIEW_OPTIONS = ["Original flags", "Reviewed flags"]
FLAG_VIEW_COLUMNS = {
    "Original flags": "label_interval",
    "Reviewed flags": "new_label_interval",
}
FLAG_VIEW_TRACE_LABELS = {
    "Original flags": "Original",
    "Reviewed flags": "Reviewed",
}


@st.cache_data(show_spinner="Loading actual oracle dataset...")
def load_review_dataframe(input_path: str) -> pd.DataFrame:
    source = core.load_source_dataset(Path(input_path))
    scoped = core.filter_review_scope(source)
    core.assert_expected_review_scope(scoped)
    return scoped


def ensure_annotation_file() -> None:
    path = core.default_annotation_path()
    if not path.exists():
        core.write_annotations(core.empty_annotations(), path)


def load_annotations() -> pd.DataFrame:
    ensure_annotation_file()
    return core.read_annotations(core.default_annotation_path())


def select_site_date(site: str, date: str) -> None:
    st.session_state["selected_site"] = site
    st.session_state["selected_date"] = date


def save_annotations(
    df: pd.DataFrame,
    message: str,
    next_selection: tuple[str, str] | None = None,
) -> None:
    core.write_annotations(df, core.default_annotation_path())
    st.session_state["annotation_save_version"] = (
        st.session_state.get("annotation_save_version", 0) + 1
    )
    if next_selection is not None:
        select_site_date(*next_selection)
    st.session_state["flash"] = message
    st.rerun()


def current_queue_row(queue: pd.DataFrame, site: str, date: str) -> pd.Series | None:
    match = queue[(queue["substation_id"] == site) & (queue["date"] == date)]
    if match.empty:
        return None
    return match.iloc[0]


def current_week_start() -> str:
    return core.review_week_start([st.session_state["selected_date"]]).iloc[0]


def current_week_row(week_queue: pd.DataFrame, site: str, week_start: str) -> pd.Series | None:
    match = week_queue[
        (week_queue["substation_id"] == site) & (week_queue["week_start"] == week_start)
    ]
    if match.empty:
        return None
    return match.iloc[0]


def initialize_selection(queue: pd.DataFrame) -> None:
    if "selected_site" in st.session_state and "selected_date" in st.session_state:
        return
    unreviewed = queue.loc[~queue["reviewed"]]
    first = (unreviewed if not unreviewed.empty else queue).iloc[0]
    select_site_date(first["substation_id"], first["date"])


def move_by_calendar_day(queue: pd.DataFrame, direction: int) -> None:
    site = st.session_state["selected_site"]
    dates = sorted(queue.loc[queue["substation_id"] == site, "date"].tolist())
    current = st.session_state["selected_date"]
    idx = dates.index(current)
    next_idx = min(max(idx + direction, 0), len(dates) - 1)
    select_site_date(site, dates[next_idx])
    st.rerun()


def move_by_queue(queue: pd.DataFrame, direction: int, unreviewed_only: bool = False) -> None:
    work = queue.loc[~queue["reviewed"]].copy() if unreviewed_only else queue
    if work.empty:
        return
    site = st.session_state["selected_site"]
    date = st.session_state["selected_date"]
    matches = work[(work["substation_id"] == site) & (work["date"] == date)]
    if matches.empty:
        row = work.iloc[0]
    else:
        pos = work.index.get_loc(matches.index[0])
        if not isinstance(pos, int):
            pos = int(pos[0])
        next_pos = min(max(pos + direction, 0), len(work) - 1)
        row = work.iloc[next_pos]
    select_site_date(row["substation_id"], row["date"])
    st.rerun()


def week_selection(
    week_queue: pd.DataFrame,
    site: str,
    week_start: str,
    direction: int,
    unreviewed_only: bool = False,
) -> tuple[str, str] | None:
    work = week_queue.loc[week_queue["substation_id"] == site].copy()
    if unreviewed_only:
        work = work.loc[~work["reviewed"]]
    if work.empty:
        return None

    matches = work[work["week_start"] == week_start]
    if matches.empty:
        row = work.iloc[0]
    else:
        pos = work.index.get_loc(matches.index[0])
        if not isinstance(pos, int):
            pos = int(pos[0])
        next_pos = min(max(pos + direction, 0), len(work) - 1)
        row = work.iloc[next_pos]
    return row["substation_id"], row["first_date"]


def move_by_week(week_queue: pd.DataFrame, direction: int, unreviewed_only: bool = False) -> None:
    selection = week_selection(
        week_queue,
        st.session_state["selected_site"],
        current_week_start(),
        direction,
        unreviewed_only=unreviewed_only,
    )
    if selection is None:
        return
    select_site_date(*selection)
    st.rerun()


def build_daily_figure(day_df: pd.DataFrame, title: str, flag_view: str) -> go.Figure:
    flag_column = FLAG_VIEW_COLUMNS[flag_view]
    trace_label = FLAG_VIEW_TRACE_LABELS[flag_view]
    corrected = core.corrected_net_load(day_df, flag_column=flag_column)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=day_df["_timestamp_dt"],
            y=day_df["net_load_MW"],
            mode="lines+markers",
            name="Raw net load",
            line={"color": "#2563eb", "width": 2},
            marker={"size": 4},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=day_df["_timestamp_dt"],
            y=corrected,
            mode="lines",
            name=f"{trace_label}-flag corrected net load",
            line={"color": "#dc2626", "width": 2, "dash": "dash"},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=day_df["_timestamp_dt"],
            y=day_df["solar_MW"],
            mode="lines",
            name="Solar generation",
            line={"color": "#16a34a", "width": 2},
        ),
        secondary_y=True,
    )
    for start, end in core.flag_spans(day_df, flag_column=flag_column):
        fig.add_vrect(x0=start, x1=end, fillcolor="#f59e0b", opacity=0.18, line_width=0)
    fig.add_hline(y=0, line_dash="dot", line_color="#111827", opacity=0.75)
    fig.update_layout(
        title=title,
        height=520,
        hovermode="x unified",
        margin={"l": 30, "r": 30, "t": 60, "b": 30},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_yaxes(title_text="Net load MW", secondary_y=False)
    fig.update_yaxes(title_text="Solar MW", secondary_y=True)
    return fig


def build_weekly_figure(
    week_df: pd.DataFrame,
    selected_date: str,
    title: str,
    flag_view: str,
) -> go.Figure:
    flag_column = FLAG_VIEW_COLUMNS[flag_view]
    trace_label = FLAG_VIEW_TRACE_LABELS[flag_view]
    corrected = core.corrected_net_load(week_df, flag_column=flag_column)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=week_df["_timestamp_dt"],
            y=week_df["net_load_MW"],
            mode="lines",
            name="Raw net load",
            line={"color": "#2563eb", "width": 1.6},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=week_df["_timestamp_dt"],
            y=corrected,
            mode="lines",
            name=f"{trace_label}-flag corrected net load",
            line={"color": "#dc2626", "width": 1.5, "dash": "dash"},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=week_df["_timestamp_dt"],
            y=week_df["solar_MW"],
            mode="lines",
            name="Solar generation",
            line={"color": "#16a34a", "width": 1.6},
        ),
        secondary_y=True,
    )
    for start, end in core.flag_spans(week_df, flag_column=flag_column):
        fig.add_vrect(x0=start, x1=end, fillcolor="#f59e0b", opacity=0.16, line_width=0)

    for day in sorted(week_df["date"].unique()):
        day_start = pd.Timestamp(day)
        fig.add_vline(x=day_start, line_dash="dot", line_color="#94a3b8", opacity=0.35)

    day_start = pd.Timestamp(selected_date)
    fig.add_vrect(
        x0=day_start,
        x1=day_start + pd.Timedelta(days=1),
        fillcolor="#0f172a",
        opacity=0.06,
        line_width=0,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#111827", opacity=0.75)
    fig.update_layout(
        title=title,
        height=460,
        hovermode="x unified",
        margin={"l": 30, "r": 30, "t": 60, "b": 30},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_yaxes(title_text="Net load MW", secondary_y=False)
    fig.update_yaxes(title_text="Solar MW", secondary_y=True)
    return fig


def existing_annotation(
    annotations: pd.DataFrame, site: str, date: str
) -> pd.Series | None:
    row = annotations[(annotations["substation_id"] == site) & (annotations["date"] == date)]
    if row.empty:
        return None
    return row.iloc[0]


def annotation_status_text(row: pd.Series | None) -> str:
    if row is None:
        return "unreviewed"
    action = row["review_action"]
    if action == core.ACTION_MANUAL_WINDOW:
        return f"manual window {row['rpf_start_time']} to {row['rpf_end_time']}"
    return ACTION_LABELS[action].lower()


def review_save_message(was_reviewed: bool, site: str, date: str) -> str:
    verb = "Updated" if was_reviewed else "Saved"
    return f"{verb} review for {site} {date}."


def save_daily_review(
    annotations: pd.DataFrame,
    site: str,
    date: str,
    action: str,
    start: str,
    end: str,
    was_reviewed: bool,
) -> None:
    try:
        updated = core.upsert_annotation(annotations, site, date, action, start, end)
    except ValueError as exc:
        st.error(f"Could not save {site} {date}: {exc}")
        return
    save_annotations(updated, review_save_message(was_reviewed, site, date))


def render_annotation_controls(
    day_df: pd.DataFrame,
    annotations: pd.DataFrame,
    site: str,
    date: str,
) -> None:
    row = existing_annotation(annotations, site, date)
    was_reviewed = row is not None
    action_default, start_default, end_default = core.review_control_defaults(day_df, row)
    if row is None:
        st.warning("Manual review status: unreviewed")
    else:
        st.success(f"Manual review status: {annotation_status_text(row)}")

    widget_prefix = (
        f"{site}_{date}_{st.session_state.get('annotation_save_version', 0)}"
    )
    action_options = [
        core.ACTION_ACCEPT_OLD,
        core.ACTION_MANUAL_WINDOW,
        core.ACTION_NO_RPF,
    ]
    action = st.radio(
        "Review action",
        action_options,
        format_func=lambda value: ACTION_LABELS[value],
        horizontal=True,
        index=action_options.index(action_default),
        key=f"review_action_{widget_prefix}",
    )

    options = core.time_options_for_day(day_df)
    if action == core.ACTION_MANUAL_WINDOW:
        left, right = st.columns(2)
        start_index = options.index(start_default) if start_default in options else 0
        end_index = options.index(end_default) if end_default in options else len(options) - 1
        start = left.selectbox(
            "RPF start time",
            options,
            index=start_index,
            key=f"rpf_start_{widget_prefix}",
        )
        end = right.selectbox(
            "RPF end time",
            options,
            index=end_index,
            key=f"rpf_end_{widget_prefix}",
        )
    else:
        start = ""
        end = ""

    action_cols = st.columns([1, 1, 1, 1, 3])
    if action_cols[0].button("Save day", type="primary", key=f"save_day_{widget_prefix}"):
        save_daily_review(annotations, site, date, action, start, end, was_reviewed)
    if action_cols[1].button("Accept old", key=f"accept_old_{widget_prefix}"):
        save_daily_review(annotations, site, date, core.ACTION_ACCEPT_OLD, "", "", was_reviewed)
    if action_cols[2].button("No RPF", key=f"no_rpf_{widget_prefix}"):
        save_daily_review(annotations, site, date, core.ACTION_NO_RPF, "", "", was_reviewed)
    if action_cols[3].button("Clear", key=f"clear_day_{widget_prefix}"):
        updated = core.clear_annotation(annotations, site, date)
        save_annotations(updated, f"Cleared review for {site} {date}.")


def upsert_week_action(
    annotations: pd.DataFrame,
    site: str,
    week_dates: list[str],
    action: str,
) -> pd.DataFrame:
    updated = annotations
    for date in week_dates:
        updated = core.upsert_annotation(updated, site, date, action)
    return updated


def week_save_message(annotations: pd.DataFrame, site: str, week_dates: list[str]) -> str:
    reviewed_dates = set(
        annotations.loc[annotations["substation_id"] == site, "date"].astype(str).tolist()
    )
    existing_count = sum(date in reviewed_dates for date in week_dates)
    if existing_count:
        return (
            f"Updated {existing_count} existing review(s) and saved "
            f"{len(week_dates)} day(s) for {site}."
        )
    return f"Saved {len(week_dates)} review(s) for {site}."


def clear_week_action(
    annotations: pd.DataFrame,
    site: str,
    week_dates: list[str],
) -> pd.DataFrame:
    updated = annotations
    for date in week_dates:
        updated = core.clear_annotation(updated, site, date)
    return updated


def render_weekly_controls(
    annotations: pd.DataFrame,
    week_queue: pd.DataFrame,
    site: str,
    week_start: str,
    week_dates: list[str],
) -> None:
    next_selection = week_selection(week_queue, site, week_start, 1)
    cols = st.columns([1, 1, 1, 1, 1])

    if cols[0].button("Accept old week", type="primary"):
        updated = upsert_week_action(annotations, site, week_dates, core.ACTION_ACCEPT_OLD)
        save_annotations(updated, week_save_message(annotations, site, week_dates))
    if cols[1].button("Accept + next"):
        updated = upsert_week_action(annotations, site, week_dates, core.ACTION_ACCEPT_OLD)
        save_annotations(
            updated,
            week_save_message(annotations, site, week_dates),
            next_selection=next_selection,
        )
    if cols[2].button("No RPF week"):
        updated = upsert_week_action(annotations, site, week_dates, core.ACTION_NO_RPF)
        save_annotations(updated, week_save_message(annotations, site, week_dates))
    if cols[3].button("No RPF + next"):
        updated = upsert_week_action(annotations, site, week_dates, core.ACTION_NO_RPF)
        save_annotations(
            updated,
            week_save_message(annotations, site, week_dates),
            next_selection=next_selection,
        )
    if cols[4].button("Clear week"):
        updated = clear_week_action(annotations, site, week_dates)
        save_annotations(updated, f"Cleared reviews for {site} week {week_start}.")


def render_weekly_daily_editor(
    week_df: pd.DataFrame,
    annotations: pd.DataFrame,
    site: str,
    week_dates: list[str],
) -> None:
    st.markdown("#### Daily review")
    st.caption(
        "Edit the week freely; the app only saves and refreshes after "
        "Save all 7 days. Start/end are used only for Manual RPF window rows."
    )

    action_options = [
        core.ACTION_ACCEPT_OLD,
        core.ACTION_MANUAL_WINDOW,
        core.ACTION_NO_RPF,
    ]
    save_version = st.session_state.get("annotation_save_version", 0)
    form_key = f"weekly_daily_form_{site}_{week_dates[0]}_{save_version}"

    with st.form(form_key, clear_on_submit=False):
        header = st.columns([1.1, 2.0, 2.0, 1.25, 1.25, 0.9])
        for column, label in zip(
            header,
            ["Date", "Current review", "Action", "Start", "End", "Clear"],
            strict=True,
        ):
            column.markdown(f"**{label}**")

        updates: list[dict[str, object]] = []
        for row_date in week_dates:
            day_df = week_df[week_df["date"] == row_date].copy()
            annotation_row = existing_annotation(annotations, site, row_date)
            action_default, start_default, end_default = core.review_control_defaults(
                day_df,
                annotation_row,
            )
            options = core.time_options_for_day(day_df)
            start_index = options.index(start_default) if start_default in options else 0
            end_index = options.index(end_default) if end_default in options else len(options) - 1
            widget_prefix = f"weekly_{site}_{row_date}_{save_version}"

            cols = st.columns([1.1, 2.0, 2.0, 1.25, 1.25, 0.9])
            cols[0].write(row_date)
            cols[1].caption(annotation_status_text(annotation_row))
            action = cols[2].selectbox(
                f"Action for {row_date}",
                action_options,
                format_func=lambda value: ACTION_LABELS[value],
                index=action_options.index(action_default),
                key=f"weekly_action_{widget_prefix}",
                label_visibility="collapsed",
            )
            start = cols[3].selectbox(
                f"RPF start for {row_date}",
                options,
                index=start_index,
                key=f"weekly_start_{widget_prefix}",
                label_visibility="collapsed",
            )
            end = cols[4].selectbox(
                f"RPF end for {row_date}",
                options,
                index=end_index,
                key=f"weekly_end_{widget_prefix}",
                label_visibility="collapsed",
            )
            clear = cols[5].checkbox(
                f"Clear {row_date}",
                value=False,
                key=f"weekly_clear_{widget_prefix}",
                label_visibility="collapsed",
            )
            is_manual = action == core.ACTION_MANUAL_WINDOW
            updates.append(
                {
                    "substation_id": site,
                    "date": row_date,
                    "review_action": action,
                    "rpf_start_time": start if is_manual else "",
                    "rpf_end_time": end if is_manual else "",
                    "clear": clear,
                }
            )

        submitted = st.form_submit_button(
            "Save all 7 days",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            updated = core.apply_annotation_batch(annotations, updates)
        except ValueError as exc:
            st.error(f"Could not save weekly daily reviews: {exc}")
            return
        save_annotations(updated, f"Saved daily reviews for {site} week {week_dates[0]}.")


def render_sidebar(queue: pd.DataFrame, week_queue: pd.DataFrame) -> None:
    st.sidebar.header("Navigation")
    sites = [site for site in core.SITE_ORDER if site in set(queue["substation_id"])]
    site_index = sites.index(st.session_state["selected_site"])
    site = st.sidebar.selectbox("Substation", sites, index=site_index)
    if site != st.session_state["selected_site"]:
        first_date = queue.loc[queue["substation_id"] == site, "date"].iloc[0]
        select_site_date(site, first_date)
        st.rerun()

    dates = sorted(queue.loc[queue["substation_id"] == site, "date"].tolist())
    date_index = dates.index(st.session_state["selected_date"])
    date = st.sidebar.selectbox("Date", dates, index=date_index)
    if date != st.session_state["selected_date"]:
        select_site_date(site, date)
        st.rerun()

    prev_week, next_week = st.sidebar.columns(2)
    if prev_week.button("Previous week"):
        move_by_week(week_queue, -1)
    if next_week.button("Next week"):
        move_by_week(week_queue, 1)

    prev_day, next_day = st.sidebar.columns(2)
    if prev_day.button("Previous day"):
        move_by_calendar_day(queue, -1)
    if next_day.button("Next day"):
        move_by_calendar_day(queue, 1)

    prev_queue, next_queue = st.sidebar.columns(2)
    if prev_queue.button("Previous queue"):
        move_by_queue(queue, -1)
    if next_queue.button("Next queue"):
        move_by_queue(queue, 1)

    if st.sidebar.button("Next unreviewed week"):
        move_by_week(week_queue, 1, unreviewed_only=True)
    if st.sidebar.button("Next unreviewed day"):
        move_by_queue(queue, 1, unreviewed_only=True)

    st.sidebar.divider()
    if st.sidebar.button("Export reflagged dataset"):
        result = core.export_reflagged_dataset()
        st.sidebar.success(
            f"Exported {result.csv_path.name}; "
            f"{result.reviewed_site_days}/{result.total_site_days} site-days reviewed."
        )


def render_progress(queue: pd.DataFrame, week_queue: pd.DataFrame) -> None:
    reviewed = int(queue["reviewed"].sum())
    total = int(len(queue))
    current = current_queue_row(
        queue, st.session_state["selected_site"], st.session_state["selected_date"]
    )
    week = current_week_row(
        week_queue,
        st.session_state["selected_site"],
        current_week_start(),
    )
    day_rank = int(current["queue_rank"]) if current is not None else 0
    week_rank = int(week["week_queue_rank"]) if week is not None else 0
    st.metric("Review progress", f"{reviewed}/{total}", f"Day {day_rank}; week {week_rank}")
    st.progress(reviewed / total if total else 0)
    if "flash" in st.session_state:
        st.info(st.session_state.pop("flash"))


def weekly_status_table(
    queue: pd.DataFrame,
    annotations: pd.DataFrame,
    preview_summary: pd.DataFrame,
    site: str,
    week_dates: list[str],
) -> pd.DataFrame:
    rows = queue[(queue["substation_id"] == site) & (queue["date"].isin(week_dates))].copy()
    action_by_date = {}
    for _, row in annotations[annotations["substation_id"] == site].iterrows():
        action_by_date[row["date"]] = annotation_status_text(row)
    rows["review_status"] = rows["date"].map(action_by_date).fillna("unreviewed")

    rows = rows.merge(preview_summary, on=["substation_id", "date"], how="left")
    rows["new_label_day"] = rows["new_label_day"].fillna(rows["old_label_day"]).astype(bool)
    rows["changed_from_old"] = rows["changed_from_old"].fillna(False).astype(bool)
    return rows[
        [
            "date",
            "review_status",
            "new_label_day",
            "changed_from_old",
            "old_label_day",
            "old_positive_intervals",
            "positive_net_load_daytime_intervals",
            "obviousness_score",
        ]
    ]


def main() -> None:
    input_path = core.default_input_path()
    df = load_review_dataframe(str(input_path))
    annotations = load_annotations()
    preview_df = core.with_reviewed_preview_labels(df, annotations)
    preview_summary = core.review_preview_summary(preview_df)
    queue = core.build_review_queue(df, annotations)
    week_queue = core.build_week_queue(queue)
    initialize_selection(queue)
    render_sidebar(queue, week_queue)

    site = st.session_state["selected_site"]
    date = st.session_state["selected_date"]
    week_start = current_week_start()
    week_row = current_week_row(week_queue, site, week_start)
    if week_row is None:
        st.error("Could not find the selected week in the review queue.")
        return

    week_dates = queue[
        (queue["substation_id"] == site) & (queue["week_start"] == week_start)
    ]["date"].tolist()
    day_df = preview_df[(preview_df["substation_id"] == site) & (preview_df["date"] == date)].copy()
    week_df = preview_df[
        (preview_df["substation_id"] == site) & (preview_df["date"].isin(week_dates))
    ].copy()

    st.title("Oracle RPF Review")
    render_progress(queue, week_queue)
    flag_view = st.radio(
        "Flag view",
        FLAG_VIEW_OPTIONS,
        horizontal=True,
        key="flag_view",
    )

    tab_weekly, tab_daily, tab_queue = st.tabs(["Weekly review", "Daily override", "Queue"])
    with tab_weekly:
        st.plotly_chart(
            build_weekly_figure(
                week_df,
                date,
                f"{site} week {week_row['first_date']} to {week_row['last_date']}",
                flag_view,
            ),
            width="stretch",
        )
        render_weekly_controls(annotations, week_queue, site, week_start, week_dates)
        render_weekly_daily_editor(week_df, annotations, site, week_dates)
        st.dataframe(
            weekly_status_table(queue, annotations, preview_summary, site, week_dates),
            width="stretch",
            hide_index=True,
        )
        day_choice = st.selectbox(
            "Open day from this week",
            week_dates,
            index=week_dates.index(date),
        )
        if day_choice != date and st.button("Open selected day"):
            select_site_date(site, day_choice)
            st.rerun()

    with tab_daily:
        st.plotly_chart(
            build_daily_figure(day_df, f"{site} on {date}", flag_view),
            width="stretch",
        )
        render_annotation_controls(day_df, annotations, site, date)

    with tab_queue:
        st.dataframe(
            queue[
                [
                    "queue_rank",
                    "substation_id",
                    "date",
                    "week_start",
                    "reviewed",
                    "old_label_day",
                    "old_positive_intervals",
                    "positive_net_load_daytime_intervals",
                    "obviousness_score",
                ]
            ],
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()

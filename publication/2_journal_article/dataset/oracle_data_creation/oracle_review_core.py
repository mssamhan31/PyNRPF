from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "substation_id",
    "date",
    "timestamp",
    "net_load_MW",
    "solar_MW",
    "label_interval",
    "label_day",
]
ANNOTATION_COLUMNS = [
    "substation_id",
    "date",
    "review_action",
    "rpf_start_time",
    "rpf_end_time",
]
LEGACY_ANNOTATION_COLUMNS = [
    "substation_id",
    "date",
    "rpf_present",
    "rpf_start_time",
    "rpf_end_time",
]
INTERNAL_COLUMNS = ["_timestamp_dt", "_time_hhmm"]

REVIEW_START = "2023-10-01"
REVIEW_END = "2024-09-30"
EXPECTED_REVIEW_ROWS = 280_800
EXPECTED_REVIEW_SITE_DAYS = 2_928
SITE_ORDER = ["act_D", "act_F", "act_B", "act_G", "act_A", "act_E", "act_H", "act_C"]
OUTPUT_BASENAME = "actual_pynrpf_dataset_reflagged"
TIME_RE = re.compile(r"^\d{2}:\d{2}$")
ACTION_ACCEPT_OLD = "accept_old"
ACTION_MANUAL_WINDOW = "manual_window"
ACTION_NO_RPF = "no_rpf"
REVIEW_ACTIONS = {ACTION_ACCEPT_OLD, ACTION_MANUAL_WINDOW, ACTION_NO_RPF}


@dataclass(frozen=True)
class ExportResult:
    csv_path: Path
    parquet_path: Path
    summary_path: Path
    status_path: Path
    checksum_path: Path
    reviewed_site_days: int
    total_site_days: int
    complete: bool


def workflow_root() -> Path:
    return Path(__file__).resolve().parent


def article_root() -> Path:
    return workflow_root().parents[1]


def default_input_path() -> Path:
    return article_root() / "dataset" / "processed" / "actual_pynrpf_dataset.parquet"


def default_annotation_path() -> Path:
    return workflow_root() / "manual_oracle_annotations.csv"


def default_output_dir() -> Path:
    return workflow_root() / "outputs"


def empty_annotations() -> pd.DataFrame:
    return pd.DataFrame(columns=ANNOTATION_COLUMNS)


def coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)

    normalized = series.astype("string").fillna("").str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f", "", "nan", "none"}
    unknown = sorted(set(normalized.unique()) - truthy - falsy)
    if unknown:
        raise ValueError(f"Unknown boolean values: {unknown}")
    return normalized.isin(truthy)


def validate_source_schema(df: pd.DataFrame) -> None:
    actual = list(df.columns)
    if actual != EXPECTED_COLUMNS:
        raise ValueError(
            "Expected source columns "
            f"{EXPECTED_COLUMNS}, but found {actual}. The workflow expects the exact "
            "oracle dataset format."
        )


def _parse_timestamp_wall_clock(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    stripped = text.str.replace(r"(Z|[+-]\d{2}:\d{2})$", "", regex=True)
    return pd.to_datetime(stripped, errors="raise")


def prepare_source_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    validate_source_schema(df)
    work = df.copy()
    work["_timestamp_dt"] = _parse_timestamp_wall_clock(work["timestamp"])
    work["_time_hhmm"] = work["_timestamp_dt"].dt.strftime("%H:%M")

    parsed_dates = work["_timestamp_dt"].dt.strftime("%Y-%m-%d")
    stored_dates = work["date"].astype(str)
    if not (stored_dates == parsed_dates).all():
        bad = int((stored_dates != parsed_dates).sum())
        raise ValueError(f"Found {bad} rows where date does not match timestamp.")

    work["date"] = stored_dates
    work["label_interval"] = coerce_bool(work["label_interval"])
    work["label_day"] = work.groupby(["substation_id", "date"])["label_interval"].transform(
        "any"
    )
    return work.sort_values(["substation_id", "_timestamp_dt"]).reset_index(drop=True)


def load_source_dataset(path: Path | str | None = None) -> pd.DataFrame:
    input_path = Path(path) if path is not None else default_input_path()
    if input_path.suffix.lower() == ".parquet":
        return prepare_source_dataframe(pd.read_parquet(input_path))
    if input_path.suffix.lower() == ".csv":
        return prepare_source_dataframe(pd.read_csv(input_path))
    raise ValueError(f"Unsupported source dataset format: {input_path}")


def filter_review_scope(df: pd.DataFrame) -> pd.DataFrame:
    dates = df["date"].astype(str)
    mask = (dates >= REVIEW_START) & (dates <= REVIEW_END)
    return df.loc[mask].copy().reset_index(drop=True)


def assert_expected_review_scope(df: pd.DataFrame) -> None:
    site_days = df[["substation_id", "date"]].drop_duplicates()
    if len(df) != EXPECTED_REVIEW_ROWS:
        raise ValueError(
            f"Review scope should contain {EXPECTED_REVIEW_ROWS} rows, found {len(df)}."
        )
    if len(site_days) != EXPECTED_REVIEW_SITE_DAYS:
        raise ValueError(
            "Review scope should contain "
            f"{EXPECTED_REVIEW_SITE_DAYS} site-days, found {len(site_days)}."
        )


def _site_rank(site: str) -> int:
    try:
        return SITE_ORDER.index(site)
    except ValueError:
        return len(SITE_ORDER)


def _sort_annotations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return empty_annotations()
    work = df.copy()
    work["_site_rank"] = work["substation_id"].map(_site_rank)
    work = work.sort_values(["_site_rank", "substation_id", "date"]).drop(
        columns="_site_rank"
    )
    return work[ANNOTATION_COLUMNS].reset_index(drop=True)


def _normalize_time(value: str) -> str:
    value = str(value).strip()
    if not TIME_RE.match(value):
        raise ValueError(f"Time values must be in HH:MM format, got {value!r}.")
    hour, minute = (int(part) for part in value.split(":"))
    if hour > 23 or minute > 59:
        raise ValueError(f"Time is outside a valid 24-hour clock: {value!r}.")
    if minute % 15 != 0:
        raise ValueError(f"Time must be on a 15-minute boundary: {value!r}.")
    return f"{hour:02d}:{minute:02d}"


def time_options_for_day(day_df: pd.DataFrame) -> list[str]:
    return sorted(day_df["_time_hhmm"].dropna().astype(str).unique().tolist())


def default_review_window(day_df: pd.DataFrame) -> tuple[str, str]:
    spans = flag_spans(day_df)
    if spans:
        start, end = spans[0]
        return start.strftime("%H:%M"), end.strftime("%H:%M")

    options = time_options_for_day(day_df)
    if not options:
        raise ValueError("Cannot choose review defaults for an empty site-day.")
    preferred_start = "10:00" if "10:00" in options else options[len(options) // 3]
    preferred_end = "14:00" if "14:00" in options else options[(2 * len(options)) // 3]
    return preferred_start, preferred_end


def review_control_defaults(
    day_df: pd.DataFrame,
    annotation_row: pd.Series | dict[str, object] | None,
) -> tuple[str, str, str]:
    default_start, default_end = default_review_window(day_df)
    if annotation_row is None:
        return ACTION_ACCEPT_OLD, default_start, default_end

    action = str(annotation_row["review_action"]).strip().lower()
    if action not in REVIEW_ACTIONS:
        raise ValueError(f"Unknown review_action: {action!r}.")
    if action == ACTION_MANUAL_WINDOW:
        return (
            action,
            str(annotation_row["rpf_start_time"]).strip(),
            str(annotation_row["rpf_end_time"]).strip(),
        )
    return action, default_start, default_end


def infer_weekly_review_update(
    update: dict[str, object],
    default_start: str,
    default_end: str,
) -> dict[str, object]:
    normalized = dict(update)
    if bool(normalized.get("clear", False)):
        return normalized

    action = str(normalized["review_action"]).strip().lower()
    if action not in REVIEW_ACTIONS:
        raise ValueError(f"Unknown review_action: {action!r}.")
    if action == ACTION_NO_RPF:
        normalized["rpf_start_time"] = ""
        normalized["rpf_end_time"] = ""
        return normalized

    start = str(normalized.get("rpf_start_time", "")).strip()
    end = str(normalized.get("rpf_end_time", "")).strip()
    changed_window = (start, end) != (str(default_start).strip(), str(default_end).strip())
    if changed_window:
        normalized["review_action"] = ACTION_MANUAL_WINDOW
    elif action != ACTION_MANUAL_WINDOW:
        normalized["rpf_start_time"] = ""
        normalized["rpf_end_time"] = ""
    return normalized


def validate_window_for_day(
    day_df: pd.DataFrame, start_time: str, end_time: str
) -> tuple[str, str]:
    start = _normalize_time(start_time)
    end = _normalize_time(end_time)
    if start > end:
        raise ValueError("RPF start time must be earlier than or equal to end time.")

    available_times = set(time_options_for_day(day_df))
    missing = [value for value in [start, end] if value not in available_times]
    if missing:
        raise ValueError(f"Selected time is not present in this site-day: {missing}.")
    return start, end


def read_annotations(path: Path | str | None = None) -> pd.DataFrame:
    annotation_path = Path(path) if path is not None else default_annotation_path()
    if not annotation_path.exists():
        return empty_annotations()

    df = pd.read_csv(annotation_path, dtype=str, keep_default_na=False)
    return read_annotations_from_dataframe(df)


def _legacy_annotations_to_current(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["rpf_present"] = coerce_bool(work["rpf_present"])
    out = pd.DataFrame(
        {
            "substation_id": work["substation_id"],
            "date": work["date"],
            "review_action": np.where(
                work["rpf_present"], ACTION_MANUAL_WINDOW, ACTION_NO_RPF
            ),
            "rpf_start_time": work["rpf_start_time"],
            "rpf_end_time": work["rpf_end_time"],
        },
        columns=ANNOTATION_COLUMNS,
    )
    out.loc[out["review_action"] == ACTION_NO_RPF, ["rpf_start_time", "rpf_end_time"]] = ""
    return out


def _validate_and_normalize_annotations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return empty_annotations()

    work = df.copy()
    work["substation_id"] = work["substation_id"].astype(str).str.strip()
    work["date"] = work["date"].astype(str).str.strip()
    work["review_action"] = work["review_action"].astype(str).str.strip().str.lower()
    work["rpf_start_time"] = work["rpf_start_time"].astype(str).str.strip()
    work["rpf_end_time"] = work["rpf_end_time"].astype(str).str.strip()

    unknown_actions = sorted(set(work["review_action"]) - REVIEW_ACTIONS)
    if unknown_actions:
        raise ValueError(f"Unknown review_action values: {unknown_actions}.")

    duplicates = work.duplicated(["substation_id", "date"], keep=False)
    if duplicates.any():
        duplicate_keys = work.loc[duplicates, ["substation_id", "date"]].drop_duplicates()
        raise ValueError(f"Duplicate annotation rows found: {duplicate_keys.to_dict('records')}")

    for row in work.itertuples(index=False):
        if row.review_action == ACTION_MANUAL_WINDOW:
            start = _normalize_time(row.rpf_start_time)
            end = _normalize_time(row.rpf_end_time)
            if start > end:
                raise ValueError(
                    f"Annotation {row.substation_id} {row.date} has start after end."
                )
            work.loc[
                (work["substation_id"] == row.substation_id) & (work["date"] == row.date),
                ["rpf_start_time", "rpf_end_time"],
            ] = [start, end]

    non_window = work["review_action"] != ACTION_MANUAL_WINDOW
    work.loc[non_window, ["rpf_start_time", "rpf_end_time"]] = ""
    return _sort_annotations(work)


def read_annotations_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(df.columns)
    if columns == LEGACY_ANNOTATION_COLUMNS:
        df = _legacy_annotations_to_current(df)
    elif columns != ANNOTATION_COLUMNS:
        raise ValueError(
            "Expected annotation columns "
            f"{ANNOTATION_COLUMNS} or legacy columns {LEGACY_ANNOTATION_COLUMNS}, "
            f"found {columns}."
        )
    return _validate_and_normalize_annotations(df)


def write_annotations(df: pd.DataFrame, path: Path | str | None = None) -> Path:
    annotation_path = Path(path) if path is not None else default_annotation_path()
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    out = read_annotations_from_dataframe(df)

    tmp_path = annotation_path.with_name(f"{annotation_path.name}.tmp")
    out.to_csv(tmp_path, index=False)
    tmp_path.replace(annotation_path)
    return annotation_path


def upsert_annotation(
    annotations: pd.DataFrame,
    substation_id: str,
    date: str,
    review_action: str | bool,
    rpf_start_time: str = "",
    rpf_end_time: str = "",
) -> pd.DataFrame:
    work = read_annotations_from_dataframe(annotations)
    key_mask = (work["substation_id"] == substation_id) & (work["date"] == date)
    work = work.loc[~key_mask].copy()

    if isinstance(review_action, bool):
        action = ACTION_MANUAL_WINDOW if review_action else ACTION_NO_RPF
    else:
        action = str(review_action).strip().lower()
    if action not in REVIEW_ACTIONS:
        raise ValueError(f"Unknown review_action: {review_action!r}.")

    if action == ACTION_MANUAL_WINDOW:
        start = _normalize_time(rpf_start_time)
        end = _normalize_time(rpf_end_time)
        if start > end:
            raise ValueError("RPF start time must be earlier than or equal to end time.")
    else:
        start = ""
        end = ""

    row = pd.DataFrame(
        [
            {
                "substation_id": substation_id,
                "date": date,
                "review_action": action,
                "rpf_start_time": start,
                "rpf_end_time": end,
            }
        ],
        columns=ANNOTATION_COLUMNS,
    )
    return _sort_annotations(pd.concat([work, row], ignore_index=True))


def clear_annotation(annotations: pd.DataFrame, substation_id: str, date: str) -> pd.DataFrame:
    work = read_annotations_from_dataframe(annotations)
    keep = ~((work["substation_id"] == substation_id) & (work["date"] == date))
    return _sort_annotations(work.loc[keep].copy())


def apply_annotation_batch(
    annotations: pd.DataFrame,
    updates: Iterable[dict[str, object]],
) -> pd.DataFrame:
    work = read_annotations_from_dataframe(annotations)
    for update in updates:
        substation_id = str(update["substation_id"]).strip()
        date = str(update["date"]).strip()
        if bool(update.get("clear", False)):
            work = clear_annotation(work, substation_id, date)
            continue

        work = upsert_annotation(
            work,
            substation_id,
            date,
            str(update["review_action"]),
            str(update.get("rpf_start_time", "")),
            str(update.get("rpf_end_time", "")),
        )
    return read_annotations_from_dataframe(work)


def annotation_key_set(annotations: pd.DataFrame) -> set[tuple[str, str]]:
    work = read_annotations_from_dataframe(annotations)
    return set(zip(work["substation_id"], work["date"], strict=True))


def build_review_queue(df: pd.DataFrame, annotations: pd.DataFrame | None = None) -> pd.DataFrame:
    work = df.copy()
    work["_old_label_interval"] = coerce_bool(work["label_interval"])
    work["_candidate"] = (
        work["_time_hhmm"].between("06:00", "18:00")
        & (work["solar_MW"].fillna(0) >= 0.25)
        & (work["net_load_MW"].fillna(-np.inf) > 0)
    )
    work["_candidate_solar"] = np.where(work["_candidate"], work["solar_MW"].fillna(0), 0)

    queue = (
        work.groupby(["substation_id", "date"], as_index=False)
        .agg(
            old_label_day=("_old_label_interval", "any"),
            old_positive_intervals=("_old_label_interval", "sum"),
            max_solar_MW=("solar_MW", "max"),
            positive_net_load_daytime_intervals=("_candidate", "sum"),
            obviousness_score=("_candidate_solar", "sum"),
        )
        .reset_index(drop=True)
    )
    queue["site_order"] = queue["substation_id"].map(_site_rank)

    reviewed_keys = annotation_key_set(annotations) if annotations is not None else set()
    queue["reviewed"] = [
        (site, date) in reviewed_keys for site, date in zip(queue["substation_id"], queue["date"])
    ]
    queue["week_start"] = review_week_start(queue["date"])
    queue["week_end"] = (
        pd.to_datetime(queue["week_start"]) + pd.Timedelta(days=6)
    ).dt.strftime("%Y-%m-%d")

    queue = queue.sort_values(
        ["site_order", "date"],
        ascending=[True, True],
    ).reset_index(drop=True)
    queue.insert(0, "queue_rank", np.arange(1, len(queue) + 1))
    return queue.drop(columns=["site_order"])


def review_week_start(dates: pd.Series | list[str]) -> pd.Series:
    date_values = pd.to_datetime(pd.Series(dates), errors="raise")
    review_start = pd.Timestamp(REVIEW_START)
    offsets = ((date_values - review_start).dt.days // 7) * 7
    return (review_start + pd.to_timedelta(offsets, unit="D")).dt.strftime("%Y-%m-%d")


def build_week_queue(daily_queue: pd.DataFrame) -> pd.DataFrame:
    work = daily_queue.copy()
    work["site_order"] = work["substation_id"].map(_site_rank)
    week_queue = (
        work.groupby(["substation_id", "week_start", "week_end"], as_index=False)
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            n_days=("date", "count"),
            reviewed_days=("reviewed", "sum"),
            old_label_days=("old_label_day", "sum"),
            old_positive_intervals=("old_positive_intervals", "sum"),
            obviousness_score=("obviousness_score", "sum"),
            site_order=("site_order", "first"),
        )
        .reset_index(drop=True)
    )
    week_queue["reviewed"] = week_queue["reviewed_days"] == week_queue["n_days"]
    week_queue = week_queue.sort_values(["site_order", "week_start"]).reset_index(drop=True)
    week_queue.insert(0, "week_queue_rank", np.arange(1, len(week_queue) + 1))
    return week_queue.drop(columns=["site_order"])


def next_week_selection(
    week_queue: pd.DataFrame,
    substation_id: str,
    week_start: str,
) -> tuple[str, str] | None:
    if week_queue.empty:
        return None
    matches = week_queue[
        (week_queue["substation_id"] == substation_id)
        & (week_queue["week_start"] == week_start)
    ]
    if matches.empty:
        row = week_queue.iloc[0]
        return str(row["substation_id"]), str(row["first_date"])

    pos = week_queue.index.get_loc(matches.index[0])
    if not isinstance(pos, int):
        pos = int(pos[0])
    next_pos = pos + 1
    if next_pos >= len(week_queue):
        return None
    row = week_queue.iloc[next_pos]
    return str(row["substation_id"]), str(row["first_date"])


def flag_spans(
    day_df: pd.DataFrame, flag_column: str = "label_interval"
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if day_df.empty:
        return []
    flags = coerce_bool(day_df[flag_column]).to_numpy()
    times = day_df["_timestamp_dt"].to_list()
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = None
    end = None
    for flag, timestamp in zip(flags, times, strict=True):
        if flag and start is None:
            start = timestamp
        if flag:
            end = timestamp
        if not flag and start is not None:
            spans.append((start, end))
            start = None
            end = None
    if start is not None:
        spans.append((start, end))
    return spans


def corrected_net_load(df: pd.DataFrame, flag_column: str = "label_interval") -> pd.Series:
    flags = coerce_bool(df[flag_column])
    values = np.where(flags, -df["net_load_MW"], df["net_load_MW"])
    result = pd.Series(values, index=df.index, name=f"{flag_column}_corrected_net_load_MW")
    result.loc[df["net_load_MW"].isna()] = np.nan
    return result


def _validate_annotation_keys(scoped_df: pd.DataFrame, annotations: pd.DataFrame) -> None:
    valid_keys = set(zip(scoped_df["substation_id"], scoped_df["date"], strict=True))
    unknown = sorted(annotation_key_set(annotations) - valid_keys)
    if unknown:
        raise ValueError(f"Annotations outside the review dataset were found: {unknown[:5]}")


def with_reviewed_preview_labels(
    scoped_df: pd.DataFrame,
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    out = scoped_df.copy()
    out["label_interval"] = coerce_bool(out["label_interval"])
    out["new_label_interval"] = out["label_interval"].copy()
    anns = read_annotations_from_dataframe(annotations)
    _validate_annotation_keys(out, anns)

    for row in anns.itertuples(index=False):
        day_mask = (out["substation_id"] == row.substation_id) & (out["date"] == row.date)
        day_df = out.loc[day_mask]

        if row.review_action == ACTION_ACCEPT_OLD:
            continue

        out.loc[day_mask, "new_label_interval"] = False

        if row.review_action == ACTION_MANUAL_WINDOW:
            start, end = validate_window_for_day(day_df, row.rpf_start_time, row.rpf_end_time)
            window_mask = day_mask & out["_time_hhmm"].between(start, end)
            out.loc[window_mask, "new_label_interval"] = True

    out["new_label_day"] = out.groupby(["substation_id", "date"])[
        "new_label_interval"
    ].transform("any")
    return out


def review_preview_summary(preview_df: pd.DataFrame) -> pd.DataFrame:
    if "new_label_interval" not in preview_df.columns:
        raise ValueError("review_preview_summary expects a new_label_interval column.")

    work = preview_df.copy()
    work["_old_label_interval"] = coerce_bool(work["label_interval"])
    work["_new_label_interval"] = coerce_bool(work["new_label_interval"])
    work["_changed_from_old"] = work["_old_label_interval"] != work["_new_label_interval"]

    return (
        work.groupby(["substation_id", "date"], as_index=False)
        .agg(
            new_label_day=("_new_label_interval", "any"),
            new_positive_intervals=("_new_label_interval", "sum"),
            changed_from_old=("_changed_from_old", "any"),
        )
        .reset_index(drop=True)
    )


def apply_annotations(scoped_df: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    out = with_reviewed_preview_labels(scoped_df, annotations)
    out["label_interval"] = out["new_label_interval"]
    out["label_day"] = out["new_label_day"]
    return out[EXPECTED_COLUMNS].copy()


def review_status(scoped_df: pd.DataFrame, annotations: pd.DataFrame) -> dict[str, object]:
    total = int(scoped_df[["substation_id", "date"]].drop_duplicates().shape[0])
    reviewed = len(annotation_key_set(annotations))
    remaining = total - reviewed
    return {
        "review_start": REVIEW_START,
        "review_end": REVIEW_END,
        "total_site_days": total,
        "reviewed_site_days": reviewed,
        "remaining_site_days": remaining,
        "complete": remaining == 0,
        "uses_old_labels_for_unreviewed_site_days": remaining > 0,
    }


def dataset_summary(df: pd.DataFrame, dataset_name: str) -> dict[str, object]:
    site_day = df[["substation_id", "date", "label_day"]].drop_duplicates()
    return {
        "dataset": dataset_name,
        "n_rows": int(len(df)),
        "n_stations": int(df["substation_id"].nunique()),
        "min_timestamp": str(df["timestamp"].min()),
        "max_timestamp": str(df["timestamp"].max()),
        "n_dates": int(df["date"].nunique()),
        "null_timestamp": int(df["timestamp"].isna().sum()),
        "null_net_load_MW": int(df["net_load_MW"].isna().sum()),
        "null_solar_MW": int(df["solar_MW"].isna().sum()),
        "duplicate_substation_timestamp_keys": int(
            df.duplicated(["substation_id", "timestamp"], keep=False).sum()
        ),
        "positive_label_interval": int(coerce_bool(df["label_interval"]).sum()),
        "positive_label_day_site_days": int(coerce_bool(site_day["label_day"]).sum()),
    }


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)
    return path


def _atomic_write_json(payload: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)
    return path


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256(paths: Iterable[Path], checksum_path: Path) -> Path:
    lines = [f"{_sha256(path)}  {path.name}" for path in paths]
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checksum_path.with_name(f"{checksum_path.name}.tmp")
    tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp_path.replace(checksum_path)
    return checksum_path


def export_reflagged_dataset(
    input_path: Path | str | None = None,
    annotation_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> ExportResult:
    source = load_source_dataset(input_path)
    scoped = filter_review_scope(source)
    assert_expected_review_scope(scoped)

    annotations = read_annotations(annotation_path)
    reflagged = apply_annotations(scoped, annotations)
    status_payload = review_status(scoped, annotations)

    out_dir = Path(output_dir) if output_dir is not None else default_output_dir()
    csv_path = out_dir / f"{OUTPUT_BASENAME}.csv"
    parquet_path = out_dir / f"{OUTPUT_BASENAME}.parquet"
    summary_path = out_dir / "dataset_summary.csv"
    status_path = out_dir / "review_status.json"
    checksum_path = out_dir / "sha256.txt"

    _atomic_write_csv(reflagged, csv_path)
    _atomic_write_parquet(reflagged, parquet_path)
    _atomic_write_csv(pd.DataFrame([dataset_summary(reflagged, OUTPUT_BASENAME)]), summary_path)

    status_payload.update(
        {
            "input_path": str(Path(input_path) if input_path is not None else default_input_path()),
            "annotation_path": str(
                Path(annotation_path) if annotation_path is not None else default_annotation_path()
            ),
            "csv_path": str(csv_path),
            "parquet_path": str(parquet_path),
        }
    )
    _atomic_write_json(status_payload, status_path)
    write_sha256([csv_path, parquet_path], checksum_path)

    return ExportResult(
        csv_path=csv_path,
        parquet_path=parquet_path,
        summary_path=summary_path,
        status_path=status_path,
        checksum_path=checksum_path,
        reviewed_site_days=int(status_payload["reviewed_site_days"]),
        total_site_days=int(status_payload["total_site_days"]),
        complete=bool(status_payload["complete"]),
    )

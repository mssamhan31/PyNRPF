from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


WORKFLOW_DIR = (
    Path(__file__).resolve().parents[1]
    / "publication"
    / "2_journal_article"
    / "dataset"
    / "oracle_data_creation"
)
sys.path.insert(0, str(WORKFLOW_DIR))

import oracle_review_core as core  # noqa: E402


def _sample_source() -> pd.DataFrame:
    rows = []
    for site in ["act_D", "act_A"]:
        for date in ["2023-10-01", "2023-10-02"]:
            for timestamp, label in [
                (f"{date} 00:15:00+00:00", False),
                (f"{date} 00:30:00+00:00", False),
                (f"{date} 00:45:00+00:00", True),
                (f"{date} 01:00:00+00:00", False),
            ]:
                rows.append(
                    {
                        "substation_id": site,
                        "date": date,
                        "timestamp": timestamp,
                        "net_load_MW": 5.0,
                        "solar_MW": 1.0,
                        "label_interval": label,
                        "label_day": True,
                    }
                )
    return core.prepare_source_dataframe(pd.DataFrame(rows, columns=core.EXPECTED_COLUMNS))


def test_apply_annotations_uses_inclusive_15_minute_window() -> None:
    source = _sample_source()
    annotations = pd.DataFrame(
        [
            {
                "substation_id": "act_D",
                "date": "2023-10-01",
                "review_action": core.ACTION_MANUAL_WINDOW,
                "rpf_start_time": "00:30",
                "rpf_end_time": "01:00",
            }
        ],
        columns=core.ANNOTATION_COLUMNS,
    )

    out = core.apply_annotations(source, annotations)
    day = out[(out["substation_id"] == "act_D") & (out["date"] == "2023-10-01")]

    assert day["label_interval"].tolist() == [False, True, True, True]
    assert day["label_day"].tolist() == [True, True, True, True]


def test_no_rpf_annotation_clears_interval_and_day_labels() -> None:
    source = _sample_source()
    annotations = pd.DataFrame(
        [
            {
                "substation_id": "act_D",
                "date": "2023-10-01",
                "review_action": core.ACTION_NO_RPF,
                "rpf_start_time": "",
                "rpf_end_time": "",
            }
        ],
        columns=core.ANNOTATION_COLUMNS,
    )

    out = core.apply_annotations(source, annotations)
    day = out[(out["substation_id"] == "act_D") & (out["date"] == "2023-10-01")]

    assert day["label_interval"].tolist() == [False, False, False, False]
    assert day["label_day"].tolist() == [False, False, False, False]


def test_unreviewed_days_keep_old_labels_and_raw_values() -> None:
    source = _sample_source()
    annotations = pd.DataFrame(
        [
            {
                "substation_id": "act_D",
                "date": "2023-10-01",
                "review_action": core.ACTION_NO_RPF,
                "rpf_start_time": "",
                "rpf_end_time": "",
            }
        ],
        columns=core.ANNOTATION_COLUMNS,
    )

    out = core.apply_annotations(source, annotations)
    unreviewed = out[(out["substation_id"] == "act_A") & (out["date"] == "2023-10-02")]

    assert list(out.columns) == core.EXPECTED_COLUMNS
    assert unreviewed["label_interval"].tolist() == [False, False, True, False]
    assert unreviewed["net_load_MW"].tolist() == [5.0, 5.0, 5.0, 5.0]


def test_queue_uses_site_order_before_obviousness() -> None:
    source = _sample_source()
    queue = core.build_review_queue(source, core.empty_annotations())

    assert queue.iloc[0]["substation_id"] == "act_D"
    assert queue.iloc[0]["date"] == "2023-10-01"
    assert queue.iloc[1]["date"] == "2023-10-02"
    assert queue.iloc[-1]["substation_id"] == "act_A"
    assert queue["queue_rank"].tolist() == list(range(1, len(queue) + 1))


def test_accept_old_review_action_preserves_old_interval_shape() -> None:
    source = _sample_source()
    annotations = pd.DataFrame(
        [
            {
                "substation_id": "act_D",
                "date": "2023-10-01",
                "review_action": core.ACTION_ACCEPT_OLD,
                "rpf_start_time": "",
                "rpf_end_time": "",
            }
        ],
        columns=core.ANNOTATION_COLUMNS,
    )

    out = core.apply_annotations(source, annotations)
    day = out[(out["substation_id"] == "act_D") & (out["date"] == "2023-10-01")]

    assert day["label_interval"].tolist() == [False, False, True, False]
    assert day["label_day"].tolist() == [True, True, True, True]


def test_review_control_defaults_use_old_flags_for_unreviewed_day() -> None:
    source = _sample_source()
    day = source[(source["substation_id"] == "act_D") & (source["date"] == "2023-10-01")]

    action, start, end = core.review_control_defaults(day, None)

    assert action == core.ACTION_ACCEPT_OLD
    assert start == "00:45"
    assert end == "00:45"


def test_review_control_defaults_keep_existing_manual_window() -> None:
    source = _sample_source()
    day = source[(source["substation_id"] == "act_D") & (source["date"] == "2023-10-01")]
    annotation = pd.Series(
        {
            "substation_id": "act_D",
            "date": "2023-10-01",
            "review_action": core.ACTION_MANUAL_WINDOW,
            "rpf_start_time": "00:30",
            "rpf_end_time": "01:00",
        }
    )

    action, start, end = core.review_control_defaults(day, annotation)

    assert action == core.ACTION_MANUAL_WINDOW
    assert start == "00:30"
    assert end == "01:00"


def test_infer_weekly_review_update_converts_changed_times_to_manual_window() -> None:
    update = {
        "substation_id": "act_D",
        "date": "2023-10-01",
        "review_action": core.ACTION_ACCEPT_OLD,
        "rpf_start_time": "00:30",
        "rpf_end_time": "01:00",
        "clear": False,
    }

    inferred = core.infer_weekly_review_update(update, "00:45", "00:45")

    assert inferred["review_action"] == core.ACTION_MANUAL_WINDOW
    assert inferred["rpf_start_time"] == "00:30"
    assert inferred["rpf_end_time"] == "01:00"


def test_infer_weekly_review_update_keeps_unchanged_accept_old_and_no_rpf() -> None:
    accept_old = {
        "substation_id": "act_D",
        "date": "2023-10-01",
        "review_action": core.ACTION_ACCEPT_OLD,
        "rpf_start_time": "00:45",
        "rpf_end_time": "00:45",
        "clear": False,
    }
    no_rpf = {
        **accept_old,
        "review_action": core.ACTION_NO_RPF,
        "rpf_start_time": "00:30",
        "rpf_end_time": "01:00",
    }
    clear = {**accept_old, "clear": True, "rpf_start_time": "00:30"}

    inferred_accept = core.infer_weekly_review_update(accept_old, "00:45", "00:45")
    inferred_no_rpf = core.infer_weekly_review_update(no_rpf, "00:45", "00:45")
    inferred_clear = core.infer_weekly_review_update(clear, "00:45", "00:45")

    assert inferred_accept["review_action"] == core.ACTION_ACCEPT_OLD
    assert inferred_accept["rpf_start_time"] == ""
    assert inferred_accept["rpf_end_time"] == ""
    assert inferred_no_rpf["review_action"] == core.ACTION_NO_RPF
    assert inferred_no_rpf["rpf_start_time"] == ""
    assert inferred_no_rpf["rpf_end_time"] == ""
    assert inferred_clear["clear"] is True
    assert inferred_clear["rpf_start_time"] == "00:30"


def test_next_week_selection_moves_within_and_across_sites() -> None:
    week_queue = pd.DataFrame(
        [
            {"substation_id": "act_D", "week_start": "2023-10-01", "first_date": "2023-10-01"},
            {"substation_id": "act_D", "week_start": "2023-10-08", "first_date": "2023-10-08"},
            {"substation_id": "act_A", "week_start": "2023-10-01", "first_date": "2023-10-01"},
        ]
    )

    assert core.next_week_selection(week_queue, "act_D", "2023-10-01") == (
        "act_D",
        "2023-10-08",
    )
    assert core.next_week_selection(week_queue, "act_D", "2023-10-08") == (
        "act_A",
        "2023-10-01",
    )
    assert core.next_week_selection(week_queue, "act_A", "2023-10-01") is None


def test_reviewed_preview_labels_apply_actions_without_changing_old_labels() -> None:
    source = _sample_source()
    annotations = pd.DataFrame(
        [
            {
                "substation_id": "act_D",
                "date": "2023-10-01",
                "review_action": core.ACTION_MANUAL_WINDOW,
                "rpf_start_time": "00:30",
                "rpf_end_time": "01:00",
            },
            {
                "substation_id": "act_D",
                "date": "2023-10-02",
                "review_action": core.ACTION_NO_RPF,
                "rpf_start_time": "",
                "rpf_end_time": "",
            },
            {
                "substation_id": "act_A",
                "date": "2023-10-01",
                "review_action": core.ACTION_ACCEPT_OLD,
                "rpf_start_time": "",
                "rpf_end_time": "",
            },
        ],
        columns=core.ANNOTATION_COLUMNS,
    )

    preview = core.with_reviewed_preview_labels(source, annotations)

    manual = preview[(preview["substation_id"] == "act_D") & (preview["date"] == "2023-10-01")]
    no_rpf = preview[(preview["substation_id"] == "act_D") & (preview["date"] == "2023-10-02")]
    accept_old = preview[
        (preview["substation_id"] == "act_A") & (preview["date"] == "2023-10-01")
    ]
    unreviewed = preview[
        (preview["substation_id"] == "act_A") & (preview["date"] == "2023-10-02")
    ]

    assert manual["label_interval"].tolist() == [False, False, True, False]
    assert manual["new_label_interval"].tolist() == [False, True, True, True]
    assert no_rpf["new_label_interval"].tolist() == [False, False, False, False]
    assert no_rpf["new_label_day"].tolist() == [False, False, False, False]
    assert accept_old["new_label_interval"].tolist() == [False, False, True, False]
    assert unreviewed["new_label_interval"].tolist() == [False, False, True, False]


def test_review_preview_summary_detects_changed_days() -> None:
    source = _sample_source()
    annotations = pd.DataFrame(
        [
            {
                "substation_id": "act_D",
                "date": "2023-10-01",
                "review_action": core.ACTION_MANUAL_WINDOW,
                "rpf_start_time": "00:30",
                "rpf_end_time": "01:00",
            },
            {
                "substation_id": "act_A",
                "date": "2023-10-01",
                "review_action": core.ACTION_ACCEPT_OLD,
                "rpf_start_time": "",
                "rpf_end_time": "",
            },
        ],
        columns=core.ANNOTATION_COLUMNS,
    )

    preview = core.with_reviewed_preview_labels(source, annotations)
    summary = core.review_preview_summary(preview)

    changed = summary[
        (summary["substation_id"] == "act_D") & (summary["date"] == "2023-10-01")
    ].iloc[0]
    unchanged = summary[
        (summary["substation_id"] == "act_A") & (summary["date"] == "2023-10-01")
    ].iloc[0]

    assert bool(changed["new_label_day"]) is True
    assert int(changed["new_positive_intervals"]) == 3
    assert bool(changed["changed_from_old"]) is True
    assert bool(unchanged["changed_from_old"]) is False


def test_upsert_annotation_replaces_existing_site_day() -> None:
    annotations = core.empty_annotations()

    first = core.upsert_annotation(
        annotations,
        "act_D",
        "2023-10-01",
        core.ACTION_MANUAL_WINDOW,
        "00:30",
        "01:00",
    )
    second = core.upsert_annotation(first, "act_D", "2023-10-01", core.ACTION_NO_RPF)

    assert len(second) == 1
    assert second.iloc[0]["review_action"] == core.ACTION_NO_RPF
    assert second.iloc[0]["rpf_start_time"] == ""
    assert second.iloc[0]["rpf_end_time"] == ""
    assert second.iloc[0]["confidence"] == core.CONFIDENCE_SURE


def test_apply_annotation_batch_upserts_and_clears_multiple_days() -> None:
    annotations = pd.DataFrame(
        [
            {
                "substation_id": "act_D",
                "date": "2023-10-01",
                "review_action": core.ACTION_ACCEPT_OLD,
                "rpf_start_time": "",
                "rpf_end_time": "",
            },
            {
                "substation_id": "act_A",
                "date": "2023-10-01",
                "review_action": core.ACTION_NO_RPF,
                "rpf_start_time": "",
                "rpf_end_time": "",
            },
        ],
        columns=core.ANNOTATION_COLUMNS,
    )

    updated = core.apply_annotation_batch(
        annotations,
        [
            {
                "substation_id": "act_D",
                "date": "2023-10-01",
                "review_action": core.ACTION_MANUAL_WINDOW,
                "rpf_start_time": "00:30",
                "rpf_end_time": "01:00",
            },
            {
                "substation_id": "act_D",
                "date": "2023-10-02",
                "review_action": core.ACTION_NO_RPF,
            },
            {
                "substation_id": "act_A",
                "date": "2023-10-01",
                "clear": True,
            },
        ],
    )

    assert len(updated) == 2
    assert set(zip(updated["substation_id"], updated["date"])) == {
        ("act_D", "2023-10-01"),
        ("act_D", "2023-10-02"),
    }
    manual = updated[updated["date"] == "2023-10-01"].iloc[0]
    no_rpf = updated[updated["date"] == "2023-10-02"].iloc[0]

    assert manual["review_action"] == core.ACTION_MANUAL_WINDOW
    assert manual["rpf_start_time"] == "00:30"
    assert manual["rpf_end_time"] == "01:00"
    assert no_rpf["review_action"] == core.ACTION_NO_RPF
    assert no_rpf["rpf_start_time"] == ""
    assert no_rpf["rpf_end_time"] == ""


def test_legacy_annotations_migrate_to_review_actions() -> None:
    legacy = pd.DataFrame(
        [
            {
                "substation_id": "act_D",
                "date": "2023-10-01",
                "rpf_present": "True",
                "rpf_start_time": "00:30",
                "rpf_end_time": "01:00",
            },
            {
                "substation_id": "act_D",
                "date": "2023-10-02",
                "rpf_present": "False",
                "rpf_start_time": "",
                "rpf_end_time": "",
            },
        ],
        columns=core.LEGACY_ANNOTATION_COLUMNS,
    )

    migrated = core.read_annotations_from_dataframe(legacy)

    assert list(migrated.columns) == core.ANNOTATION_COLUMNS
    assert migrated["review_action"].tolist() == [
        core.ACTION_MANUAL_WINDOW,
        core.ACTION_NO_RPF,
    ]
    assert migrated["confidence"].tolist() == [
        core.CONFIDENCE_SURE,
        core.CONFIDENCE_SURE,
    ]


def test_validate_source_schema_requires_exact_columns() -> None:
    bad = pd.DataFrame({"substation_id": ["act_D"]})

    try:
        core.prepare_source_dataframe(bad)
    except ValueError as exc:
        assert "Expected source columns" in str(exc)
        return
    raise AssertionError("Expected exact schema validation to fail.")

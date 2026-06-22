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


def test_validate_source_schema_requires_exact_columns() -> None:
    bad = pd.DataFrame({"substation_id": ["act_D"]})

    try:
        core.prepare_source_dataframe(bad)
    except ValueError as exc:
        assert "Expected source columns" in str(exc)
        return
    raise AssertionError("Expected exact schema validation to fail.")

"""The run entry point, the command line and the Spark adapter on synthetic frames."""

import json

import numpy as np
import pandas as pd
import pytest
from fixtures.planted_days import clean_day, clean_flip

import pynrpf
from pynrpf import schemas
from pynrpf.cli import main
from pynrpf.m9.decision import AUTO_CORRECT, UNCERTAIN


def frame_for(days: dict, site_col="substation_id", ts_col="timestamp", y_col="net_load_MW",
              s_col="solar_MW") -> pd.DataFrame:
    """Readings for {(site, date): (y, s)} with the given column names."""
    parts = []
    for (site, date), (y, s) in days.items():
        ts = pd.date_range(f"{date} 00:00", periods=len(y), freq="15min")
        parts.append(pd.DataFrame({site_col: site, ts_col: ts, y_col: y, s_col: s}))
    return pd.concat(parts, ignore_index=True)


def test_run_end_to_end_with_default_columns():
    y1, s1, truth = clean_flip()
    y0, s0, _ = clean_day()
    result = pynrpf.run(frame_for({("A", "2024-01-01"): (y1, s1), ("A", "2024-01-02"): (y0, s0)}))
    assert list(result.site_days.columns) == schemas.names(schemas.SITE_DAYS)
    assert list(result.intervals.columns) == schemas.names(schemas.INTERVALS)
    d = result.site_days.set_index("date")
    assert d.loc["2024-01-01", "outcome"] == AUTO_CORRECT and d.loc["2024-01-02", "outcome"] != AUTO_CORRECT
    assert d.loc["2024-01-01", "proposed_mwh"] > 0 and d.loc["2024-01-01", "min_change_mw"] < 0
    corrected = result.intervals[result.intervals["corrected"]]
    assert len(corrected) == d.loc["2024-01-01", "window_end"] - d.loc["2024-01-01", "window_start"] + 1
    assert (corrected["net_load_corrected_mw"] == -corrected["net_load_mw"]).all()
    assert result.summary()["days_auto_correct"] == 1


def test_run_accepts_a_column_mapping_and_keeps_timestamps_as_given():
    y, s, _ = clean_flip()
    frame = frame_for({("B", "2024-03-05"): (y, s)}, site_col="site_id", ts_col="ts", y_col="mw", s_col="pv")
    result = pynrpf.run(frame, columns=dict(site="site_id", timestamp="ts", net_load="mw", solar="pv"))
    assert result.site_days["site"].iloc[0] == "B"
    assert result.intervals["timestamp"].iloc[0] == str(frame["ts"].iloc[0])


def test_incomplete_day_is_reported_but_not_scored():
    y, s, _ = clean_flip()
    result = pynrpf.run(frame_for({("A", "2024-01-01"): (y[:80], s[:80])}))
    row = result.site_days.iloc[0]
    assert row["n_slots"] == 80 and not row["input_ok"] and row["outcome"] == UNCERTAIN and np.isnan(row["p"])
    assert not result.intervals["corrected"].any()


def test_bad_inputs_are_refused():
    y, s, _ = clean_flip()
    frame = frame_for({("A", "2024-01-01"): (y, s)})
    with pytest.raises(KeyError):
        pynrpf.run(frame.drop(columns=["solar_MW"]))
    with pytest.raises(ValueError):
        pynrpf.run(pd.concat([frame, frame.iloc[:1]]))               # a duplicated reading
    off_grid = frame.copy()
    off_grid.loc[0, "timestamp"] = off_grid.loc[0, "timestamp"] + pd.Timedelta(minutes=7)
    with pytest.raises(ValueError):
        pynrpf.run(off_grid)


def test_cli_writes_the_tables_and_summary(tmp_path):
    y, s, _ = clean_flip()
    csv = tmp_path / "readings.csv"
    frame_for({("A", "2024-01-01"): (y, s)}).to_csv(csv, index=False)
    assert main(["run", str(csv), "--out", str(tmp_path / "out"), "--c", "0.8"]) == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["c"] == 0.8 and summary["site_days"] == 1
    assert (tmp_path / "out" / "site_days.csv").exists() and (tmp_path / "out" / "intervals.csv").exists()


def test_spark_adapter_matches_pandas():
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[1]").appName("pynrpf-test").getOrCreate()
    y1, s1, _ = clean_flip()
    y0, s0, _ = clean_day()
    frame = frame_for({("A", "2024-01-01"): (y1, s1), ("B", "2024-01-01"): (y0, s0)})
    from pynrpf.spark import run_per_site

    site_days, intervals = run_per_site(spark.createDataFrame(frame))
    got = site_days.toPandas().sort_values("site").reset_index(drop=True)
    want = pynrpf.run(frame).site_days.sort_values("site").reset_index(drop=True)
    assert (got["outcome"] == want["outcome"]).all()
    assert np.allclose(got["evidence"].fillna(0), want["evidence"].fillna(0))
    assert intervals.count() == len(frame)
    spark.stop()

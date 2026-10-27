# Databricks notebook source
# MAGIC %md
# MAGIC # PyNRPF smoke test
# MAGIC
# MAGIC **Purpose.** Prove that the pinned release of `pynrpf` installs and runs on this cluster, in pandas
# MAGIC and through the Spark adapter, before any real readings are touched.
# MAGIC
# MAGIC **Inputs.** Nothing from the workspace. Two synthetic site-days are built below: one clean day and
# MAGIC one day whose midday export was stored as an import (the recorded trace is the absolute value of
# MAGIC the true net load inside the export span), the same construction as the package's test fixtures.
# MAGIC
# MAGIC **Outputs.** Displayed tables and the provenance record. Nothing is written.
# MAGIC
# MAGIC **Runtime.** Under a minute on any cluster with Python 3.10 or later.
# MAGIC
# MAGIC **Steps.**
# MAGIC 1. Install the pinned release.
# MAGIC 2. Build the synthetic readings.
# MAGIC 3. Run the method in pandas.
# MAGIC 4. Run it through the Spark adapter.
# MAGIC 5. Check that both paths agree and print the provenance record.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install the pinned release
# MAGIC
# MAGIC Two routes give the same files, built from the same tagged commit. Use PyPI when the workspace can
# MAGIC reach it; otherwise copy the wheel from the GitHub Release into a Unity Catalog Volume and install it
# MAGIC by path. Keep the line that applies and leave the other commented out.

# COMMAND ----------

# MAGIC %pip install pynrpf==0.4.0
# MAGIC # %pip install /Volumes/<catalog>/<schema>/<volume>/pynrpf-0.4.0-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821  (dbutils is provided by Databricks)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Two synthetic site-days
# MAGIC
# MAGIC A smooth demand curve and a solar bell peaking at noon make a true net load. Site A exports at
# MAGIC midday and its export was stored as an import, so its recorded trace dips towards zero instead of
# MAGIC crossing it. Site B never exports, so there is nothing to flip. The frame has the four columns the
# MAGIC package expects by default.

# COMMAND ----------

import json

import numpy as np
import pandas as pd

import pynrpf

print("pynrpf", pynrpf.__version__)

SLOTS = 96                                          # fifteen-minute readings in a day


def base_day(demand_level: float, solar_peak: float) -> tuple[np.ndarray, np.ndarray]:
    """A smooth demand day (MW) and a solar bell (MW) peaking at noon."""
    t = np.arange(SLOTS) / 4.0
    demand = demand_level + 1.5 * np.sin((t - 6) / 24 * 2 * np.pi)
    solar = solar_peak * np.clip(np.sin((t - 6) / 12 * np.pi), 0.0, None) ** 2
    return demand, solar


def readings(site: str, date: str, net_load: np.ndarray, solar: np.ndarray) -> pd.DataFrame:
    """One site-day in the package's default column names."""
    timestamps = pd.date_range(f"{date} 00:00", periods=SLOTS, freq="15min")
    return pd.DataFrame({"substation_id": site, "timestamp": timestamps,
                         "net_load_MW": net_load, "solar_MW": solar})


demand_a, solar_a = base_day(demand_level=8.0, solar_peak=12.0)
true_a = demand_a - solar_a
recorded_a = np.abs(true_a)                         # the export stored as an import

demand_b, solar_b = base_day(demand_level=15.0, solar_peak=6.0)
recorded_b = demand_b - solar_b                     # never negative: nothing to flip

frame = pd.concat([readings("A", "2024-01-01", recorded_a, solar_a),
                   readings("B", "2024-01-01", recorded_b, solar_b)], ignore_index=True)
display(frame.head())  # noqa: F821  (display is provided by Databricks)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The method in pandas
# MAGIC
# MAGIC One call scores every site-day. Site A should come out `AUTO_CORRECT` with a midday window and a
# MAGIC negative change in minimum demand; site B should not be corrected.

# COMMAND ----------

result = pynrpf.run(frame)
display(result.site_days)  # noqa: F821

by_site = result.site_days.set_index("site")
assert by_site.loc["A", "outcome"] == "AUTO_CORRECT", by_site.loc["A", "outcome"]
assert by_site.loc["B", "outcome"] != "AUTO_CORRECT", by_site.loc["B", "outcome"]
print(json.dumps(result.summary(), indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. The same method through the Spark adapter
# MAGIC
# MAGIC The adapter groups the frame by site and runs the pandas implementation on each group, so the
# MAGIC numbers are the same ones the package's tests pin.

# COMMAND ----------

from pynrpf.spark import run_per_site

sdf = spark.createDataFrame(frame)  # noqa: F821  (spark is provided by Databricks)
site_days, intervals = run_per_site(sdf)
display(site_days)  # noqa: F821
print("interval rows:", intervals.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Agreement and provenance
# MAGIC
# MAGIC The Spark outcomes and evidence must equal the pandas ones. The provenance record names the package
# MAGIC version and the settings that produced the tables; store it with every real output so a result can
# MAGIC always be traced to the release that made it.

# COMMAND ----------

got = site_days.toPandas().set_index("site").sort_index()
want = result.site_days.set_index("site").sort_index()
assert (got["outcome"] == want["outcome"]).all()
assert np.allclose(got["evidence"].fillna(0), want["evidence"].fillna(0))

print(json.dumps(pynrpf.provenance().as_dict(), indent=2))   # the defaults run_per_site used above

# COMMAND ----------

# MAGIC %md
# MAGIC ## Result
# MAGIC
# MAGIC If every cell ran, the pinned release installs on this cluster, the pandas and Spark paths agree,
# MAGIC and the provenance record shows which version and settings produced the tables.

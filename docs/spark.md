# Spark

M9 scores one site-day at a time and needs nothing across sites or days, so Spark's job is
distribution, not arithmetic. The adapter groups the frame by site and runs the pandas
implementation on each group, so the numbers are exactly the ones the tests pin.

```python
from pynrpf.spark import run_per_site

site_days, intervals = run_per_site(sdf, c=0.7)
site_days.write.mode("overwrite").saveAsTable("rpf.site_days")
intervals.write.mode("overwrite").saveAsTable("rpf.intervals")
```

`sdf` is a Spark DataFrame with the same four columns as the pandas call, mapped with
`columns` if they are named differently. Both outputs carry the schemas of the
[site-day and interval tables](quickstart.md).

## Group size

Each group is one site's readings: a year of one site is 35,000 rows, a bounded pandas
frame. For very long histories, `group_by=("site", "month")` bounds the group further; the
result is unchanged because days are scored independently.

## Two passes

The two tables are produced by two `applyInPandas` passes, each scoring the group again. A
site-day costs about six milliseconds, so a year of one site scores in a few seconds per
pass; on a cluster the passes run per site in parallel. Ask for one table only if the other
is not needed.

## Idempotence

The same day scores the same in any batch: the evidence floor is fixed (`RELEASE_PHI`) and
the calibration is two fixed numbers. Re-running over an overlapping period gives identical
rows, so a `MERGE` keyed on site and date is safe.

## What the calling pipeline owns

Reading the source tables, choosing the population, materialising the two tables, the
review workflow (see [Overrides](overrides.md)) and any audit trail. The package returns
proposals and never alters raw readings.

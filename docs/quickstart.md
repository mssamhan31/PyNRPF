# Quick start

## One call

```python
import pandas as pd
import pynrpf

frame = pd.read_parquet("readings.parquet")   # substation_id, timestamp, net_load_MW, solar_MW
result = pynrpf.run(frame)

result.site_days.head()      # one row per site and day
result.intervals.head()      # one row per reading
result.summary()             # days per outcome, MWh flipped
```

The frame needs four columns: a site identifier, a timestamp on the fifteen-minute grid, the
recorded net load in MW (positive = import as stored) and a solar generation estimate in MW.
Other column names are mapped with `columns`:

```python
result = pynrpf.run(frame, columns=dict(site="site_id", timestamp="ts", net_load="mw", solar="pv"))
```

A day is scored when all 96 readings are present; missing values inside a complete day are
allowed and only disqualify the windows that touch them. Timestamps with a time zone are
converted to UTC before the calendar day is taken; naive timestamps are taken as they are.

## The control

```python
result = pynrpf.run(frame, c=0.9)
```

`c` is the only control. A day is corrected automatically when its probability reaches `c`,
kept automatically when the probability is at most `1 - c`, and sent to review in between.
The release default is 0.7. Raw readings are never altered: the corrected series is a
separate column.

## The site-day table

| column | type | meaning |
|---|---|---|
| `site` | string | site identifier as given |
| `date` | string | calendar day, YYYY-MM-DD |
| `n_slots` | int64 | readings found for the day; 96 is complete |
| `input_ok` | bool | the day was complete and had at least one admissible window |
| `n_admissible` | int64 | candidate windows that could be scored |
| `evidence` | float64 | r*: evidence of the best window; NaN when not scored |
| `p` | float64 | calibrated probability that the day carries a wrong sign |
| `outcome` | string | AUTO_CORRECT, AUTO_KEEP or UNCERTAIN |
| `window_start` | int64 | first slot of the best window (0 = 00:00), -1 when none |
| `window_end` | int64 | last slot of the best window, inclusive, -1 when none |
| `runner_start` | int64 | first slot of the runner-up window, -1 when none |
| `runner_end` | int64 | last slot of the runner-up window, -1 when none |
| `runner_evidence` | float64 | evidence of the runner-up window |
| `proposed_mwh` | float64 | energy the best window would flip, MWh |
| `recorded_min_mw` | float64 | minimum recorded net load of the day, MW |
| `corrected_min_mw` | float64 | minimum after flipping the best window, MW |
| `min_change_mw` | float64 | corrected minus recorded minimum, MW |

Slots count fifteen-minute intervals from midnight: slot 24 is 06:00, slot 71 is 17:45.
The window and the energy are reported for every scored day, whatever the outcome, so a
reviewer sees what the tool would have done on an UNCERTAIN day.

## The interval table

| column | type | meaning |
|---|---|---|
| `site` | string | site identifier as given |
| `timestamp` | string | the reading's timestamp as given, ISO 8601 |
| `net_load_mw` | float64 | recorded net load, MW |
| `in_window` | bool | the slot lies inside the best window |
| `corrected` | bool | the sign was flipped: in the window and the day is AUTO_CORRECT |
| `net_load_corrected_mw` | float64 | recorded net load with the sign flipped where corrected, MW |

## A new population

The release calibration was fitted on eight Ausgrid substations with real, reviewed errors.
For a population whose ordinary days look different, refit the intercept from about 27
reviewed days and keep the slope; see [Calibration](calibration.md).

```python
from pynrpf import RELEASE_CALIBRATION, fit_calibration

pair = fit_calibration(reviewed["evidence"], reviewed["label"], slope=RELEASE_CALIBRATION.slope)
result = pynrpf.run(frame, calibration=pair)
```

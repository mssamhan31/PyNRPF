# Command line

```powershell
pynrpf run readings.csv --out results/
```

Reads a CSV with the four columns (`substation_id`, `timestamp`, `net_load_MW`, `solar_MW`
by default) and writes three files to the output folder:

| file | content |
|---|---|
| `site_days.csv` | the [site-day table](quickstart.md#the-site-day-table) |
| `intervals.csv` | the [interval table](quickstart.md#the-interval-table) |
| `summary.json` | days per outcome, MWh flipped, the control and calibration used |

## Options

| option | meaning | default |
|---|---|---|
| `--c` | the control | 0.7 |
| `--calibration a,b` | intercept and slope replacing the release calibration | release pair |
| `--phi` | the evidence floor, MW | release value |
| `--site`, `--timestamp`, `--net-load`, `--solar` | column names | `substation_id`, `timestamp`, `net_load_MW`, `solar_MW` |

```powershell
pynrpf run readings.csv --out results/ --c 0.9 --site site_id --net-load mw --solar pv
pynrpf --version
```

The same command reproduces the package's part of the paper: run it on
`publication/2_journal_article/dataset/final/dataset_beta.parquet` exported to CSV, and the
site-day rows equal the reference run's M9 rows for the release calibration.

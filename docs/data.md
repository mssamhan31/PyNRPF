# Data

All datasets in this repository derive from **anonymised Ausgrid distribution
network data**. Substation identifiers are replaced with opaque labels, so no
site, customer or location can be identified from what is published here.

Datasets are committed deliberately, so the published results can be reproduced
from the repository alone.

## Conference paper dataset

`publication/1_conference_paper/dataset/raw/rpf_dataset.parquet`, with a
`sha256.txt` checksum sidecar beside it.

1,011,264 rows: 10 substations at 15-minute resolution, 1 November 2021 to
30 September 2024.

| Column | Type | Meaning |
|---|---|---|
| `substation_id` | string | Anonymised substation label, `A` to `J` |
| `timestamp` | datetime, UTC | Interval start |
| `net_load_MW` | float | Recorded net load, megawatts. This is the series carrying wrong RPF signs. |
| `solar_MW` | float | Estimated local solar generation, megawatts |
| `net_load_ground_truth` | float | Manually reviewed correct net load, megawatts |

## Journal article datasets

Under `publication/2_journal_article/dataset/`, in three stages.

`processed/` holds the intermediate frames, both the actual dataset and a
synthetic counterpart; `final/` holds the three datasets the journal analysis
uses; `oracle_data_creation/` holds the manually reviewed oracle dataset that
Beta and Gamma are built from.

| Dataset | Sites | Period | Notes |
|---|---|---|---|
| Alpha | 10 | Nov 2021 – Sep 2024 | The full period, matching the conference dataset |
| Beta | 8 | Oct 2023 – Sep 2024 | Carries day-level reviewer `confidence` labels |
| Gamma | 1 | Oct 2023 – Sep 2024 | Single site, used for the forecast impact study |

`dataset_final_summary.csv` in `final/` records row counts, date ranges, null
counts and positive label counts for each.

## Ground truth

Labels come from manual review rather than a physical measurement. The
publication-ready export lives in `dataset/oracle_data_creation/`; the review
tooling, the per-reviewer annotation files and superseded outputs are retained
under `archive/` beside it, including the Streamlit application used to conduct
the review.

The export carries a day-level reviewer `confidence` field, repeated on every
15-minute row of the same site-day, valued `sure` or `unsure`. Beta and Gamma
retain it, so an analysis can be repeated on the confident subset alone.

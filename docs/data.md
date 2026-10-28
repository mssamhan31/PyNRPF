# Data

Every dataset in the repository derives from anonymised Ausgrid distribution network data.
Substation identifiers are replaced with opaque labels, so no site, customer or location can
be identified. The datasets are committed on purpose, so that the published results
reproduce from the repository alone.

## The journal datasets

Under `publication/2_journal_article/dataset/final/`, with a `sha256.txt` sidecar that the
evaluation checks before it runs.

| dataset | sites | period | rows | labelled days | what it is |
|---|---|---|---|---|---|
| Alpha | 10 | Nov 2021 to Sep 2024 | 1,011,264 | 3,423 | meters that read reverse flow negative correctly; the wrong sign is imitated by taking the absolute value, so every slot where the true value was negative is a label. Exact, plentiful, but includes tiny exports no reviewer would call |
| Beta | 8 | Oct 2023 to Sep 2024 | 280,800 | 630 | real meters with real wrong-sign errors, labelled by two reviewers who marked the span to flip and a confidence, `sure` or `unsure`, for every site-day (2,310 sure, 618 unsure) |
| Gamma | 1 | Oct 2023 to Sep 2024 | 35,136 | 152 | one Beta station used for the forecasting case study |

| column | type | meaning |
|---|---|---|
| `substation_id` | string | anonymised label, `alpha_A` to `alpha_J`, `beta_A` to `beta_H` |
| `timestamp` | datetime, UTC | interval start |
| `net_load_MW` | float | recorded net load, MW; the series that carries the wrong sign |
| `solar_MW` | float | estimated local solar generation, MW |
| `label_interval` | bool | the slot should be flipped |
| `label_day` | bool | the day carries a wrong sign |
| `confidence` | string | Beta and Gamma only: `sure` or `unsure`, repeated on every row of the day |

`dataset_final_summary.csv` records row counts, date ranges, null counts and label counts.

## How the Beta labels were made

Two reviewers inspected every site-day of the eight stations with the recorded trace and
the solar estimate, marked the span whose sign was wrong, and recorded whether they were
sure. Unsure days are cloudy days with a jagged solar estimate, meters that clip at zero,
and shallow midday dips that could be either story. The paper judges every method on the
sure days; unsure days are reported separately and never used to fit or tune anything. The
publication-ready export of that review, with its summary and review status, is
`publication/2_journal_article/dataset/oracle_data_creation/`; the review tooling and the
per-reviewer files are kept under `publication/2_journal_article/sandbox/oracle_review/`.

## The conference dataset

`publication/1_conference_paper/dataset/raw/rpf_dataset.parquet`: the same ten Alpha
stations with a manually reviewed correct net load in `net_load_ground_truth`, as used by
the conference paper. It is not read by the journal work.

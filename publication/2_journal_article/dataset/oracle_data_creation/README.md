# Final Oracle Dataset

This folder contains the final manually reviewed actual PyNRPF oracle dataset
used to build the journal article Beta and Gamma datasets.

Visible files are the publication-ready export:

- `actual_pynrpf_dataset_reflagged.csv`
- `actual_pynrpf_dataset_reflagged.parquet`
- `dataset_summary.csv`
- `review_status.json`
- `sha256.txt`

The exported dataset includes the original data and label columns plus
day-level reviewer `confidence`, repeated on every 15-minute row for the same
site-day. Confidence values are `sure` or `unsure`.

All review tooling, reviewer-specific annotation files, logs, and superseded
outputs are retained under `archive/`.

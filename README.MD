# PyNRPF v0.1.0

PyNRPF is a reproducible research repository for detecting and correcting reverse power flow (RPF) polarity errors (reverse flow recorded as positive). v0.1.0 targets reproducible headline metrics at both day-level and timestamp-level.

## What you do
1) Set `zenodo.record_id` in `configs/run.yaml`.
2) Run `notebooks/01_reproduce_key_numbers.ipynb`.

If `data/raw/rpf_dataset.parquet` is missing, the notebook downloads it from Zenodo automatically, then loads it with PySpark.

## Repo structure
- `configs/run.yaml` — configuration (Zenodo record id, file name, column mapping, split)
- `notebooks/01_reproduce_key_numbers.ipynb` — end-to-end entry point
- `data/raw/` — downloaded parquet (not committed)
- `outputs/` — generated metrics (not committed)

## Output
The notebook writes a metrics JSON to `outputs/metrics__<run_tag>.json`. In Chunk 1 this file is a stub; later chunks populate full evaluation metrics.

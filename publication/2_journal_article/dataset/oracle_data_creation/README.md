# Oracle Data Creation Review Tool

This folder contains the local manual-review workflow for rebuilding the actual
RPF oracle labels for 2023-10-01 through 2024-09-30.

## Setup

From this folder:

```powershell
python -m pip install -r requirements.txt
streamlit run review_app.py
```

The app reads the immutable source dataset from:

```text
../processed/actual_pynrpf_dataset.csv
```

Manual decisions are stored in:

```text
manual_oracle_annotations.csv
```

## Review Rules

- Review only 2023-10-01 through 2024-09-30.
- Review site order is `act_D`, `act_F`, `act_B`, `act_G`, `act_A`, `act_E`,
  `act_H`, `act_C`, with dates sorted chronologically inside each site.
- A reviewed site-day has one row in `manual_oracle_annotations.csv`.
- `review_action=accept_old` keeps the original interval flags exactly.
- `review_action=manual_window` uses one inclusive 15-minute window from
  `rpf_start_time` through `rpf_end_time`.
- `review_action=no_rpf` means no intervals are flagged for that site-day.
- Unreviewed site-days retain their original labels in draft exports.

The Streamlit app opens on weekly review. Use `Accept old week` or
`Accept + next` when the current labels look correct, and switch to the daily
override tab only for days that need manual start/end correction.

## Export

To generate the reviewed-year dataset:

```powershell
python export_reflagged_dataset.py
```

Outputs are written to `outputs/`:

- `actual_pynrpf_dataset_reflagged.csv`
- `actual_pynrpf_dataset_reflagged.parquet`
- `dataset_summary.csv`
- `review_status.json`
- `sha256.txt`

The exported dataset keeps the original seven columns and never changes raw
`net_load_MW` values.

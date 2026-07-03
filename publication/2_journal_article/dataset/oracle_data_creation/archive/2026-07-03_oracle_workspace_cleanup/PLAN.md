# Oracle RPF Review Workflow Plan

## Summary
Build a Python review workflow under `publication/2_journal_article/dataset/oracle_data_creation/`.

Use Streamlit + Plotly for a fast local visual review app. The review scope is only
2023-10-01 to 2024-09-30, covering 280,800 rows and 2,928 site-days. The final
regenerated dataset contains only that reviewed year, with the exact original
seven-column format.

## Key Changes
- Create a simple manual side table: `manual_oracle_annotations.csv`
  - Columns: `substation_id,date,review_action,rpf_start_time,rpf_end_time`
  - One row per reviewed site-day.
  - No reviewer, confidence, notes, or reviewed timestamp.
  - Absence from the table means unreviewed.
  - `accept_old` keeps the original interval flags exactly.
  - `manual_window` uses `HH:MM` start/end values at 15-minute resolution,
    inclusive.
  - `no_rpf` clears all interval flags for that site-day.

- Create a Streamlit review app:
  - Load `actual_pynrpf_dataset.parquet` with cached pandas loading.
  - Treat source timestamps as dataset wall-clock time; do not convert `+00:00` to
    Sydney time.
  - Weekly review is the default app view for fast review.
  - Daily view shows raw `net_load_MW`, old-label corrected net load,
    `solar_MW`, zero line, and shaded current `label_interval=True` regions.
  - Provide previous/next day controls plus direct jump to `substation_id` and
    `date`.
  - Weekly view shows seven-day context for the selected site and lets the user
    select a day for daily annotation.
  - Save buttons: mark RPF window, mark no RPF, clear annotation.

- Review queue:
  - Scope: `2023-10-01` through `2024-09-30` only.
  - Site order: `act_D`, `act_F`, `act_B`, `act_G`, `act_A`, `act_E`,
    `act_H`, `act_C`.
  - Within each site, sort dates chronologically.

- Export workflow:
  - Generate `outputs/actual_pynrpf_dataset_reflagged.csv` and `.parquet`.
  - Output contains only reviewed-year rows and the original columns:
    `substation_id,date,timestamp,net_load_MW,solar_MW,label_interval,label_day`.
  - Only labels change; raw `net_load_MW` is never sign-flipped.
  - For reviewed days, replace interval labels from the manual side table.
  - For unreviewed days, retain old labels until review is complete.
  - Always use the same output filename, plus write `outputs/review_status.json`,
    `dataset_summary.csv`, and `sha256.txt`.

## Test Plan
- Validate input schema exactly matches the seven expected columns.
- Confirm reviewed-year filter returns 280,800 rows and 2,928 site-days.
- Test `accept_old` preserves old interval labels exactly.
- Test inclusive 15-minute start/end conversion into `label_interval`.
- Test `rpf_present=False` clears all interval labels for that site-day.
- Test `label_day` is recomputed from grouped `label_interval`.
- Test exported CSV preserves column order and raw measurement values.
- Smoke-test the app: load data, jump to site/date, navigate previous/next,
  save/clear annotation, and open weekly context.

## Assumptions
- A single continuous RPF window per site-day is sufficient.
- The timestamp clock in the CSV is the review clock, despite the `+00:00`
  suffix.
- The original dataset is immutable; all manual review artifacts and regenerated
  outputs stay under `oracle_data_creation`.
- The one-year reviewed output is considered complete only when all 2,928
  site-days have a manual annotation.

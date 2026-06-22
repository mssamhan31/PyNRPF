# Journal Article Notebook Experiment Plan

## Summary

This notebook workflow implements the Alpha/Beta/Gamma experiment design for the
journal article on reverse-power-flow sign errors. It replaces the previous
time-split/station-split notebook set with a clearer sequence:

1. `00_prepare_datasets.ipynb`
2. `01_characterisation.ipynb`
3. `02_correction_validation.ipynb`
4. `03_gamma_forecast_impact.ipynb`
5. `04_publication_tables_figures.ipynb`

The notebooks use shared helpers in `_experiment_helpers.py` so that notebooks
remain readable while validation, metrics, training, forecasting, and plotting
logic stay testable.

## Dataset Definitions

- **Dataset Alpha**: `dataset/processed/synthetic_pynrpf_dataset.csv`.
  This is the simulated sign-error dataset from correctly signed substations.
- **Dataset Beta**: for now,
  `dataset/processed/actual_pynrpf_dataset.csv` filtered to
  `2023-10-01` through `2024-09-30`. Later, set
  `datasets.use_reviewed_beta: true` to use the completed manual-review oracle.
- **Dataset Gamma**: selected automatically from Beta. The primary ranking is
  raw-vs-reference data-error RMSE, then RPF days, then RPF intervals. Current
  labels select `act_B`.

## Notebook Outputs

All notebooks write beneath `publication/2_journal_article/outputs/`:

- `intermediate/`: reusable per-notebook CSVs.
- `metrics/`: correction and forecasting metrics.
- `tables/`: paper table exports.
- `figures/`: publication figures.
- `manifests/`: run metadata and config snapshots.

## Notebook Roles

### 00 Prepare Datasets

- Load and validate Alpha and Beta.
- Filter Beta to the one-year study period.
- Compute dataset summaries.
- Rank Alpha sites by RPF days and select top-3 LOSO folds.
- Rank Beta sites for Gamma and select Gamma.
- Write:
  - `intermediate/dataset_summary.csv`
  - `intermediate/alpha_site_rankings.csv`
  - `intermediate/beta_gamma_site_rankings.csv`
  - `manifests/00_prepare_datasets.json`

### 01 Characterisation

- Characterise Alpha observed negative/RPF-labelled intervals and Beta sign
  errors.
- Summarise site, month, season, hour, weekday/weekend, event duration, and
  magnitude patterns.
- Write:
  - `intermediate/rpf_occurrence_by_site.csv`
  - `intermediate/rpf_temporal_summary.csv`
  - `intermediate/rpf_event_summary.csv`
  - `figures/site_rpf_day_counts.png`
  - `figures/month_hour_heatmap_alpha.png`
  - `figures/month_hour_heatmap_beta.png`
  - `figures/event_duration_distribution.png`

### 02 Correction Validation

- Evaluate `m8_xgb` as the main trainable method.
- Evaluate `m7_dtr` as the deterministic benchmark.
- Alpha validation: top-3 leave-one-station-out spatiotemporal folds.
- Beta validation: train `m8_xgb` on Alpha training data and test on Beta.
- Metrics:
  - Day-level precision, recall, F1, TP, FP, FN, TN.
  - Interval-level metrics over all daytime rows, 06:00-18:00.
- Write:
  - `metrics/correction_metrics.csv`
  - `metrics/correction_confusion_matrices.csv`
  - `intermediate/correction_predictions_*.csv`
  - `figures/correction_confusion_matrices.png`
  - `manifests/02_correction_validation.json`

### 03 Gamma Forecast Impact

- Select Gamma from Beta automatically.
- Compare raw uncorrected, `m8_xgb` corrected, and reference-labelled net load.
- Include a data-error-only benchmark where the forecast perfectly predicts raw
  data and is evaluated against reference.
- Forecast task: exactly 7 days ahead, one target point per 15-minute timestamp.
- Test targets: all 15-minute timestamps in September 2024.
- Features: 14-day observed lookback ending at target minus 7 days, plus
  calendar features; no future solar.
- Models: seasonal naive, linear regression, XGBoost.
- Write:
  - `intermediate/gamma_forecast_examples_train.csv`
  - `intermediate/gamma_forecast_examples_test.csv`
  - `intermediate/gamma_forecasts.csv`
  - `metrics/gamma_forecast_metrics.csv`
  - `metrics/gamma_data_error_benchmark.csv`
  - `figures/gamma_series_raw_corrected_reference.png`
  - `figures/gamma_forecast_rmse.png`
  - `figures/gamma_forecast_residuals.png`

### 04 Publication Tables Figures

- Read intermediate and metric CSVs.
- Produce final publication-facing table and figure files.
- Write:
  - `tables/table1_dataset_summary.csv`
  - `tables/table2_characterisation_summary.csv`
  - `tables/table3_correction_metrics.csv`
  - `tables/table4_forecast_impact.csv`
  - final paper figures in `figures/`.

## Execution Notes

- `execution.run_full_correction_validation` defaults to `false` so notebooks can
  smoke-run quickly.
- `execution.run_full_forecast` defaults to `false`; the data-error-only Gamma
  benchmark still runs in smoke mode.
- Full `m8_xgb` training should be enabled only after dataset labels are final.
- PyNNLF is intentionally excluded because this workflow needs custom raw,
  corrected, and reference data-condition comparisons.

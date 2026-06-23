# Journal Article Notebook Experiment Plan v2

## Summary

This notebook workflow implements the paper-facing Alpha/Beta/Gamma experiment
design using the v2 final dataset layer. The notebooks read Parquet inputs from
`dataset/final/`, write audit-friendly intermediate outputs into
notebook-specific folders, and keep final tables/figures deliberately compact.

The notebook sequence is:

1. `00_prepare_datasets.ipynb`
2. `01_characterisation.ipynb`
3. `02_correction_validation.ipynb`
4. `03_gamma_forecast_impact.ipynb`
5. `04_publication_tables_figures.ipynb`

Shared logic lives in `_experiment_helpers.py` so the notebooks can remain
readable experiment narratives with numbered markdown sections and short,
commented code cells.

## Dataset Definitions

- **Dataset Alpha**: `dataset/final/dataset_alpha.parquet`, copied from the
  processed synthetic Parquet and kept at the full Alpha date range.
- **Dataset Beta**: `dataset/final/dataset_beta.parquet`, copied from the
  processed actual Parquet and filtered to `2023-10-01` through `2024-09-30`.
  This is provisional until the manual oracle review output replaces it.
- **Dataset Gamma**: `dataset/final/dataset_gamma.parquet`, a one-site extract
  from Beta selected by Gamma candidate ranking. The current automatic ranking
  uses raw-vs-reference data-error RMSE and selects `beta_B` on current labels.

Final dataset site IDs are paper-facing aliases: Alpha sites use `alpha_*` and
Beta/Gamma sites use `beta_*`. The processed source files retain their original
raw IDs.

All final datasets preserve the original seven-column schema:
`substation_id,date,timestamp,net_load_MW,solar_MW,label_interval,label_day`.

## Output Convention

Every notebook writes into its own subfolders:

```text
outputs/intermediate/<notebook_slug>/
outputs/metrics/<notebook_slug>/
outputs/tables/<notebook_slug>/
outputs/figures/<notebook_slug>/
outputs/manifests/<notebook_slug>.json
```

Intermediate outputs can be detailed. Final tables and figures should be the
small set most likely to appear in the paper or directly support paper choices.

## Notebook Roles

### 00 Prepare Datasets

Purpose: create and validate final Alpha, Beta, and Gamma datasets.

Key outputs:

- `dataset/final/dataset_alpha.parquet`
- `dataset/final/dataset_beta.parquet`
- `dataset/final/dataset_gamma.parquet`
- `dataset/final/dataset_final_summary.csv`
- `dataset/final/gamma_selection_summary.csv`
- `dataset/final/sha256.txt`
- `outputs/intermediate/00_prepare_datasets/01_dataset_summary.csv`
- `outputs/intermediate/00_prepare_datasets/02_alpha_site_rankings.csv`
- `outputs/intermediate/00_prepare_datasets/03_beta_gamma_site_rankings.csv`
- `outputs/intermediate/00_prepare_datasets/04_final_dataset_validation.csv`

No final paper-facing figures or tables are required from this notebook.

### 01 Characterisation

Purpose: characterise RPF/sign-error occurrence in Alpha and Beta only.

Key outputs:

- `outputs/intermediate/01_characterisation/01_rpf_occurrence_by_dataset.csv`
- `outputs/intermediate/01_characterisation/02_rpf_occurrence_by_site.csv`
- `outputs/intermediate/01_characterisation/03_rpf_temporal_summary.csv`
- `outputs/intermediate/01_characterisation/04_rpf_event_summary.csv`
- `outputs/tables/01_characterisation/table01_rpf_occurrence_summary_alpha_beta.csv`
- `outputs/tables/01_characterisation/table02_rpf_event_summary_alpha_beta.csv`
- `outputs/figures/01_characterisation/fig01_site_rpf_day_counts_alpha_beta.png`
- `outputs/figures/01_characterisation/fig02_month_hour_heatmap_alpha_beta.png`
- `outputs/figures/01_characterisation/fig03_event_duration_distribution_alpha_beta.png`

Gamma-specific final artefacts are intentionally left to notebook 03.

### 02 Correction Validation

Purpose: evaluate `m8_xgb` as the main correction model and `m7_dtr` as the
deterministic benchmark.

Key outputs:

- `outputs/intermediate/02_correction_validation/01_correction_validation_plan.csv`
- `outputs/intermediate/02_correction_validation/*_correction_predictions_*.csv`
- `outputs/metrics/02_correction_validation/01_correction_metrics.csv`
- `outputs/metrics/02_correction_validation/02_correction_confusion_matrices.csv`
- `outputs/tables/02_correction_validation/table01_correction_metrics_summary.csv`
- `outputs/tables/02_correction_validation/table02_beta_transfer_key_metrics.csv`
- `outputs/figures/02_correction_validation/fig01a_confusion_matrices_day.png`
- `outputs/figures/02_correction_validation/fig01b_confusion_matrices_interval.png`
- `outputs/figures/02_correction_validation/fig02a_precision_recall_f1_day.png`
- `outputs/figures/02_correction_validation/fig02b_precision_recall_f1_interval.png`

The config keeps full model training disabled by default so the notebook can be
opened and inspected before running expensive work.

### 03 Gamma Forecast Impact

Purpose: quantify how a one-site Gamma RPF case affects 7-day-ahead point
forecasting.

Key outputs:

- `outputs/intermediate/03_gamma_forecast_impact/01_gamma_series.csv`
- `outputs/intermediate/03_gamma_forecast_impact/*_gamma_forecast_examples_*.csv`
- `outputs/intermediate/03_gamma_forecast_impact/*_gamma_forecasts.csv`
- `outputs/metrics/03_gamma_forecast_impact/01_gamma_perfect_model_baseline.csv`
- `outputs/metrics/03_gamma_forecast_impact/02_gamma_forecast_metrics.csv`
- `outputs/tables/03_gamma_forecast_impact/table01_forecast_impact.csv`
- `outputs/tables/03_gamma_forecast_impact/table02_gamma_perfect_model_baseline.csv`
- `outputs/figures/03_gamma_forecast_impact/fig01_gamma_series_raw_corrected_reference.png`
- `outputs/figures/03_gamma_forecast_impact/fig02_gamma_forecast_rmse.png`
- `outputs/figures/03_gamma_forecast_impact/fig03_gamma_forecast_residuals.png`

Smoke mode produces the real perfect-model baseline plus placeholder forecast
rows for layout checks. Full mode trains the forecast models and constructs the
real `m8_xgb`-corrected Gamma series.

### 04 Publication Tables Figures

Purpose: consolidate upstream notebook outputs into a small paper-facing table
set and inventories of available figures.

Key outputs:

- `outputs/tables/04_publication_tables_figures/table01_dataset_summary.csv`
- `outputs/tables/04_publication_tables_figures/table02_characterisation_summary.csv`
- `outputs/tables/04_publication_tables_figures/table03_correction_metrics.csv`
- `outputs/tables/04_publication_tables_figures/table04_forecast_impact.csv`
- matching `.md` and `.tex` exports for the four final tables
- `outputs/intermediate/04_publication_tables_figures/01_table_inventory.csv`
- `outputs/intermediate/04_publication_tables_figures/02_figure_inventory.csv`
- `outputs/intermediate/04_publication_tables_figures/03_missing_upstream_outputs.csv`

## Execution Notes

- `execution.run_full_correction_validation` defaults to `false`.
- `execution.run_full_forecast` defaults to `false`.
- Full training should be enabled only after the final oracle-reviewed Beta is
  ready.
- PyNNLF remains excluded because this workflow needs custom raw, corrected, and
  reference data-condition comparisons.

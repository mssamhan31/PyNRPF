# PyNRPF v0.1.2
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18764944.svg)](https://doi.org/10.5281/zenodo.18764944)

PyNRPF reproduces reverse power flow (RPF) detection and correction results using:
- `m7_dtr` (deterministic threshold rule; config key `m7_threshold`)
- `m8_xgb` (two-stage XGBoost)

The repository publishes three artifact groups:
- core metrics and models
- publication figures
- publication tables

## Contributors

- M. Syahman Samhan (contact: m.samhan@unsw.edu.au)
- Anna Bruce
- Baran Yildiz

## Quick Start (Windows)

```powershell
# 1) Clone and enter repo
git clone <repo-url>
cd PyNRPF

# 2) Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt

# 4) Place dataset
#    data/raw/rpf_dataset.parquet

# 5) Run notebooks in order (01 -> 02 -> 03)
```

## Reproduce Outputs

Run in this order so downstream notebooks use current upstream outputs.

### Option A: Jupyter Run All

1. `notebooks/01_reproduce_key_numbers.ipynb`
2. `notebooks/02_publication_figures.ipynb`
3. `notebooks/03_publication_tables.ipynb`

### Option B: CLI execution

```powershell
.\.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/01_reproduce_key_numbers.ipynb --output 01_reproduce_key_numbers.executed.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=7200
.\.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/02_publication_figures.ipynb --output 02_publication_figures.executed.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=7200
.\.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/03_publication_tables.ipynb --output 03_publication_tables.executed.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=7200
```

## Notebook Map

### 01_reproduce_key_numbers.ipynb
Purpose:
- load config and dataset
- validate data
- run `m7_dtr` and `m8_xgb`
- evaluate train/test metrics
- write metrics and trained models

Primary outputs:
- `outputs/metrics__<run_tag>.json`
- `outputs/metrics__<run_tag>.yaml`
- `outputs/xgb1_day.pkl`
- `outputs/xgb2_timestamp.pkl`

### 02_publication_figures.ipynb
Purpose:
- generate publication figures
- create Figure 1 and Figure 2 from dataset/model outputs
- create confusion-matrix figures using **metrics JSON counts only**

Primary outputs:
- `outputs/publication_figures/fig01_sample_day_mw_vs_ground_truth.png`
- `outputs/publication_figures/fig02_dtr_illustration.png`
- `outputs/publication_figures/archived/confusion_matrices_all_daytime.png`
- `outputs/publication_figures/archived/confusion_matrices_tp_days_only_interval.png`
- `outputs/publication_figures/fig03_confusion_matrices_1x4.png`

### 03_publication_tables.ipynb
Purpose:
- export publication tables as CSV
- Table 2 is read from metrics JSON

Primary outputs:
- `outputs/publication_tables/table1_dataset_split_summary.csv`
- `outputs/publication_tables/table2_model_performance_summary.csv`
- `outputs/publication_tables/table3_implementation_mw_difference.csv`
- `outputs/publication_tables/table4_implementation_ts_difference.csv`

## Single Source of Truth for Metrics

`outputs/metrics__<run_tag>.json` is the single truth source for publication performance numbers.

- Notebook 3 Table 2 reads from this file.
- Notebook 2 confusion-matrix figure counts/annotations read from this file.

## Metric Scope Definitions

Day-level:
- one prediction per `(site, date)`
- days with already-negative raw `net_load_MW` are excluded

Interval-level (daytime only, 06:00 <= hour < 18:00):
- `interval_all_days`: intervals on predicted-positive days (`pred_day == 1`)  
  (this is the Option A definition used in figures/tables)
- `interval_tp_days_only`: intervals on true-positive days (`pred_day == 1` and true day RPF)

## Configuration

Runtime configuration is in `config/run.yaml`:
- `run`: run tag and seed
- `paths`: dataset and output paths
- `data`: column names and interval
- `validation`: validation behavior
- `split`: train/test date ranges
- `m7_threshold`: deterministic rule parameters (method label: `m7_dtr`)
- `m8_xgb`: XGB stage-1 and stage-2 parameters
- `evaluation`: rounding and metric fields

Default split:
- train: 2021-11-01 to 2023-09-30
- test: 2023-10-01 to 2024-09-30

## Repository Layout

```text
config/run.yaml
notebooks/
  01_reproduce_key_numbers.ipynb
  02_publication_figures.ipynb
  03_publication_tables.ipynb
src/
  io.py
  validate.py
  m7_threshold.py
  m8_xgb.py
  features.py
  evaluate.py
data/raw/
outputs/
```

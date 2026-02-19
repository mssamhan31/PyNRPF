# PyNRPF v0.1.0

PyNRPF is a reproducible research repository for detecting and correcting reverse power flow (RPF) polarity errors — cases where reverse flow is recorded as positive load. v0.1.0 targets reproducible headline metrics at both day-level and timestamp-level.

## Quick start (Windows)

```powershell
# 1. Clone and enter the repo
git clone <repo-url>
cd PyNRPF

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure the dataset is in place
#    Place rpf_dataset.parquet at data/raw/rpf_dataset.parquet
#    (or set zenodo.record_id in config/run.yaml for auto-download)

# 5. Open notebooks/01_reproduce_key_numbers.ipynb and Run All
```

The notebook runs end-to-end: it loads the config, downloads or validates the dataset, runs both detection methods, evaluates them, and writes metrics + plots to `outputs/`.

## Configuration

All parameters live in `config/run.yaml`:

| Section | Purpose |
|---|---|
| `run` | Run tag and random seed |
| `paths` | Dataset path, SHA-256 file, output directory |
| `zenodo` | Zenodo record ID and filename (for auto-download) |
| `data` | Interval settings, column name mapping |
| `validation` | Flags for SHA-256 check, timezone stripping, alignment, uniqueness |
| `split` | Train/test date boundaries |
| `m7_threshold` | Deterministic method hyperparameters |
| `m8_xgb` | XGBoost day + timestamp model hyperparameters |
| `evaluation` | Metrics list, rounding, filtering |

Changing column names or hyperparameters requires editing **only** the YAML file, not the notebook code.

### Train / test split

- **Train:** 2021-11-01 to 2023-09-30
- **Test:** 2023-10-01 to 2024-09-30

## Methods

### m7_threshold — Deterministic Threshold Rule (DTR)

A rule-based method that identifies RPF by examining the shape of daytime MW profiles relative to the solar generation curve. For each (site, date):

1. Locate the solar generation peak during midday hours.
2. Search for candidate MW peaks near the solar peak, then find left and right minima flanking each candidate.
3. Apply configurable thresholds (as a fraction of the day's MW maximum) to decide whether the MW shape is consistent with masked reverse flow.
4. Flag the intervals strictly between the two minima.

This method does **not** use the ground-truth column — it operates solely on observed `net_load_MW` and `solar_MW`.

### m8_xgb — XGBoost Two-Stage Classifier

A machine-learning approach using two XGBoost models trained on historical labels:

- **Stage 1 (XGB1):** Day-level binary classifier. Uses 221 features (substation identity, calendar features, 96-interval MW/solar profiles, statistical aggregates, missingness). Predicts whether a given (site, date) contains any RPF.
- **Stage 2 (XGB2):** Interval-level binary classifier. Runs only on days flagged by XGB1. Uses 665 features (all day-level features plus time-of-day encoding, underlying MW estimates, rate-of-change features). Predicts which specific 15-minute intervals are affected.

Trained models are saved as `.pkl` files in `outputs/`.

## Evaluation

Both methods are scored on precision, recall, and F1 at two granularities:

- **Day-level:** One prediction per (site, date). Days where the raw input `net_load_MW` already has negative values are excluded (these are already-obvious RPF, not the target of detection).
- **Interval-level:** One prediction per 15-minute interval, restricted to daytime hours (06:00–18:00). Reported in two variants:
  - `interval_all` — all eligible daytime intervals.
  - `interval_rpf_days_only` — only intervals on days that truly contain RPF (ground truth).

Confusion matrix plots (test set, 4 panels: m7 day, m7 interval, m8 day, m8 interval) are saved to `outputs/confusion_matrices.png`.

Ground truth (`net_load_ground_truth`) is used **only** in the evaluation module — it is never accessed by the detection methods.

## Repo structure

```
config/run.yaml             — all configuration (single source of truth)
notebooks/
  01_reproduce_key_numbers.ipynb — end-to-end entry point
src/
  io.py                    — config loading, SHA-256, parquet I/O, JSON/YAML export
  validate.py              — column checks, timestamp parsing, duplicate/missingness
  m7_threshold.py          — deterministic threshold rule (DTR)
  m8_xgb.py                — two-stage XGBoost pipeline
  features.py              — feature engineering for XGB1 + XGB2
  evaluate.py              — metrics computation + confusion matrix plots
data/raw/                   — parquet dataset (not committed)
outputs/                    — generated artifacts (not committed)
```

## Outputs

After a full notebook run, `outputs/` contains:

| File | Description |
|---|---|
| `metrics__<run_tag>.json` | Full evaluation results (JSON) |
| `metrics__<run_tag>.yaml` | Full evaluation results (YAML) |
| `confusion_matrices.png` | 2×2 confusion matrix plot (test set) |
| `xgb1_day.pkl` | Trained XGB1 day-level model |
| `xgb2_timestamp.pkl` | Trained XGB2 interval-level model |

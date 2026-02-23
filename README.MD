# PyNRPF v0.1.0

PyNRPF is a reproducible repository for detecting and correcting reverse power flow (RPF) polarity errors (reverse flow recorded as positive load). The pipeline reports day-level and interval-level metrics for both deterministic and ML methods.

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

# 5. Open notebooks/01_reproduce_key_numbers.ipynb and Run All
```

The notebook runs end-to-end: it loads config, validates and reads the local parquet, runs `m7_threshold` and `m8_xgb`, evaluates both, and writes artifacts to `outputs/`.

## Optional: VS Code + Databricks Connect hello check

This repository now includes a minimal Databricks Connect setup so you can validate cluster connectivity from VS Code without changing the existing PyNRPF pipeline structure.

```powershell
# 1. Create local connect config
Copy-Item config/databricks_connect.example.yaml config/databricks_connect.yaml

# 2. One-time setup (venv + dependencies + Databricks CLI + OAuth + smoke test)
.\scripts\setup_local_connect.ps1 `
  -WorkspaceHost "https://<your-workspace-host>" `
  -Profile "dbx" `
  -ClusterId "<your-cluster-id>"

# 3. Run hello check through Databricks Connect
.\scripts\run_local_connect.ps1 `
  -ConfigPath "config/databricks_connect.yaml" `
  -Profile "dbx"
```

Notes:
- `setup_local_connect.ps1` installs `databricks-connect` with default version `16.4.*`.
- If your cluster runtime is different, pass `-DatabricksConnectVersion "<major.minor>.*"` to setup.

## Configuration

All runtime settings are in `config/run.yaml`:

| Section | Purpose |
|---|---|
| `run` | Run tag and random seed |
| `paths` | Local dataset path, SHA-256 file, output directory |
| `data` | Interval settings and column name mapping |
| `validation` | SHA-256 check, timezone stripping, alignment, uniqueness controls |
| `split` | Train/test date boundaries |
| `m7_threshold` | Deterministic method hyperparameters |
| `m8_xgb` | XGBoost stage-1 and stage-2 hyperparameters |
| `evaluation` | Metric names and rounding |

### Train / test split

- **Train:** 2021-11-01 to 2023-09-30
- **Test:** 2023-10-01 to 2024-09-30

## Methods

### `m7_threshold` — Deterministic Threshold Rule (DTR)

`m7_threshold` operates per `(site, date)` and follows these exact gates/steps from `src/m7_threshold.py`:

1. **Gate: missing MW** — skip day if any `net_load_MW` is null.
2. **Gate: already negative MW** — skip day if any `net_load_MW < 0`.
3. **Gate: midday sufficiency** — require at least 3 intervals in midday window (`06:00` to `<18:00`).
4. **Daily thresholds** based on day max MW:
   - `th = MW_max * min_threshold`
   - `th_both = MW_max * min_threshold_both`
5. **Solar peak selection** in midday:
   - choose max `solar_MW`
   - ties broken by nearest configured `solar_peak_tiebreak_time` (then earliest timestamp)
   - if no valid midday solar, fallback candidate window is `10:00–14:59:59`
6. **Candidate MW peaks** in midday:
   - strict local maxima (`mw[t] > mw[t-1]` and `mw[t] > mw[t+1]`), boundaries excluded
   - must lie inside `peak_window_minutes` around solar peak (or fallback window)
7. **Left/right minima search** for each candidate (over full day, not midday-only):
   - left min: `timestamp < peak` and `MW < th_both`
   - right min: `timestamp > peak` and `MW < th_both`
   - choose lowest MW on each side
8. **Best pair selection**:
   - minimize raw `(left_mw + right_mw)`
   - tie-break by larger peak MW
9. **Final decision gates**:
   - at least one minimum is below `th`
   - both minimum timestamps are within `[06:00:00, 18:00:00]`
10. **Interval flagging rule**:
   - mark intervals strictly between minima (`left_ts < ts < right_ts`)

Outputs added by `m7`:
- `m7_rpf_flag` (interval bool)
- `m7_rpf_day` (day bool = any flagged interval)
- `net_load_MW_m7` (sign-corrected series)

### `m8_xgb` — Two-stage XGBoost

`m8_xgb` in `src/m8_xgb.py` trains two classifiers and applies them end-to-end:

1. **Stage 1 (`xgb1_day`)**
   - Input: one row per `(site, date)` from `build_xgb1_features`
   - Label: `is_having_reverse_power_flow_issue` (`1` if any ground truth interval is negative)
   - Output: `m8_prob_day`, `m8_rpf_day`
   - Threshold: `m8_xgb.xgb1_day.threshold`

2. **Stage 2 (`xgb2_timestamp`)**
   - Runs only on site-days predicted positive by Stage 1
   - Input: interval rows from `build_xgb2_features`
   - Label: `sign_is_wrong` (`1` if interval ground truth is negative)
   - Output: `m8_prob_ts`, `m8_rpf_flag`
   - Threshold: `m8_xgb.xgb2_timestamp.threshold`

3. **Mapping and correction**
   - predictions are mapped back to original interval table
   - corrected load is written to `net_load_MW_m8`

Model artifacts:
- `outputs/xgb1_day.pkl`
- `outputs/xgb2_timestamp.pkl`

`paths.output_dir` is resolved from repository root, so running from notebook CWD still writes to repo `outputs/`.

## Feature schema (`src/features.py`)

### XGB1 day-level features

Base keys: `(substation_id, date)`

Label:
- `is_having_reverse_power_flow_issue`

Feature groups:
1. **Substation OHE**: `is_<substation_name_sanitized>` (count = number of substations, denoted `N`)
2. **Calendar (13 cols)**:
   - `is_holiday`, `is_weekend`
   - `is_Jan` ... `is_Nov`
3. **MW profile pivots (96 cols)**:
   - `MW_0000` ... `MW_2345`
4. **MW aggregates (2 cols)**:
   - `MW_max`, `MW_noon_min`
5. **Solar profile pivots (96 cols)**:
   - `Solar_MW_0000` ... `Solar_MW_2345`
6. **Solar aggregates (2 cols)**:
   - `Solar_MW_max`, `Solar_MW_sum`
7. **Missingness (2 cols)**:
   - `mw_missing_count`, `solar_missing_count`

Total XGB1 features: `211 + N`.

### XGB2 interval-level features

Primary keys: `(substation_id, date, timestamp)`

Label:
- `sign_is_wrong`

Feature groups:
1. **Substation OHE**: same `N` columns as XGB1
2. **Interval time OHE (52 cols)**:
   - `hhmm_0600` ... `hhmm_1845` (15-min daytime slots)
3. **Calendar (13 cols)**:
   - `is_holiday`, `is_weekend`, `is_Jan` ... `is_Nov`
4. **MW profile + aggs (98 cols)**:
   - `MW_0000` ... `MW_2345`
   - `MW_max`, `MW_noon_min`
5. **Solar profile + aggs (98 cols)**:
   - `Solar_MW_0000` ... `Solar_MW_2345`
   - `Solar_MW_max`, `Solar_MW_sum`
6. **Underlying MW profile + aggs (98 cols)**:
   - `MW_underlying_0000` ... `MW_underlying_2345`
   - `MW_underlying_max`, `MW_underlying_noon_min`
7. **Rate-of-change pivots (288 cols)**:
   - `roc_MW_0000` ... `roc_MW_2345`
   - `roc_Solar_MW_0000` ... `roc_Solar_MW_2345`
   - `roc_MW_underlying_0000` ... `roc_MW_underlying_2345`
8. **ROC aggregates (6 cols)**:
   - `roc_MW_max`, `roc_MW_min`
   - `roc_Solar_MW_max`, `roc_Solar_MW_min`
   - `roc_MW_underlying_max`, `roc_MW_underlying_min`
9. **Missingness (2 cols)**:
   - `mw_missing_count`, `solar_missing_count`

Total XGB2 features: `655 + N`.

## Evaluation

Both methods are reported at day and interval granularity:

- **Day-level:** one prediction per `(site, date)`; days with already-negative raw `net_load_MW` are excluded from scoring.
- **Interval-level (daytime only, 06:00–18:00):**
  - `interval_all_days`: intervals on predicted-positive days (`pred_day == 1`)
  - `interval_tp_days_only`: intervals on true-positive days (`pred_day == 1` and true RPF day)

Metrics include precision, recall, F1, and confusion components (TP/FP/FN/TN) for train and test windows.

## Repo structure

```
config/run.yaml             — central configuration
notebooks/
  01_reproduce_key_numbers.ipynb
src/
  io.py                    — config/parquet/hash I/O helpers
  validate.py              — schema/time/uniqueness validation
  m7_threshold.py          — deterministic DTR implementation
  m8_xgb.py                — two-stage XGBoost training + inference
  features.py              — XGB1/XGB2 feature engineering
  evaluate.py              — metrics + confusion matrix plots
data/raw/
outputs/
```

## Outputs

After a full run, `outputs/` contains:

| File | Description |
|---|---|
| `metrics__<run_tag>.json` | Full metrics payload (JSON) |
| `metrics__<run_tag>.yaml` | Full metrics payload (YAML) |
| `confusion_matrices.png` | 2×2 test confusion matrix figure |
| `xgb1_day.pkl` | Trained Stage-1 XGB model + metadata |
| `xgb2_timestamp.pkl` | Trained Stage-2 XGB model + metadata |

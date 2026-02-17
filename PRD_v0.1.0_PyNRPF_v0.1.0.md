# PRD v0.1.0 — PyNRPF v0.1.0 (Reverse Power Flow Polarity Error Detection & Correction)

## 1. Purpose
PyNRPF v0.1.0 is a reproducible research repository to accompany a conference paper and a journal paper on correcting reverse power flow (RPF) polarity reading errors (RPF recorded as positive). The repository’s primary goal is academic reproducibility: an external reader can re-run the notebook end-to-end on a normal local computer and reproduce the key reported performance numbers.

This repository is also subject to an internal review gate: Ausgrid must be able to check the repo (code + included data + documentation) before any public release.

## 2. Goals and non-goals

### 2.1 Goals (v0.1.0)
1) Reproducibility: a single notebook reproduces key performance numbers for both methods:
   - Deterministic method (m7_threshold)
   - ML method using XGBoost (m8_xgb), including day-level and timestamp-level classification
2) Local execution: runnable on a standard local machine (Windows supported) using a pinned Python environment.
3) Centralised configuration: the notebook contains one dedicated “CONFIG” cell controlling all constants (column names, file paths, hyperparameters, thresholds, seeds, and evaluation settings).
4) Schema/column-name adaptability: all input column keys (e.g., substation/site ID, timestamp, net load MW, solar MW, ground truth) must be adjustable via configuration so the notebook can be rerun after schema changes without editing logic code.
5) Auditable outputs: notebook writes metrics outputs to disk with stable filenames.

### 2.2 Non-goals (explicitly out of scope for v0.1.0)
- Packaging/distribution as a pip-installable library (target for v0.2.0+).
- Production deployment and integration into Ausgrid pipelines.
- Producing paper-ready figures/tables inside the repo (v0.1.0 outputs are key numbers only).
- Automatic missing-value imputation (missing data is left as null; models must handle it).

## 3. Stakeholders and user requirements

### 3.1 Stakeholders
- Samhan (author): repository must reproduce key results reported in publication.
- Ausgrid (review gate): must be able to review content before public sharing.
- UNSW supervisor: no explicit additional requirements, but expects reproducible methodology and clear documentation.

### 3.2 User requirements
Samhan:
- Must be able to reproduce the same key performance numbers reported in the paper(s) from a clean clone + run-all workflow.

Ausgrid:
- Repo must be reviewable before public sharing.
- Repo must not contain sensitive identifiers, internal system names/paths, or proprietary data.

UNSW supervisor:
- N/A (no explicit requirement), but reproducibility standards must be met.

## 4. Dataset and inputs

### 4.1 Included dataset (repo-provided)
The repo will include interval data for 10 substations (IDs anonymised, e.g., A–J). Data frequency is **30-minute intervals**, units are **MW only**.

Required columns (canonical names; configurable via CONFIG):
- `substation_id` : str (anonymised A–J)
- `timestamp` : datetime (interval start timestamp)
- `net_load_MW` : float (synthetic “wrong sign” series; constructed as abs of ground truth)
- `solar_MW` : float
- `net_load_ground_truth` : float (true signed net load; RPF is negative)

Data-generation note (documented in README and notebook):
- `net_load_MW = abs(net_load_ground_truth)` to simulate a polarity error series while preserving a known ground truth.

### 4.2 Missing data constraint
Some interval rows may have missing values in `net_load_MW` or `solar_MW`.
- v0.1.0 requirement: **leave missing values as null** (no imputation).
- Algorithms must handle missingness gracefully:
  - Deterministic method: skip/ignore affected points as required; must not crash.
  - XGB: may use NaN handling natively; features may incorporate missingness indicators if already in your method, but do not impute values in v0.1.0.

Ground truth requirement:
- Rows used for evaluation must have non-null `net_load_ground_truth`. Any row with missing ground truth must be excluded from evaluation with a logged count.

## 5. Train / test split (publication-aligned)
The repository must reproduce results using the same time split used in the paper:

- Train set: **Nov 2021 – Sep 2023**
- Test set: **Oct 2023 – Sep 2024**

Split must be implemented as a **time-based split** using `timestamp` (converted to date for day-level tasks). The exact date boundaries must be configurable in the CONFIG cell.

## 6. Functional scope (what the notebook must do)

### 6.1 End-to-end workflow (single notebook)
The primary notebook (e.g., `notebooks/01_reproduce_key_numbers.ipynb`) must run end-to-end and produce the key performance numbers. Workflow stages:

1) Environment + imports (and deterministic seeds)
2) CONFIG cell (single source of truth)
3) Load data (CSV/Parquet supported; at least one provided format in repo)
4) Basic validation checks
   - required columns exist
   - timestamp parsable and monotonicity sanity checks
   - duplicate key checks (`substation_id`, `timestamp`)
   - missingness summary
5) Derive day-level dataset (per `substation_id`, per date)
6) Run deterministic method (m7_threshold)
   - day classification (optional if your deterministic method is day-only or both day + timestamp)
   - timestamp classification and sign correction logic (if implemented in your method)
7) Run ML method (m8_xgb)
   - Model 1: day classification
   - Model 2: timestamp classification (conditional on day-positive if that matches the paper)
8) Evaluate (test set primarily; training results optional but must not be required)
9) Export outputs (key numbers) to `outputs/`

### 6.2 Required outputs (key numbers only)
The notebook must export machine-readable results files (CSV or JSON), at minimum:

A) Data summary (for the evaluated period)
- number of sites, days, timestamps
- count of positive (needs correction) days/timestamps based on ground truth
- missingness counts per key column

B) Performance numbers — day classification (per method)
- confusion matrix counts (TP, TN, FP, FN)
- precision, recall, F1

C) Performance numbers — timestamp classification (per method)
- confusion matrix counts (TP, TN, FP, FN)
- precision, recall, F1

D) Energy impact metric (Navid error)
Because data are 30-minute intervals, compute energy per interval as:
- `MWh = MW * 0.5`

Compute at day level (per substation-day):
- `MWh_corrected_pred`: energy magnitude corrected by model
- `MWh_corrected_true`: energy magnitude that should be corrected (from ground truth)
- `Navid_error = MWh_corrected_pred - MWh_corrected_true`

Export at least:
- summary stats of Navid_error on test set (count, mean, std, p5/p50/p95 or similar)
- optionally, filtered summaries (e.g., excluding winter / excluding TN+TP) **only if those are reported in the paper**

### 6.3 Determinism and reproducibility requirements
- Fixed random seeds for:
  - numpy
  - Python random
  - XGBoost (and any CV if used)
- Pinned dependency versions via `requirements.txt` (and/or `environment.yml`).
- Outputs must be reproducible on repeated runs (exact match or within a defined numeric tolerance in README).

## 7. Configuration requirements (single CONFIG cell)
The notebook must contain one dedicated CONFIG cell that controls all constants. Minimum required config fields:

### 7.1 Paths and I/O
- `INPUT_PATH` (relative path under repo, default `data/...`)
- `OUTPUT_DIR` (default `outputs/`)
- `RUN_TAG` (optional, appended to output filenames)

### 7.2 Column mapping
- `COL_SITE` (default `substation_id`)
- `COL_TS` (default `timestamp`)
- `COL_NET_LOAD` (default `net_load_MW`)
- `COL_SOLAR` (default `solar_MW`)
- `COL_GT` (default `net_load_ground_truth`)

Requirement: all downstream logic must reference these config keys only (no hard-coded column names elsewhere). Changing column names must not require edits outside the CONFIG cell.

### 7.3 Time settings
- `INTERVAL_MINUTES = 30`
- `MWH_FACTOR = INTERVAL_MINUTES / 60` (expected 0.5)
- `TIMEZONE` (if needed; otherwise explicitly “naive timestamps assumed local”)

### 7.4 Train/test split
- `TRAIN_START_DATE`, `TRAIN_END_DATE` (Nov 2021 – Sep 2023)
- `TEST_START_DATE`, `TEST_END_DATE` (Oct 2023 – Sep 2024)

### 7.5 Deterministic model hyperparameters (m7_threshold)
All thresholds/windows used by your algorithm must be configurable here, e.g.:
- solar peak search window definition
- reading peak window definition
- minima search constraints
- threshold(s) for minima validation
- any time-of-day constraints (e.g., minima must lie within 6am–6pm if applicable)

### 7.6 ML model hyperparameters (m8_xgb)
- model objective and evaluation metric
- learning rate, max_depth, n_estimators, subsample, colsample_bytree, etc.
- classification threshold for converting probabilities to labels (if not default 0.5)
- seed(s)

### 7.7 Evaluation settings
- which metrics to compute
- any filtering rules for additional summaries (e.g., “exclude winter”)
- numeric rounding for reporting (e.g., 3 decimals)

## 8. Non-functional requirements

### 8.1 Compatibility
- Must run on a local Windows machine using Python (preferably 3.10+).
- No Databricks-only dependencies in the primary notebook path.

### 8.2 Performance expectations
- v0.1.0 uses a small included dataset (10 sites) intended to be runnable locally.
- If future dataset size increases significantly, local runtime may degrade; however, v0.1.0 acceptance is based on the included dataset.

### 8.3 Usability
- “Run All” from top executes without manual edits (given environment setup steps in README).
- Clear printed progress/log messages for each stage and key counts.

## 9. Repository structure (v0.1.0)

Recommended structure:
- `README.md`
- `requirements.txt` (pinned)
- `notebooks/01_reproduce_key_numbers.ipynb`
- `data/` (included synthetic dataset + `DATA_DICTIONARY.md`)
- `src/` (optional helper modules; not packaged)
- `outputs/` (generated; gitignored)
- `expected_outputs/` (optional: a reference metrics file for regression checks)
- `LICENSE` (must be resolved before public release)

## 10. Documentation requirements

### 10.1 README must include
- Project description and scope (v0.1.0)
- Exact reproduction steps (Windows-first)
- Environment setup commands
- Where outputs are written and which files represent “key numbers”
- Train/test split dates and how they map to code
- “Ausgrid review gate” checklist (below)

### 10.2 Ausgrid review gate checklist (must be included)
- No real substation names/IDs; only A–J
- No internal paths, table names, or system references
- No credentials or secrets (git-secrets scan recommended)
- Confirm dataset is synthetic/derived and approved for release
- Confirm license and publication permissions

## 11. Acceptance criteria (Definition of Done)
v0.1.0 is complete when:

1) From a clean clone, a user can:
   - create environment from `requirements.txt`
   - open `01_reproduce_key_numbers.ipynb`
   - Run All successfully

2) The notebook generates (at minimum) exported metrics files in `outputs/` containing:
   - day classification P/R/F1 + confusion matrix (m7_threshold and m8_xgb)
   - timestamp classification P/R/F1 + confusion matrix (m7_threshold and m8_xgb)
   - Navid error summary stats on the test set

3) Results are deterministic across repeated runs (same commit + same environment + same data).

4) Column-name adaptability is verified: switching the input schema (by changing the configured column names) does not require code edits outside the CONFIG cell, and the notebook still runs end-to-end.

5) Repo passes Ausgrid review gate checklist prior to publication.

## 12. Future roadmap (not part of v0.1.0)
- v0.2.0+: refactor into an installable Python package (pip), add CLI entry points, add more robust testing, and optional Databricks connectors.

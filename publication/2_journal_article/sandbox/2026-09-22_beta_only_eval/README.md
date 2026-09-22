> The outputs of this run were promoted to `publication/2_journal_article/results/` on 22 September 2026 (with the Gamma stage added); this folder keeps the config and the results note.

# Sandbox: Beta-only-trained evaluation (22 September 2026)

Question: what do M7, M8 and M9 score when every fitted quantity is learnt from Beta alone
and Alpha is a pure transfer cohort? Phase 3 (16 September 2026) trained M8 on the other
17 stations of both cohorts and calibrated M9 within cohort; here nothing is ever fitted
on Alpha. See `RESULTS.md` for the numbers and the reproduction checks.

The run is the ordinary `final_eval` engine driven by `config.yaml`, which is the Phase 3
configuration with three differences (below). Nothing outside this folder and the engine
is modified; the Phase 3 release under `sandbox/2026-09-16_phase3_release/` is untouched.

## The rule

| held-out station | M8 trains on | M9 calibrates (cal_intercept, cal_slope) on |
|---|---|---|
| a Beta station | the other 7 Beta stations | the other 7 Beta stations |
| an Alpha station | all 8 Beta stations (one shared bundle) | all 8 Beta stations (one shared pair) |

M7 fits nothing and is unchanged. Beta 'unsure' days are never fitted on and never enter
a headline number (as in Phase 3). Nine M8 bundles are fitted: eight Beta folds plus one
bundle that the ten Alpha folds share.

## Inputs

- `dataset/final/dataset_alpha.parquet` and `dataset_beta.parquet`, hash-checked by the
  engine against `config.yaml` (identical to Phase 3).
- `m9_dev/` (the frozen M9 revision 2 scorer), imported, never copied.
- `sandbox/2026-09-16_phase3_release/outputs/` and
  `sandbox/2026-09-18_intercept_transfer/curve.csv` are read only by the reproduction
  checks in `RESULTS.md`, not by the run.

## Outputs

Everything under `outputs/`, in the Phase 3 layout: `01_folds` (fold manifest with the
Beta-only training sets and the `m8_training_signature` column), `02_bundles` (nine
bundles, gitignored, and eighteen kept manifests; a shared manifest names its donor in
`shared_with`), `03_m7`, `04_m8`, `05_m9`, `06_site_days`, `07_metrics` (now with
`macro.csv`), `08_tables` (now with `table10_pooled_macro_stations`), `08_figures`,
`10_operating_points` and `manifests/`. The Gamma stage was not run. Stage logs
(`train_m8.log`, `predict_*.log`, ...) sit beside them.

## Reproduce

From `publication/2_journal_article/` with the project environment active:

```powershell
..\..\.venv\Scripts\Activate.ps1
$cfg = "sandbox/2026-09-22_beta_only_eval/config.yaml"
python -m final_eval folds            --config $cfg
python -m final_eval train-m8         --config $cfg    # 9 fits, about 10 minutes
python -m final_eval predict-m7       --config $cfg
python -m final_eval predict-m8       --config $cfg
python -m final_eval predict-m9       --config $cfg
python -m final_eval outcomes         --config $cfg
python -m final_eval metrics          --config $cfg
python -m final_eval report           --config $cfg
python -m final_eval operating-points --config $cfg
```

`train-m8` is resumable: a fold whose manifest and bundle exist is skipped, and a fold
whose training signature already has a bundle in this run shares it.

## Config differences from Phase 3

Every difference is marked `SANDBOX` in `config.yaml`; the rest is byte-for-byte the
Phase 3 file.

| key | Phase 3 | this run |
|---|---|---|
| `paths.output_dir` | `outputs/01_final_evaluation` | `sandbox/2026-09-22_beta_only_eval/outputs` |
| `folds.m8_training_scope` | `all_other_stations_both_cohorts` | `beta_only` |
| `folds.m9_calibration_scope` | `other_stations_same_cohort` | `beta_only` |
| `m8.split.train_start` / `train_end` | 2021-11-01 / 2024-05-31 | 2023-10-01 / 2024-07-31 |
| `m8.split.validation_start` / `validation_end` | 2024-06-01 / 2024-09-30 | 2024-08-01 / 2024-09-30 |

The Beta dataset spans 2023-10-01 to 2024-09-30, so the M8 fit window is its first ten
months and the in-bundle validation window the last two. Validation metrics are reported
only; nothing is tuned. Thresholds, hyperparameters, the M9 scorer and control c, the
population rules and the metric definitions are all Phase 3.

## Engine changes this run needed

Kept as defaults so the Phase 3 configuration behaves exactly as before:

- `final_eval/folds.py`: the `beta_only` scope for both `folds.m8_training_scope` and
  `folds.m9_calibration_scope`; `training_signature()` and the manifest column
  `m8_training_signature`; the manifest's M9 calibration counts draw from every cohort.
- `final_eval/m8.py`: folds with an equal training signature share one bundle.
- `final_eval/m9.py`, `final_eval/operating_points.py`: the calibration stations are
  drawn from a pool of both cohorts, because they may lie outside the held-out cohort.
- `final_eval/metrics.py`: `macro_table()` (mean over stations);
  `final_eval/tables.py`: `pooled_macro_table()` (table 10).

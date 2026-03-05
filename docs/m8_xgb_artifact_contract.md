# `m8_xgb` Artifact Contract (`pynrpf.m8_xgb.bundle.v2`)

The training API writes a reusable pickle bundle for inference.

## Required Top-Level Keys

- `bundle_schema`: must be `pynrpf.m8_xgb.bundle.v2`
- `model_name`: `m8_xgb`
- `created_at_utc`: ISO timestamp
- `xgb1_day`: day-stage section
- `xgb2_timestamp`: interval-stage section

## Required Stage Section Keys

For both `xgb1_day` and `xgb2_timestamp`:

- `model`: fitted XGBoost classifier object
- `feature_columns`: ordered feature list used at fit time
- `threshold`: classification threshold used at inference

Inference remains backward-compatible with older bundles that use `feat_cols`.

## Metadata

- `training_metadata.feature_pipeline`
- `training_metadata.labels.day`
- `training_metadata.labels.interval`
- `training_metadata.split.train_start`
- `training_metadata.split.train_end`
- `training_metadata.split.validation_start`
- `training_metadata.split.validation_end`
- `training_metadata.random_seed`
- `training_metadata.validation_metrics.xgb1_day`
- `training_metadata.validation_metrics.xgb2_timestamp`

## Storage Locations Supported in Databricks

Bundle writer accepts base locations such as:

- `dbfs:/...`
- `/Volumes/<catalog>/<schema>/<volume>/...`
- `dbfs:/Volumes/<catalog>/<schema>/<volume>/...` (alias)

Versioned artifacts are written under:

- `<base_uri>/m8_xgb/<YYYYMMDDTHHMMSSZ>/bundle.pkl`
- `<base_uri>/m8_xgb/<YYYYMMDDTHHMMSSZ>/manifest.json`

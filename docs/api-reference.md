# API reference

Every public function in the `pynrpf` package. For the shortest path to a
working call, see the Quick start and User Journey in the [main README](../README.MD).

## Classification

### Core APIs (most users)

- `run_inference(data, config)`
- `train_m8_xgb(data, config)`
- `load_config(config)`
- `list_models()`

### Additional APIs (advanced/helpers)

- `load_artifact_bundle(location)`
- `save_artifact_bundle(bundle, location)`
- `build_pipeline_config(model_id, include_training)`
- `generate_pipeline_config(output_path, model_id, include_training, overwrite)`
- `generate_model_scaffold(model_id, output_dir, overwrite, include_tests, include_pipeline_config)`

## Function guide

### Core: `run_inference(data, config)`

What it does:
- Validates and standardises input data.
- Selects model from config (`m7_dtr` or `m8_xgb`).
- Runs model inference and returns scored data plus operational summary.

Use when:
- You want corrected net load + flags on new data.

Input:
- `data`: pandas DataFrame or Spark DataFrame.
- `config`: mapping or YAML path. Can be:
  - pure inference config, or
  - full pipeline config containing `pynrpf_inference`.
- Required logical columns (configured under `columns`):
  - `site`, `timestamp`, `net_load`, `solar`.

Output:
- `data`: same table type as input (pandas in, pandas out; Spark in, Spark out).
- `summary`: row counts and monitoring stats.
- `model`: resolved model id.
- `input_type`: `"pandas"` or `"spark"`.
- `m7_dtr` note: strict day flags remain threshold-based, while interval
  corrections and corrected net load use a relaxed, threshold-free minima span,
  so day and interval flags may diverge.

Common errors:
- Missing required columns.
- Unsupported model id.
- `m8_xgb` without `artifacts.m8_pretrained_bundle_uri`.

### Core: `train_m8_xgb(data, config)`

What it does:
- Trains both internal models:
  - `xgb1_day` (day classifier)
  - `xgb2_timestamp` (interval classifier)
- Writes a versioned artifact bundle and manifest.
- Returns artifact URIs + validation metrics.

Use when:
- You need to create or refresh `m8_xgb` artifacts for inference.

Input:
- `data`: interval-level pandas/Spark DataFrame containing:
  - inference columns (`site`, `timestamp`, `net_load`, `solar`)
  - day label column
  - interval label column
- `config`: mapping or YAML path containing:
  - `pynrpf_inference`
  - `pynrpf_training`

Output:
- `bundle`: in-memory artifact dictionary.
- `bundle_schema`: currently `pynrpf.m8_xgb.bundle.v2`.
- `artifact_uri`: bundle file URI to use for inference.
- `artifact_dir_uri`, `manifest_uri`.
- `validation_metrics` for both stages.

Common errors:
- Missing day/interval labels.
- Invalid training split window.
- Invalid threshold values.
- Unsupported training model id (currently only `m8_xgb`).

### Core: `load_config(config)`

What it does:
- Loads and validates inference config.
- Accepts mapping or YAML path.
- If full pipeline config is provided, extracts `pynrpf_inference`.
- Applies defaults and normalises model selection fields.

Use when:
- You want to inspect/validate final effective inference config before execution.

### Core: `list_models()`

What it does:
- Returns currently registered inference model ids.

Use when:
- You want to see which model names are valid for `selected_model`.

### Additional: `load_artifact_bundle(location)`

What it does:
- Reads and deserialises a pickle artifact bundle.
- Supports local paths, `file://`, `dbfs:/`, and `http(s)://` for reads.

Use when:
- You want to inspect/debug a trained artifact payload.

### Additional: `save_artifact_bundle(bundle, location)`

What it does:
- Serialises and writes bundle payload to a local or DBFS/Volumes-backed path.

Use when:
- You need explicit one-file bundle writes outside training API orchestration.

### Additional: `build_pipeline_config(model_id, include_training)`

What it does:
- Builds an in-memory pipeline-style config dictionary with `pynrpf_inference`.
- Optionally includes `pynrpf_training` (currently only for `m8_xgb`).

Use when:
- You want a Python-first config object without writing a file.

### Additional: `generate_pipeline_config(output_path, model_id, include_training, overwrite)`

What it does:
- Writes a pipeline YAML template to disk using the same schema as `build_pipeline_config(...)`.

Use when:
- You want a starter config file for Databricks/notebook use.

### Additional: `generate_model_scaffold(model_id, output_dir, overwrite, include_tests, include_pipeline_config)`

What it does:
- Creates a starter plugin module under `src/pynrpf/plugins/`.
- Optionally creates a plugin test and pipeline config template.
- Auto-wires model import/export and registry entries:
  - `src/pynrpf/plugins/__init__.py`
  - `src/pynrpf/registry.py`

Use when:
- You want to add a new model quickly and start editing logic immediately.

## Input and output schemas

Input:
- `run_inference`:
  - `data` (pandas/Spark) + config
- `train_m8_xgb`:
  - labeled interval data + training/inference config blocks

Output:
```python
{
  "run_inference": {
    "data": "<same type as input>",
    "summary": {...},
    "model": "<model_id>",
    "input_type": "pandas|spark",
  },
  "train_m8_xgb": {
    "bundle_schema": "pynrpf.m8_xgb.bundle.v2",
    "artifact_uri": "<base>/m8_xgb/<utc_ts>/bundle.pkl",
    "validation_metrics": {
      "xgb1_day": {...},
      "xgb2_timestamp": {...},
    },
  }
}
```


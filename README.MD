# PyNRPF
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18764944.svg)](https://doi.org/10.5281/zenodo.18764944)

PyNRPF provides an implementation package (`pynrpf`) for reverse power flow
inference and `m8_xgb` training workflows.

## Contributors

- M. Syahman Samhan (contact: m.samhan@unsw.edu.au)
- Anna Bruce
- Baran Yildiz

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .[dev]
```

## User Journey

### Step 1: Choose your workflow

- Inference only:
  - Use `run_inference(...)`.
  - If model is `m8_xgb`, provide a trained bundle URI.
- Train + inference (`m8_xgb`):
  - Use `train_m8_xgb(...)` first.
  - Take returned `artifact_uri`.
  - Put it into `pynrpf_inference.artifacts.m8_pretrained_bundle_uri`.
  - Run `run_inference(...)`.

### Step 2: Keep one config file (`pipeline.yaml`)

PyNRPF reads only:
- `pynrpf_inference`
- `pynrpf_training` (for training)

Other pipeline keys (tables, orchestration, write targets) are ignored by PyNRPF.

### Step 3: Run training and inference

```python
import yaml
from pynrpf import run_inference, train_m8_xgb

with open("config/pipeline.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

train_out = train_m8_xgb(train_df, cfg)

cfg.setdefault("pynrpf_inference", {}).setdefault("artifacts", {})
cfg["pynrpf_inference"]["artifacts"]["m8_pretrained_bundle_uri"] = train_out["artifact_uri"]

result = run_inference(score_df, cfg)
scored_df = result["data"]
summary = result["summary"]
```

If you already have a trained bundle, skip `train_m8_xgb(...)` and set
`m8_pretrained_bundle_uri` directly in YAML.

## API Classification

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

## Detailed API Function Guide

### Core: `run_inference(data, config)`

What it does:
- Validates and standardizes input data.
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
- Applies defaults and normalizes model selection fields.

Use when:
- You want to inspect/validate final effective inference config before execution.

### Core: `list_models()`

What it does:
- Returns currently registered inference model ids.

Use when:
- You want to see which model names are valid for `selected_model`.

### Additional: `load_artifact_bundle(location)`

What it does:
- Reads and deserializes a pickle artifact bundle.
- Supports local paths, `file://`, `dbfs:/`, and `http(s)://` for reads.

Use when:
- You want to inspect/debug a trained artifact payload.

### Additional: `save_artifact_bundle(bundle, location)`

What it does:
- Serializes and writes bundle payload to a local or DBFS/Volumes-backed path.

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

## API Input/Output Schemas (Quick View)

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

## `m8_xgb` Notes

`m8_xgb` is a two-stage model family:
- `xgb1_day` (day-level)
- `xgb2_timestamp` (interval-level)

Training consumes one interval-level labeled dataset and internally builds both
feature schemas.

Artifact contract details:
- `docs/m8_xgb_artifact_contract.md`

## Scaffold Helpers

Generate starter model logic/test/config files:

```python
from pynrpf import generate_model_scaffold

created = generate_model_scaffold("m9_custom", output_dir=".")
print(created)
```

`generate_model_scaffold(...)` now auto-wires:
- `src/pynrpf/plugins/__init__.py` import/export list
- `src/pynrpf/registry.py` model registry entry

Generate only a pipeline config template:

```python
from pynrpf import generate_pipeline_config

generate_pipeline_config(
    output_path="config/pynrpf_pipeline_m8_xgb.yaml",
    model_id="m8_xgb",
    include_training=True,
    overwrite=True,
)
```

## Conference Publication Archive

Publication artifacts are isolated and frozen under:
- `publication/1_conference_paper`

Archive run instructions:
- `publication/1_conference_paper/README.md`

## Continuous Integration and Release

- `ci`: lint, tests, build
- `release`: publish on version tags (`v*`)
- `publication_archive_smoke` (nightly): non-blocking publication notebook smoke

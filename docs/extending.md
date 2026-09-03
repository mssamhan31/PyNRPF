# Extending PyNRPF with a new model

How the `m8_xgb` model family is structured, and how to add your own model
plugin. The artefact format itself is specified in
[m8_xgb_artifact_contract.md](m8_xgb_artifact_contract.md).

## The `m8_xgb` two-stage family

`m8_xgb` is a two-stage model family:
- `xgb1_day` (day-level)
- `xgb2_timestamp` (interval-level)

Training consumes one interval-level labelled dataset and internally builds both
feature schemas. The bundle it writes is specified in
[m8_xgb_artifact_contract.md](m8_xgb_artifact_contract.md).

## Scaffold helpers

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


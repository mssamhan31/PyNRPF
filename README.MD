# PyNRPF
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18764944.svg)](https://doi.org/10.5281/zenodo.18764944)

PyNRPF provides an implementation package (`pynrpf`) for reverse power flow
inference workflows and a frozen conference-paper archive for reproducibility.

## Contributors

- M. Syahman Samhan (contact: m.samhan@unsw.edu.au)
- Anna Bruce
- Baran Yildiz

## Install (development/local)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .[dev]
```

## Package Usage (`pynrpf`)

Canonical import path:

```python
from pynrpf import run_inference

result = run_inference(df, "config/pynrpf_inference.yaml")
corrected_df = result["data"]
summary = result["summary"]
```

Supported package API:
- `pynrpf.run_inference(...)`
- `pynrpf.train_m8_xgb(...)`
- `pynrpf.load_config(...)`
- `pynrpf.load_artifact_bundle(...)`
- `pynrpf.save_artifact_bundle(...)`
- `pynrpf.list_models(...)`

## `m8_xgb` Training (Two-Stage)

`m8_xgb` is trained as one model family with two internal estimators:
- `xgb1_day` (day-level)
- `xgb2_timestamp` (interval-level)

Training consumes one interval-level labeled table and derives both internal
feature schemas during training.

```python
from pynrpf import run_inference, train_m8_xgb

full_cfg = {
    "pynrpf_inference": {
        "columns": {
            "site": "substation_id",
            "timestamp": "timestamp",
            "net_load": "net_load_MW",
            "solar": "solar_MW",
        },
        "runtime": {"interval_minutes": 15, "strict_validation": True},
        "model": {"selected_model": "m8_xgb"},
    },
    "pynrpf_training": {
        "model_id": "m8_xgb",
        "labels": {"day": "label_day", "interval": "label_interval"},
        "split": {
            "train_start": "2024-01-01",
            "train_end": "2024-01-31",
            "validation_start": "2024-02-01",
            "validation_end": "2024-02-29",
        },
        "features": {"pipeline": "legacy_v1"},
        "thresholds": {"xgb1_day": 0.585985, "xgb2_timestamp": 0.892234},
        "random_seed": 9,
        "output": {"base_uri": "dbfs:/FileStore/pynrpf/artifacts", "versioning": "timestamp_dir"},
    },
}

train_out = train_m8_xgb(train_df, full_cfg)
bundle_uri = train_out["artifact_uri"]

inference_cfg = {
    "pynrpf_inference": {
        **full_cfg["pynrpf_inference"],
        "artifacts": {"m8_pretrained_bundle_uri": bundle_uri},
    }
}
result = run_inference(score_df, inference_cfg)
```

Artifact key contract and Databricks storage URI details:
- `docs/m8_xgb_artifact_contract.md`

## Conference Publication Archive

Publication artifacts are isolated and frozen under:

- `publication/1_conference_paper`

This archive contains its own:
- `dataset/`
- `notebooks/`
- `models/`
- `outputs/`
- `config/run.yaml`
- `requirements.txt`
- `src/` snapshot used by archived notebooks

Archive run instructions are documented in:

- `publication/1_conference_paper/README.md`

## Continuous Integration and Release

- `ci`: runs lint, tests, and build on pushes/pull requests
- `release`: builds and publishes on version tags (`v*`)
- `publication_archive_smoke` (nightly): non-blocking notebook-chain smoke for
  `publication/1_conference_paper`

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
- `pynrpf.train_model(...)`
- `pynrpf.load_config(...)`
- `pynrpf.load_artifact_bundle(...)`
- `pynrpf.save_artifact_bundle(...)`
- `pynrpf.list_models(...)`

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

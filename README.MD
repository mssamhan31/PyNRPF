# PyNRPF — Python for Network Reverse Power Flow

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19211355.svg)](https://doi.org/10.5281/zenodo.19211355)

PyNRPF detects and corrects a wrong reverse power flow (RPF) sign in electricity
distribution network interval meter data.

## What problem it solves

A distribution substation meter records **net load** — demand minus local
generation. When rooftop solar output exceeds demand, real power flows backwards
into the network: a *reverse power flow* event, and net load is genuinely
negative. Some meters record the magnitude of that flow but not its direction, so
the reading is stored with the wrong sign — positive where it should be negative.

The reading looks plausible and passes ordinary validation, but it is wrong by
twice the true value, and it corrupts every minimum-demand study and net load
forecast built on top of it.

PyNRPF finds the intervals where the sign is wrong and returns a corrected series
alongside the flags, so the correction is auditable rather than silent.

It is built for distribution network planners and forecasting analysts working
with interval meter data, and for researchers reproducing the published results
in [`publication/`](publication/). Two models are available:

- **`m7_dtr`** — a deterministic threshold rule (daytime threshold rule). No
  training required; runs on any conforming dataset.
- **`m8_xgb`** — a two-stage gradient-boosted classifier (`xgb1_day` for
  day-level detection, `xgb2_timestamp` for interval-level). Requires a trained
  artefact bundle.

## Install

Requires **Python 3.10 or later**. Continuous integration tests 3.10, 3.11 and 3.12.

For the full repository environment, including tests, journal notebooks, and the
oracle review Streamlit app:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For package-only usage, install just the runtime package:

```powershell
python -m pip install -e .
```

## Quick start

Run the deterministic `m7_dtr` model over one day of synthetic readings. This
needs no trained artefact, so it works immediately after install. Run it from the
repository root, so that the configuration path resolves.

```python
import pandas as pd
from pynrpf import run_inference

# One substation, one day of 15-minute readings. Solar peaks at midday, so true
# net load goes negative then -- but the meter has recorded it with the wrong sign.
timestamps = pd.date_range("2024-01-01", periods=96, freq="15min", tz="UTC")
solar = [max(0.0, 3.0 - abs(i - 48) * 0.12) for i in range(96)]
readings = pd.DataFrame({
    "substation_id": "A",
    "timestamp": timestamps,
    "net_load_MW": [abs(2.0 - s) for s in solar],  # sign error: never negative
    "solar_MW": solar,
})

result = run_inference(readings, "config/pynrpf_inference.yaml")
scored, summary = result["data"], result["summary"]

print(f"model:             {result['model']}")
print(f"rows processed:    {summary['rows_processed']}")
print(f"rows corrected:    {summary['rows_corrected']}")
print(f"intervals flagged: {int(scored['pynrpf_interval_flag'].sum())}")
```

After a progress block from the model, this prints:

```text
model:             m7_dtr
rows processed:    96
rows corrected:    15
intervals flagged: 15
```

`scored` is the input frame plus the correction columns, chiefly
`pynrpf_interval_flag`, `pynrpf_day_flag`, `pynrpf_corrected_net_load` and
`pynrpf_confidence`. For the `m8_xgb` workflow, which needs a trained bundle,
see the User Journey below.

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

### Step 2: Keep one config file

A single pipeline configuration file carries both blocks PyNRPF needs.
[config/pynrpf_training_m8_xgb.yaml](config/pynrpf_training_m8_xgb.yaml) is a
worked example. PyNRPF reads only:
- `pynrpf_inference`
- `pynrpf_training` (for training)

Other pipeline keys (tables, orchestration, write targets) are ignored by PyNRPF.

### Step 3: Run training and inference

```python
import yaml
from pynrpf import run_inference, train_m8_xgb

with open("config/pynrpf_training_m8_xgb.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# train_df and score_df are your own labelled interval frames -- see Data below
# for the expected columns.
train_out = train_m8_xgb(train_df, cfg)

cfg.setdefault("pynrpf_inference", {}).setdefault("artifacts", {})
cfg["pynrpf_inference"]["artifacts"]["m8_pretrained_bundle_uri"] = train_out["artifact_uri"]

result = run_inference(score_df, cfg)
scored_df = result["data"]
summary = result["summary"]
```

If you already have a trained bundle, skip `train_m8_xgb(...)` and set
`m8_pretrained_bundle_uri` directly in YAML.

## Repository structure

```text
src/pynrpf/       The installable package: config, validation, model registry,
                  artefacts, monitoring, and one plugin per model.
config/           Example inference and training configuration files.
tests/            Pytest suite.
docs/             Documentation — see the index below.
publication/      Frozen conference paper archive, and ongoing journal work.
```

## Data

All datasets derive from **anonymised Ausgrid distribution network data**.
Substation identifiers are replaced with opaque labels, so no site, customer or
location can be identified from what is published here.

The conference dataset is 10 substations at 15-minute resolution, November 2021
to September 2024, carrying recorded net load in megawatts, estimated solar, and
a manually reviewed ground truth. The journal work splits into Alpha, Beta and
Gamma. Full schema, provenance and label process: [docs/data.md](docs/data.md).

## Tests

```powershell
python -m pip install -e .[dev]
pytest -q
```

89 tests. Lint with `ruff check src/pynrpf tests`. See
[docs/development.md](docs/development.md).

## Documentation

| Document | What it covers |
|---|---|
| [docs/api-reference.md](docs/api-reference.md) | Every public function: arguments, returns, errors, schemas. |
| [docs/data.md](docs/data.md) | Dataset provenance, columns, units, and the ground-truth process. |
| [docs/extending.md](docs/extending.md) | The `m8_xgb` family, and scaffolding a new model plugin. |
| [docs/m8_xgb_artifact_contract.md](docs/m8_xgb_artifact_contract.md) | The artefact bundle format. |
| [docs/development.md](docs/development.md) | Environment, tests, lint, and continuous integration. |

## Conference paper archive

The conference publication artefacts are frozen and self-contained under
[`publication/1_conference_paper/`](publication/1_conference_paper/README.md),
which documents how to reproduce the published figures and tables.

## Licence

MIT. See [LICENSE](LICENSE).

## Acknowledgements

This project is part of Samhan's PhD study, supported by the University International Postgraduate Award (UIPA) Scholarship from UNSW, the Industry Collaboration Project Scholarship from Ausgrid, the RACE for 2030 Scholarship, and the NSW Decarbonisation Innovation Hub (NSW Decarb Hub).

## Contributors

- M. Syahman Samhan (contact: m.samhan@unsw.edu.au)
- Anna Bruce
- Baran Yildiz

## Citation

If you use this software, please cite it using the metadata in
[CITATION.cff](CITATION.cff), or the archived release:

> Samhan, M. S., Bruce, A., & Yildiz, B. (2026). *PyNRPF: Detecting and
> rectifying wrong reverse power flow readings* (version 0.3.0) [Software].
> Zenodo. https://doi.org/10.5281/zenodo.19211355

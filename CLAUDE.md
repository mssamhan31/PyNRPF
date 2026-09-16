# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

PyNRPF (Python for Network Reverse Power Flow) detects and corrects a wrong reverse
power flow (RPF) sign in distribution network interval meter data. See
[README.MD](README.MD) for what it does and [docs/](docs/) for the API reference.

## The red line: results are the author's

`publication/` holds research results, not build artefacts. The committed
figures, tables, metrics and manifests are the numbers that go into a paper.

**Never re-execute a notebook to "refresh" an output, and never edit a committed
result value.** Re-running changes numbers, and a silently changed metric is
indistinguishable from a corrupted one six months later. Fixing formatting,
documentation, structure and dead code is fine. Changing a model, metric, feature
set, split, hyperparameter, transformation or tolerance is methodology — propose
it, do not do it.

If a fix needs a methodology change to be correct, stop and say so rather than
making a partial edit.

## Things that look wrong but are deliberate

- **`publication/1_conference_paper/` is a frozen archive** with its own forked
  copy of the package under `src/`. The duplication is the point: it is the code
  that produced the published paper. Do not refactor it, deduplicate it against
  `src/pynrpf/`, or tidy it.
- **Large datasets are committed on purpose** — `.parquet` and `.csv` files over
  10 MB under `publication/*/dataset/`. They are anonymised Ausgrid data,
  cleared for publication, and committed so the results reproduce from the repo
  alone. Do not flag them as leakage or propose gitignoring them.
- **`publication/` is outside CI's lint scope**, and `src/pynrpf/_legacy` is
  excluded from ruff. Experiment and archive code is held to a looser standard.
- **`99_Misc/` notebooks are working material**, judged more leniently than the
  numbered publication notebooks.

## Commands

```powershell
# Full environment (tests, notebooks, Streamlit review app)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

pytest -q                                    # 89 tests
pytest tests/test_api.py -q                  # one file
pytest tests/test_api.py::test_run_inference_m7_pandas_dataframe -q   # one test
ruff check src/pynrpf tests                  # CI's lint scope
python -m build --sdist --wheel
```

`ci.yml` installs `.[dev]` only. The `dev` extra therefore carries `matplotlib`
and `pyarrow` as well as the test toolchain, because the journal and oracle tests
import helper modules from `publication/` by `sys.path` manipulation. If you add
a test that reaches into `publication/`, check its imports are satisfied by `dev`
alone, or CI will fail at collection.

## Architecture

One dispatch path, worth tracing once. `src/pynrpf/api.py` is the only public
entry point; everything else is called from it.

```
run_inference(data, config)
  config.load_config        unwrap pynrpf_inference, adapt legacy schema, merge DEFAULT_CONFIG
  validation.to_pandas_input    pandas or Spark in, remember which
  validation.validate_dataframe schema, interval alignment, key uniqueness
  registry.get_model            model id -> plugin instance
  <plugin>.run_inference        the only model-specific step
  monitoring.build_operational_summary
  validation.from_pandas_output restore the caller's frame type
```

`train_m8_xgb` follows the same shape, adding `training_config.load_training_config`
and ending at `artifacts.save_versioned_artifact_bundle`, which writes a
timestamped `bundle.pkl` plus `manifest.json`.

**Adding a model means adding a plugin, not touching `api.py`.** Subclass
`BaseModelPlugin` in `plugins/base.py` — `run_inference` is abstract, `train` is
optional and raises by default — then register it in `registry.py`. Do not
hand-write the wiring: `generate_model_scaffold(model_id)` creates the module, a
test and a config template, and edits `plugins/__init__.py` and `registry.py` for
you. See [docs/extending.md](docs/extending.md).

Config is a single resolved dictionary. `columns` maps logical names (`site`,
`timestamp`, `net_load`, `solar`) to physical column names, and plugins receive
that mapping rather than hardcoding column names. A full pipeline config with a
`pynrpf_inference` block is accepted anywhere an inference config is, so
Databricks pipeline files work unchanged.

`src/pynrpf/_legacy/` holds feature building, validation and the m7 threshold
rule carried over from the conference codebase. The conference results depend on
it, so it stays.

## Conventions

**Australian spelling** in all prose — README, docstrings, comments, notebook
markdown. American spelling stays in code identifiers and library interfaces
(`color=`, `normalize=`).

**Notebooks are committed without outputs**, uniformly. `*.executed.ipynb` is
gitignored. Every notebook opens with a markdown cell giving purpose, inputs,
outputs and approximate runtime, and expands abbreviations on first use in that
notebook — RPF included, each notebook separately.

**Manifests and inventories must not carry absolute paths.** Machine account
names leaked into committed manifests once. Use `relative_to_article()` in
`_experiment_helpers.py`, or `relative_to(paths.article)` as `_m9_pbm_data.py`
does. Anything written into `outputs/manifests/` or an inventory CSV is
repository-relative.

**Docstrings**: every module opens with purpose, inputs, outputs, key steps.
Every public function documents arguments, returns and units — MW, MWh, minutes,
15-minute intervals.

## Branches and releases

`main` is the released, Zenodo-archived version; a `v*` tag triggers
`release.yml`, which publishes to PyPI. Work happens on `dev` and on feature
branches off `dev`. Do not merge working branches into `main` — that decision is
a release decision.

`.ai/` and `planning/` are gitignored working directories.

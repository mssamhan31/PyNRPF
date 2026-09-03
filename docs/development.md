# Development

## Environment

Requires Python 3.10 or later. Continuous integration tests 3.10, 3.11 and 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs the package in editable mode with the `dev` and
`journal` extras. The `dev` extra carries the test and lint toolchain, plus
`matplotlib` and `pyarrow`, which the journal article tests need. The `journal`
extra adds the heavier notebook and Streamlit dependencies.

## Tests

```powershell
pytest -q
```

89 tests, covering:

| Area | Files |
|---|---|
| Package API, config and training validation | `test_api.py`, `test_config.py`, `test_training_config.py` |
| Artefact bundle round-trips | `test_artifacts.py`, `test_training_m8_xgb.py` |
| Model behaviour | `test_m7_dtr_behavior.py`, `test_m8_xgb_behavior.py` |
| Scaffold generation | `test_scaffold.py` |
| Journal article helpers and figure rendering | `test_journal_article_experiment_helpers.py`, `test_m9_pbm_helpers.py`, `test_journal_figure_rendering.py` |
| Oracle review workflow | `test_oracle_review_core.py` |

The journal and oracle tests import helper modules from `publication/` by path,
so they need the `dev` extra installed rather than the runtime package alone.

## Lint

```powershell
ruff check src/pynrpf tests
```

Configured in `pyproject.toml`: line length 100, rules `E`, `F`, `I`, `B`, with
`src/pynrpf/_legacy` excluded. Code under `publication/` is outside the linted
scope — it is experiment and archive material held to a looser standard.

## Continuous integration and release

Three workflows under `.github/workflows/`:

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | every push and pull request | Builds under setuptools 69 for compatibility, then lints, tests and builds across Python 3.10, 3.11 and 3.12. |
| `release.yml` | version tags matching `v*` | Lints, tests, builds, and publishes to PyPI via OpenID Connect. |
| `publication_archive_smoke.yml` | nightly | Executes conference notebooks 01 to 03 with nbconvert and uploads the outputs as a workflow artefact. Non-blocking. |

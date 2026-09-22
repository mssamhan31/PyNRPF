# Development

## Environment

Python 3.10 or later; continuous integration tests 3.10, 3.11 and 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev,paper]
```

## Tests

```powershell
pytest -q
ruff check src/pynrpf tests publication/2_journal_article/paper
```

| tests | what they protect |
|---|---|
| `test_m9_steps.py` | each step of the method on planted days: admissibility, the edge rule with a gap, the bridge, the misfit, the evidence sign and floor, the tie order |
| `test_m9_reference.py` | the whole chain reproduces the reference run to the last digit on committed slices of two stations (`tests/fixtures/`) |
| `test_calibration_decision_impact.py` | the release numbers, the intercept refit, the decision bands, the energy and minimum-demand numbers |
| `test_run_cli_spark.py` | the pandas entry point, the column mapping, incomplete days, refused inputs, the command line, the Spark adapter (skipped without pyspark) |
| `test_paper_*.py` | the paper code: folds and leakage guards, reference terms, metrics, operating points, and that `results/` matches its manifests |

`tests/fixtures/make_fixtures.py` rebuilds the reference slices from the datasets and
`results/`; run it only when the reference run changes.

## Documentation

```powershell
mkdocs serve          # local preview at http://127.0.0.1:8000
mkdocs build --strict
```

The site is built locally only; it is not published.

## Continuous integration

`ci.yml` lints, tests and builds the package on every push; `release.yml` publishes to PyPI
when a `v*` tag is pushed; a nightly workflow smoke-tests the conference paper archive.

## Releasing

Bump `version` in `pyproject.toml`, `__version__` in `src/pynrpf/__init__.py` and
`CITATION.cff`; add the entry to `CHANGELOG.md`; tag `vX.Y.Z` on `main`.

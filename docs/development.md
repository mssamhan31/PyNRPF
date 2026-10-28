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

/// details | Show details
    type: more

## What the tests protect

| tests | what they protect |
|---|---|
| `test_m9_steps.py` | each step of the method on planted days: admissibility, the edge rule with a gap, the bridge, the misfit, the evidence sign and floor, the tie order |
| `test_m9_reference.py` | the whole chain reproduces the reference run to the last digit on committed slices of two stations (`tests/fixtures/`) |
| `test_calibration_decision_impact.py` | the release numbers, the intercept refit, the decision bands, the energy and minimum-demand numbers |
| `test_run_cli_spark.py` | the pandas entry point, the column mapping, incomplete days, refused inputs, the command line, the provenance record, the Spark adapter (skipped without pyspark; run in CI) |
| `test_paper_*.py` | the paper code: folds and leakage guards, reference terms, metrics, operating points, and that `results/` matches its manifests |

`tests/fixtures/make_fixtures.py` rebuilds the reference slices from the datasets and
`results/`; run it only when the reference run changes.

## Documentation

```powershell
mkdocs serve          # local preview at http://127.0.0.1:8000
mkdocs build --strict
```

`docs-pages.yml` builds the site on every pull request to `main` and publishes it to GitHub
Pages at <https://mssamhan31.github.io/PyNRPF/> when `main` changes. The landing page
(`docs/index.md`) is written for a general reader; its figures live in `docs/assets/`,
copied from `publication/2_journal_article/results/paper/figures/`.

Every documentation page shows what an analyst running PyNRPF needs first and puts the rest in
one `/// details | Show details` block (type `more`) at its foot. Pages left out of the menu are
listed under `not_in_nav` in `mkdocs.yml` and reached from the Overview's full page list.

## Continuous integration

`ci.yml` runs on every push and pull request: lint, tests and a build on Python 3.10, 3.11
and 3.12 with the `dev` and `paper` extras; a build under setuptools 69; and a Spark job
(Java 17, the `spark` extra) that runs `test_run_cli_spark.py`, so the adapter is exercised
before a cluster sees it. `release.yml` runs on a `v*` tag. A nightly workflow smoke-tests
the conference paper archive.

## Releasing

1. Bump the version in `pyproject.toml`, `src/pynrpf/_version.py` and `CITATION.cff` (with
   `date-released`), and date the entry in `CHANGELOG.md`.
2. Open a pull request to `main` and merge it once `ci.yml` is green.
3. Create a GitHub Release `vX.Y.Z` from `main`, titled `vX.Y.Z`, with notes from the
   changelog. The tag starts `release.yml`: lint, tests, build, publish to PyPI, then attach
   the wheel, the sdist and `SHA256SUMS` to the release. Zenodo archives the release under
   the concept DOI `10.5281/zenodo.18764943`.
4. Check in a fresh environment: `pip install pynrpf==X.Y.Z`, `pynrpf --version`, the release
   assets, and the new version on Zenodo. PyPI never accepts a re-upload of the same version,
   so a failed publish means the next patch number.

///

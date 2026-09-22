# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

PyNRPF (Python for Network Reverse Power Flow) detects and corrects a wrong reverse
power flow (RPF) sign in fifteen-minute substation net-load data with one method, M9.
See [README.MD](README.MD) for what it does and [docs/](docs/) for the method, the API and
the paper.

## The red line: the reference run is the paper's

`publication/2_journal_article/results/` is the reference run every number in the journal
paper traces to, with a manifest per stage (repository-relative paths, SHA-256 hashes).
Regenerate it only through the notebooks or `paper.stages`, and only when asked; then
confirm the manifests reproduce (`pytest tests/test_paper_results.py`). Never edit a
committed result value by hand. Changing a model, metric, feature set, split,
hyperparameter, transformation or tolerance is methodology: propose it, do not do it.

If a fix needs a methodology change to be correct, stop and say so rather than making a
partial edit.

## Things that look wrong but are deliberate

- `publication/1_conference_paper/` is a self-contained archive with its own forked
  copy of the old package under `src/`. The duplication is the point: it is the code that
  produced the published conference paper. Do not refactor it, deduplicate it, or tidy it.
- Large datasets are committed on purpose (`.parquet` and `.csv` over 10 MB under
  `publication/*/dataset/`). They are anonymised Ausgrid data, cleared for publication,
  committed so the results reproduce from the repository alone. Do not flag them as
  leakage or propose gitignoring them.
- `publication/2_journal_article/sandbox/` is development history and dated studies. It is
  not maintained, not linted, not cited, and may carry process wording. Nothing outside
  `sandbox/` may reference it except the sandbox README.
- The M8 bundles under `results/02_baselines/bundles/` are gitignored: they are regenerated
  by notebook 02 and their hashes live in the fold manifests.

## Commands

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev,paper]

pytest -q                                                   # package and paper tests
pytest tests/test_m9_reference.py -q                        # one file
ruff check src/pynrpf tests publication/2_journal_article/paper
python -m build --sdist --wheel
mkdocs build --strict                                       # docs, built locally only
```

`ci.yml` installs `.[dev,paper]`; the paper tests import `publication/2_journal_article/paper`
by `sys.path`, so a new test that reaches into it must be satisfied by those extras.

## Architecture

The package is read in the order of the method. `src/pynrpf/m9/` holds one module per step
(`stories`, `windows`, `edges`, `bridge`, `misfit`, `evidence`, `winner`, `calibration`,
`decision`), composed by `score_siteday` in `m9/__init__.py`. `run.py` validates a frame
(`validate.py`), scores every site-day, calibrates, decides, computes the energy and
minimum-demand impact (`impact.py`) and returns the two tables of `schemas.py`. `spark.py`
runs `run` per site through `applyInPandas`; `cli.py` wraps it for a CSV.

Released constants live where they are used: `RELEASE_PHI` in `m9/evidence.py`,
`RELEASE_CALIBRATION` in `m9/calibration.py`, each with its provenance. Changing either is
methodology.

The paper's code is `publication/2_journal_article/paper/`, a local package the notebooks
call; it imports `pynrpf` for M9 and adds the M7 and M8 baselines, the station-held-out
folds, the reference-side metrics, the operating-point and Gamma studies, and the figure
and table registries. It has no public API and no version.

## Conventions

Australian spelling in all prose: README, docstrings, comments, notebook markdown.
American spelling stays in code identifiers and library interfaces (`color=`).

No process wording outside `sandbox/`: no "Phase", "round", "revision", "frozen", "locked",
"Track". Say "release", "reference run", "development history".

Notebooks are committed without outputs. Every notebook opens with a markdown cell giving
purpose, inputs, outputs and approximate runtime, and expands abbreviations on first use,
RPF included.

Manifests must not carry absolute paths: every path written under `results/` is relative
to `publication/2_journal_article/`.

Docstrings: every module opens with purpose, inputs, outputs and key steps; every public
function documents arguments, returns and units (MW, MWh, slots of fifteen minutes).

## Branches and releases

`main` is the released, Zenodo-archived version; a `v*` tag triggers `release.yml`, which
publishes to PyPI. Work happens on `dev` and feature branches. Do not merge working
branches into `main`; that is a release decision. Do not commit or push unless asked.

`.ai/` and `planning/` are gitignored working directories.

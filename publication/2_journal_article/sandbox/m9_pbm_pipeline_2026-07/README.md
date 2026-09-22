# Archive — legacy M9-PBM journal pipeline (frozen 2026-09-16)

This folder holds the journal-article work as it stood before the compact
counterfactual M9 (revision 2, `../m9_dev/`) replaced the nine-feature
"physical bridge model" (M9-PBM). It is kept intact for provenance and is not
developed further. Nothing here is imported by the live Phase 3 evaluation.

## What is here

| path | what it was |
|---|---|
| `notebooks/00_prepare_datasets.ipynb` … `04_publication_tables_figures.ipynb` | the numbered M9-PBM pipeline: dataset preparation, characterisation, method example, candidate features, training regimes, feature ablation, weight optimisation, ML comparison, final evaluation, Gamma forecast impact, publication tables and figures |
| `notebooks/_experiment_helpers.py`, `_m9_pbm_*.py`, `_build_*.py`, `_cached_figure_rendering.py`, `_figure_sources.py`, `_journal_figure_style.py`, `_gamma_forecast.py` | the helper modules those notebooks import |
| `notebooks/99_Misc/` | working experiments, specs and journals from June–July 2026 |
| `outputs/` | every table, figure, metric, intermediate file and manifest those notebooks produced; the M7/M8 correction-validation tables under `outputs/*/02_correction_validation/` were produced on the 2023-10 to 2024-09 test window with M8 transferred from Alpha, and are superseded by the Phase 3 identical-fold evaluation |
| `config/experiment_config.yaml` | the configuration those notebooks read; the dataset hashes it records were carried into `../config/final_evaluation.yaml` |
| `tests/` | the three tests that imported these helpers, moved out of continuous integration with the code they test |

## What stayed live

`../dataset/` (unchanged; the frozen Alpha, Beta and Gamma datasets and the oracle
review material), `../m9_dev/` (the locked M9), and the repository test for the
oracle review workflow, which depends only on `dataset/`.

## Gamma

The Gamma downstream forecasting study in `notebooks/03_gamma_forecast_impact.ipynb`
was produced with M9-PBM corrections. Its forecast harness and four plots were
ported to `../final_eval/gamma.py` so the same figures and tables are regenerated
against the locked M9 in Phase 6.

## Running anything here

The notebooks expect the helper modules beside them and `config/experiment_config.yaml`
relative to this folder; the moved tests have their `sys.path` lines repointed. They
are not part of `pytest` from the repository root.

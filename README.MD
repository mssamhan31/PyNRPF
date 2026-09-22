# PyNRPF: Python for Network Reverse Power Flow

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19211355.svg)](https://doi.org/10.5281/zenodo.19211355)

PyNRPF detects and corrects a wrong reverse power flow (RPF) sign in fifteen-minute
substation net-load data.

## The problem

A substation meter records net load: demand minus local generation. When rooftop solar
exceeds demand the substation exports and net load is genuinely negative. Some meters store
that export as an import, so the recorded trace dips towards zero instead of crossing it.
The reading passes ordinary validation but is wrong by twice its value, and it corrupts
every minimum-demand study and net-load forecast built on it.

## The method

M9 asks one physical question of every day: for which daytime window would flipping the
recorded sign make the implied demand more plausible than keeping it, how strong is that
evidence, and is it strong enough to act on? The score needs no training; two fitted
numbers turn it into a probability; one control decides whether a day is corrected
automatically, kept automatically, or sent to a person. Raw readings are never altered.
The nine steps are one file each under `src/pynrpf/m9/` and are explained in
[docs/method.md](docs/method.md).

## Install

Python 3.10 or later. The package depends on numpy and pandas only.

```powershell
python -m pip install pynrpf
```

## Quick start

```python
import pandas as pd
import pynrpf

frame = pd.read_parquet("readings.parquet")   # substation_id, timestamp, net_load_MW, solar_MW
result = pynrpf.run(frame, c=0.7)
result.site_days      # one row per site and day: outcome, probability, window, MWh
result.intervals      # one row per reading: flag and corrected net load
```

On Spark: `pynrpf.spark.run_per_site(sdf)`. From a CSV without code:
`pynrpf run readings.csv --out results/`. Column names, the control and the calibration are
documented in [docs/quickstart.md](docs/quickstart.md).

## Repository layout

```
src/pynrpf/                 the package: m9/ (one file per step), run, validate, schemas, impact, spark, cli
docs/                       one page per topic; mkdocs serve for a local site
tests/                      package tests on fixtures, paper tests on the reference run
publication/2_journal_article/
    paper/                  the code the paper's notebooks call: baselines, folds, metrics, figures, tables
    notebooks/              five notebooks that regenerate the reference run, two that draw every figure and table
    results/                the reference run, with a manifest per stage
    dataset/                the Alpha, Beta and Gamma datasets (anonymised Ausgrid data)
    config/evaluation.yaml  the one configuration of the reference run
    sandbox/                development history and dated studies; not maintained, not cited
publication/1_conference_paper/   the conference paper's archive, self-contained
```

## Data

All datasets derive from anonymised Ausgrid distribution network data and are committed so
that the paper reproduces from the repository alone: Alpha (ten stations, three years, the
wrong sign imitated by taking the absolute value), Beta (eight stations, one year, real
errors labelled by two reviewers) and Gamma (one station for the forecasting case study).
Details in [docs/data.md](docs/data.md).

## Reproducing the paper

`python -m pip install -e .[dev,paper]`, then open the notebooks under
`publication/2_journal_article/notebooks/`: `paper_figures.ipynb` and `paper_tables.ipynb`
regenerate every figure and table from the committed results in minutes; notebooks 01 to
05 regenerate the results themselves. See [docs/reproduce.md](docs/reproduce.md).

## Tests

```powershell
python -m pip install -e .[dev,paper]
pytest -q
ruff check src/pynrpf tests publication/2_journal_article/paper
```

## Documentation

| page | what it covers |
|---|---|
| [docs/quickstart.md](docs/quickstart.md) | one call, the output tables column by column |
| [docs/spark.md](docs/spark.md) | the per-site Spark adapter |
| [docs/cli.md](docs/cli.md) | the command line |
| [docs/api.md](docs/api.md) | every public name |
| [docs/method.md](docs/method.md) | the nine steps of M9 with the formulas and a worked day |
| [docs/calibration.md](docs/calibration.md) | the release numbers and refitting for a new population |
| [docs/overrides.md](docs/overrides.md) | the review-table schema for human decisions |
| [docs/data.md](docs/data.md) | the datasets and how they were labelled |
| [docs/evaluation.md](docs/evaluation.md) | how the paper judges a method |
| [docs/reproduce.md](docs/reproduce.md) | the notebooks and the reference run |
| [docs/development.md](docs/development.md) | tests, lint, continuous integration, releases |

## Licence

MIT. See [LICENSE](LICENSE).

## Acknowledgements

This project is part of Samhan's PhD study, supported by the University International
Postgraduate Award (UIPA) Scholarship from UNSW, the Industry Collaboration Project
Scholarship from Ausgrid, the RACE for 2030 Scholarship, and the NSW Decarbonisation
Innovation Hub (NSW Decarb Hub).

## Contributors

- M. Syahman Samhan (contact: m.samhan@unsw.edu.au)
- Anna Bruce
- Baran Yildiz

## Citation

If you use this software, please cite it using the metadata in [CITATION.cff](CITATION.cff),
or the archived release:

> Samhan, M. S., Bruce, A., & Yildiz, B. (2026). *PyNRPF: Detecting and rectifying wrong
> reverse power flow readings* [Software]. Zenodo. https://doi.org/10.5281/zenodo.19211355

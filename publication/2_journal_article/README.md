# PyNRPF journal article: Phase 3 final evaluation

The frozen station-held-out evaluation of three methods for detecting and correcting a
wrong reverse power flow (RPF) sign in substation interval data: M7 (deterministic
threshold rule), M8 (two-stage XGBoost classifier) and M9 (compact counterfactual
method, revision 2). Every table, figure and number in the journal article and on the
public website traces to the files this folder produces.

## Problem and audience

Rooftop solar can push a distribution substation into reverse power flow. Some meters
store that export with the wrong sign, so recorded net load looks like demand. The
Python for Network Reverse Power Flow (PyNRPF) package detects and corrects such days;
this folder measures how well each method does it on stations it has never seen.
It is written for a peer reviewer reproducing the paper, for Ausgrid deciding which
method to run by default, and for the author returning to it later.

## Installation

From the repository root, the full environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Python 3.10 or later. Nothing beyond `requirements.txt` is needed.

## Quick start

Run the notebooks in `notebooks/` in numerical order. Notebooks 01, 03, 05, 06, 07
and 08 reproduce every M7 and M9 number in a few minutes; notebook 02 trains the
eighteen M8 bundles (about forty minutes) and notebook 04 scores them. Notebook 09
is the Gamma downstream forecasting study and notebook 10 the operating-point
sensitivity study of the M9 decision policy. Each notebook opens with its purpose,
inputs, outputs and runtime.

The same stages run from the command line, from this folder:

```powershell
python -m final_eval folds
python -m final_eval train-m8          # heavy; --fold beta_beta_A for one fold
python -m final_eval predict-m7
python -m final_eval predict-m8
python -m final_eval predict-m9
python -m final_eval outcomes
python -m final_eval metrics
python -m final_eval report
python -m final_eval gamma
python -m final_eval operating-points
```

Expected result: `<output_dir>/08_tables/table01_headline.md` holds the headline table
and `07_metrics/gate.json` the release-gate decision, where `<output_dir>` is `paths.output_dir`
in the config (`outputs/01_final_evaluation` by default; pass `--config` for a sandbox run).
The Phase 3 run of 16 September 2026 is kept as it was under
`sandbox/2026-09-16_phase3_release/outputs/`; the Beta-only evaluation of 22 September
lives in `sandbox/2026-09-22_beta_only_eval/`.

## Structure

```
config/final_evaluation.yaml   the one configuration: dataset hashes, fold rule, M7/M8 settings,
                               the frozen M9 settings, c, the gate, Gamma forecast settings
dataset/                       frozen Alpha, Beta and Gamma datasets and the oracle review material
m9_dev/                        the locked M9 revision 2 scorer and its development record; imported, never copied
final_eval/                    the evaluation package, one module per stage (see below)
notebooks/                     ten thin notebooks, one per stage, committed without outputs
outputs/                       default output root for a fresh run (gitignored bundles and logs)
sandbox/                       dated studies; 2026-09-16_phase3_release/ holds the Phase 3 outputs unchanged
archive_m9_pbm_2026-07/        the previous pipeline, frozen for provenance; its own README explains it
```

| module | does |
|---|---|
| `final_eval/config.py` | loads the YAML, verifies dataset hashes, resolves paths |
| `final_eval/data.py` | the common site-day population (complete 96-slot days) and interval frames |
| `final_eval/folds.py` | the 18 leave-one-station-out folds and the fold manifest |
| `final_eval/m7.py`, `m8.py`, `m9.py` | held-out predictions per method in one interval schema |
| `final_eval/outcomes.py`, `impact.py` | decision policy, applied series, energy terms, minimum-demand impact |
| `final_eval/metrics.py` | pooled and per-station metrics, bootstrap, coverage, gate, sensitivity |
| `final_eval/tables.py`, `figures.py` | paper tables and figures, including 27 sample site-days per method |
| `final_eval/gamma.py` | the Gamma forecasting case study (ported from the archived study, same figures) |
| `final_eval/operating_points.py` | precision targets and the two-control policy: threshold selection per fold, trade-off tables and figures |
| `final_eval/manifest.py` | stage manifests with repository-relative paths and SHA-256 hashes |
| `final_eval/cli.py` | the stage entry points the notebooks call |

## Data

Alpha: ten stations, three years, correctly signed readings with controlled sign
errors, so the reference is exact. Beta: eight stations, one year, real sign errors
labelled by two reviewers; `sure` days are the headline reference, `unsure` days are
reported only as a sensitivity analysis and never enter a fit. Gamma: Beta station B
for the downstream forecasting study. All three are anonymised Ausgrid data cleared
for publication and committed in `dataset/final/` with their hashes; see the
repository's `docs/data.md` for provenance. M8 bundles are regenerated by notebook 02
and are not committed; their manifests are.

## Evaluation design, in brief

Every station is held out once. M7 needs no fitting. M8 is trained per fold on all
seventeen other stations of both cohorts (Beta `sure` days only) with frozen
thresholds and no inner tuning. M9 is scored label-free once and, per fold, fits two
calibration coefficients (`cal_intercept`, `cal_slope`) on the other stations of the
same cohort; the public control c = 0.7 gives AUTO_CORRECT, AUTO_KEEP or UNCERTAIN.
M7 and M8 are binary at their native thresholds. Only AUTO_CORRECT days carry applied
energy. Energy per slot is 2·y·0.25 MWh; Reference Energy IoU is the main metric and
reference energy precision the safety gate (0.90 on Beta `sure`; Alpha reported
against its label-contiguity ceiling of 0.904). Station-level bootstrap intervals
accompany every pooled headline.

The operating-point study (notebook 10) keeps the headline at c = 0.7 and asks, as
a sensitivity analysis, what correction threshold delivers a minimum precision on
unseen stations. The threshold is chosen per fold on the calibration stations only,
either from the calibrated probabilities alone (rule L, label-free) or from the
calibration stations' labels (rule E), and the achieved precision, recall, Energy IoU,
false corrections and review load are reported on the held-out stations.

## Tests

From the repository root, `pytest tests/test_final_eval_*.py -q`. They prove, on
synthetic fixtures, that no fold trains on its held-out station or on Beta `unsure`
days, that the energy and minimum-demand terms match hand-worked examples, that the
decision policy behaves at the band edges, that all methods share identical site-day
keys, that manifests carry no absolute path, and, on the committed release files,
that the M9 port reproduces the frozen development run to four decimals.

## Licence

MIT, as the repository root `LICENSE`.

## Acknowledgements

This project is part of Samhan's PhD study, supported by the University International
Postgraduate Award (UIPA) Scholarship from UNSW, the Industry Collaboration Project
Scholarship from Ausgrid, the RACE for 2030 Scholarship, and the NSW Decarbonisation
Innovation Hub (NSW Decarb Hub). The datasets are anonymised Ausgrid substation data
cleared for publication.

## Citation

See `CITATION.cff` at the repository root and the Zenodo record of the released version.

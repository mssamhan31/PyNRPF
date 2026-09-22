"""The code the journal article's notebooks call.

The notebooks are the story; this folder is the code they call. It is not a second
package: it has no public API and no version, and its only callers are the notebooks
under ``notebooks/`` and the tests under ``tests/``. M9 itself comes from the installed
``pynrpf`` package, so nothing here manipulates ``sys.path``; what lives here is only
what the paper needs and the package must not carry.

Modules, in the order the notebooks use them:

    config, data, folds      notebook 01  the configuration, the datasets, the station-held-out folds
    baselines/               notebook 02  M7 (threshold rule) and M8 (two-stage XGBoost), fitted per fold
    m9                       notebook 03  M9 scored with ``pynrpf`` and calibrated per fold
    reference, metrics,      notebook 04  reference-side terms, the metric tables, the operating-point study
      operating_points
    gamma                    notebook 05  the forecasting case study
    figures, tables, style   paper_figures, paper_tables  the registries of paper artefacts
    results                  readers for the files a finished stage wrote
    manifest                 stage manifests with repository-relative paths and hashes
    stages                   one function per stage; what each notebook runs

Every stage writes its files under ``results/`` (or the directory the configuration
names) and a manifest under ``results/manifests/``; the reference run committed there is
the run the paper reports.
"""

from __future__ import annotations

from pathlib import Path

ARTICLE_ROOT = Path(__file__).resolve().parents[1]

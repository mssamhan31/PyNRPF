"""Phase 3 final evaluation of M7, M8 and M9 on identical station-held-out folds.

Purpose: produce the single frozen evaluation release that every paper and public
claim traces to. One module per stage, each callable from the numbered notebooks
or from ``python -m final_eval <stage>``.

Stages, in order:
    folds        common site-day population and the 18-fold manifest        (config, data, folds)
    train-m8     one M8 bundle per fold, the only heavy stage                (m8)
    predict-*    held-out predictions per method                            (m7, m8, m9)
    outcomes     decisions, applied series, energy and minimum-demand impact (outcomes, impact)
    metrics      pooled, per-station, bootstrap, coverage, gate, sensitivity (metrics)
    report       paper tables and figures                                   (tables, figures)
    gamma        downstream forecasting case study                          (gamma)

Every stage writes a manifest with repository-relative paths and SHA-256 hashes of
what it read and wrote (manifest). Nothing here changes M9 revision 2: the scorer
is imported from ``m9_dev`` and never copied.
"""

from __future__ import annotations

import sys
from pathlib import Path

ARTICLE_ROOT = Path(__file__).resolve().parents[1]
__version__ = "1.0.0"

# The frozen M9 modules (m9_scorer, m9_metrics) are imported by name from m9_dev/;
# registering the folder here keeps every module of this package importable on its own.
_M9_DEV = str(ARTICLE_ROOT / "m9_dev")
if _M9_DEV not in sys.path:
    sys.path.insert(0, _M9_DEV)

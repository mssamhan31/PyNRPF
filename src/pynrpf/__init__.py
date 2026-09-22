"""PyNRPF: detect and correct a wrong reverse-power-flow sign in fifteen-minute net-load data.

    import pynrpf
    result = pynrpf.run(frame)          # pandas frame in, two tables out
    result.site_days, result.intervals

The method (M9) lives in ``pynrpf.m9``, one file per step; ``pynrpf.spark.run_per_site``
runs it on a Spark frame; ``pynrpf run`` is the command line.
"""

from .m9 import RELEASE_CALIBRATION, RELEASE_PHI, Calibration, Score, fit_calibration, score_siteday
from .run import Result, run

__version__ = "0.4.0"

__all__ = ["Calibration", "RELEASE_CALIBRATION", "RELEASE_PHI", "Result", "Score", "__version__",
           "fit_calibration", "run", "score_siteday"]

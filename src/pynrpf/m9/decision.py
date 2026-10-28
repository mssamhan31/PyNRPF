"""Step 9 of M9: one control, three outcomes, and the corrected series.

    AUTO_CORRECT   if p >= c          the sign is flipped inside the best window
    AUTO_KEEP      if p <= 1 - c      the day is left alone
    UNCERTAIN      otherwise          the day is left alone and listed for review

``c`` is the only control an operator sets; the release default is 0.7. Raw data is never
overwritten: the corrected series is a second series beside the recorded one.

Inputs:  the probability of step 8, the control ``c``, the recorded net load and the window.
Outputs: the outcome name; the net load with the sign flipped inside the window (MW).
"""

from __future__ import annotations

import numpy as np

from .winner import Candidate

AUTO_CORRECT = "AUTO_CORRECT"
AUTO_KEEP = "AUTO_KEEP"
UNCERTAIN = "UNCERTAIN"
OUTCOMES = (AUTO_CORRECT, AUTO_KEEP, UNCERTAIN)
DEFAULT_C = 0.7


def decide(p: float, c: float = DEFAULT_C) -> str:
    """The outcome for a probability ``p`` under the control ``c`` (``0.5 < c < 1``)."""
    if not 0.5 < c < 1.0:
        raise ValueError(f"c must lie strictly between 0.5 and 1, got {c}")
    if not np.isfinite(p):
        return UNCERTAIN
    if p >= c:
        return AUTO_CORRECT
    if p <= 1.0 - c:
        return AUTO_KEEP
    return UNCERTAIN


def corrected_series(y: np.ndarray, window: Candidate) -> np.ndarray:
    """The recorded net load with the sign flipped inside ``window`` (MW); a copy when there is none."""
    out = np.array(y, dtype=float, copy=True)
    if not window.is_null:
        out[window.start : window.end + 1] = -out[window.start : window.end + 1]
    return out

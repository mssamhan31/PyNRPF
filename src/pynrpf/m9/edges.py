"""Step 3 of M9: a window may only start and end where the recorded trace kinks.

A wrong sign reflects the true trace about zero, so where the error starts and ends the
recorded net load touches its own reflection: a cusp, which is a local minimum. The rule is
applied to the candidate set, not to the score, with one slot of tolerance inward:

    a slot t is a local minimum when   y_t <= y_{t-1}  and  y_t <= y_{t+1}   (plateaus count)
    a start a is valid when            t = a  or  t = a - 1  is a local minimum
    an end b is valid when             t = b  or  t = b + 1  is a local minimum

The tolerance points inward because outward tolerance would re-admit the over-wide windows
the rule exists to remove. An edge beside a missing reading is exempt: a cusp cannot be seen
against a gap.

Inputs:  ``y`` recorded net load and ``s`` solar estimate, MW, 96 slots each.
Outputs: two boolean vectors over the 48 scan slots, valid starts and valid ends.
"""

from __future__ import annotations

import numpy as np

from .windows import SCAN_END, SCAN_START, SLOTS


def local_minima(y: np.ndarray) -> np.ndarray:
    """Boolean per slot: the recorded net load is at a local minimum (non-strict).

    Slots 0 and 95 have one neighbour only and are never minima; a slot whose value or
    neighbour is missing is not a minimum.
    """
    m = np.zeros(SLOTS, dtype=bool)
    with np.errstate(invalid="ignore"):
        m[1:-1] = (y[1:-1] <= y[:-2]) & (y[1:-1] <= y[2:])
    return m & np.isfinite(y)


def valid_edges(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Valid window starts and ends over the scan slots, under the inward rule and the gap exemption."""
    minimum = local_minima(y)
    start_ok = minimum.copy()
    start_ok[1:] |= minimum[:-1]            # a start may sit one slot after a minimum
    end_ok = minimum.copy()
    end_ok[:-1] |= minimum[1:]              # an end may sit one slot before a minimum
    finite = np.isfinite(y) & np.isfinite(s)
    gap_before = np.zeros(SLOTS, dtype=bool)
    gap_after = np.zeros(SLOTS, dtype=bool)
    gap_before[1:] = ~finite[:-1]           # the reading just before this slot is missing
    gap_after[:-1] = ~finite[1:]            # the reading just after this slot is missing
    scan = slice(SCAN_START, SCAN_END)
    return (start_ok | gap_before)[scan], (end_ok | gap_after)[scan]

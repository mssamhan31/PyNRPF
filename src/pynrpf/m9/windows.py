"""Step 2 of M9: the candidate windows and which of them can be scored.

M9 does not know where an error starts or ends, so it tries every contiguous window of
slots between 06:00 and 18:00: 48 slots, hence 48 x 49 / 2 = 1,176 start-end pairs. A
window is admissible when every reading inside it is finite and a finite reading exists
somewhere before its start and somewhere after its end (those readings anchor the bridge of
step 4). Missing readings therefore disqualify the windows that touch them, not the day.

Inputs:  ``y`` recorded net load and ``s`` solar estimate, MW, 96 slots each.
Outputs: boolean matrices indexed ``[start, end]`` over the 48 scan slots, and the index of
         the nearest finite reading on each side of every slot.
"""

from __future__ import annotations

import numpy as np

SLOTS = 96                      # fifteen-minute slots in a day
SCAN_START = 24                 # 06:00, first slot a window may start on
SCAN_END = 72                   # 18:00, exclusive: slots 71 is the last a window may end on
N_WINDOWS = SCAN_END - SCAN_START
N_CANDIDATES = N_WINDOWS * (N_WINDOWS + 1) // 2   # 1,176


def grid() -> tuple[np.ndarray, np.ndarray]:
    """Slot numbers of every window start (column vector) and end (row vector)."""
    a = SCAN_START + np.arange(N_WINDOWS)[:, None]
    b = SCAN_START + np.arange(N_WINDOWS)[None, :]
    return a, b


def nearest_finite(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every slot, the last finite reading before it and the first after it (-1 if none).

    A reading is finite when both ``y`` and ``s`` are finite.
    """
    ok = np.isfinite(y) & np.isfinite(s)
    before = np.full(SLOTS, -1)
    last = -1
    for k in range(SLOTS):
        before[k] = last
        if ok[k]:
            last = k
    after = np.full(SLOTS, -1)
    first = -1
    for k in range(SLOTS - 1, -1, -1):
        after[k] = first
        if ok[k]:
            first = k
    return before, after


def admissible(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Boolean ``[start, end]`` matrix of the windows that can be scored.

    True when ``end >= start``, no reading inside the window is missing, and a finite
    reading exists before the start and after the end.
    """
    finite = np.isfinite(y) & np.isfinite(s)
    missing_before = np.concatenate([[0], np.cumsum(~finite)])   # missing readings in slots < k
    a, b = grid()
    n_missing_inside = missing_before[b + 1] - missing_before[a]
    before, after = nearest_finite(y, s)
    return (n_missing_inside == 0) & (b >= a) & (before[a] >= 0) & (after[b] >= 0)

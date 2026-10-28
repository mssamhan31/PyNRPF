"""What a correction changes on a site-day: the energy it flips and the daily minimum.

Flipping the sign of a fifteen-minute reading of ``y`` MW changes the recorded energy by
``2 * y * 0.25`` MWh. Summed over the proposed window that is the proposed correction energy,
the quantity the paper's energy metrics are built on. The minimum of the day is the number
network planning rests on, so the change a correction makes to it is reported as well.

Inputs:  ``y`` recorded net load, MW, 96 slots (NaN for a missing reading, which carries no
         energy); the proposed window.
Outputs: MWh and MW numbers for one site-day.
"""

from __future__ import annotations

import numpy as np

from .m9.winner import Candidate

HOURS_PER_SLOT = 0.25


def slot_energy(y: np.ndarray) -> np.ndarray:
    """Energy a flip changes in each slot, MWh: ``2 * y * 0.25``. Missing readings carry none.

    The signed reading is used, as in the reference evaluation; inside a genuine wrong-sign
    window the readings are positive, so this equals ``2 * |y| * 0.25`` there.
    """
    return 2.0 * np.nan_to_num(np.asarray(y, dtype=float), nan=0.0) * HOURS_PER_SLOT


def proposed_energy_mwh(y: np.ndarray, window: Candidate) -> float:
    """Energy the proposed window would flip, MWh (0 when there is no window)."""
    return float(slot_energy(y)[window.mask(len(y))].sum())


def minimum_demand(y: np.ndarray, window: Candidate) -> tuple[float, float, float]:
    """``(recorded minimum, minimum after flipping the window, change)`` in MW.

    Missing readings are ignored; a day with no finite reading returns NaN throughout.
    """
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).any():
        return np.nan, np.nan, np.nan
    recorded = float(np.nanmin(y))
    corrected = float(np.nanmin(np.where(window.mask(len(y)), -y, y)))
    return recorded, corrected, corrected - recorded

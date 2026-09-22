"""Step 1 of M9: the two stories of underlying demand.

Underlying demand is what the customers behind a substation consumed: the solar generation
estimate plus the net load the meter recorded. If the meter's sign is right, demand is
``s + y`` in every slot. If the sign is wrong inside a window, the export in that window was
stored as an import, so demand there is ``s - y`` instead.

    Story A, keep the sign:              U0_t = s_t + y_t
    Story B, flip it inside window W:    UW_t = s_t - y_t  for t in W,  s_t + y_t elsewhere

Inputs:  ``y`` recorded net load, MW, 96 fifteen-minute slots, positive = import as stored;
         ``s`` solar generation estimate, MW, 96 slots.
Outputs: the two full-day demand series in MW. Story B is returned with the sign flipped in
         every slot; step 5 measures it only inside each candidate window.
"""

from __future__ import annotations

import numpy as np


def demand_if_kept(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Story A: underlying demand if the recorded sign is right, ``U0 = s + y`` (MW)."""
    return s + y


def demand_if_flipped(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Story B in every slot: underlying demand if the recorded sign is wrong, ``s - y`` (MW).

    Computed as ``U0 - 2y`` rather than ``s - y`` so that the two stories share the same
    floating-point rounding; the reference evaluation used this form.
    """
    return demand_if_kept(y, s) - 2.0 * y

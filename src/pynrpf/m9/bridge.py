"""Step 4 of M9: the straight bridge between the readings just outside a window.

For a window ``[a, b]`` the anchors are the nearest finite readings before ``a`` (slot
``l``) and after ``b`` (slot ``r``). The bridge is the straight line between the underlying
demand at those two slots:

    g_W(t) = U0_l + (t - l) / (r - l) * (U0_r - U0_l)

Both stories share one bridge, because they agree outside the window. The bridge does not
claim demand is linear; it asks which story sits closer to a smooth transition between the
surrounding demand levels.

Inputs:  the kept-sign demand ``U0`` (MW, 96 slots) for the anchors; anchor slots and values
         and the slots to evaluate for the line.
Outputs: anchor indices per slot; the line values as a matrix ``[end, slot]``.
"""

from __future__ import annotations

import numpy as np

from .windows import nearest_finite


def anchors(u0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest finite reading of ``U0`` before and after every slot (-1 where none)."""
    return nearest_finite(u0, u0)


def line(left_slot: int, left_value: float, right_slots: np.ndarray, right_values: np.ndarray,
         slots: np.ndarray) -> np.ndarray:
    """Bridge values at ``slots`` for one left anchor and one right anchor per window end.

    Args:
        left_slot, left_value: the left anchor, shared by every end.
        right_slots, right_values: one right anchor per end (a missing anchor is NaN-valued).
        slots: the slots to evaluate, shared by every end.

    Returns:
        Matrix ``[end, slot]`` of bridge values in MW.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        span = (right_slots - left_slot).astype(float)
        fraction = (slots[None, :] - left_slot) / span[:, None]
        return left_value + (right_values[:, None] - left_value) * fraction

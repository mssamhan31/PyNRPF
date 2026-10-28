"""Step 5 of M9: how far each story sits from the bridge, as a residual sum of squares.

Inside a window the residual of a story is its demand minus the bridge value. Squared and
summed over the window's slots:

    RSS_u(W) = sum over t in W of (U0_t - g_W(t))^2      the kept-sign story
    RSS_c(W) = sum over t in W of (UW_t - g_W(t))^2      the flipped story

Smaller means closer to the bridge. On a genuine wrong-sign window the flipped story hugs
the bridge and RSS_c is far smaller than RSS_u.

Inputs:  the two demand series (MW, 96 slots) and the anchor indices from step 4.
Outputs: three ``[start, end]`` matrices over the 48 scan slots: RSS_u and RSS_c in MW^2
         (NaN where a window has no anchor) and the window length in slots.
"""

from __future__ import annotations

import numpy as np

from .bridge import line
from .windows import N_WINDOWS, SCAN_END, SCAN_START, SLOTS


def _sum_of_squares(series: np.ndarray, bridge: np.ndarray, slots: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Squared residuals of ``series`` against ``bridge``, summed over the slots inside each window."""
    residual = np.where(inside, series[slots][None, :] - bridge, 0.0)
    return (residual**2).sum(1)


def residual_matrices(u0: np.ndarray, uw: np.ndarray, before: np.ndarray, after: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RSS of both stories and the length for every window, vectorised over window ends.

    Args:
        u0: kept-sign demand, MW, 96 slots.
        uw: flipped demand in every slot, MW, 96 slots (step 1).
        before, after: anchor slot before and after every slot (step 4), -1 where none.

    Returns:
        ``(rss_u, rss_c, length)``, each ``[start, end]`` over the scan slots.
    """
    rss_u = np.full((N_WINDOWS, N_WINDOWS), np.nan)
    rss_c = np.full((N_WINDOWS, N_WINDOWS), np.nan)
    length = np.zeros((N_WINDOWS, N_WINDOWS))
    scan = np.arange(SCAN_START, SCAN_END)
    for i, a in enumerate(scan):
        left = before[a]
        if left < 0:
            continue                                        # no reading before the window: unscorable
        ends = np.arange(a, SCAN_END)                       # every end for this start
        right = after[ends]
        right_safe = np.where(right >= 0, right, SLOTS - 1)
        right_value = np.where(right >= 0, u0[right_safe], np.nan)
        slots = scan[i:]                                    # slots a .. 71
        inside = slots[None, :] <= ends[:, None]            # [end, slot]: slot belongs to the window
        length[i, i:] = inside.sum(1)
        # The anchors lie outside the window, where the two stories agree, so both use U0 there.
        bridge = line(left, u0[left], right_safe, right_value, slots)
        rss_u[i, i:] = _sum_of_squares(u0, bridge, slots, inside)
        rss_c[i, i:] = _sum_of_squares(uw, bridge, slots, inside)
    return rss_u, rss_c, length

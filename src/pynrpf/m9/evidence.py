"""Step 6 of M9: the evidence for flipping the sign inside each window.

The two misfits are compared as a ratio on a log scale, weighted by the window length:

    r(W) = (L / 2) * log( (RSS_u(W) + lambda_L) / (RSS_c(W) + lambda_L) ),   lambda_L = L * phi^2

``r > 0`` says the flipped story fits the bridge better, ``r = 0`` that both fit equally,
``r < 0`` that keeping the sign fits better. The ratio makes the day its own yardstick (no
external noise scale), the logarithm makes a two-fold improvement and a two-fold
deterioration symmetric, and ``L/2`` is what the Gaussian profile likelihood gives: with the
variance fitted under each story, the log-likelihood difference is exactly this expression.
The floor ``lambda_L`` keeps a near-perfect fit finite and gives ``r = 0`` when both stories
fit perfectly; it scales with the window length and with the units, like RSS.

``phi`` is the smallest non-zero overnight step of demand across the population the method
was released on. It only guards perfectly flat readings and is fixed so that a day scores
the same in any batch.

Inputs:  the RSS and length matrices of step 5, ``phi`` in MW, the admissible matrix of steps 2 and 3.
Outputs: the evidence matrix ``[start, end]``; ``-inf`` where a window is inadmissible.
"""

from __future__ import annotations

import numpy as np

# Release floor: smallest non-zero overnight demand step over the eight Beta stations,
# 22 September 2026 reference run (results/manifests/05_m9_predict.json, sigma_floor_beta).
RELEASE_PHI = 8.03130184579004e-09


def evidence_matrix(rss_u: np.ndarray, rss_c: np.ndarray, length: np.ndarray, phi: float,
                    admissible: np.ndarray) -> np.ndarray:
    """Evidence ``r(W)`` for every window; ``-inf`` for inadmissible or undefined windows."""
    length = np.maximum(length, 1.0)
    floor = length * phi**2
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log((rss_u + floor) / (rss_c + floor))
        r = 0.5 * log_ratio * length
    r = np.where(np.isfinite(r), r, -np.inf)
    return np.where(admissible, r, -np.inf)

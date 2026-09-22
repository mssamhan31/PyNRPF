"""M9: a label-free counterfactual scorer for a wrong reverse-power-flow sign, one site-day at a time.

The method asks, for every daytime window, whether flipping the recorded sign inside it
makes the implied underlying demand more plausible than keeping the sign, how strong that
evidence is, and whether it is strong enough to act on. Read the step modules in order:

    stories      step 1  the two reconstructions of demand
    windows      step 2  the 1,176 candidate windows and admissibility
    edges        step 3  window edges must sit where the recorded trace kinks
    bridge       step 4  the straight bridge between the readings outside a window
    misfit       step 5  residual sum of squares of each story against the bridge
    evidence     step 6  r(W) = (L/2) log((RSS_u + lambda)/(RSS_c + lambda))
    winner       step 7  the best window, no correction at evidence zero, the runner-up
    calibration  step 8  z = sign(r*) log(1 + |r*|), p = logistic(a + b z)
    decision     step 9  c = 0.7: AUTO_CORRECT, AUTO_KEEP or UNCERTAIN

``score_siteday`` composes steps 1 to 7 for one day; ``pynrpf.run`` applies steps 8 and 9
to every day of a frame.

Inputs:  ``y`` recorded net load and ``s`` solar estimate, MW, 96 fifteen-minute slots each,
         NaN for a missing reading; ``phi`` the evidence floor in MW.
Outputs: a ``Score``: whether the day could be scored, the number of admissible windows,
         the winner, the runner-up and the evidence ``r*``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import bridge, edges, evidence, misfit, stories, windows, winner
from .calibration import RELEASE_CALIBRATION, Calibration, fit_calibration
from .decision import AUTO_CORRECT, AUTO_KEEP, OUTCOMES, UNCERTAIN, corrected_series, decide
from .evidence import RELEASE_PHI
from .winner import NO_CORRECTION, Candidate


@dataclass(frozen=True)
class Score:
    """What M9 knows about one site-day before calibration."""

    input_ok: bool
    n_admissible: int
    winner: Candidate
    runner_up: Candidate
    evidence: float           # r*: the best window's evidence, NaN when the day cannot be scored


def score_siteday(y: np.ndarray, s: np.ndarray, phi: float = RELEASE_PHI) -> Score:
    """Steps 1 to 7 for one site-day.

    Args:
        y: recorded net load, MW, 96 slots, positive = import as stored, NaN where missing.
        s: solar generation estimate, MW, 96 slots.
        phi: the evidence floor, MW (``RELEASE_PHI`` unless research asks otherwise).
    """
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    admissible = windows.admissible(y, s)                               # step 2
    start_ok, end_ok = edges.valid_edges(y, s)                          # step 3
    admissible &= start_ok[:, None] & end_ok[None, :]
    if not admissible.any():
        return Score(False, 0, NO_CORRECTION, NO_CORRECTION, float("nan"))
    kept = stories.demand_if_kept(y, s)                                 # step 1
    flipped = stories.demand_if_flipped(y, s)
    before, after = bridge.anchors(kept)                                # step 4
    rss_u, rss_c, length = misfit.residual_matrices(kept, flipped, before, after)   # step 5
    r = evidence.evidence_matrix(rss_u, rss_c, length, phi, admissible)             # step 6
    best, runner = winner.rank(r)                                       # step 7
    return Score(True, int(admissible.sum()), best, runner, winner.best_evidence(best, runner))


__all__ = [
    "AUTO_CORRECT", "AUTO_KEEP", "UNCERTAIN", "OUTCOMES", "NO_CORRECTION",
    "Calibration", "Candidate", "RELEASE_CALIBRATION", "RELEASE_PHI", "Score",
    "corrected_series", "decide", "fit_calibration", "score_siteday",
]

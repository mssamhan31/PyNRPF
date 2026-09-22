"""Step 7 of M9: pick the best window, with "no correction" competing on equal terms.

The window with the most evidence wins. "No correction" is a candidate with evidence
exactly zero, so it wins whenever no window fits better than keeping the sign. Ties are
resolved in this order: no correction over a window, a shorter window over a longer one,
an earlier window over a later one. The best window that does not overlap the winner is
kept as the runner-up, a genuinely different explanation a reviewer can compare.

The evidence carried forward, ``r*``, is the best window's evidence even when no correction
wins, so that a day with a weak negative best window stays distinguishable from a day with
none at all.

Inputs:  the evidence matrix of step 6.
Outputs: the winning candidate, the runner-up, and ``r*``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .windows import N_WINDOWS, SCAN_START, grid

TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Candidate:
    """One candidate: a window of slots ``start .. end`` inclusive, or no correction (start < 0)."""

    start: int
    end: int
    evidence: float

    @property
    def is_null(self) -> bool:
        return self.start < 0

    @property
    def length(self) -> int:
        return 0 if self.is_null else self.end - self.start + 1

    def mask(self, n_slots: int = 96) -> np.ndarray:
        """Boolean per slot: inside the window."""
        inside = np.zeros(n_slots, dtype=bool)
        if not self.is_null:
            inside[self.start : self.end + 1] = True
        return inside


NO_CORRECTION = Candidate(-1, -1, 0.0)


def best_window(r: np.ndarray) -> Candidate:
    """Highest-evidence window; ties go to the shorter, then the earlier, window."""
    top = r.max()
    if not np.isfinite(top):
        return NO_CORRECTION
    starts, ends = np.nonzero(r >= top - TIE_TOLERANCE)
    order = np.lexsort((starts, ends - starts))          # primary key length, secondary start
    i, j = starts[order[0]], ends[order[0]]
    return Candidate(SCAN_START + i, SCAN_START + j, float(r[i, j]))


def runner_up(r: np.ndarray, best: Candidate) -> Candidate:
    """Best window that does not overlap ``best``."""
    if best.is_null:
        return best_window(r)
    a, b = grid()
    overlaps = ~((b < best.start) | (a > best.end))
    return best_window(np.where(overlaps, -np.inf, r))


def rank(r: np.ndarray) -> tuple[Candidate, Candidate]:
    """The winner and the runner-up, with no correction ranked jointly at evidence zero.

    When no correction wins, the best window is returned as the runner-up so it stays
    available for inspection.
    """
    best = best_window(r)
    if best.is_null or best.evidence <= 0.0 + TIE_TOLERANCE:
        return NO_CORRECTION, best
    return best, runner_up(r, best)


def best_evidence(winner: Candidate, runner: Candidate) -> float:
    """``r*``: the evidence of the best window, whether or not it beat no correction."""
    return float(runner.evidence if winner.is_null else winner.evidence)


__all__ = ["Candidate", "NO_CORRECTION", "N_WINDOWS", "best_window", "runner_up", "rank", "best_evidence"]

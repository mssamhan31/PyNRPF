"""Step 8 of M9: turn the evidence into a probability with two fitted numbers.

The winning evidence spans orders of magnitude, so it is first compressed:

    z = sign(r*) * log(1 + |r*|)

and then mapped to the probability that the day carries a wrong sign:

    p = 1 / (1 + exp(-(a + b * z)))

``a`` (the intercept) sets how much evidence a population's ordinary days produce; ``b``
(the slope) sets how fast the probability rises with evidence. Both were fitted by logistic
regression on reviewed days. The slope was found to carry over between populations while
the intercept did not, so a new population needs only its intercept, from about 27 reviewed
days, with the slope kept at the release value.

Inputs:  evidence ``r*`` per site-day; for fitting, the reviewed label per day.
Outputs: probabilities in (0, 1); a ``Calibration`` holding the two numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def signed_log(r: np.ndarray | float) -> np.ndarray | float:
    """``z = sign(r) * log(1 + |r|)``: keeps direction, zero and order, compresses the spread."""
    return np.sign(r) * np.log1p(np.abs(r))


def logistic(x: np.ndarray | float) -> np.ndarray | float:
    """The sigmoid ``1 / (1 + exp(-x))``; an overflow for very negative ``x`` gives exactly 0."""
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class Calibration:
    """The two numbers that map evidence to probability, with a note on where they came from."""

    intercept: float
    slope: float
    provenance: str = ""

    def probability(self, evidence: np.ndarray | float) -> np.ndarray | float:
        """Probability that a site-day with evidence ``r*`` carries a wrong sign."""
        return logistic(self.intercept + self.slope * signed_log(evidence))

    def evidence_at(self, p: float) -> float:
        """The raw evidence ``r*`` at which the probability equals ``p`` (for reporting thresholds)."""
        z = (np.log(p / (1.0 - p)) - self.intercept) / self.slope
        return float(np.sign(z) * np.expm1(abs(z)))


# Fitted on the reviewed 'sure' days of all eight Beta stations (2,288 days, 470 with a wrong
# sign) in the 22 September 2026 reference run: results/05_m9/calibration_fits.csv.
RELEASE_CALIBRATION = Calibration(
    intercept=-4.590095360044541,
    slope=2.1620798550382885,
    provenance="Beta, eight stations, sure days, 22 September 2026",
)


def fit_calibration(evidence: np.ndarray, labels: np.ndarray, slope: float | None = None,
                    provenance: str = "") -> Calibration:
    """Fit the calibration on reviewed days.

    Args:
        evidence: ``r*`` per reviewed site-day.
        labels: 1 for a day that carries a wrong sign, 0 otherwise.
        slope: when given, only the intercept is fitted and the slope is kept (the
            label-efficient refit for a new population); when None, both numbers are fitted
            by logistic regression, which needs scikit-learn (``pip install pynrpf[fit]``).
        provenance: free text recorded on the result.

    Returns:
        A ``Calibration``. Units: evidence is dimensionless.
    """
    z = signed_log(np.asarray(evidence, dtype=float))
    labels = np.asarray(labels, dtype=float)
    if labels.min() == labels.max():
        raise ValueError("Both labels are needed to fit a calibration.")
    if slope is None:
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Fitting both numbers needs scikit-learn: pip install pynrpf[fit]") from exc
        model = LogisticRegression(max_iter=2000).fit(z.reshape(-1, 1), labels.astype(int))
        return Calibration(float(model.intercept_[0]), float(model.coef_[0, 0]), provenance)
    return Calibration(_fit_intercept(z, labels, slope), float(slope), provenance)


def _fit_intercept(z: np.ndarray, labels: np.ndarray, slope: float) -> float:
    """Maximum-likelihood intercept with the slope fixed, by Newton's method on one parameter."""
    offset = slope * z
    intercept = 0.0
    for _ in range(100):
        p = logistic(intercept + offset)
        gradient = float((labels - p).sum())
        curvature = float((p * (1.0 - p)).sum())
        if curvature <= 0.0:
            break
        step = gradient / curvature
        intercept += step
        if abs(step) < 1e-10:
            break
    return float(intercept)

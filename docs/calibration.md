# Calibration

Step 8 of the method maps the winning evidence to a probability with two numbers:

    p = 1 / (1 + exp(−(a + b · z))),    z = sign(r*) · log(1 + |r*|)

## The release numbers

| constant | value | provenance |
|---|---|---|
| intercept `a` | −4.590 | logistic regression on the reviewed "sure" days of all eight Beta stations (2,288 days, 470 with a wrong sign), reference run of 22 September 2026 |
| slope `b` | 2.162 | same fit |
| floor `φ` | 8.03 × 10⁻⁹ MW | smallest non-zero overnight demand step over the eight Beta stations |

With these numbers a day is corrected automatically (`p ≥ 0.7`) once its evidence reaches
about 11.4, and kept automatically (`p ≤ 0.3`) below about 4.6. The held-out fits of the
evaluation (one per station, each fitted without that station) range from −5.19 to −4.27
for the intercept and 2.06 to 2.35 for the slope, so the release pair is not sensitive to
which stations were seen.

## Why the intercept belongs to the population

The intercept says how much evidence a population's ordinary days produce. On Beta, real
substations with real errors, ordinary sunny days can produce evidence of five to ten
without any error (demand rising with solar, cloudy-day solar estimates), so the rule waits
for about eleven. On Alpha, correctly signed meters with the wrong sign imitated by taking
the absolute value, ordinary days produce almost none, so a rule fitted on Alpha corrects at
about 0.3 and, applied to Beta, flips 214 clean days. The Beta-fitted rule applied to Alpha
stays precise (energy precision 0.935) and only skips the tiny planted errors no reviewer
would call. The release calibration is therefore the conservative one.

## Why the slope carries over

Freezing the slope from one population and refitting only the intercept on the other
reproduces that population's own held-out result in both directions (Beta energy precision
0.938 against 0.941 with its own fit; Alpha 0.892 against 0.889). The slope is the shape of
the evidence-to-probability curve; the intercept is its position.

## Refitting for a new population

About 27 randomly chosen reviewed days are enough: in the learning-curve study the
intercept fitted on 27 random days met the 0.90 energy-precision gate in 95% of draws; nine
days did not. Fitting both numbers on so few days is unstable (a handful of error days
cannot pin the slope), so the slope stays.

```python
from pynrpf import RELEASE_CALIBRATION, fit_calibration, run

scored = run(frame)                                    # release calibration, to get the evidence
reviewed = scored.site_days.merge(labels, on=["site", "date"])   # label: 1 wrong sign, 0 not
pair = fit_calibration(reviewed["evidence"], reviewed["label"],
                       slope=RELEASE_CALIBRATION.slope, provenance="network X, 27 reviewed days, 2027")
result = run(frame, calibration=pair)
```

Until a population has its own intercept, the release pair is the safe default: it demands
more evidence, never less.

## What the numbers are not

They are not training in the machine-learning sense: no features, no model file, two
numbers a reader can check. They do not change which window wins or how days rank; the
sigmoid is monotone. `p` is the probability that the day carries a wrong sign, not that the
window's edges are exactly right.

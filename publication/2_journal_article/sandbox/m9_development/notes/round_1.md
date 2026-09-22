# Round 1 — the noise scale

**Failure class (priority 1, missed obvious cases).** Baseline `phase2_baseline`: 128 of 470 Beta `sure` RPF days not corrected (27%), 1,260 of 9,896 MWh. By cause: calibration 100 days / 1,012 MWh (positive evidence, p below c); input 24 / 207 MWh (missing slots, 21 of them at beta_D); window 4 / 41 MWh.

**Where.** beta_A recall 0.31 (16 of 26 uncertain), beta_E 0.00 (21 days), beta_H 0.00 (4), beta_D 0.51 (input), beta_F 0.81, beta_G 0.82, beta_B 0.90.

**Why (from the site-day figures in `weak_sites/phase2_baseline/`).** beta_A's missed days are textbook RPF — net load pinned at zero for five hours under 20 MW of solar — scoring r ≈ 250–380 against an applied threshold of ~394. beta_B's smaller humps score ~7,000. The score is gain / (σ_d²·L) and σ_d is the median overnight step: 0.33–0.45 MW at beta_A, 0.07 at beta_B. The ratio of σ² is 22×; the ratio of scores is 23×. **The overnight step is being used as the yardstick for how well a straight line should fit smooth daytime demand, and on stations with jagged nights it is wrong by an order of magnitude.** beta_E is the same mechanism at smaller amplitude (σ 0.28, hump ≈ 2 MW on a 10 MW base). Not a calibration-transfer problem: the ordering across stations is wrong before calibration sees it.

**Remedy tested.** Scale definition, two label-free variants: (a) median absolute step over the full day; (b) station-level median of (a).

**Result.**

| | pooled Beta IoU | sure recall | beta_A recall | beta_B recall | beta_F / beta_G energy precision |
|---|---|---|---|---|---|
| baseline | 0.804 | 0.736 | 0.31 | 0.90 | 0.957 / 0.899 |
| (a) full-day | 0.836 | 0.768 | 0.89 | **0.78** | 0.957 / 0.873 |
| (b) station full-day | 0.846 | 0.785 | 0.89 | **0.82** | 0.951 / 0.876 |

Alpha unchanged to four decimals in both.

**Decision: REJECT both.** beta_B loses 10–16 obvious days. beta_B is rough by day and smooth by night — the mirror image of beta_A — so any single external σ mis-serves one of them. The fragile element is the external noise scale itself, not where it is measured.

**What this licenses for round 2.** Remove σ_d. Comparing two nested fits with unknown variance is a likelihood-ratio with the variance profiled out: LLR = (L/2)·log(RSS_u/RSS_c). The corrected reconstruction's own residual is the yardstick, so no station-level or day-level scale enters. Mechanism: standard; necessity: the diagnosed failure; generality: dimensionless by construction and one fewer fixed setting.

**Rule clarifications recorded before round 2 (not in response to round-2 results).**
1. Alpha baseline energy precision is 0.887, below the 0.90 gate, because of the label-contiguity ceiling documented in the recommendation. The gate is therefore applied to Beta; Alpha is held to non-regression (rule 5).
2. "No station loses" is applied at sampling-noise tolerance: one `sure` day of recall (1 / n_sure_days for that station) and 0.01 of energy precision. A zero-tolerance rule rejects 0.006 differences that no reviewer would call a loss.

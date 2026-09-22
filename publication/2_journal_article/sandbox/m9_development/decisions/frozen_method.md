# Frozen M9 method — 2026-09-16

Frozen before the Phase 5 run. Code and data hashes are in `runs/phase5_final/config.json`.

## Definition

For one site-day of 96 fifteen-minute slots with recorded net load y (MW) and estimated solar s (MW):

1. **Counterfactuals.** U0 = s + y everywhere. For a window W = [a, b], U^W = s − y inside W and s + y outside.
2. **Candidates.** `NO_CORRECTION`, plus every contiguous window with 24 ≤ a ≤ b ≤ 71 (06:00 inclusive to 18:00 exclusive): 1,176 windows.
3. **Admissibility.** A window is admissible iff y and s are finite on every interior slot and a finite reading exists somewhere before a and somewhere after b within the day. The day abstains (`UNCERTAIN`) only if no window is admissible.
4. **Bridge.** The straight line g_W joining U0 at the nearest finite reading before a to U0 at the nearest finite reading after b. With no missing data these are slots a−1 and b+1.
5. **Evidence.** RSS_u = Σ_{t∈W} (U0_t − g_W(t))², RSS_c = Σ_{t∈W} (U^W_t − g_W(t))². With λ = L·floor²,
   r(W) = (L/2) · log((RSS_u + λ) / (RSS_c + λ)),
   the profile likelihood ratio of the two fits with variance unknown. `NO_CORRECTION` scores 0 and is ranked jointly. Ties: null beats window; shorter beats longer; earlier beats later.
6. **Runner-up.** Best admissible window not overlapping the best. Margins are reported and are not confidence.
7. **Calibration.** z = sign(r)·log(1 + |r|) of the best window's score; p = logistic(α + β·z), α and β fitted on training stations of the same cohort with `sure` (Beta) or all (Alpha) days; β > 0 required.
8. **Decision.** One public control c = 0.7: `AUTO_CORRECT` if p ≥ c; `AUTO_KEEP` if p ≤ 1 − c; `UNCERTAIN` otherwise. Only `AUTO_CORRECT` applies the window; the corrected series negates y inside it.

## Counts

| | |
|---|---|
| Plausibility components | 1 (bridge misfit ratio) |
| Fitted scoring parameters | 0 |
| Fixed physical settings | 3: scan range 24–71; RSS floor = smallest non-zero overnight step in the cohort's frozen data; overnight slots 0–23 used only for that floor |
| Calibration parameters | 2 per fold (α, β) |
| Public controls | 1 (c) |

Relative to the 15 September recommendation: the external noise scale σ_d is removed (round 2), missing readings disqualify windows rather than days (round 3), and anchors are the nearest finite readings (round 4). Nothing was added.

## Not in this method, and why

- Any second component: tested (solar consistency, joint window choice) and it degrades every metric on both cohorts.
- Curvature, total variation, absolute-deviation bridge: screened; weaker than the squared bridge on recall at equal or lower precision.
- A minimum window duration: short windows carry noise-level energy; c already removes them.
- A second correction model for clipped-to-zero flanks (beta_D): a different error mode; documented as a limitation, decision for Samhan.

## Known limitations

1. Cloudy days on which the solar estimate is jagged: both reconstructions inherit the error and the evidence is modest (beta_B, seven of eleven land `UNCERTAIN`).
2. Meters that clip at zero on the flanks of an RPF window (beta_D): the one-window sign-flip cannot explain the flank.
3. Alpha energy precision is capped near 0.89 by non-contiguous synthetic labels; this is a label property, not a method property.
4. Labelled spans containing an interior positive spike cannot be corrected by any single window.

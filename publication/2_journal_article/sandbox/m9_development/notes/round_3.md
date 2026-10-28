# Round 3 — missing readings disqualify windows, not days

**Failure class (priority 1, input).** 24 Beta `sure` RPF days abstained because a slot in 23–72 was missing; 21 of them at beta_D (47% of its obvious days, 171 MWh). The figures show the missing block sits hours before the RPF window, which is intact.

**Remedy.** A window is admissible iff its interior and both anchors are finite; the day abstains only when no window is admissible. Parameter-free; it can only add scorable days. Phase 0's "abstain only on missing values" is kept, applied to windows rather than days. Tested on both bases so the round-2 escalation is fully informed.

**Result.**

| base | change in Beta sure recall | beta_D recall | beta_A | other station deltas | rule check |
|---|---|---|---|---|---|
| baseline statistic | +0.014 | 0.51 → 0.62 | 0.31 → 0.38 | beta_B −0.016 (2 of 125 days) | REJECT on beta_B by one day beyond tolerance |
| full LLR (r2b) | +0.013 | 0.47 → 0.51 | unchanged | beta_F +0.018, beta_B +0.008, beta_G energy precision −0.010 | REJECT on beta_G at exactly the 0.01 tolerance |

Alpha within tolerance on both. Neither failure is caused by masking itself — masking cannot remove a scorable day — but by the recalibration ripple when newly scorable days enter the training folds and shift a threshold past one or two near-threshold days elsewhere.

**Decision.** Masking is adopted into the method definition on mechanism grounds: it removes a rule that was discarding intact evidence, adds nothing tunable, and the two rule failures are single-day ripples at the tolerance boundary. Recorded as an adoption under judgement, not under the mechanical rule, for Samhan to confirm.

**What it did not fix.** beta_D reaches only 0.51–0.62 although its 21 abstained days are now scorable. The round-1 figures show flat *exact-zero* net-load blocks immediately before beta_D's windows. Round 4 checks whether exact zeros are dropout artefacts rather than readings.

**Combined position after three rounds** — full LLR + window masking versus baseline, leave-one-station-out, c = 0.7:

| | Beta IoU | Beta energy precision | Beta sure recall | Beta uncertain | Beta day precision | Alpha IoU / precision |
|---|---|---|---|---|---|---|
| baseline | 0.804 | 0.922 | 0.736 | 0.194 | 0.906 | 0.852 / 0.887 |
| LLR + masking | **0.867** | 0.925 | **0.868** | 0.068 | **0.956** | 0.864 / 0.881 |

# Round 4 — bridge anchors are the nearest finite readings

**Failure class (priority 1, missed obvious cases at beta_D).** After round 3, 22 of beta_D's 45 obvious days were still not corrected although all were now scorable. Most were localised nearly correctly with weak evidence (LLR 2–12 against a threshold of ~13); one (2024-09-06) chose a single-slot fragment at 14:30 against a true window 11:00–13:45, with only 451 of 1,176 windows admissible.

**Why.** At beta_D the missing block ends exactly where the RPF starts, so under the round-3 rule the true window's left anchor (slot a−1) is missing and the true window is inadmissible. The scorer settles for whatever fragment is admissible.

**Remedy.** The bridge needs a reference level on each side; the nearest available finite reading is the best label-free estimate of it. A window is admissible iff its interior is finite and a finite reading exists before it and after it within the day; the bridge runs between those two readings. With no missing data this is exactly the adjacent-anchor rule, so nothing changes elsewhere. No parameter. The total-variation variant keeps adjacent anchors because its edge terms need them.

**Result, versus round 3 (leave-one-station-out, c = 0.7).**

| station | sure recall before → after | energy precision before → after |
|---|---|---|
| beta_D | **0.511 → 0.756** | 0.963 → 0.973 |
| beta_A | 0.885 → 0.923 | 0.975 → 0.975 |
| beta_B | 0.840 → 0.848 | 0.933 → 0.932 |
| others | unchanged | unchanged |

Pooled Beta: sure recall 0.868 → 0.891, Energy IoU 0.867 → 0.884, day F1 0.910 → 0.925, uncertain rate 0.068 → 0.062. Alpha within tolerance.

**Decision: ADOPT.** Passes every rule mechanically — the first remedy to do so.

**What remains at beta_D, and why it is out of scope for this method.** The still-missed days show the recorded net load flat at ≈0 for an hour or more *after* the labelled window while solar is still 5–7 MW: a meter clipping at zero, not a sign flip. The reviewer labelled only the mirrored middle. Under a one-window sign-flip counterfactual the clipped flank cannot be explained — corrected demand steps from ≈4.5 inside the window to ≈7 outside — so RSS_c is inflated and the evidence stays modest. This is a second error mode. It is documented with figures in `weak_sites/r3b_beta_D/` and left as a known limitation; representing it would need a second correction model, which is a methodology decision for Samhan and not a round-5 remedy.

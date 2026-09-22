# Round 5 — false corrections and the root-length exponent

**Context.** Samhan adopted round 2, confirmed round 3, and asked for further rounds aimed at the headline metrics (pooled Beta `sure` Energy IoU, energy precision, site-day F1 and precision), with per-station results reported rather than gating. Rounds beyond the plan's cap of five are at Samhan's instruction.

**Priority-2 diagnosis — the 19 false corrections** (`weak_sites/round5_false_corrections/`). All at beta_F (8 days, 39 MWh) and beta_G (11 days, 228 MWh); long windows (median 21–26 slots); confident (p 0.72–0.95); evidence 12–26 against thresholds of ≈12, overlapping the lowest decile of true positives (q10 ≈ 20). The figures show days on which net load dips to 1–2 MW at midday but never approaches zero (minimum inside the window 0.76 MW versus 0.12 on true positives; edge values 1.0 versus 0.4) while demand rises with solar — cooling load on sunny days. A reflected trace must pass through zero; these do not. The counterfactual treats a 3 MW step at the window edge as ordinary misfit, no less plausible than a smooth midday hump, so it cannot tell the two apart. Five parameter-free zero-crossing vetoes were tested; every one passes false corrections at least as often as true ones. **No remedy without a tuned tolerance; closed and documented.** Removing these 19 days would raise Beta IoU from 0.881 to 0.904.

**Where the Energy IoU is actually lost** (Beta `sure`, 1,267 MWh between the current result and a perfect one): over-proposal on correctly found days 38%, missed days 36%, false days 21%, under-coverage 5%. **All of the over-proposal is edge extension beyond the labelled span** — median one slot each side, on 76% of found days (beta_D 8 slots on average, from its clipped flanks). Mechanism: the full LLR multiplies the log-ratio by L, so an edge slot whose flip barely changes the misfit still raises the evidence.

**Remedy tested.** The middle exponent, r = (√L / 2)·log(RSS_u/RSS_c): evidence for a sustained effect growing like the square root of its length, which penalises uninformative edge slots.

**Result.** Beta Energy IoU 0.881 → 0.865, energy precision 0.926 → 0.950, sure recall 0.896 → 0.855 (beta_B −13 days, beta_G −5, beta_E −1). Alpha IoU +0.015, precision +0.027. It shortens windows as intended but also loses detections, because the exponent governs both.

**Decision: REJECT.** The objective is Beta Energy IoU and it fell.

**What this licenses.** Detection and window extent are governed by the same exponent, and the two want different things. A rule that shapes only the edges after detection — trim an edge slot when flipping it does not itself reduce the misfit against the bridge — would leave detection at p = 0 untouched. Round 6 checks first whether the extension slots actually have non-positive per-slot gain; if they do not, the rule cannot fire and the idea is dead.

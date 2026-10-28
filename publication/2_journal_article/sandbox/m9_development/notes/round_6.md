# Round 6 — edge trim: probed, not run

**Target.** The 38% of Beta Energy IoU loss that is edge extension beyond the labelled span (round 5 diagnosis): 1,138 extension slots across 421 found days, 482 MWh.

**Probe before building.** Per-slot gain from flipping, (U0 − g)² − (U^W − g)², at the extension slots: median −0.099 MW², 59% non-positive, |y| median 0.48 MW. At labelled interior slots: median +7.8 MW², 6% non-positive. So the extension slots are ones whose flip does essentially nothing; the full LLR includes them because adding a zero-gain slot to a window with ratio ≈ 5 still raises (L/2)·log(RSS_u/RSS_c) by ≈ 0.3. This is inherent to a profile likelihood ratio evaluated on window-length data: windows of different length are compared on different data, and more data is weakly rewarded.

**Why the obvious rule fails.** A greedy trim — drop an edge slot while its per-slot gain is non-positive, re-anchor, repeat — cuts 5,130 true slots on 339 days in simulation. Re-anchoring moves the bridge anchor onto the slot just trimmed, which the hypothesis now says is an ordinary reading; when that slot is really part of the hump, the line is wrong and the trim cascades inward. A fixed-line variant (trim without re-anchoring) would recover roughly half the extension, about 0.02 IoU, but adds a two-stage rule whose second stage contradicts its first stage's anchors.

**The principled fix is a different method.** Scoring every window on one fixed span with a full-day demand model removes the length dependence entirely, but replaces the local bridge with a global demand model — the smooth-demand approach found weak on 15 September because a solar-shaped hump is smooth. Not a round; a separately versioned study if wanted.

**Decision: not run.** Frozen method stands at `phase5_final`. Remaining Beta loss, with its cause and status:

| loss | MWh | share | cause | status |
|---|---|---|---|---|
| edge extension on found days | 482 | 38% | length bias of the profile LLR | understood; clean fix is a different method |
| missed days | 459 | 36% | cloudy days with jagged solar estimate (beta_B); clipped-zero flanks (beta_D); no evidence (beta_H) | input-limited / second error mode / documented |
| false days | 267 | 21% | demand rising with solar on sunny days, net load never near zero | no parameter-free veto separates them |
| under-coverage | 57 | 5% | — | negligible |

# Round 2 — remove the external noise scale

**Failure class.** Same as round 1: obvious days with compressed evidence on stations whose overnight roughness misrepresents their daytime smoothness.

**Remedy.** Replace gain / (σ_d²·L) with the profile likelihood ratio for two nested fits with unknown variance, LLR = (L/2)·log(RSS_u/RSS_c), each RSS floored at L·floor². The corrected reconstruction's own misfit is the yardstick. No external scale enters; one fixed setting removed. Tested in per-slot form (p = 1) and full form (p = 0).

**Result (leave-one-station-out, c = 0.7).**

| | pooled Beta IoU | Beta energy precision | sure recall | uncertain rate | day precision | Alpha IoU / precision |
|---|---|---|---|---|---|---|
| baseline | 0.804 | 0.922 | 0.736 | 0.194 | 0.906 | 0.852 / 0.887 |
| (a) LLR per slot | 0.430 | 0.972 | 0.492 | 0.445 | 0.760 | 0.859 / 0.937 |
| (b) full LLR | **0.864** | 0.928 | **0.855** | 0.104 | **0.959** | 0.864 / 0.881 |

(a) is rejected outright: the per-slot log-ratio does not accumulate a sustained effect and beta_B collapses to 0.19 recall.

(b) per station, sure-day recall: beta_A 0.31→0.89, beta_E 0.00→1.00 (21 of 21), beta_F 0.81→0.93, beta_G 0.82→0.93, beta_H 0.00→0.25; **beta_B 0.90→0.83 (−9 days), beta_D 0.51→0.47 (−2 days)**; beta_F energy precision −0.015. Alpha within tolerance. Net: 70 obvious days gained, 14 lost.

**Why the 11 beta_B days are lost** (`weak_sites/r2b_lost_days/`). Seven of eleven land `UNCERTAIN` at 0.52–0.99 of the threshold, not `AUTO_KEEP`. They are cloudy days on which the solar *estimate* is jagged; that error enters both reconstructions, so the corrected demand is as rough as the uncorrected one and the ratio is modest. One is a clean day (2024-05-29) with a 3 MW step at the window edges under both hypotheses. One labelled span contains an interior positive spike, so no single window is plausible — a label/requirement collision of the same kind as Alpha's gaps. The baseline corrected these only because beta_B's near-zero overnight σ inflated any gain; it was right for the wrong reason, and that same inflation is what failed beta_A and beta_E.

**Decision under the rules as written: REJECT (b)** — rule 2a fails at beta_B and beta_D.

**Escalation for Samhan.** This is the case the rules did not anticipate: a change that removes a fragile setting, is more principled, gains 70 human-obvious days and loses 14 — most of them cloudy days where the limiting factor is the solar input, not the method. Adopting (b) is a methodology decision and yours; I recommend it, with the beta_B cloudy-day limitation documented as a known failure mode tied to solar-estimate quality. Round 3 is run on both bases so the choice is fully informed either way.

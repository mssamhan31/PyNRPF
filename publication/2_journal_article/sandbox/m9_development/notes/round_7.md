# Round 7 — window edges must be local minima of recorded net load

**Origin.** Samhan's proposal after the 2026-09-16 freeze, from M7 experience: a sign-error window's borders should sit at local minima (plateaus included) of the recorded net load, within one slot.

**Physics.** A sign flip reflects the true trace about zero, so at each true edge the recorded net load reaches a cusp minimum at the crossing. On Beta, reviewer-labelled windows have both edges at a strict local minimum 84% of the time and within one slot 95%; on Alpha 29% and 94%, because the synthetic generator labels the first negative sample and the cusp is often one slot earlier. The constraint is applied to the candidate set, not the score: zero parameters beyond the one-slot tolerance option.

**Variants run** (all on the frozen method: full LLR, window masking, nearest-finite anchors, c = 0.7):

| variant | Beta IoU | Beta energy precision | Beta sure recall | Alpha IoU / precision / recall |
|---|---|---|---|---|
| frozen (`phase5_final`) | 0.881 | 0.926 | 0.896 | 0.864 / 0.881 / 0.917 |
| E1 strict minima | 0.879 | 0.942 | 0.857 | 0.878 / 0.914 / 0.800 |
| E2 within one slot | 0.871 | 0.932 | 0.868 | 0.867 / 0.885 / 0.915 |
| E3 within one slot + gap exemption | 0.883 | 0.931 | 0.894 | 0.867 / 0.885 / 0.915 |
| **E4 strict minima + gap exemption** | **0.894** | **0.942** | 0.887 | 0.878 / 0.914 / **0.800** |

The gap exemption: an edge whose outside neighbour is a missing reading is exempt from the test, because a cusp cannot be observed against a gap. Without it beta_D collapses (0.76 → 0.51): 42% of its labelled edges abut a missing block.

**Edge quality on found Beta days** (frozen → E4): exact window 13% → 48%; days with any extension 76% → 46%; over-proposed energy 484 → 338 MWh. The one-slot tolerance (E2, E3) does almost nothing to edges because it lets the extension slot back in; the cusp itself is what carries the information.

**Per station, E4 vs frozen** (sure-day recall): beta_D 0.76 → 0.80; beta_F 0.95 → 0.93 (−3 days); beta_G 0.93 → 0.89 (−3 days); others unchanged. Energy precision up at A, B, F, G, H; down at D (−0.04) and E (−0.05).

**Alpha.** IoU and precision rise; day recall falls 0.917 → 0.800 with 18% of RPF days `UNCERTAIN`. The 410 days no longer corrected are noise-floor events: median reference 0.83 MWh (Alpha median 5.14), 2.2% of Alpha reference energy, median true span 7 slots, evidence r ≈ 3 even under the frozen method; with the constraint no cusp-edged window exists near them and the null wins. 55% have non-contiguous labels.

**Assessment against the rules.** Mechanism: named and verified (84% of reviewer edges are cusps). Necessity: it removes the edge-extension failure class that was 38% of the remaining Beta loss, and the ablation (E2 vs E4) shows the class returns without the strict form. Generality: helps five Beta stations, costs beta_F and beta_G three obvious days each, and costs Alpha day recall on events below the noise floor. Headline Beta Energy IoU +0.013, precision +0.017, sure recall −0.009.

**Decision: for Samhan.** This is a candidate-set constraint from domain knowledge with a measurable mechanism; it improves the locked main metric on the human-labelled cohort and makes the proposed windows markedly more accurate, at the price of six Beta days and Alpha's smallest events. Recommendation: adopt as method revision 2, with the Alpha noise-floor behaviour documented; or hold the freeze if Alpha day recall is a claim you need to keep.

## Addendum — asymmetric tolerance (E5), at Samhan's suggestion

Start may sit one slot after a cusp, end one slot before; gap-adjacent edges exempt. The extension slot lies on the descending limb just outside the cusp, so a symmetric tolerance re-admits it while an inward tolerance does not; the inward tolerance still absorbs the sampling jitter that made strict minima disqualify one-slot-offset labels.

| variant | Beta IoU | Beta energy precision | Beta sure recall | Alpha IoU / precision / recall | exact window | over MWh |
|---|---|---|---|---|---|---|
| frozen | 0.881 | 0.926 | 0.896 | 0.864 / 0.881 / 0.917 | 13% | 484 |
| E4 strict + gap | 0.894 | 0.942 | 0.887 | 0.878 / 0.914 / 0.800 | 48% | 338 |
| **E5 inward + gap** | **0.895** | **0.941** | **0.896** | 0.869 / 0.889 / 0.905 | 35% | 347 |

E5 keeps every day the frozen method found (beta_B +2, beta_F +1, beta_G −3), keeps E4's headline gain, and reduces the Alpha recall cost from 12 points to 1.2. Recommended for adoption as revision 2; decision brief at `m9_dev_Claude/M9_edge_rule_decision_brief.md`.

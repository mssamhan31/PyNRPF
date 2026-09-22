---
title: "M9 — as-built method"
status: "locked — revision 2, 2026-09-16"
version: "revision 2 (edge rule); revision 1 was commit 126f989. Hashes in runs/phase5_final_rev2/config.json"
scope: "What the method is, with every equation; how it was evaluated; what it achieves; where it fails. Not paper prose."
---

# M9 — as-built method

## 0. What changed in revision 2

One addition to the candidate set, proposed by Samhan from M7 experience and tested in round 7: **a window's edges must sit at local minima of the recorded net load**, with an inward one-slot tolerance and an exemption for edges that abut a missing reading (§4). Nothing in the scoring, calibration or decision changed. Effect on Beta `sure`: Energy IoU 0.881 → **0.895**, energy precision 0.926 → **0.941**, obvious-day recall unchanged at 0.896; windows now match the reviewer's edges exactly on 35% of corrected days against 13%. Alpha: IoU and precision up, day recall −1.2 points on events below the noise floor. Full derivation and the four rejected variants are in `M9_edge_rule_decision_brief.md`.

## 1. What M9 does, in one paragraph

A distribution substation meter records net load $y_t$ (MW, positive = import). When rooftop solar pushes real power back into the network, true net load is negative; some meters record the magnitude but lose the sign, so a negative $y_t$ is stored as $+|y_t|$. M9 takes one site-day of 96 fifteen-minute readings together with a solar estimate $s_t$ and asks, for every admissible contiguous daytime window, a single question: *if the readings inside this window had their sign flipped back, would the implied underlying demand look more like ordinary demand than it does now?* The window for which the answer is strongest is proposed; if no window beats "leave it alone", nothing is proposed. The strength of the answer is converted to a probability, and one public confidence setting turns that into `AUTO_CORRECT`, `AUTO_KEEP` or `UNCERTAIN`.

It has one plausibility component and no fitted scoring parameters. What is learned from data is two calibration coefficients that map evidence to probability, nothing else.

## 2. Definitions

| symbol | meaning | unit |
|---|---|---|
| $t = 0,\dots,95$ | fifteen-minute slot index; slot 24 is 06:00, slot 72 is 18:00 | — |
| $y_t$ | recorded net load | MW |
| $s_t$ | reliable solar generation estimate | MW |
| $\mathcal{T} = \{24,\dots,71\}$ | eligible slots, 06:00 inclusive to 18:00 exclusive | — |
| $W = [a,b]$ | a candidate window, $24 \le a \le b \le 71$, length $L = b-a+1$ | — |
| $\varnothing$ | the `NO_CORRECTION` candidate | — |
| $U^{0}_t$ | underlying demand if the recorded sign is kept | MW |
| $U^{W}_t$ | underlying demand if the sign is corrected inside $W$ | MW |
| $c$ | the public confidence control, fixed at 0.7 | — |

The candidate set is $\mathcal{C} = \{\varnothing\} \cup \{[a,b] : 24 \le a \le b \le 71\}$: 1,176 windows plus the null. Every window of every length is considered; the admissibility rules in §4 remove windows on physical grounds, never on solar position or duration.

## 3. The two counterfactual reconstructions

Underlying demand is what the customers behind the substation actually consumed: net load plus local generation. Under the hypothesis that the recorded sign is right,

$$U^{0}_t = s_t + y_t \qquad \text{for all } t .$$

Under the hypothesis that the sign is wrong exactly inside $W$,

$$U^{W}_t = \begin{cases} s_t - y_t, & t \in W \\ s_t + y_t, & t \notin W \end{cases}$$

Both use the same $s_t$. On a true sign-error day $U^{0}$ carries a solar-shaped hump of height $2|y_t|$ across the window — the mirror image of the reverse flow — while $U^{W}$ recovers the smooth demand underneath. On a normal day a false window does the opposite: it carves a solar-shaped hole into $U^{W}$. The method's whole job is to tell which of these two pictures is the more ordinary demand curve.

The proposed corrected net load, when a correction is applied, is

$$\tilde{y}_t = \begin{cases} -\,y_t, & t \in W \\ y_t, & t \notin W \end{cases}$$

Raw readings are never overwritten; $\tilde{y}$ is a separate series.

## 4. Admissibility, anchors and the edge rule

A reading is *finite* when both $y_t$ and $s_t$ are present. For a window $W=[a,b]$ define the nearest finite readings on each side,

$$\ell(W) = \max\{k < a : \text{slot } k \text{ finite}\}, \qquad r(W) = \min\{k > b : \text{slot } k \text{ finite}\}.$$

Define the local-minimum indicator on recorded net load, plateaus included and comparisons against a missing reading false:

$$m_t = \bigl[\,y_t \le y_{t-1}\,\bigr] \wedge \bigl[\,y_t \le y_{t+1}\,\bigr].$$

**$W$ is admissible iff**

1. every slot in $[a,b]$ is finite;
2. $\ell(W)$ and $r(W)$ exist;
3. *start*: $m_a \vee m_{a-1}$, or the reading at $a-1$ is missing; and
4. *end*: $m_b \vee m_{b+1}$, or the reading at $b+1$ is missing.

Conditions 1–2 are revision 1: a missing reading disqualifies only the windows that contain it, and the day is `UNCERTAIN` only if no window is admissible. Conditions 3–4 are revision 2.

*Why the edge rule.* A sign flip reflects the true trace about zero, so at each true edge the recorded net load comes down to a cusp — the crossing — and rises on the far side. On Beta, 84% of reviewer-labelled windows have both edges at a strict local minimum and 95% within one slot. The rule removes the one-slot **edge extension** that was 38% of revision 1's remaining Beta loss: the slot just outside the cusp sits on the descending limb and is not a minimum. The tolerance is **inward only** — the start may sit one slot *after* a cusp, the end one slot *before* — so the sampling jitter of the crossing is absorbed without re-admitting the outward slot; a symmetric tolerance was tested and does almost nothing. The gap exemption exists because a cusp cannot be observed against a missing reading; without it beta_D, where 42% of labelled edges abut a gap, collapses.

*Why nearest-finite anchors.* Two earlier rules were rejected on real data: abstaining the whole day on any missing slot discarded 21 of beta_D's 45 obvious days; requiring the adjacent slots $a-1$, $b+1$ to be finite made beta_D's true windows inadmissible because its missing block ends exactly where the reverse flow begins.

## 5. The plausibility criterion: a straight bridge

Ordinary demand over a few hours is close to a straight line. For an admissible window, draw the line through the reconstructed demand at its two anchors:

$$g_W(t) = U^{0}_{\ell} + \frac{t-\ell}{r-\ell}\,\bigl(U^{0}_{r} - U^{0}_{\ell}\bigr), \qquad t \in [a,b].$$

Both reconstructions share this line because they agree outside the window. Measure how far each sits from it, summed over the window:

$$\mathrm{RSS}_u(W) = \sum_{t=a}^{b}\bigl(U^{0}_t - g_W(t)\bigr)^2, \qquad \mathrm{RSS}_c(W) = \sum_{t=a}^{b}\bigl(U^{W}_t - g_W(t)\bigr)^2 .$$

Units MW². On a true window $\mathrm{RSS}_u$ is dominated by the hump and $\mathrm{RSS}_c$ is small; on a false window $\mathrm{RSS}_c$ carries the hole and the two edge steps. The bridge integrates edge continuity and within-window shape into one number, which is why it was the strongest single statistic when the true window was given (AUC 0.997 on both cohorts) and why no second component was needed — every second component tested made the result worse.

## 6. The evidence score: a profile likelihood ratio

Model demand inside the window as the line plus Gaussian noise of *unknown* variance. Under either hypothesis the maximised log-likelihood, after estimating the variance from that hypothesis's own residuals, is $-\tfrac{L}{2}\log(\mathrm{RSS}/L)$ plus a common constant. Their difference is the evidence for correction:

$$r(W) = \frac{L}{2}\,\log\!\left(\frac{\mathrm{RSS}_u(W) + \lambda_L}{\mathrm{RSS}_c(W) + \lambda_L}\right), \qquad \lambda_L = L\,\phi^2 ,$$

with $\phi$ a fixed resolution floor — the smallest non-zero absolute overnight step of $U^{0}$ in the cohort's frozen data — so a near-perfect fit on a short window cannot produce an unbounded ratio. The null scores $r(\varnothing) = 0$.

Read $r$ as *by what factor does correcting improve the straight-line fit, weighted by how many slots sustain that improvement*. A 20-slot window in which correction makes the line fit five times better scores $10\ln 5 \approx 16$; two times better scores $\approx 7$. The applied thresholds land around 12–15.

*Why this and not an external noise scale.* The 15 September recommendation used $(\mathrm{RSS}_u - \mathrm{RSS}_c)/(\sigma_d^2 L)$ with $\sigma_d$ the median overnight step of demand: an external guess at how rough ordinary demand is, taken at night. On stations with jagged nights and smooth days (beta_A, beta_E) it understated the evidence on textbook days by an order of magnitude; on the mirror-image station (beta_B) it overstated it. No place to measure $\sigma_d$ served both. The profile likelihood ratio uses the corrected fit's own residual as the yardstick, so no external scale enters and one threshold transfers across stations. It is the standard statistic for comparing two fits with unknown noise — the same object as an F-test. The exponent on $L$ was tested at 1 (adopted), ½ and 0; ½ trades recall for precision and loses Energy IoU, 0 collapses detection.

## 7. Ranking, best window, runner-up

$$W^\star = \arg\max_{W \in \mathcal{C}} r(W)$$

with ties resolved: the null beats any window; the shorter beats the longer; the earlier start beats the later. If $r(W^\star) \le 0$ the null wins and the best window is retained for inspection. Otherwise the runner-up is the best admissible window not overlapping $W^\star$ — a genuinely different explanation of the day. Two margins are stored, against the null and against the runner-up; **neither is confidence.**

## 8. Calibration and decision

Let $R_d = r(W^\star)$ for site-day $d$ (the best window's score even when the null wins). Apply a variance-stabilising transform and a two-coefficient logistic:

$$z_d = \operatorname{sign}(R_d)\,\log\bigl(1+|R_d|\bigr), \qquad p_d = \frac{1}{1+e^{-(\alpha + \beta z_d)}}, \quad \beta > 0 .$$

$\alpha, \beta$ are fitted by maximum likelihood on training stations of the same cohort — Beta `sure` days, or all Alpha days. The transform matters: on raw $R_d$, whose 99th percentile on Beta is $10^5$, the logistic saturates and Beta collapses (ECE 0.18); on $z_d$ both cohorts calibrate to ECE ≤ 0.02.

With one public control $c \in (0.5,1)$, fixed at $0.7$:

$$\text{outcome}_d = \begin{cases} \texttt{AUTO\_CORRECT}, & p_d \ge c \\ \texttt{AUTO\_KEEP}, & p_d \le 1-c \\ \texttt{UNCERTAIN}, & \text{otherwise, or no admissible window} \end{cases}$$

Only `AUTO_CORRECT` applies $\tilde y$. At $p_d=0.5$ the outcome is `UNCERTAIN`, so a tie never forces a correction. Because $\alpha,\beta$ are fixed after fitting, $c$ maps to a raw threshold reported per fold and can be moved without retraining:

$$R^{*}(c) = \operatorname{sign}(z^{*})\,\bigl(e^{|z^{*}|}-1\bigr), \qquad z^{*} = \frac{\operatorname{logit}(c) - \alpha}{\beta}.$$

$p_d$ means one thing: *the probability that this site-day requires a correction, under the reviewer's labels and the training population.* It makes no claim that $a$ and $b$ are exactly right.

## 9. Outputs per site-day

Best candidate and score; runner-up and score; both margins; $z_d$, $p_d$, the applied $c$ and its raw thresholds; outcome; proposed window and interval flags; both reconstructions; proposed corrected net load $\tilde y$; proposed correction energy $\sum_{t\in W} 2\,y_t \cdot 0.25$ MWh; and a reason code (`window_wins`, `null_wins`, `uncertain_band`, `no_admissible_window`).

## 10. Complexity accounting

| | count | what |
|---|---|---|
| Plausibility components | 1 | straight-bridge misfit ratio |
| Fitted scoring parameters | 0 | — |
| Fixed physical settings | 4 | scan range 24–71; floor $\phi$; overnight slots 0–23 used only to compute $\phi$; edge rule (§4) with its one-slot inward tolerance = the sampling interval |
| Calibration parameters | 2 per fold | $\alpha, \beta$ |
| Public controls | 1 | $c$ |

Rejected with evidence: a second component (solar consistency, post-hoc and with joint window selection); curvature and total-variation smoothness; an absolute-deviation bridge; every external noise scale tried; the $L^{1/2}$ exponent; a minimum window duration; five parameter-free zero-crossing vetoes for false corrections; trimming edge slots after detection; strict and symmetric-tolerance forms of the edge rule.

## 11. Evaluation protocol as run

Leave-one-station-out within each cohort: for each held-out station, $\alpha,\beta$ are fitted on the other stations of that cohort and the held-out station is scored once. Nothing else is fitted. Alpha: 10 stations, 10,425 complete site-days, 3,381 RPF, controlled truth. Beta: 8 stations, 2,305 `sure` site-days, 470 RPF; `unsure` days scored and reported as sensitivity only. Energy metrics follow the evaluation PRD's provisional definitions, pooled by summing energies before dividing.

**This is a development-conditioned, locked station-held-out comparison, not untouched validation.** The scorer, statistic, admissibility and edge rules were chosen with all eighteen stations visible over five adopted rounds; only $\alpha,\beta$ are fitted out of station.

## 12. Results, revision 2

### Headline, $c = 0.7$ (revision 1 in brackets)

| | Alpha | Beta `sure` |
|---|---|---|
| Reference Energy IoU | 0.869 (0.864) | **0.895** (0.881) |
| Reference energy precision | 0.889 (0.881) | **0.941** (0.926) |
| Sure-day recall (human-obvious cases found) | 0.905 (0.917) | **0.896** (0.896) |
| Site-day precision | 0.977 (0.975) | 0.953 (0.957) |
| Site-day F1 | 0.940 (0.945) | 0.923 (0.925) |
| `AUTO_CORRECT` rate, all days | 0.300 | 0.192 |
| `UNCERTAIN` rate, all days | 0.026 | 0.043 |
| Calibration ECE | 0.022 | 0.009 |

Alpha energy precision sits below the 0.90 gate by label construction (42% of Alpha RPF days have non-contiguous synthetic labels; a perfect one-window localiser scores 0.904); by your decision the gate applies to Beta and Alpha is reported with that ceiling. On reviewer-`unsure` days the model is `UNCERTAIN` 18% of the time against 4% on `sure` days.

### Per station

Alpha:

| station | days | RPF | ref MWh | Energy IoU | energy precision | sure recall | day precision | day F1 | correct rate |
|---|---|---|---|---|---|---|---|---|---|
| alpha_A | 1039 | 0 | 0 | — | — | — | — | — | 0.000 |
| alpha_B | 1055 | 27 | 30 | 0.502 | 0.518 | 0.667 | 0.947 | 0.783 | 0.018 |
| alpha_C | 1055 | 627 | 3752 | 0.845 | 0.870 | 0.890 | 0.965 | 0.926 | 0.548 |
| alpha_D | 1035 | 0 | 0 | — | — | — | — | — | 0.000 |
| alpha_E | 1035 | 700 | 6327 | 0.895 | 0.912 | 0.920 | 0.989 | 0.953 | 0.629 |
| alpha_F | 1033 | 842 | 13357 | 0.887 | 0.902 | 0.931 | 0.992 | 0.961 | 0.765 |
| alpha_G | 1037 | 650 | 5829 | 0.863 | 0.889 | 0.920 | 0.966 | 0.942 | 0.597 |
| alpha_H | 1039 | 0 | 0 | — | — | — | — | — | 0.000 |
| alpha_I | 1042 | 148 | 1200 | 0.771 | 0.784 | 0.872 | 0.970 | 0.918 | 0.128 |
| alpha_J | 1055 | 387 | 1542 | 0.800 | 0.829 | 0.848 | 0.965 | 0.902 | 0.322 |

Beta `sure`:

| station | days | RPF | ref MWh | Energy IoU | energy precision | sure recall | day precision | day F1 | correct rate |
|---|---|---|---|---|---|---|---|---|---|
| beta_A | 337 | 26 | 755 | 0.956 | 0.988 | 0.923 | 1.000 | 0.960 | 0.071 |
| beta_B | 231 | 125 | 4769 | 0.889 | 0.948 | 0.864 | 1.000 | 0.927 | 0.468 |
| beta_C | 339 | 0 | 0 | — | — | — | — | — | 0.000 |
| beta_D | 188 | 45 | 475 | 0.902 | 0.966 | 0.756 | 0.971 | 0.850 | 0.186 |
| beta_E | 306 | 21 | 87 | 0.899 | 0.911 | 1.000 | 0.955 | 0.977 | 0.072 |
| beta_F | 286 | 164 | 1578 | 0.930 | 0.950 | 0.957 | 0.946 | 0.952 | 0.580 |
| beta_G | 283 | 85 | 2202 | 0.870 | 0.902 | 0.894 | 0.884 | 0.889 | 0.304 |
| beta_H | 335 | 4 | 30 | 0.173 | 0.818 | 0.250 | 1.000 | 0.400 | 0.003 |

The four zero-RPF stations correctly receive essentially no corrections. beta_H (four labelled days) and alpha_B (27 days at the noise floor) are not meaningful.

### Edge quality on corrected Beta `sure` days (revision 1 → 2)

Exact window match 13% → **35%**; extension beyond the labelled span 2.70 → **1.95** slots per day; over-proposed energy 484 → **347** MWh; under-covered 57 → 77 MWh.

### Against M7 and M8 — headline only, with two caveats

The evaluation PRD requires all three methods on identical folds; that run is scheduled for the next cycle. What exists is the journal notebook's earlier run of M7 and M8, **day-level only**, and for Beta a *transfer* — M8 trained on Alpha and applied to Beta — against an older label vintage (557 `sure` positives, now 470). Only the Alpha column is like-for-like.

| site-day level | Alpha precision / recall / F1 | Beta precision / recall / F1 |
|---|---|---|
| M7 (deterministic threshold rule) | 0.530 / 0.989 / 0.690 | 0.192 / 0.761 / 0.307 (transfer, old labels) |
| M8 (two-stage XGBoost) | 0.974 / 0.819 / 0.890 | 0.606 / 0.420 / 0.496 (transfer, old labels) |
| **M9 revision 2** | **0.977 / 0.905 / 0.940** | **0.953 / 0.896 / 0.923** (station-held-out, current labels) |

## 13. Twenty-seven site-day samples

Beta `sure` days from `runs/phase5_final_rev2`, so every panel is a real error or a real non-error a reviewer was confident about. Each panel shows recorded net load (black), the solar estimate (orange), underlying demand with the sign kept (grey dashed), underlying demand if corrected (blue), the window M9 proposed (blue shading if applied, purple if proposed but not applied), and the reviewer's labelled window (red). Titles give the outcome, $p$, $r$, reference and proposed MWh. Index: `figures/samples_index.csv`.

**True positives — 9.** Three largest corrections, three closest to the threshold, three at random.

![TP](figures/samples_TP.png)

**False negatives — 6.** Three `UNCERTAIN` and three `AUTO_KEEP`, largest reference energy first. Cloudy beta_B days (jagged solar estimate) and days whose net load never reaches zero.

![FN](figures/samples_FN.png)

**False positives — 6.** The largest applied corrections. Net load dips to 1–2 MW but never approaches zero while demand rises with solar.

![FP](figures/samples_FP.png)

**True negatives — 6.** Three closest to the threshold, three at random.

![TN](figures/samples_TN.png)

## 14. Known limitations

1. **Jagged solar estimates.** On cloudy days the estimate's error enters both reconstructions equally; the ratio is modest even when the hump is large. The reviewer's cue — the raw net-load trace mirroring the solar trace — does not need the estimate to be smooth; the counterfactual does.
2. **Meters that clip at zero.** beta_D's flanks read ≈0 after the labelled window while solar is 5–7 MW. A one-window sign flip cannot explain a clipped flank. Documented as a second error mode outside scope, by your decision.
3. **Demand that rises with solar.** Sunny days on which cooling load produces a midday hump and net load never reaches zero: confident false corrections at beta_F and beta_G. No parameter-free veto separates them from true corrections.
4. **Noise-floor events.** Very short, very small events with no clean cusp can leave no admissible window under the edge rule; this costs Alpha 1.2 points of day recall on events holding about 2% of its reference energy.
5. **Alpha's label contiguity** caps its energy precision near 0.90 for any one-window method.

## 15. Provenance

Implementation: `PyNRPF/publication/2_journal_article/m9_dev/m9_scorer.py` (reference and vectorised versions checked equal in `tests/`), `m9_metrics.py`, `m9_eval.py`; revision 1 remains reproducible with `--edges any`. Frozen records with hashes: `decisions/frozen_method.md` (rev 1), `decisions/frozen_method_rev2.md` (rev 2); runs `phase5_final` and `phase5_final_rev2`. Development trail: `notes/round_1.md` to `round_7.md`; diagnosis figures under `weak_sites/`. Twenty-two synthetic fixtures; `pytest` and `ruff` clean.

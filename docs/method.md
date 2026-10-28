# The method, M9

M9 is a counterfactual comparison, not a classifier. For one site-day of 96 fifteen-minute
readings it follows nine steps; each has its own module under `pynrpf.m9`, and reading them
in order is reading the method.

    readings (y, s) → two stories → candidate windows → bridge misfits → r(W) → W*, r* → z → p → decision

## Notation

| symbol | meaning |
|---|---|
| `t` | fifteen-minute slot of the day, 0 to 95 |
| `y_t` | recorded net load, MW; positive means import as stored |
| `s_t` | solar generation estimate, MW |
| `W = [a, b]` | a candidate window from slot `a` to slot `b` inclusive |
| `L = b − a + 1` | slots in the window |
| `U0_t`, `UW_t` | underlying demand if the sign is kept, or flipped inside `W` |
| `ℓ`, `r` | anchor slots: the nearest finite readings before `a` and after `b` |
| `g_W(t)` | the bridge, a straight line from `U0_ℓ` to `U0_r` |
| `RSS_u`, `RSS_c` | misfit of the kept and flipped stories against the bridge, MW² |
| `φ` | the evidence floor, MW |
| `r(W)`, `r*` | evidence for window `W`; the winner's evidence |
| `z`, `p`, `c` | compressed evidence; probability of a wrong sign; the control |

## Step 1, the two stories (`stories`)

Underlying demand is the solar generation plus the net load the meter recorded. If the sign
is right that is `U0_t = s_t + y_t` everywhere. If the sign is wrong inside `W`, the export
there was stored as an import, and demand inside `W` is `s_t − y_t`.

    Story A, keep the sign:     U0_t = s_t + y_t
    Story B, flip inside W:     UW_t = s_t − y_t  for t in W,  s_t + y_t elsewhere

On a real wrong-sign day Story A shows a solar-shaped hump (demand would have to rise and
fall with the sun); Story B gives an ordinary smooth day.

## Step 2, every window (`windows`)

M9 does not know where the error starts or ends, so it tries every contiguous window between
06:00 and 18:00: 48 slots, hence 48 × 49 / 2 = 1,176 start-end pairs, plus "no correction".
A window is admissible when every reading inside it is finite and a finite reading exists
somewhere before its start and after its end. Missing readings disqualify the windows that
touch them, not the day.

## Step 3, edges at kinks (`edges`)

A wrong sign reflects the true trace about zero, so where the error starts and ends the
recorded net load touches its reflection: a cusp, a local minimum. On the Beta dataset 84% of
reviewer-labelled edges sit exactly on such a minimum and 95% within one slot.

    m_t = [y_t ≤ y_{t−1}] and [y_t ≤ y_{t+1}]         (plateaus count)
    a start a is valid when m_a or m_{a−1};  an end b is valid when m_b or m_{b+1}

The one-slot tolerance points inward: outward tolerance would re-admit the over-wide windows
the rule exists to remove. An edge beside a missing reading is exempt.

## Step 4, the straight bridge (`bridge`)

    g_W(t) = U0_ℓ + (t − ℓ) / (r − ℓ) · (U0_r − U0_ℓ)

Both stories share the bridge, because they agree outside the window. The bridge does not
claim demand is linear; it asks which story looks like a smooth transition between the
surrounding demand levels and which like a hump.

## Step 5, the misfit (`misfit`)

    RSS_u(W) = Σ_{t∈W} (U0_t − g_W(t))²        RSS_c(W) = Σ_{t∈W} (UW_t − g_W(t))²

Smaller means closer to the bridge. On a genuine wrong-sign window `RSS_c ≪ RSS_u`.

## Step 6, the evidence (`evidence`)

    r(W) = (L / 2) · log( (RSS_u(W) + λ_L) / (RSS_c(W) + λ_L) ),      λ_L = L · φ²

`r > 0`: flipping fits better; `r = 0`: the stories tie; `r < 0`: keeping fits better.

Why a ratio: the same 100 MW² of improvement is nothing on a window whose misfit is 1,000
and everything on one whose misfit is 110; the ratio sees that, and it has no units, so no
external noise scale is needed. Why the logarithm: a two-fold improvement and a two-fold
deterioration become +0.69 and −0.69. Why `L/2`: treat the residuals as Gaussian noise with
unknown variance, fit the variance under each story, and the log-likelihood difference is
exactly `(L/2) log(RSS_u / RSS_c)`; evidence sustained over more slots counts for more. Why
the floor: two perfect fits must tie (`log 1 = 0`), and a floor on one side only would send
the ratio to infinity. `φ` is the smallest non-zero overnight demand step over the release
population; it only guards perfectly flat readings, and it is fixed (`RELEASE_PHI`) so a
day scores the same in any batch.

## Step 7, the winner (`winner`)

    W* = argmax_W r(W),   r* = r(W*),   with "no correction" at r = 0

Ties: no correction over a window, a shorter window over a longer one, an earlier over a
later. The best window that does not overlap the winner is kept as the runner-up. `r*` is
the best window's evidence even when no correction wins, so a weak negative day stays
distinguishable from a day with no candidate at all. A positive `r*` does not by itself
cause a correction; the next two steps decide.

## Step 8, from evidence to probability (`calibration`)

    z = sign(r*) · log(1 + |r*|)
    p = 1 / (1 + exp(−(a + b · z)))

The signed log compresses winning scores that span orders of magnitude, so a few
near-perfect days do not dominate the fit. `a` and `b` are two numbers fitted by logistic
regression on reviewed days; see [Calibration](calibration.md) for the release values and
why `b` carries over between populations while `a` does not.

## Step 9, the decision (`decision`)

    AUTO_CORRECT if p ≥ c,   AUTO_KEEP if p ≤ 1 − c,   UNCERTAIN otherwise,   c = 0.7

Only AUTO_CORRECT flips the sign, inside `W*`, into a separate corrected series. UNCERTAIN
leaves the day alone and lists it for review with the window and the runner-up. Three
different statements: `r(W) > 0` says flipping inside `W` fits better; `p > 0.5` says an
error is more likely than not; `p ≥ 0.7` says the confidence set for acting automatically
has been reached.

## One day end to end

Beta station B, 20 September 2024, from the reference run:

| step | value |
|---|---|
| admissible windows | 78 |
| best window `W*` | slots 33 to 63 (08:15 to 15:45); the reviewers' span was the same |
| `r*` | 55.9; runner-up −27.1 |
| `z` | 4.04 |
| `p` with the release calibration | logistic(−4.59 + 2.16 × 4.04) = 0.984 |
| outcome at `c = 0.7` | AUTO_CORRECT, 86 MWh flipped, 86 MWh in the reference |

## Moving parts

One physical test (the bridge), no fitted scoring parameters, four fixed settings (the
06:00 to 18:00 scan, the floor `φ`, the edge rule, the tie order), two calibration numbers,
one control.

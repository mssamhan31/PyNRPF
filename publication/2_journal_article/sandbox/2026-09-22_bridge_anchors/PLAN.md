# Plan — bridge anchors at the window edge (2026-09-22)

**Question.** The M9 bridge is drawn between the nearest finite readings outside a
candidate window. Where the reading adjacent to the window is missing, the anchor
moves further out, past the gap. At beta_D the missing block typically ends where the
reverse power flow (RPF) starts: 19 of its 45 `sure` labelled RPF days (38 of 76 with
`unsure` included) have a labelled span that abuts a gap on at least one side, and on
those days the evidence is often weak or the window misplaced. Does anchoring the
bridge on the window's own edge slots help there, and what does it cost elsewhere?

**Variants.** One new option, `--anchors`, on the frozen revision-2 method; everything
else fixed.

- `nearest` — the frozen behaviour, bit for bit. Both stories share one bridge between
  the nearest finite readings outside the window; the misfit covers all L window slots.
- `edge` — the bridge runs between the window's own end slots: story A through U0 at a
  and b, story B through S − y at a and b; the misfit covers the L − 2 interior slots.
  Windows shorter than three slots are inadmissible.
- `gap_edge` — nearest-finite anchors, except that a side whose adjacent reading (a − 1
  or b + 1) is missing anchors on the edge slot as in `edge`, and that slot leaves the
  misfit. Each side is decided on its own: with a gap on one side only, the bridge runs
  from the window's own edge on the gap side to the adjacent reading on the other,
  over L − 1 slots. On a day without gaps it is identical to `nearest`.

In every variant the L in the likelihood ratio is the number of residual slots, since
the profile likelihood counts residuals and a slot that sits on its own line carries
none; the floor λ = L φ² follows the same count. The admissibility conditions of
revision 2 (finite interior, a finite reading somewhere on each side, edges at local
minima with the gap exemption) are kept for all three, so the candidate sets differ
only by the minimum-length rule.

**Fixed settings.** `--variant sq --p 0 --sigma overnight --c 0.7 --stat llr
--missing mask_windows --edges inwardx` from `m9_dev/decisions/frozen_method_rev2.md`;
leave-one-station-out calibration as in `m9_eval.py`; data hashes as in
`m9_dev/runs/phase5_final_rev2/config.json`.

**What is compared.**

1. `nearest` against `m9_dev/runs/phase5_final_rev2`: it must reproduce the pooled Beta
   `sure` and Alpha metrics to four decimals and, row by row, the held-out predictions.
2. `edge` and `gap_edge` against `nearest` through `compare_runs.py` (Phase 0 rules 1,
   2 and 5) and per-station tables for both cohorts.
3. beta_D's gap-adjacent labelled RPF days, `sure` and `unsure`: per day and rule the
   best window, evidence r, held-out p, decision, exact match and slot IoU ≥ 0.8.
4. Days elsewhere whose outcome or window changed, per station.

**Success reading.** A variant is worth carrying forward to a methodology proposal only
if beta_D's gap-adjacent days gain evidence or window accuracy, no other station with
`sure` RPF days loses sure-day recall or energy precision beyond the round-1
tolerances, Alpha stays within 0.01 on Energy IoU and energy precision, and Beta energy
precision stays at or above 0.90. Anything short of that is a finding. This sandbox
makes no adoption decision and the default stays `nearest`.

**What could go wrong.** Anchoring on the edge removes edge continuity from the misfit:
under `edge`, story B's line always passes through its own edge values, so the steps a
false window creates at its edges no longer count against it, and false corrections
may rise on stations with a midday demand hump (beta_F, beta_G). `gap_edge` confines
the change to windows beside a gap, so its risk is confined to stations with gaps and
its benefit, if any, to the beta_D failure class. Neither variant changes a fitted
quantity; only the calibration coefficients are refitted per fold, as always.

# Frozen M9 method — revision 2, 2026-09-16

Supersedes `frozen_method.md` (revision 1, commit 126f989). One change; everything else identical. Code and data hashes in `runs/phase5_final_rev2/config.json`.

## What changed from revision 1

**Window edges must sit at local minima of the recorded net load, with an inward one-slot tolerance and a gap exemption.** Proposed by Samhan from M7 experience; tested in round 7.

Let $m_t$ be true when $y_t \le y_{t-1}$ and $y_t \le y_{t+1}$ (plateaus count; comparisons against missing readings are false). A window $[a,b]$ is admissible only if, in addition to the revision-1 conditions,

- **start**: $m_a$ or $m_{a-1}$ is true, or the reading at $a-1$ is missing; and
- **end**: $m_b$ or $m_{b+1}$ is true, or the reading at $b+1$ is missing.

Physics: a sign flip reflects the true trace about zero, so each true edge is a cusp minimum of the recorded net load; 84% of reviewer-labelled Beta edges are strict minima, 95% within one slot. The tolerance is inward only — the start may sit one slot after a cusp and the end one slot before — so sampling jitter of the crossing is absorbed without re-admitting the slot on the descending limb outside the cusp, which was the source of the one-slot edge extension (38% of revision 1's remaining Beta loss). A cusp cannot be observed against a missing reading, hence the exemption; without it beta_D (42% of edges abut a gap) collapses.

Applied to the candidate set only. No parameter is fitted; the one-slot tolerance is the sampling interval.

## Counts

| | | 
|---|---|
| Plausibility components | 1 (bridge misfit ratio) |
| Fitted scoring parameters | 0 |
| Fixed physical settings | 4: scan range 24–71; RSS floor from the cohort's smallest non-zero overnight step; overnight slots 0–23 used only for that floor; **edge rule as above** |
| Calibration parameters | 2 per fold (α, β) |
| Public controls | 1 (c = 0.7) |

## Headline, leave-one-station-out, c = 0.7 (revision 1 in brackets)

| | Alpha | Beta `sure` |
|---|---|---|
| Reference Energy IoU | 0.869 (0.864) | **0.895** (0.881) |
| Reference energy precision | 0.889 (0.881) | **0.941** (0.926) |
| Sure-day recall | 0.905 (0.917) | **0.896** (0.896) |
| Site-day precision | 0.977 (0.975) | 0.953 (0.957) |
| Site-day F1 | 0.940 (0.945) | 0.923 (0.925) |

Edge quality on corrected Beta `sure` days: exact window match 35% (13%); extension 1.95 slots per day (2.70); over-proposed 347 MWh (484).

## Known limitations carried forward

Those of revision 1 (jagged solar estimates; meters clipping at zero; demand rising with solar; Alpha label contiguity), plus: on very short, very small events with no clean cusp — Alpha's noise-floor events — the edge rule can leave no admissible window near the event and the day becomes `UNCERTAIN`; this costs 1.2 points of Alpha day recall on events holding about 2% of Alpha reference energy.

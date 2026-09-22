# Within-day window ranking of M9: results

Run of 22 September 2026 (`run.py`, 22 s, deterministic). The frozen revision-2 scorer was re-run label-free on every labelled reverse power flow (RPF) day of the Phase 3 population; labels were used only to locate the truth in each day's ranking. Observations first, interpretation after, separated on purpose. Definitions are in `README.md`; every number below is in `tables/`.

## 1. Self-check against Phase 3

| cohort | days checked | admissible-count mismatches | best-window mismatches | null-decision mismatches | tie-break mismatches | max abs score difference | energy days compared | max abs MWh difference | days differing |
|---|---|---|---|---|---|---|---|---|---|
| alpha | 3,381 | 0 | 0 | 0 | 0 | 0.0 | 3,093 | 6.4e-14 | 0 |
| beta | 629 | 0 | 0 | 0 | 0 | 0.0 | 597 | 1.2e-13 | 0 |

On every one of the 4,010 labelled RPF days the recomputed admissible count equals the committed `n_admissible`, the masked argmax equals the committed best window (or, on the 320 days where the null won, the committed runner-up, which holds the best window in that case), its score equals `r_best` exactly, and the ranking's top window equals the scorer's own `best_window` with its tie tolerance. The sigma floors equal the manifest values. On the 3,690 days with a committed candidate window the rank-1 candidate, correct and required energies equal the Phase 3 `site_days` values to floating-point precision; the |y| convention therefore makes no difference on these days. The 320 null-won days carry no candidate in Phase 3 and could not be energy-checked. No mismatch, so the study proceeds on the committed scorer.

## 2. Observations

### 2.1 Where the truth sits in the ranking (admissible windows)

| population | target | n days | 1 | 2 | 3 | 4–10 | >10 | not admissible | none ≥ 0.8 |
|---|---|---|---|---|---|---|---|---|---|
| Alpha | exact span | 3,381 | 813 (24.0%) | 479 (14.2%) | 269 (8.0%) | 596 (17.6%) | 618 (18.3%) | 606 (17.9%) | — |
| Alpha | IoU ≥ 0.8 | 3,381 | 1,844 (54.5%) | 153 (4.5%) | 90 (2.7%) | 276 (8.2%) | 188 (5.6%) | — | 830 (24.5%) |
| Beta sure | exact span | 470 | 157 (33.4%) | 53 (11.3%) | 36 (7.7%) | 113 (24.0%) | 82 (17.4%) | 29 (6.2%) | — |
| Beta sure | IoU ≥ 0.8 | 470 | 387 (82.3%) | 13 (2.8%) | 8 (1.7%) | 22 (4.7%) | 33 (7.0%) | — | 7 (1.5%) |
| Beta unsure | exact span | 159 | 17 (10.7%) | 10 (6.3%) | 5 (3.1%) | 32 (20.1%) | 71 (44.7%) | 24 (15.1%) | — |
| Beta unsure | IoU ≥ 0.8 | 159 | 67 (42.1%) | 12 (7.5%) | 6 (3.8%) | 34 (21.4%) | 39 (24.5%) | — | 1 (0.6%) |

Cumulatively, the exact truth span is within the top 3 on 46.2% of Alpha days and 52.3% of Beta sure days, and within the top 10 on 63.8% and 76.4%. A window with IoU ≥ 0.8 is within the top 3 on 61.7% and 86.8%.

Alpha split by label contiguity (`rank_distribution_alpha_contiguity_admissible.md`):

| Alpha days | target | n days | 1 | 2 | 3 | 4–10 | >10 | not admissible | none ≥ 0.8 |
|---|---|---|---|---|---|---|---|---|---|
| contiguous | exact span | 1,951 | 694 (35.6%) | 376 (19.3%) | 180 (9.2%) | 336 (17.2%) | 138 (7.1%) | 227 (11.6%) | — |
| contiguous | IoU ≥ 0.8 | 1,951 | 1,484 (76.1%) | 83 (4.3%) | 47 (2.4%) | 135 (6.9%) | 78 (4.0%) | — | 124 (6.4%) |
| non-contiguous | exact span | 1,430 | 119 (8.3%) | 103 (7.2%) | 89 (6.2%) | 260 (18.2%) | 480 (33.6%) | 379 (26.5%) | — |
| non-contiguous | IoU ≥ 0.8 | 1,430 | 360 (25.2%) | 70 (4.9%) | 43 (3.0%) | 141 (9.9%) | 110 (7.7%) | — | 706 (49.4%) |

### 2.2 The same ranks with admissibility ignored (all scoreable windows)

| population | target | n days | 1 | 2 | 3 | 4–10 | >10 | not scoreable | none ≥ 0.8 |
|---|---|---|---|---|---|---|---|---|---|
| Alpha | exact span | 3,381 | 656 (19.4%) | 392 (11.6%) | 236 (7.0%) | 709 (21.0%) | 1,386 (41.0%) | 2 (0.1%) | — |
| Alpha | IoU ≥ 0.8 | 3,381 | 1,804 (53.4%) | 153 (4.5%) | 88 (2.6%) | 282 (8.3%) | 414 (12.2%) | — | 640 (18.9%) |
| Beta sure | exact span | 470 | 57 (12.1%) | 39 (8.3%) | 36 (7.7%) | 116 (24.7%) | 218 (46.4%) | 4 (0.9%) | — |
| Beta sure | IoU ≥ 0.8 | 470 | 366 (77.9%) | 15 (3.2%) | 12 (2.6%) | 24 (5.1%) | 48 (10.2%) | — | 5 (1.1%) |
| Beta unsure | exact span | 159 | 2 (1.3%) | 7 (4.4%) | 5 (3.1%) | 26 (16.4%) | 119 (74.8%) | — | — |
| Beta unsure | IoU ≥ 0.8 | 159 | 67 (42.1%) | 8 (5.0%) | 4 (2.5%) | 19 (11.9%) | 61 (38.4%) | — | — |

### 2.3 Why truth spans are inadmissible

| population | n days | truth admissible | non-contiguous labels | outside slots 24–71 | missing reading | edge rule (start) | edge rule (end) | edge rule (both) |
|---|---|---|---|---|---|---|---|---|
| Alpha | 3,381 | 2,775 | 1,430 | 2 | 0 | 251 | 304 | 49 |
| Beta sure | 470 | 441 | 2 | 0 | 4 | 9 | 14 | 2 |
| Beta unsure | 159 | 135 | 1 | 0 | 0 | 7 | 13 | 4 |

A window with IoU ≥ 0.8 exists but only among inadmissible windows on 190 Alpha days (123 of them contiguous), 2 Beta sure days and 1 Beta unsure day. No window of any kind reaches 0.8 on 640 Alpha days (639 non-contiguous; mean 8.1 labelled slots) and 5 Beta sure days.

### 2.4 Quality of the rank-1 window on positive days

| population | n days | null wins | exact share | mean IoU | median IoU | IoU ≥ 0.8 | ceiling: mean best admissible IoU | interval precision | recall | F1 | energy IoU | energy precision | candidate MWh | correct MWh | required MWh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Alpha | 3,381 | 288 | 0.240 | 0.708 | 0.833 | 0.545 | 0.872 | 0.830 | 0.929 | 0.877 | 0.874 | 0.893 | 35,053 | 31,290 | 32,036 |
| Alpha (contiguous) | 1,951 | 208 | 0.356 | 0.810 | 0.947 | 0.761 | 0.961 | 0.920 | 0.969 | 0.944 | 0.954 | 0.959 | 23,123 | 22,182 | 22,321 |
| Alpha (non-contiguous) | 1,430 | 80 | 0.083 | 0.569 | 0.615 | 0.252 | 0.750 | 0.703 | 0.864 | 0.775 | 0.727 | 0.763 | 11,930 | 9,109 | 9,714 |
| Beta sure | 470 | 6 | 0.334 | 0.884 | 0.946 | 0.823 | 0.990 | 0.915 | 0.970 | 0.942 | 0.951 | 0.961 | 10,178 | 9,786 | 9,896 |
| Beta unsure | 159 | 26 | 0.107 | 0.640 | 0.731 | 0.421 | 0.985 | 0.795 | 0.821 | 0.808 | 0.733 | 0.876 | 1,461 | 1,279 | 1,564 |

"Null wins" counts positive days whose best window scored at or below zero; the rank-1 window is still measured on them. Those days hold 173 MWh of Alpha's 32,036 MWh required (0.5%), 57 of 9,896 MWh on Beta sure (0.6%) and 231 of 1,564 MWh on Beta unsure (14.8%). These energy metrics are conditional on the day being positive and a window being proposed, so they are not the Phase 3 headline numbers (Alpha 0.869 / 0.889, Beta sure 0.895 / 0.941), which also carry the decision: positive days not corrected and non-RPF days wrongly corrected.

Per station (headline populations; `rank1_quality_station.md`, `rank_distribution_station_admissible.md`):

| station | n days | exact rank 1 | exact rank > 10 | truth not admissible | IoU ≥ 0.8 at rank 1 | mean IoU | interval F1 | energy IoU | energy precision |
|---|---|---|---|---|---|---|---|---|---|
| alpha_B | 27 | 25.9% | 22.2% | 18.5% | 48.1% | 0.621 | 0.797 | 0.490 | 0.501 |
| alpha_C | 627 | 25.2% | 19.3% | 19.3% | 52.5% | 0.692 | 0.870 | 0.860 | 0.884 |
| alpha_E | 700 | 29.0% | 16.4% | 16.0% | 57.6% | 0.738 | 0.890 | 0.902 | 0.918 |
| alpha_F | 842 | 22.8% | 18.2% | 17.7% | 56.9% | 0.726 | 0.881 | 0.887 | 0.901 |
| alpha_G | 650 | 22.0% | 19.8% | 16.2% | 55.2% | 0.712 | 0.879 | 0.866 | 0.891 |
| alpha_I | 148 | 18.2% | 18.2% | 24.3% | 45.9% | 0.648 | 0.843 | 0.784 | 0.796 |
| alpha_J | 387 | 21.4% | 17.3% | 20.2% | 49.9% | 0.666 | 0.857 | 0.816 | 0.842 |
| beta_A | 26 | 38.5% | 15.4% | 7.7% | 76.9% | 0.853 | 0.937 | 0.978 | 0.989 |
| beta_B | 125 | 13.6% | 18.4% | 3.2% | 86.4% | 0.881 | 0.938 | 0.934 | 0.944 |
| beta_D | 45 | 22.2% | 51.1% | 6.7% | 33.3% | 0.684 | 0.777 | 0.967 | 0.971 |
| beta_E | 21 | 14.3% | 38.1% | 0.0% | 85.7% | 0.862 | 0.926 | 0.916 | 0.929 |
| beta_F | 164 | 40.9% | 10.4% | 9.1% | 87.8% | 0.918 | 0.958 | 0.961 | 0.974 |
| beta_G | 85 | 57.6% | 8.2% | 3.5% | 91.8% | 0.943 | 0.976 | 0.972 | 0.982 |
| beta_H | 4 | 25.0% | 0.0% | 50.0% | 100.0% | 0.912 | 0.954 | 0.927 | 0.939 |

### 2.5 Oracles: best of the top k admissible windows

| population | k | exact within top k | IoU ≥ 0.8 within top k | mean IoU (best by IoU) | interval precision | recall | F1 | energy IoU (best by energy IoU) | energy precision |
|---|---|---|---|---|---|---|---|---|---|
| Alpha | 1 | 0.240 | 0.545 | 0.708 | 0.830 | 0.929 | 0.877 | 0.874 | 0.893 |
| Alpha | 3 | 0.462 | 0.617 | 0.773 | 0.867 | 0.950 | 0.907 | 0.900 | 0.915 |
| Alpha | 5 | 0.555 | 0.654 | 0.800 | 0.888 | 0.955 | 0.920 | 0.917 | 0.931 |
| Beta sure | 1 | 0.334 | 0.823 | 0.884 | 0.915 | 0.970 | 0.942 | 0.951 | 0.961 |
| Beta sure | 3 | 0.523 | 0.868 | 0.919 | 0.941 | 0.982 | 0.961 | 0.968 | 0.976 |
| Beta sure | 5 | 0.626 | 0.896 | 0.937 | 0.956 | 0.987 | 0.971 | 0.979 | 0.985 |
| Beta unsure | 1 | 0.107 | 0.421 | 0.640 | 0.795 | 0.821 | 0.808 | 0.733 | 0.876 |
| Beta unsure | 3 | 0.201 | 0.535 | 0.708 | 0.849 | 0.855 | 0.852 | 0.788 | 0.918 |
| Beta unsure | 5 | 0.277 | 0.610 | 0.755 | 0.888 | 0.878 | 0.883 | 0.830 | 0.946 |

### 2.6 Geometry of the rank-1 window against the truth span

| population | n days | exact | too long (start) | too long (end) | too long (both) | too short | shifted earlier | shifted later | disjoint | both edges within 1 slot | mean extra slots when too long |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Alpha | 3,381 | 813 (24.0%) | 408 (12.1%) | 371 (11.0%) | 321 (9.5%) | 879 (26.0%) | 231 (6.8%) | 186 (5.5%) | 172 (5.1%) | 58.6% | 3.1 |
| Alpha (contiguous) | 1,951 | 694 (35.6%) | 267 (13.7%) | 245 (12.6%) | 177 (9.1%) | 304 (15.6%) | 64 (3.3%) | 60 (3.1%) | 140 (7.2%) | 75.8% | 2.8 |
| Alpha (non-contiguous) | 1,430 | 119 (8.3%) | 141 (9.9%) | 126 (8.8%) | 144 (10.1%) | 575 (40.2%) | 167 (11.7%) | 126 (8.8%) | 32 (2.2%) | 35.2% | 3.6 |
| Beta sure | 470 | 157 (33.4%) | 51 (10.9%) | 55 (11.7%) | 75 (16.0%) | 97 (20.6%) | 23 (4.9%) | 10 (2.1%) | 2 (0.4%) | 63.0% | 4.4 |
| Beta unsure | 159 | 17 (10.7%) | 24 (15.1%) | 18 (11.3%) | 33 (20.8%) | 40 (25.2%) | 9 (5.7%) | 7 (4.4%) | 11 (6.9%) | 28.9% | 5.5 |

Supplementary counts from `per_day.csv`: of the too-long windows, 502 of 1,100 on Alpha and 40 of 181 on Beta sure extend by exactly one slot (medians 2 and 3 slots). Of the too-short windows, 362 of 879 on Alpha and 65 of 97 on Beta sure fall short by exactly one slot; Beta sure too-short windows still average IoU 0.884, Alpha's average 0.604 because 575 of the 879 are non-contiguous days. Days with the exact truth at rank 2–3 are near-misses: rank-1 IoU averages 0.828 (Alpha) and 0.932 (Beta sure), with both edges within one slot on 85% and 82% of them. Days with the exact truth beyond rank 10 are not: rank-1 IoU averages 0.444 (Alpha) and 0.675 (Beta sure), and only 10.5% and 30.5% of them reach IoU 0.8. The 172 disjoint Alpha days are tiny events (mean 2.5 labelled slots, 0.6 MWh required against a population mean of 9.5 MWh); 104 of them are null-won days. When the truth span is inadmissible, the rank-1 window averages IoU 0.484 on Alpha (21% reach 0.8) and 0.747 on Beta sure (69% reach 0.8).

Figures: `figures/fig01_rank_distribution.png` (section 2.1), `fig02_rank1_iou_cdf.png` (section 2.4), `fig03_station_exact_rank1.png` (per-station exact share).

## 3. Interpretation

1. **The candidate set is not the limit on Beta; the ranking is.** The best admissible window averages IoU 0.990 on Beta sure and 0.985 on Beta unsure, so an almost perfect window is nearly always on offer. M9 puts a window with IoU ≥ 0.8 at rank 1 on 82% of Beta sure days and the exact reviewer span at rank 1 on a third; the exact span is within the top 5 on 63%. On Alpha the candidate set itself is capped at mean IoU 0.872, entirely by the 42% of days with non-contiguous labels (ceiling 0.750 on those days, 0.961 on contiguous days). Contiguous Alpha days behave like Beta sure days on every measure (exact 35.6% against 33.4%, IoU ≥ 0.8 at rank 1 76% against 82%, energy IoU 0.954 against 0.951).

2. **What rank-1 failures look like.** On Beta sure, 181 of the 313 non-exact days (58%) are windows that are too long, extended on one or both sides by a median of 3 slots; 97 (31%) are one-slot-short windows that still overlap the truth almost completely; only 35 (11%) are shifted or disjoint. On contiguous Alpha days the pattern is the same (too long 55% of non-exact days, too short 24%, shifted 10%, disjoint 11%; the disjoint ones are sub-MWh events). On non-contiguous Alpha days the dominant category is "too short" (575 of 1,311 non-exact days, 44%): the rank-1 window covers one labelled segment and the outer span, which is what a single-window method is scored against, cannot be matched. Beta unsure shows both longer extensions (median 4 extra slots) and more shifts and disjoint picks, consistent with labels the reviewer was not sure of.

3. **Near-miss and genuine failure are separable by rank.** Truth at rank 2–3 means a one-slot edge disagreement (rank-1 IoU 0.83–0.93, both edges within one slot on more than 80% of those days). Truth beyond rank 10 means the scorer preferred a materially different window (rank-1 IoU 0.44–0.68). That second group is 18% of Alpha and 17% of Beta sure days, and is where localisation is actually wrong; on beta_D it is 51% of days, because that station's missing blocks abut the RPF and the gap exemption admits long, low-energy extensions (interval precision 0.656 but energy precision 0.971).

4. **The edge rule earns its place on this measure.** Removing admissibility (section 2.2) drops the exact-rank-1 share from 33.4% to 12.1% on Beta sure and from 24.0% to 19.4% on Alpha, and moves the truth beyond rank 10 on 46% of Beta sure days. Its cost is the 6.2% of Beta sure days and 17.9% of Alpha days whose truth span is itself inadmissible (mostly one edge failing the cusp test on Alpha), and 190 Alpha days on which only an inadmissible window would reach IoU 0.8. The rev-2 note reported exact matches on 35% of corrected Beta sure days; here the equivalent over all Beta sure RPF days is 33.4%, and the comparable figure without the edge rule is 12.1%, consistent with that note's 13%.

5. **What a reviewer could gain from the top few.** If a reviewer always chose the best of the top 3, Beta sure energy IoU would rise from 0.951 to 0.968 and Alpha from 0.874 to 0.900; the top 5 gives 0.979 and 0.917, and roughly doubles the exact-match share on both cohorts. Those are oracle upper bounds: the choice uses the truth. They say the top 5 usually contains a better window than rank 1; they do not say a reviewer would find it.

6. **Localisation against decision.** On positive days with a proposed window, Beta sure energy IoU is 0.951 here against the 0.895 headline; the difference is the decision layer (sure-day recall 0.896 at c = 0.7, plus false corrections on non-RPF days), not window placement. On Alpha the corresponding gap is small (0.874 against 0.869) because the localisation ceiling from non-contiguous labels dominates.

## 4. What this supports and what it does not

It supports: that M9's ranking places a window with IoU ≥ 0.8 first on four Beta sure days in five and the exact reviewer span first on one in three; that most rank-1 disagreements are edge extensions of one to three slots or one-slot shortfalls rather than misplacements; that the edge rule is responsible for most of the exact-match rate; that Alpha's lower figures are largely a property of its non-contiguous synthetic labels, since contiguous Alpha days match Beta sure; and that the top-5 list is a reasonable review aid, with a bounded gain.

It does not support: any held-out claim (the scorer and its rules were chosen with all eighteen stations visible, and this study fits nothing but also holds nothing out); any claim about a reviewer's actual gain (the oracles know the truth); any statement about detection, calibration or false corrections on non-RPF days, none of which enters here; comparability of these energy metrics with the Phase 3 headline (these are conditional on positive days); or conclusions from Beta unsure beyond "worse and noisier", since those labels are themselves uncertain. The IoU threshold of 0.8 and the top-k values are reporting choices, not tuned settings. Nothing here proposes a change to the method.

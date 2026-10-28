# Results — bridge anchors at the window edge (2026-09-22)

Three runs of the frozen revision-2 method (`--variant sq --p 0 --sigma overnight
--c 0.7 --stat llr --missing mask_windows --edges inwardx`), differing only in the
`--anchors` rule. Leave-one-station-out calibration, c = 0.7. Data hashes and the
scorer hash (`24022361…`, the scorer with the `anchors` option) are in each
`runs/<rule>/config.json`. The frozen reference `m9_dev/runs/phase5_final_rev2/` was
present and was not re-created. No default was changed: `nearest` remains the method.

## 1. Observations

### 1.1 Reproduction of the frozen reference

`runs/nearest` reproduces `phase5_final_rev2` exactly: every numeric cell of
`summary_pooled.csv` and `summary_station.csv` differs by 0, and all 13,347 held-out
site-day rows are identical in `r_best`, `p`, `best_start`, `best_end`, `runner_start`,
`runner_end`, `n_admissible`, `proposed_mwh` and `outcome` (`tables/reproduction_check.md`;
`compare_runs.py` deltas all 0.0000 in `tables/compare_nearest.csv`). The synthetic
fixtures under `m9_dev/tests/fixtures/`, written by the pre-change scorer, are also
matched bit for bit.

### 1.2 Pooled, per anchor rule

Beta `sure` (2,305 days, 470 RPF):

| anchors | Energy IoU | energy precision | sure-day recall | sure-day uncertain | day precision | day F1 | AUTO_CORRECT rate | UNCERTAIN rate | ECE |
|---|---|---|---|---|---|---|---|---|---|
| nearest (frozen) | **0.8945** | **0.9407** | **0.8957** | 0.0532 | 0.9525 | 0.9232 | 0.1918 | 0.0430 | 0.0088 |
| edge | 0.5864 | 0.6924 | 0.6468 | 0.2064 | 0.9297 | 0.7629 | 0.1419 | 0.1020 | 0.0200 |
| gap_edge | 0.8892 | 0.9400 | 0.8851 | 0.0489 | 0.9498 | 0.9163 | 0.1900 | 0.0438 | 0.0089 |

Alpha (10,425 days, 3,381 RPF):

| anchors | Energy IoU | energy precision | day recall | day uncertain | day precision | day F1 | AUTO_CORRECT rate | UNCERTAIN rate | ECE |
|---|---|---|---|---|---|---|---|---|---|
| nearest (frozen) | **0.8692** | **0.8886** | **0.9048** | 0.0497 | 0.9773 | 0.9396 | 0.3002 | 0.0261 | 0.0218 |
| edge | 0.2670 | 0.3022 | 0.3144 | 0.4167 | 0.8292 | 0.4559 | 0.1230 | 0.3369 | 0.0504 |
| gap_edge | 0.8692 | 0.8886 | 0.9048 | 0.0497 | 0.9773 | 0.9396 | 0.3002 | 0.0261 | 0.0218 |

Beta `unsure` (617 days, 159 RPF; sensitivity only): Energy IoU 0.386 / 0.288 / 0.383,
energy precision 0.534 / 0.839 / 0.531, day recall 0.497 / 0.132 / 0.497 for nearest /
edge / gap_edge.

`compare_runs.py` verdicts against `nearest`: `edge` fails rules 1, 2a, 2b, 2c and 5;
`gap_edge` fails 2a (beta_D loses sure-day recall) and 2c, passes 2b, 5 and 1
(`tables/compare_edge.csv`, `tables/compare_gap_edge.csv`).

### 1.3 Per station

Energy IoU / energy precision / sure-day recall, nearest → edge → gap_edge
(full table with day F1 and correction rates in `tables/per_station.md`).

Beta `sure`:

| station | RPF days | Energy IoU | energy precision | sure-day recall |
|---|---|---|---|---|
| beta_A | 26 | 0.956 → 0.453 → 0.956 | 0.988 → 0.513 → 0.988 | 0.923 → 0.692 → 0.923 |
| beta_B | 125 | 0.889 → 0.806 → 0.889 | 0.948 → 0.993 → 0.948 | 0.864 → 0.696 → 0.864 |
| beta_C | 0 | — | — | — |
| **beta_D** | 45 | 0.902 → 0.348 → **0.792** | 0.966 → 0.485 → 0.958 | 0.756 → 0.356 → **0.644** |
| beta_E | 21 | 0.899 → 0.063 → 0.899 | 0.911 → 0.073 → 0.911 | 1.000 → 0.190 → 1.000 |
| beta_F | 164 | 0.930 → 0.820 → 0.929 | 0.950 → 0.923 → 0.948 | 0.957 → 0.726 → 0.957 |
| beta_G | 85 | 0.870 → 0.747 → 0.870 | 0.902 → 0.955 → 0.902 | 0.894 → 0.706 → 0.894 |
| beta_H | 4 | 0.173 → 0.000 → 0.173 | 0.818 → 0.000 → 0.818 | 0.250 → 0.000 → 0.250 |

Alpha (`gap_edge` is identical to `nearest` on every Alpha station: the Alpha days are
complete, so no window has a gap-adjacent edge):

| station | RPF days | Energy IoU nearest → edge | energy precision nearest → edge | day recall nearest → edge |
|---|---|---|---|---|
| alpha_B | 27 | 0.502 → 0.000 | 0.518 → 0.000 | 0.667 → 0.000 |
| alpha_C | 627 | 0.845 → 0.644 | 0.870 → 0.898 | 0.890 → 0.282 |
| alpha_E | 700 | 0.895 → 0.715 | 0.912 → 0.986 | 0.920 → 0.334 |
| alpha_F | 842 | 0.887 → 0.690 | 0.902 → 0.979 | 0.931 → 0.340 |
| alpha_G | 650 | 0.863 → 0.621 | 0.889 → 0.942 | 0.920 → 0.306 |
| alpha_I | 148 | 0.771 → 0.083 | 0.784 → 0.086 | 0.872 → 0.365 |
| alpha_J | 387 | 0.800 → 0.615 | 0.829 → 0.827 | 0.848 → 0.292 |
| alpha_A, D, H | 0 | — | — | AUTO_CORRECT rate 0.000 → 0.000 / 0.047 / 0.088 (49 and 91 false corrections at D and H) |

### 1.4 beta_D: labelled RPF days whose span abuts a missing reading

beta_D has 76 labelled RPF days (45 `sure`, 31 `unsure`). 38 have a missing reading
(net load or solar) immediately before the first labelled slot or immediately after the
last: 29 before, 25 after, 16 both. Among `sure` days it is 19 (16 before, 14 after, 11
both). `truth` is the labelled span in slots (inclusive). Under each rule: the best window,
or in brackets the best window when the null won; evidence r; held-out p; decision (AC =
AUTO_CORRECT, AK = AUTO_KEEP, UNC = UNCERTAIN); and whether the shown window equals the
truth exactly, has slot IoU ≥ 0.8, or its IoU otherwise. Every labelled span is contiguous.
Source: `tables/beta_D_gap_days.csv`.

| date | conf | truth | gap | nearest window | r | p | dec | match | edge window | r | p | dec | match | gap_edge window | r | p | dec | match |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2023-10-05 | sure | 45–60 | before | 45–62 | 32.5 | 1.0 | AC | IoU≥0.8 | 45–67 | 20.0 | 0.5 | UNC | IoU 0.70 | 45–66 | 20.4 | 0.9 | AC | IoU 0.73 |
| 2023-10-06 | sure | 46–59 | before | 46–62 | 28.6 | 0.9 | AC | IoU≥0.8 | 46–62 | 14.2 | 0.3 | AK | IoU≥0.8 | 46–61 | 13.8 | 0.8 | AC | IoU≥0.8 |
| 2023-10-11 | sure | 51–59 | before | 51–67 | 25.2 | 0.9 | AC | IoU 0.53 | 51–62 | 4.7 | 0.0 | AK | IoU 0.75 | 51–67 | 6.2 | 0.4 | UNC | IoU 0.53 |
| 2023-10-12 | sure | 45–52 | both | 45–52 | 5.3 | 0.4 | UNC | exact | 45–52 | 1.1 | 0.0 | AK | exact | 45–52 | 1.1 | 0.1 | AK | exact |
| 2023-10-14 | sure | 48–55 | both | 48–51 | 5.9 | 0.4 | UNC | IoU 0.50 | 48–55 | 0.9 | 0.0 | AK | exact | 48–55 | 0.9 | 0.0 | AK | exact |
| 2023-10-20 | sure | 46–58 | both | 46–58 | 13.4 | 0.8 | AC | exact | 53–55 | 4.6 | 0.0 | AK | IoU 0.23 | 46–58 | 3.6 | 0.2 | AK | exact |
| 2023-10-23 | sure | 44–58 | both | 44–58 | 18.8 | 0.9 | AC | exact | 44–55 | 5.5 | 0.0 | AK | IoU≥0.8 | 44–58 | 5.5 | 0.4 | UNC | exact |
| 2023-11-10 | sure | 43–56 | both | 47–56 | 6.8 | 0.5 | UNC | IoU 0.71 | 30–33 | 4.7 | 0.0 | AK | IoU 0.00 | 47–56 | 3.6 | 0.2 | AK | IoU 0.71 |
| 2023-11-18 | sure | 47–59 | both | 47–59 | 14.8 | 0.8 | AC | exact | 28–32 | 6.7 | 0.0 | AK | IoU 0.00 | 47–59 | 2.5 | 0.1 | AK | exact |
| 2023-12-22 | sure | 48–59 | after | 48–59 | 15.5 | 0.8 | AC | exact | 48–59 | 12.6 | 0.2 | AK | exact | 49–59 | 12.4 | 0.7 | AC | IoU≥0.8 |
| 2024-04-11 | sure | 45–52 | both | 45–52 | 7.6 | 0.5 | UNC | exact | 45–52 | 1.4 | 0.0 | AK | exact | 45–52 | 1.4 | 0.1 | AK | exact |
| 2024-08-22 | sure | 49–54 | after | 45–54 | 11.0 | 0.7 | UNC | IoU 0.60 | 47–54 | 7.9 | 0.0 | AK | IoU 0.75 | 43–54 | 9.2 | 0.6 | UNC | IoU 0.50 |
| 2024-08-26 | sure | 46–54 | both | 46–54 | 4.6 | 0.3 | UNC | exact | 46–54 | 5.0 | 0.0 | AK | exact | 46–54 | 5.0 | 0.3 | UNC | exact |
| 2024-08-30 | sure | 45–52 | both | 45–52 | 6.7 | 0.5 | UNC | exact | 45–52 | 1.3 | 0.0 | AK | exact | 45–52 | 1.3 | 0.1 | AK | exact |
| 2024-08-31 | sure | 47–50 | both | 47–50 | 0.9 | 0.0 | AK | exact | 27–29 | 0.2 | 0.0 | AK | IoU 0.00 | (47–50) | -0.3 | 0.0 | AK | exact |
| 2024-09-02 | sure | 43–55 | before | 43–57 | 24.9 | 0.9 | AC | IoU≥0.8 | 43–61 | 16.4 | 0.4 | UNC | IoU 0.68 | 43–63 | 18.0 | 0.9 | AC | IoU 0.62 |
| 2024-09-17 | sure | 41–54 | after | 34–54 | 29.3 | 0.9 | AC | IoU 0.67 | 34–54 | 15.2 | 0.3 | UNC | IoU 0.67 | 35–54 | 15.3 | 0.8 | AC | IoU 0.70 |
| 2024-09-21 | sure | 44–56 | before | 44–57 | 23.2 | 0.9 | AC | IoU≥0.8 | 44–62 | 14.1 | 0.2 | AK | IoU 0.68 | 44–61 | 14.1 | 0.8 | AC | IoU 0.72 |
| 2024-09-22 | sure | 44–54 | both | 44–54 | 12.0 | 0.7 | AC | exact | 44–54 | 4.2 | 0.0 | AK | exact | 44–54 | 4.2 | 0.3 | AK | exact |
| 2023-10-04 | unsure | 49–60 | before | 49–56 | 5.1 | 0.3 | UNC | IoU 0.67 | 51–59 | 5.5 | 0.0 | AK | IoU 0.75 | 49–56 | 3.6 | 0.2 | AK | IoU 0.67 |
| 2023-10-07 | unsure | 55–57 | after | 43–57 | 2.9 | 0.2 | AK | IoU 0.20 | 48–52 | 2.4 | 0.0 | AK | IoU 0.00 | 52–53 | 2.1 | 0.1 | AK | IoU 0.00 |
| 2023-10-08 | unsure | 53–57 | both | 53–57 | 1.2 | 0.1 | AK | exact | 32–35 | 2.7 | 0.0 | AK | IoU 0.00 | 53–57 | 0.4 | 0.0 | AK | exact |
| 2023-10-16 | unsure | 45–58 | before | 45–61 | 12.8 | 0.8 | AC | IoU≥0.8 | 45–64 | 7.1 | 0.0 | AK | IoU 0.70 | 45–62 | 7.9 | 0.5 | UNC | IoU 0.78 |
| 2023-10-19 | unsure | 58–60 | before | 61–64 | 7.9 | 0.6 | UNC | IoU 0.00 | 40–46 | 1.1 | 0.0 | AK | IoU 0.00 | 61–64 | 7.9 | 0.6 | UNC | IoU 0.00 |
| 2023-10-24 | unsure | 46–55 | both | 46–55 | 10.2 | 0.7 | UNC | exact | 46–55 | 7.6 | 0.0 | AK | exact | 46–55 | 7.6 | 0.5 | UNC | exact |
| 2023-10-29 | unsure | 48–57 | both | 48–57 | 4.5 | 0.3 | UNC | exact | 48–57 | 2.6 | 0.0 | AK | exact | 48–57 | 2.6 | 0.2 | AK | exact |
| 2023-10-30 | unsure | 44–54 | both | 44–54 | 9.6 | 0.6 | UNC | exact | 44–54 | 2.9 | 0.0 | AK | exact | 44–54 | 2.9 | 0.2 | AK | exact |
| 2023-11-03 | unsure | 46–59 | after | 36–59 | 20.9 | 0.9 | AC | IoU 0.58 | 35–59 | 13.0 | 0.2 | AK | IoU 0.56 | 36–59 | 13.6 | 0.8 | AC | IoU 0.58 |
| 2023-11-14 | unsure | 40–54 | after | 37–54 | 26.0 | 0.9 | AC | IoU≥0.8 | 33–54 | 18.6 | 0.5 | UNC | IoU 0.68 | 34–54 | 19.6 | 0.9 | AC | IoU 0.71 |
| 2023-11-19 | unsure | 49–53 | after | 48–53 | 3.3 | 0.2 | AK | IoU≥0.8 | 47–53 | 0.4 | 0.0 | AK | IoU 0.71 | 47–53 | 0.4 | 0.0 | AK | IoU 0.71 |
| 2023-12-04 | unsure | 54–56 | before | 57–59 | 3.4 | 0.2 | AK | IoU 0.00 | 33–38 | 2.3 | 0.0 | AK | IoU 0.00 | 34–38 | 1.6 | 0.1 | AK | IoU 0.00 |
| 2023-12-07 | unsure | 55–58 | before | 55–60 | 6.3 | 0.4 | UNC | IoU 0.67 | 58–67 | 4.1 | 0.0 | AK | IoU 0.08 | 55–59 | 1.3 | 0.1 | AK | IoU≥0.8 |
| 2024-02-21 | unsure | 51–55 | before | 51–64 | 8.3 | 0.6 | UNC | IoU 0.36 | 68–71 | 2.2 | 0.0 | AK | IoU 0.00 | 51–61 | 4.6 | 0.3 | UNC | IoU 0.45 |
| 2024-02-22 | unsure | 42–44 | after | 41–44 | 3.1 | 0.2 | AK | IoU 0.75 | 40–44 | 1.7 | 0.0 | AK | IoU 0.60 | 41–44 | 1.7 | 0.1 | AK | IoU 0.75 |
| 2024-04-09 | unsure | 46–50 | before | 46–62 | 7.1 | 0.5 | UNC | IoU 0.29 | 46–62 | 6.8 | 0.0 | AK | IoU 0.29 | 46–61 | 6.1 | 0.4 | UNC | IoU 0.31 |
| 2024-04-16 | unsure | 49–52 | before | 52–55 | 7.1 | 0.5 | UNC | IoU 0.14 | 35–38 | 2.4 | 0.0 | AK | IoU 0.00 | 52–55 | 7.1 | 0.5 | UNC | IoU 0.14 |
| 2024-05-16 | unsure | 44–48 | after | 41–42 | 3.9 | 0.3 | AK | IoU 0.00 | 38–48 | 2.2 | 0.0 | AK | IoU 0.45 | 41–42 | 3.9 | 0.3 | AK | IoU 0.00 |
| 2024-09-14 | unsure | 48–52 | both | 48–52 | 1.5 | 0.1 | AK | exact | 48–52 | 0.4 | 0.0 | AK | exact | 48–52 | 0.4 | 0.0 | AK | exact |

Summary over those days:

| confidence | anchors | days | AUTO_CORRECT | UNCERTAIN | exact window | IoU ≥ 0.8 | median r | median p |
|---|---|---|---|---|---|---|---|---|
| sure | nearest | 19 | **11** | 7 | 10 | **14** | **13.4** | 0.77 |
| sure | edge | 19 | 0 | 3 | 7 | 9 | 5.0 | 0.01 |
| sure | gap_edge | 19 | 6 | 4 | 10 | 12 | 5.0 | 0.35 |
| unsure | nearest | 19 | 3 | 9 | 5 | 8 | 6.3 | 0.44 |
| unsure | edge | 19 | 0 | 1 | 4 | 4 | 2.6 | 0.00 |
| unsure | gap_edge | 19 | 2 | 6 | 5 | 6 | 3.6 | 0.23 |

On the 11 `sure` days with a gap on both sides, `gap_edge` and `edge` bridge between
the same two edge slots (they differ only in the candidate set, which keeps one- and
two-slot windows under `gap_edge` beside a one-sided gap). The window is exact on 9 of
the 11 under `nearest` and on 10 under `gap_edge` (2023-10-14 becomes exact; 2023-11-10
is 47–56 against 43–56 under both), so placement is not what changes; the evidence is:
its median falls from 6.8 (nearest) to 2.5 (gap_edge), taking all four corrections on
these days (2023-10-20, 10-23, 11-18, 2024-09-22) to AUTO_KEEP or UNCERTAIN. On the 8
one-sided days the evidence falls on every day, typically by a third to a half, and the
window under `gap_edge` tends to extend further on the open side (2023-10-05 45–62 →
45–66; 2024-09-02 43–57 → 43–63; 2024-09-21 44–57 → 44–61; 2023-10-11 loses its
correction); the placement improves on 2023-10-06 (46–62 → 46–61) and worsens on
2023-12-22 (48–59, exact, → 49–59).

### 1.5 What changed elsewhere

`gap_edge` against `nearest`, headline days (`tables/outcome_changes.md`): Alpha, no
site-day changes window or outcome (0 of 10,425). Beta: beta_D 12 outcomes changed, 5
true positives lost, none gained, 18 windows changed, −52 MWh correctly corrected;
beta_F 4 outcomes changed, one false correction gained (2.2 MWh, energy precision 0.950
→ 0.948); beta_E and beta_G 1 and 2 changes between AUTO_KEEP and UNCERTAIN; beta_A 5
windows changed with no outcome change; beta_B, beta_C, beta_H unchanged.

`edge` against `nearest`: 1,996 of 3,381 Alpha RPF days and 120 of 470 Beta `sure` RPF
days lose their correction (3 Beta days gain one); false corrections appear on the
zero-RPF stations alpha_D (49), alpha_H (91), beta_C (3) and beta_H (9); the best window
changes on 73–99% of days at every station where one is proposed. `edge` also removes
one- and two-slot windows from the candidate set, which accounts for at most 139 of the
lost Alpha corrections (the days `nearest` corrects with such a window) and none of
Beta's.

### 1.6 Where the evidence sits (`tables/evidence.md`)

Quartiles of the best-window evidence r on headline days, and the raw AUTO_CORRECT
threshold across the leave-one-station-out folds:

| anchors | cohort | RPF days r (q25 / median / q75) | non-RPF days r (q25 / median / q75 / q99) | raw threshold (median, range) |
|---|---|---|---|---|
| nearest | alpha | 5.6 / 19.1 / 39.0 | −4.2 / −3.1 / −2.1 / 0.3 | 0.3 (0.3–0.4) |
| nearest | beta | 19.7 / 32.1 / 47.0 | −2.7 / −1.9 / −1.1 / 12.5 | 11.2 (10.6–13.1) |
| edge | alpha | 8.0 / 18.7 / 38.4 | 1.2 / 4.2 / 8.8 / 32.9 | 29.2 (23.4–35.5) |
| edge | beta | 21.6 / 35.1 / 49.9 | 2.0 / 4.0 / 7.3 / 25.6 | 24.8 (23.0–27.2) |
| gap_edge | alpha | 5.6 / 19.1 / 39.0 | −4.2 / −3.1 / −2.1 / 0.3 | 0.3 (0.3–0.4) |
| gap_edge | beta | 18.7 / 32.0 / 47.0 | −2.7 / −1.9 / −1.1 / 12.5 | 11.0 (10.3–12.8) |

## 2. Interpretation

**Edge continuity is what the bridge statistic mostly measures on ordinary days, and edge
anchoring removes it.** Under `nearest` both stories share one line drawn from readings
outside the window, so a false correction pays for the 2y step it creates at each edge:
on non-RPF days the best window scores a median of −3 and the null wins by a margin.
Under `edge`, story B's line passes through story B's own edge values, so the steps
vanish from the misfit and what remains is a comparison of interior curvature, which is
weakly positive almost everywhere (non-RPF median +4, 99th percentile 26–33). The
evidence on true days barely moves (median 19 → 19 on Alpha, 32 → 35 on Beta), but the
calibration must raise the raw threshold from 0.3 to 29 on Alpha and from 11 to 25 on
Beta to hold precision, and every true day with moderate evidence falls under it. That
is the whole of the collapse: it is the false-window distribution shifting, not the
true-window one. The residual-count L (L − 2 under `edge`) and the loss of one- and
two-slot windows are second-order.

**On beta_D's gap-adjacent days the problem was never window placement.** Under
`nearest` the window is already exact on 10 of the 19 `sure` days and within IoU ≥ 0.8
on 14; `gap_edge` leaves the exact count at 10 and lowers the other to 12. What those
days lack is evidence, and `gap_edge` lowers it further (median r 13.4 → 5.0), because
on a labelled edge that abuts a gap the recorded net load is generally not at a clean
zero crossing — the missing block or the clipped ≈0 flank starts while the reverse flow
is still under way — so anchoring on that slot's own value puts the line on the shoulder
of the hump and cuts off its base for story A while story B loses the outside reference
level it is being compared with. The reading beyond the gap, though hours away, is an
independent estimate of the demand level, which is what the comparison needs. The
one-sided days show the same thing from the other direction: with the gap side anchored
on the edge, extending the window into the open side costs little, and the proposed
window grows (2023-10-05, 2024-09-02, 2024-09-21).

**Cost outside beta_D is small under `gap_edge` and total under `edge`.** `gap_edge`
is identical to `nearest` wherever no window edge abuts a gap, which is all of Alpha and
nearly all of Beta; its only material effect is at beta_D, and that effect is a loss of
five `sure` corrections (sure-day recall 0.756 → 0.644, Energy IoU 0.902 → 0.792).
`edge` fails every adoption rule on both cohorts.

**Reading against the plan's success criteria.** Neither variant meets the first
condition, a gain in evidence or placement on beta_D's gap-adjacent days, so the other
conditions do not come into play. The question is answered in the negative: for this
statistic the anchor should be a reading the correction does not touch, and moving it
onto the window makes the days it was meant to help less certain. The remaining beta_D
loss stays where round 4 left it — the clipped flank, a second error mode outside a
one-window sign flip — and is not a matter of where the bridge is anchored.

No adoption decision is taken here. The default stays `nearest`; `edge` and `gap_edge`
remain available in `m9_scorer.py` and `m9_eval.py` as documented variants.

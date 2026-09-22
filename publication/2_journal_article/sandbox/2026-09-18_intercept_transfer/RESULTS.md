# Intercept transfer: results

Run of 18 September 2026 (`run.py`, seed 9, 20 draws per cell). Committed Phase 3 scores, labels and energies only; the scorer was not re-run. Observations first, interpretations after.

## 1. Observations

### 1.1 Benchmarks (`curve.csv`, k ≤ 0 rows)

| direction | configuration | Energy IoU | energy precision | sure-day recall | false days |
|---|---|---:|---:|---:|---:|
| Alpha slope → Beta | R0 within-cohort (reference) | 0.895 | 0.941 | 0.896 | 21 |
| Alpha slope → Beta | Alpha slope, intercept from all other Beta labels | 0.894 | 0.938 | 0.900 | 24 |
| Alpha slope → Beta | Alpha pair untouched (k = 0) | 0.870 | 0.881 | 0.987 | 214 |
| Beta slope → Alpha | R0 within-cohort (reference) | 0.869 | 0.889 | 0.905 | 71 |
| Beta slope → Alpha | Beta slope, intercept from all other Alpha labels | 0.871 | 0.892 | 0.882 | 52 |
| Beta slope → Alpha | Beta pair untouched (k = 0) | 0.874 | 0.935 | 0.630 | 2 |

The transferred slopes are not the within-cohort ones (Alpha's slope is 3.24 against Beta's own 2.06 to 2.35, and Beta's 2.16 against Alpha's 3.1 to 3.5), yet with the intercept refitted the reference is recovered in both directions to within 0.003 of energy precision and 0.002 of Energy IoU.

### 1.2 Learning curves, Alpha slope → Beta (`curve_summary.csv`; median and 5th percentile over 20 draws)

| level | sampling | k | Energy IoU | energy precision | precision 5th pct | sure-day recall | false days |
|---|---|---:|---:|---:|---:|---:|---:|
| population | random | 9 | 0.878 | 0.916 | 0.893 | 0.928 | 97 |
| population | random | 18 | 0.874 | 0.921 | 0.902 | 0.910 | 55 |
| population | random | 27 | 0.884 | 0.924 | 0.908 | 0.894 | 49 |
| population | random | 54 | 0.888 | 0.933 | 0.916 | 0.885 | 35 |
| population | random | 108 | 0.884 | 0.933 | 0.921 | 0.893 | 30 |
| population | review design | 9 | 0.892 | 0.932 | 0.925 | 0.909 | 60 |
| population | review design | 27 | 0.887 | 0.911 | 0.903 | 0.941 | 83 |
| population | review design | 108 | 0.888 | 0.920 | 0.914 | 0.923 | 50 |
| station | random | 27 | 0.896 | 0.930 | 0.906 | 0.913 | 39 |
| station | random | 108 | 0.904 | 0.943 | 0.928 | 0.906 | 14 |
| station | review design | 27 | 0.879 | 0.901 | 0.894 | 0.934 | 41 |

Beta slope → Alpha: every cell from k = 9 upward sits at Energy IoU 0.868 to 0.873 and energy precision 0.89 to 0.90; recall climbs from 0.81 at k = 9 to 0.88 at k = 108 (population, random). Alpha is easy to adapt to because its ordinary days carry almost no evidence.

### 1.3 Intercepts (`intercepts.csv`)

- All-labels intercept with the Alpha slope on Beta: −7.06 (range over folds −7.49 to −6.87). With 27 review-design days at population level: median −3.8, 5th to 95th percentile −7.8 to −3.1. With 27 random days the spread is narrower and centred nearer the all-labels value (see `intercepts.csv`).
- Single-class samples (no fit possible, default kept): 30 of 160 station-draws at k = 9 random at population level; none at k ≥ 27.

### 1.4 Per station at 27 review-design days, Alpha slope → Beta (`station_k27.csv`, medians over draws)

| station | population-level precision (R0) | population-level recall (R0) |
|---|---:|---:|
| beta_A | 0.947 (0.988) | 0.962 (0.923) |
| beta_B | 0.941 (0.948) | 0.964 (0.864) |
| beta_D | 0.934 (0.966) | 0.789 (0.756) |
| beta_E | 0.768 (0.911) | 1.000 (1.000) |
| beta_F | 0.932 (0.950) | 0.994 (0.957) |
| beta_G | 0.820 (0.902) | 0.941 (0.894) |
| beta_H | 0.655 (0.818) | 0.500 (0.250) |

## 2. Interpretations

1. The slope transfers; the intercept does not. With the slope fixed from the other population and the intercept fitted on the target's labels, both directions reproduce the within-cohort reference. This is the clean statement of what "generalises" in M9's calibration: the shape of the evidence-to-probability curve, not its position. The position is a property of the population's ordinary days.
2. Safe by default holds in one direction only. The real-data (Beta) pair applied to Alpha untouched gives energy precision 0.935 and two false corrections, at the price of recall (0.63). The Alpha pair applied to Beta untouched gives 214 false corrections. So the deployable default must be the conservative, real-data intercept; then adaptation only ever buys recall.
3. Twenty-seven random reviewed days per population are enough. At population level, random sampling, k = 27 gives median energy precision 0.924 with a 5th percentile of 0.908, above the 0.90 gate in 95% of draws, and Energy IoU 0.884 against the 0.875 target. Nine days are not enough (5th percentile 0.893, and one draw in five has no error day at all). Fifty-four days add a little (0.933, 5th percentile 0.916). This is the number the plan asked for.
4. Per-station adaptation is not needed for the claim, but it helps where labels are plentiful: at 108 random days per station it exceeds the reference (0.943, Energy IoU 0.904) because the station intercept also absorbs the station's prevalence. At 27 days it matches the population level. The population level is the simpler claim and the one to make.
5. The review design as implemented is worse than random beyond 18 days, not better. Its provisional probabilities come from the untouched foreign pair, which on Beta is permissive, so "nearest the threshold" selects days with almost no evidence and "largest corrections" selects days that are certain under any intercept; neither region determines where the Beta curve turns. Designed sampling needs a first estimate of the target intercept to aim at, which random sampling provides. A two-stage design (random, then targeted) is the fix, but random alone already meets the target, so the simpler procedure wins.
6. The weak stations stay weak. beta_E, beta_G and beta_H are below the reference at 27 days in every configuration, as they were in every earlier study; they carry the known limitations (clipped meters, cloudy solar, four RPF days). Adaptation does not create precision the evidence does not contain.

## 3. Against the reading fixed in the plan

- Population-level intercept from 27 review-design days: energy precision 0.911, Energy IoU 0.887, both above the thresholds (0.90, 0.875), but random sampling does better (0.924, 0.884) with a higher 5th percentile.
- Reverse direction: Alpha Energy IoU 0.872 at 27 days against the 0.849 target.
- The untouched foreign pair over-corrects in one direction (Alpha pair on Beta). The safe-default claim therefore attaches to the real-data pair only.

## 4. What this supports, and what it does not

Supported, on two populations from one network: M9's scorer and calibration slope are portable; one intercept per population, estimated from about 27 randomly chosen reviewed days, recovers the within-population operating point; the real-data pair is a safe default before any review. Not supported: universality across environments, any claim about populations beyond Alpha and Beta, and the review-design shortcut as implemented.

## 5. Options for the methodological decision

- A. Adopt the procedure for the release and the paper: scorer and slope frozen (slope from the real-data population), default intercept from the real-data population, one-off refit of the intercept from about 27 random reviewed days per deployment population; state the claim as cross-population portability with minimal labelled calibration.
- B. As A, with a two-stage review design (random first, then targeted) developed and tested before it is offered to Ausgrid.
- C. Do not add the procedure; keep within-cohort calibration and the Beta-8 pair only, and state the limitation.

## 6. Files

`curve.csv`, `curve_summary.csv`, `station_k27.csv`, `intercepts.csv`, `fig_learning_curve.png`, `results_summary.json`, `run.log`.

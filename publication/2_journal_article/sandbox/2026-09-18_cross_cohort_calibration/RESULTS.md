# Cross-cohort calibration for M9: results

Run of 18 September 2026 (`run.py`, seed 9, 437 s). Everything below comes from committed Phase 3 outputs; the scorer was not re-run. Observations first, interpretations after, separated on purpose.

## 1. Checks before trusting anything

- R0 (frozen within-cohort calibration) reproduces `07_metrics/pooled.csv` exactly: Alpha Energy IoU 0.8692, energy precision 0.8886; Beta 0.8945, 0.9407.
- B1 and B2 reproduce the 17 September sandbox to four decimals.

## 2. Observations

### 2.1 Pooled held-out results at c = 0.7, headline days

| candidate | cohort-blind | Beta IoU | Beta energy prec. | Beta sure recall | Beta false days | Beta day prec. | Alpha IoU | Alpha energy prec. | Alpha recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 within-cohort (reference) | no | 0.895 | 0.941 | 0.896 | 21 | 0.952 | 0.869 | 0.889 | 0.905 |
| B1 pooled logistic | yes | 0.880 | 0.893 | 0.977 | 150 | 0.754 | 0.875 | 0.897 | 0.849 |
| B2 pooled, cohorts equally weighted | fit-time weighting | 0.570 | 0.983 | 0.428 | 0 | 1.000 | 0.627 | 0.984 | 0.268 |
| B3 cohort-indicator diagnostic | no | 0.890 | 0.932 | 0.902 | 28 | 0.938 | 0.870 | 0.889 | 0.897 |
| C1 evidence per slot | yes | 0.884 | 0.900 | 0.975 | 184 | 0.713 | 0.882 | 0.905 | 0.872 |
| C2 z and log length | yes | 0.883 | 0.897 | 0.975 | 159 | 0.742 | 0.880 | 0.904 | 0.846 |
| C3 z and margin | yes | 0.874 | 0.886 | 0.987 | 197 | 0.702 | 0.872 | 0.893 | 0.871 |
| C4 null-quantile normalisation | yes | 0.867 | 0.879 | 0.987 | 240 | 0.659 | 0.867 | 0.886 | 0.915 |
| D1 per-station normalisation (transductive diagnostic) | yes | 0.884 | 0.899 | 0.957 | 123 | 0.785 | 0.878 | 0.910 | 0.777 |
| **N nested (inner-selected per fold)** | yes | **0.876** | **0.888** | 0.985 | 197 | 0.702 | 0.882 | 0.905 | 0.872 |

Station-bootstrap 95% intervals (`bootstrap.csv`): Beta energy precision R0 0.913–0.966; C1 0.833–0.935; C2 0.819–0.937; N 0.797–0.933. Alpha Energy IoU R0 0.824–0.884; C1 0.842–0.896.

### 2.2 Nested selection (`selection.csv`)

- Selected candidate per outer fold: C1 in 12 folds (all 10 Alpha folds, beta_C, beta_D); C3 in 6 Beta folds.
- Inner Beta energy-precision gate (0.90) met by at least one candidate in only 3 of the 8 Beta folds (beta_C, beta_D, beta_G). In the other 5 Beta folds no cohort-blind candidate met the gate on the inner held-out Beta stations and the fallback (lowest station-macro log loss) applied. In all 10 Alpha folds the gate was met on the inner Beta stations.
- The nested Beta energy precision is 0.888, below the gate.

### 2.3 Success criteria (point estimates)

| criterion | C1 (fixed) | C2 (fixed) | N (nested) |
|---|---|---|---|
| Beta energy precision ≥ 0.90 | 0.9004, met by 0.0004 | 0.897, not met | 0.888, not met |
| Beta Energy IoU ≥ 0.875 | 0.884, met | 0.883, met | 0.876, met by 0.001 |
| Alpha Energy IoU ≥ 0.849 | 0.882, met | 0.880, met | 0.882, met |
| Sure-day recall ≥ 0.85 in each cohort | Beta 0.975, Alpha 0.872, met | Alpha 0.846, not met | Beta 0.985, Alpha 0.872, met |
| No hidden station failure | Beta stations D, E, H energy precision 0.79, 0.74, 0.66 (R0: 0.97, 0.91, 0.82) | same pattern | beta_H 0.39 |

### 2.4 Reliability (`reliability.csv`)

| candidate | Beta log loss | Beta Brier | Beta ECE | Alpha log loss | Alpha Brier | Alpha ECE |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 0.098 | 0.027 | 0.009 | 0.089 | 0.024 | 0.022 |
| B3 | 0.102 | 0.027 | 0.010 | 0.090 | 0.025 | 0.024 |
| B1 | 0.248 | 0.067 | 0.098 | 0.111 | 0.033 | 0.043 |
| C1 | 0.286 | 0.082 | 0.101 | 0.093 | 0.026 | 0.031 |
| C2 | 0.257 | 0.070 | 0.098 | 0.107 | 0.032 | 0.039 |
| C3 | 0.264 | 0.080 | 0.101 | 0.090 | 0.026 | 0.031 |
| N | 0.276 | 0.081 | 0.101 | 0.093 | 0.026 | 0.031 |

Every cohort-blind candidate has Beta calibration-in-the-large of about +0.10: on Beta it says "error" about ten points more often than errors occur. On Alpha the same candidates are close to R0.

### 2.5 Per station (`metrics_station.csv`)

- False corrections under C1 on Beta: beta_A 47 (R0: 0), beta_D 49 (1), beta_F 35 (9), beta_E 24 (1), beta_G 23 (10). These are mostly small-energy days, which is why energy precision falls by 0.04 while day precision falls by 0.24.
- Sure-day recall under C1 on Beta rises everywhere (beta_D 0.76 → 0.98, beta_B 0.86 → 0.96, beta_H 0.25 → 0.50); on Alpha it falls at every station (alpha_C 0.89 → 0.86, alpha_I 0.87 → 0.82).
- C1's fitted coefficients are stable across folds (intercept −0.13 to +0.20, slope 3.88 to 4.20), so its behaviour is not a fitting artefact.

### 2.6 Diagnostics

- B3 (cohort indicator with a shared slope) nearly recovers R0 on both cohorts (Beta precision 0.932, Alpha unchanged). Knowing the cohort is worth almost the whole gap on its own.
- D1 (per-station null normalisation, transductive) does not beat C1 on Beta precision (0.899) and costs Alpha recall (0.78).

## 3. Interpretations

1. No cohort-blind candidate met the target in the nested result. The inner loop failed the Beta gate in five of eight Beta folds, and the nested outer Beta energy precision (0.888) is below 0.90. That is the honest outcome of the protocol as approved.
2. C1, taken as a fixed candidate, sits exactly on the gate (0.9004) and meets the other point criteria, but its bootstrap interval on Beta precision (0.83–0.94) is wide, three Beta stations fall well below R0, and it corrects 184 Beta days that carry no error against R0's 21. Energy precision survives because those days are small; a reviewer or an operator would not see it that way. It is not a pass in the sense the plan defined.
3. Every cohort-blind mapping over-predicts errors on Beta by about ten points (calibration-in-the-large) while staying honest on Alpha. That is the signature of a distribution shift that the available label-free quantities (evidence, window length, margin, a pooled null scale) do not remove: Alpha's null days and Beta's null days differ in evidence, and none of these covariates separates a Beta non-error day from an error day well enough.
4. B3 shows the gap is almost entirely "which cohort is this", and it recovers with a single intercept shift. That is consistent with the round-5 diagnosis: on real data, some non-error days produce genuine positive evidence under the bridge test. The remaining shift lives in the evidence, not in the calibration.
5. The per-slot form (C1) is nonetheless informative for the paper: it improves Alpha on every metric and matches or beats R0 on combined Energy IoU (0.882 versus 0.875). Its cost is precision on real data. It is a different operating point, not a shared calibration that works.
6. D1 says station-adaptive normalisation does not rescue the shared mapping either; its use would be as a future method idea, not as evidence for this one.

## 4. Where this leaves the question

Under the approved criteria, a single shared calibration on the frozen evidence does not generalise across Alpha and Beta without cohort-specific behaviour. The evidence supports keeping the within-cohort protocol as evaluated, shipping the real-data (Beta) calibration for deployment, and stating the limitation: the M9 evidence scale is population-dependent on its null days, so calibration must be fitted on data like the deployment data. Closing the gap would require a change to the scorer (a revision 3 that reduces positive evidence on smooth-hump non-error days), which is outside this sandbox and a methodological decision for you.

Options on the table, none adopted here:

- A. Keep within-cohort calibration; ship the Beta-8 pair; state the limitation. No change to anything frozen.
- B. Adopt C1's per-slot transform as a documented alternative operating point (better Alpha, worse Beta precision), reported beside R0, not as the default.
- C. Open a revision-3 investigation of the scorer aimed at the Beta null days (the false-correction class), with the understanding that it re-runs the whole evaluation.

## 5. Files

`predictions_outer.parquet` (per site-day, candidate, fold, p, outcome, energies, selected flag); `fits.csv` (coefficients and C4 constants per fold and candidate); `selection.csv` (inner table per outer fold); `metrics_pooled.csv`, `metrics_station.csv`, `reliability.csv`, `bootstrap.csv`; `LEAKAGE_AUDIT.md`; `results_summary.json`.

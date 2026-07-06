# Physics-Guided Structured Window Method for Reverse Power Flow Sign Correction

**Version:** v0.2, VS Code-safe Markdown  
**Purpose:** Specify a physically interpretable RPF sign-correction method with explicit equations, candidate-window generation, feature definitions, scoring, decoding, confidence flags, and validation design.  
**Design goal:** Keep the method physically meaningful, robust, and not overly dependent on hand-tuned thresholds.

This version avoids LaTeX syntax. All equations are written in plain text code blocks so they are readable in standard VS Code Markdown preview without a math-rendering extension.

---

## 1. Problem definition

For each site `i`, day `d`, and interval `t`, observe:

```text
y[i,d,t] = observed net load
S[i,d,t] = estimated behind-the-meter solar generation
```

The time resolution is usually 15 minutes, but the method does not depend on this resolution.

The goal is to decide whether a site-day contains a reverse-power-flow sign error and, if yes, identify the correction window:

```text
W = [s, e]

s = start interval of correction window
e = end interval of correction window
```

The desired outputs are:

```text
rpf_day_label        in {0, 1}
rpf_start_time       = start of correction window, if rpf_day_label = 1
rpf_end_time         = end of correction window, if rpf_day_label = 1
rpf_interval_label_t in {0, 1}
rpf_mechanism        in {reflection_wrong_sign, clipped_deadband,
                         telemetry_anomaly, normal_low_load, unclear}
rpf_confidence       in {sure, unsure}
```

The main binary decision is:

```text
z_hat[i,d] = 1  if day d at site i requires RPF sign correction
z_hat[i,d] = 0  otherwise
```

---

## 2. Physical intuition

Observed net load is approximately:

```text
net load = underlying demand - solar generation
```

Let underlying demand be:

```text
U[i,d,t] = underlying demand
```

If the meter is correctly signed:

```text
y[i,d,t] ≈ U[i,d,t] - S[i,d,t]
```

Therefore:

```text
U[i,d,t] ≈ S[i,d,t] + y[i,d,t]
```

If the meter incorrectly records reverse flow as positive during RPF, then inside the erroneous RPF window:

```text
y[i,d,t] ≈ abs(U[i,d,t] - S[i,d,t])
```

When true net load is negative:

```text
U[i,d,t] - S[i,d,t] < 0
```

then the observed value becomes:

```text
y[i,d,t] ≈ S[i,d,t] - U[i,d,t]
```

Therefore, inside the wrong-sign window:

```text
U[i,d,t] ≈ S[i,d,t] - y[i,d,t]
```

The central question is:

```text
For which candidate window W, if any, does replacing S + y with S - y
inside the window produce a more physically plausible underlying-demand curve?
```

---

## 3. Counterfactual interpretations

For each candidate window `W = [s, e]`, define two reconstructed underlying-demand curves.

### 3.1 No-correction interpretation

Assume no RPF sign error anywhere on the day:

```text
U_F[t] = S[t] + y[t]
```

Here `F` means the normal forward-flow interpretation.

### 3.2 Candidate sign-correction interpretation

Assume a wrong-sign RPF period inside candidate window `W`:

```text
if t is inside W:
    U_R[t; W] = S[t] - y[t]
else:
    U_R[t; W] = S[t] + y[t]
```

Here `R` means the reverse-flow sign-correction interpretation.

The method scores whether `U_R[.; W]` is more physically plausible than `U_F`.

---

## 4. Candidate-window generation

### 4.1 Design principle

Do not rely on local maxima/minima to generate candidate windows. Those rules can miss true RPF days. Instead:

```text
1. Generate many simple physically plausible daytime windows.
2. Score each window using physical features.
3. Select either the best window or no correction.
```

This separates the problem into:

```text
high-recall candidate generation + physically interpretable candidate scoring
```

### 4.2 Candidate set

For each site-day, define:

```text
W_set[i,d] = {NULL} union all windows [s,e] satisfying:

    s and e are daytime intervals
    min_duration <= e - s + 1 <= max_duration
```

`NULL` means no correction.

Recommended physical bounds:

```text
minimum duration = 0.5 hours
maximum duration = 8.0 hours
candidate period = broad daylight / solar-active period
```

For 15-minute data:

```text
L_min = 2 intervals
L_max = 32 intervals
```

### 4.3 Solar-active period

Recommended simple version:

```text
candidate start/end must be inside 06:00–18:00
```

Adaptive alternative:

```text
T_solar = all intervals where S[t] >= eta_S * max(S)

Example: eta_S = 0.05
```

Recommended paper-friendly choice:

```text
Use fixed broad daylight bounds, e.g. 06:00–18:00.
```

This is easy to explain and avoids one extra solar threshold.

### 4.4 Solar-peak information

Let:

```text
t_peak_S = interval t where S[t] is maximum
```

You have two choices.

Strict option:

```text
Keep only windows where t_peak_S is inside or near W.
```

Cleaner option:

```text
Generate all broad daylight windows and include solar_peak_distance as a feature.
```

Recommended version:

```text
Do not use solar peak as a hard rule unless computation is too expensive.
Use solar_peak_distance as a feature instead.
```

### 4.5 Missing values

Missing values should not automatically reject the entire day.

Recommended handling:

```text
1. Interpolate short gaps for candidate generation and feature calculation only.
2. Retain missingness indicators as features.
3. Do not fabricate corrected operational output at originally missing timestamps.
```

Missingness indicator:

```text
m[t] = 1 if y[t] is missing
m[t] = 0 if y[t] is observed
```

Candidate missingness feature:

```text
f_miss(W) = mean of m[t] over t inside W
          = sum(m[t] for t in W) / length(W)
```

---

## 5. Candidate-level physical features

For each candidate `W = [s, e]`, compute features that compare `U_F` and `U_R[.; W]`.

The notation below omits site/day subscripts for readability.

---

## 5.1 Basic candidate geometry features

Candidate duration in intervals:

```text
f_dur(W) = e - s + 1
```

Candidate duration in hours:

```text
f_dur_h(W) = delta_t_hours * (e - s + 1)

For 15-minute data:
delta_t_hours = 0.25
```

Start, end, and midpoint:

```text
f_start(W) = hour_of_day(s)
f_end(W)   = hour_of_day(e)
f_mid(W)   = (hour_of_day(s) + hour_of_day(e)) / 2
```

Distance from solar peak:

```text
f_peakdist(W) = minimum absolute distance between t_peak_S and any t inside W
```

Solar mean inside the candidate:

```text
f_solar_mean(W) = mean(S[t] for t in W)
```

Solar fraction inside the candidate:

```text
f_solar_frac(W) = sum(S[t] for t in W) / (sum(S[t] for t in whole day) + eps)
```

---

## 5.2 Bridge plausibility features

### Intuition

Underlying demand should usually be smoother than the raw wrong-sign net-load shape. If sign correction is correct, `U_R[.; W]` should connect plausibly between the pre-window and post-window underlying-demand levels.

### Anchor windows

Define pre-window and post-window context sets:

```text
B_L(W) = intervals {s-K, ..., s-1}
B_R(W) = intervals {e+1, ..., e+K}
```

Recommended default:

```text
K = 4 intervals = 1 hour for 15-minute data
```

Use robust anchor levels:

```text
a_L(W) = median(U_F[t] for t in B_L(W))
a_R(W) = median(U_F[t] for t in B_R(W))
```

The anchors use `U_F` because outside the candidate is assumed correctly signed.

### Linear bridge

For each `t` inside `W`, define a straight-line bridge between the two anchors:

```text
B[t; W] = a_L(W) + ((t - s + 1) / (e - s + 2)) * (a_R(W) - a_L(W))
```

### Bridge residuals

No-correction bridge residual:

```text
R_F(W) = median(abs(U_F[t] - B[t; W]) for t in W)
```

Correction bridge residual:

```text
R_R(W) = median(abs(U_R[t; W] - B[t; W]) for t in W)
```

Bridge improvement:

```text
f_bridge_imp(W) = (R_F(W) - R_R(W)) / (R_F(W) + R_R(W) + eps)
```

Interpretation:

```text
positive value  -> correction improves bridge plausibility
near zero       -> correction does not materially help
negative value  -> correction makes the curve less plausible
```

Optional robust-scaled version:

```text
f_bridge_z(W) = (R_F(W) - R_R(W)) / (sigma_U_site + eps)
```

where `sigma_U_site` can be a robust site-level scale, such as the median absolute deviation of daytime `U_F`.

---

## 5.3 Total-variation improvement

### Intuition

Wrong-sign reflection can create artificial bumps or cusps. Correcting the sign should reduce excessive variation in reconstructed underlying demand.

Define total variation of a series `x` over interval set `A`:

```text
TV(x; A) = sum(abs(x[t] - x[t-1]) for t in A if t > min(A))
```

Define local context around the candidate:

```text
C(W) = B_L(W) union W union B_R(W)
```

No-correction total variation:

```text
TV_F(W) = TV(U_F; C(W))
```

Correction total variation:

```text
TV_R(W) = TV(U_R[.; W]; C(W))
```

Total-variation improvement:

```text
f_tv_imp(W) = (TV_F(W) - TV_R(W)) / (TV_F(W) + TV_R(W) + eps)
```

---

## 5.4 Curvature improvement

### Intuition

A reflected sign-error window often creates artificial curvature. Correcting the sign should reduce second-difference magnitude.

First difference:

```text
dx[t] = x[t] - x[t-1]
```

Second difference:

```text
d2x[t] = x[t] - 2*x[t-1] + x[t-2]
```

Curvature over interval set `A`:

```text
Curv(x; A) = sum(abs(d2x[t]) for t in A where t-2 exists)
```

No-correction curvature:

```text
Curv_F(W) = Curv(U_F; C(W))
```

Correction curvature:

```text
Curv_R(W) = Curv(U_R[.; W]; C(W))
```

Curvature improvement:

```text
f_curv_imp(W) = (Curv_F(W) - Curv_R(W)) / (Curv_F(W) + Curv_R(W) + eps)
```

---

## 5.5 Boundary continuity features

### Intuition

The corrected underlying-demand curve should not have large discontinuities at the start and end of the correction window.

Left-boundary level discontinuity after correction:

```text
D_R_L(W) = abs(U_R[s; W] - U_F[s-1])
```

Right-boundary level discontinuity after correction:

```text
D_R_R(W) = abs(U_F[e+1] - U_R[e; W])
```

Corresponding no-correction discontinuities:

```text
D_F_L(W) = abs(U_F[s]   - U_F[s-1])
D_F_R(W) = abs(U_F[e+1] - U_F[e])
```

Boundary-level improvement:

```text
numerator   = (D_F_L(W) + D_F_R(W)) - (D_R_L(W) + D_R_R(W))
denominator = D_F_L(W) + D_F_R(W) + D_R_L(W) + D_R_R(W) + eps

f_bdry_level_imp(W) = numerator / denominator
```

---

## 5.6 Boundary reflection features

### Intuition

If the observed curve is an absolute-value reflection, the slope immediately inside the RPF window tends to be reflected relative to the slope immediately outside the window. After sign correction, the slope should become more continuous.

For the left boundary:

```text
a_L = y[s]   - y[s-1]      # slope before/at left boundary
b_L = y[s+1] - y[s]        # slope inside left boundary
```

Raw slope discontinuity:

```text
D_raw_slope_L = abs(a_L - b_L)
```

Reflected slope discontinuity:

```text
D_refl_slope_L = abs(a_L + b_L)
```

Left boundary reflection score:

```text
r_L(W) = (D_raw_slope_L - D_refl_slope_L) / (abs(a_L) + abs(b_L) + eps)
```

For the right boundary:

```text
a_R = y[e]   - y[e-1]
b_R = y[e+1] - y[e]

r_R(W) = (abs(a_R - b_R) - abs(a_R + b_R)) / (abs(a_R) + abs(b_R) + eps)
```

Conservative two-boundary reflection feature:

```text
f_refl_min(W) = min(r_L(W), r_R(W))
```

Average reflection feature:

```text
f_refl_mean(W) = (r_L(W) + r_R(W)) / 2
```

Interpretation:

```text
large positive -> sign reflection explains both boundaries well
near zero      -> weak reflection evidence
negative       -> sign reflection worsens boundary slope continuity
```

---

## 5.7 Solar and net-load co-movement features

### Intuition

In a wrong-sign reflected RPF period, the recorded positive net load can move with solar generation because export magnitude is being recorded as positive.

Inside candidate `W`, compute:

```text
f_corr_S_y(W) = correlation(S[t], y[t]) over t inside W
```

Derivative correlation:

```text
f_corr_dS_dy(W) = correlation(S[t] - S[t-1], y[t] - y[t-1]) over t inside W
```

Same-sign derivative fraction:

```text
f_same_deriv(W) = mean( sign(S[t] - S[t-1]) == sign(y[t] - y[t-1])
                        for t inside W and t > s )
```

These should generally be higher for reflected wrong-sign windows than for normal low-load days.

---

## 5.8 Pseudo-load stability features

Inside a candidate RPF window, the corrected underlying demand is:

```text
P[t; W] = S[t] - y[t]
```

This can be interpreted as pseudo underlying load inside the candidate.

Candidate pseudo-load mean:

```text
f_pseudo_mean(W) = mean(P[t; W] for t in W)
```

Pseudo-load standard deviation:

```text
f_pseudo_sd(W) = standard_deviation(P[t; W] for t in W)
```

Pseudo-load coefficient of variation:

```text
f_pseudo_cv(W) = f_pseudo_sd(W) / (abs(f_pseudo_mean(W)) + eps)
```

Fraction of negative pseudo-load:

```text
f_pseudo_negfrac(W) = mean(P[t; W] < 0 for t in W)
```

Physical interpretation:

```text
Very negative pseudo-load may indicate that sign correction is physically implausible,
unless solar generation estimates are materially biased.
```

---

## 5.9 Prominence and shape features

### Raw reflected-bump prominence

Boundary baseline:

```text
b_y(W) = (median(y[t] for t in B_L(W)) + median(y[t] for t in B_R(W))) / 2
```

Candidate raw maximum:

```text
y_max(W) = max(y[t] for t in W)
```

Raw bump prominence:

```text
f_prom_raw(W) = y_max(W) - b_y(W)
```

Scaled raw prominence:

```text
f_prom_raw_z(W) = (y_max(W) - b_y(W)) / (sigma_y_site + eps)
```

where `sigma_y_site` is a robust site-level scale.

### Corrected valley depth

Corrected net load inside `W`:

```text
if t is inside W:
    y_tilde[t; W] = -y[t]
else:
    y_tilde[t; W] = y[t]
```

Corrected minimum:

```text
y_tilde_min(W) = min(y_tilde[t; W] for t in W)
```

Corrected valley depth:

```text
f_valley(W) = b_y(W) - y_tilde_min(W)
```

---

## 5.10 Site-normalised features

Raw MW features are not directly comparable across substations. For a raw candidate feature `f(W)`, define a site-normalised z-like feature:

```text
f_z[i,d](W) = (f[i,d](W) - median_historical_f[i]) / (MAD_historical_f[i] + eps)
```

where:

```text
median_historical_f[i] = median of the same feature over historical days at site i
MAD_historical_f[i]    = median absolute deviation of the same feature over historical days at site i
```

A simpler and robust alternative is within-site percentile rank:

```text
f_rank[i,d](W) = fraction of historical days d' where f[i,d'] <= f[i,d](W)
```

Recommendation:

```text
Use site-normalised ranks for operational Beta-like deployment.
They are less sensitive to scale mismatch and solar-estimation bias.
```

---

## 5.11 Temporal-context features

RPF sign errors often persist across neighbouring days. A physically plausible isolated one-day event is less convincing than a high-scoring day inside a coherent period.

First compute a preliminary day score:

```text
q[i,d] = max(Score(W) for all non-null W on site i, day d)
```

Then compute rolling context features:

```text
N_k(d) = neighbouring days from d-k to d-1 and d+1 to d+k

f_rollmean_k(i,d) = mean(q[i,d'] for d' in N_k(d))
```

Example window sizes:

```text
k = 7 days, 15 days, 21 days
```

Recommendation:

```text
Use temporal context as a feature, not as many separate hand-written rules.
```

---

## 6. Candidate scoring

The cleanest implementation is a low-parameter candidate-ranking or candidate-scoring model.

### 6.1 Compact feature vector

For each candidate `W`, define:

```text
phi(W) = [
    f_bridge_imp(W),
    f_tv_imp(W),
    f_curv_imp(W),
    f_bdry_level_imp(W),
    f_refl_min(W),
    f_corr_S_y(W),
    f_corr_dS_dy(W),
    f_pseudo_negfrac(W),
    f_prom_raw_z(W),
    f_peakdist(W),
    f_miss(W)
]
```

This is the recommended compact feature set.

### 6.2 Linear physical score

A simple score is:

```text
Score(W) = beta_0 + beta_1*f_bridge_imp(W)
                  + beta_2*f_tv_imp(W)
                  + beta_3*f_curv_imp(W)
                  + beta_4*f_bdry_level_imp(W)
                  + beta_5*f_refl_min(W)
                  + beta_6*f_corr_S_y(W)
                  + beta_7*f_corr_dS_dy(W)
                  + beta_8*f_pseudo_negfrac(W)
                  + beta_9*f_prom_raw_z(W)
                  + beta_10*f_peakdist(W)
                  + beta_11*f_miss(W)
```

where the `beta` coefficients are learned from training/validation data.

To keep the method interpretable:

```text
1. Use a linear model, logistic regression, or shallow gradient-boosted tree.
2. Apply strong regularisation.
3. Prefer monotonic constraints where appropriate.
4. Avoid site-specific thresholds in the main result.
```

### 6.3 Expected feature directions

| Feature | Expected direction for RPF sign error |
|---|---:|
| `f_bridge_imp` | positive |
| `f_tv_imp` | positive |
| `f_curv_imp` | positive |
| `f_bdry_level_imp` | positive |
| `f_refl_min` | positive |
| `f_corr_S_y` | positive |
| `f_corr_dS_dy` | positive |
| `f_pseudo_negfrac` | negative |
| `f_miss` | negative or uncertain |
| `f_peakdist` | negative |

### 6.4 Null candidate score

The null candidate `NULL` represents no correction.

Recommended simple option:

```text
Score(NULL) = 0
```

Then a non-null window is selected only if it provides positive evidence relative to no correction.

More complex option:

```text
Score(NULL) = gamma_0 + gamma^T * day_level_non_RPF_features
```

But this adds complexity. The recommended initial version is:

```text
Score(NULL) = 0
```

with one calibrated decision threshold.

---

## 7. Day-level decoding

For each site-day, select the best non-null candidate:

```text
W_star = argmax Score(W) over all non-null candidate windows W
```

Define the day score:

```text
Q = Score(W_star) - Score(NULL)
```

With `Score(NULL) = 0`:

```text
Q = Score(W_star)
```

Day-level prediction:

```text
z_hat = 1 if Q >= tau
z_hat = 0 if Q < tau
```

where `tau` is a single threshold selected on validation data.

If `z_hat = 1`, predict interval labels:

```text
r_hat[t] = 1 if t is inside W_star
r_hat[t] = 0 otherwise
```

If `z_hat = 0`:

```text
r_hat[t] = 0 for all t
```

---

## 8. Corrected net-load output

If the predicted mechanism is `reflection_wrong_sign`, corrected net load is:

```text
if r_hat[t] = 1:
    y_corrected[t] = -y[t]
else:
    y_corrected[t] = y[t]
```

For `clipped_deadband`, do not automatically apply simple sign flip as the final corrected value. The magnitude may have been lost.

Recommended output for clipped/deadband:

```text
flag as RPF-censored
```

Optional reconstruction:

```text
y_corrected[t] = U_hat[t] - S[t]
```

where `U_hat[t]` is a reconstructed underlying-demand estimate from a bridge or historical profile.

This should be reported separately from pure sign correction.

---

## 9. Mechanism classification

The binary RPF label and mechanism should be separated.

### 9.1 reflection_wrong_sign

Use this when the day looks like absolute-value sign loss:

```text
y[t] ≈ abs(U[t] - S[t])
```

Typical evidence:

```text
- Clear positive midday reflected bump.
- Near-zero boundaries around the event.
- Strong boundary reflection score.
- Sign flip produces smooth underlying demand.
- Solar and raw net load move together inside the window.
```

Operational action:

```text
Apply sign correction: y[t] -> -y[t] inside the window.
```

### 9.2 clipped_deadband

Use this when reverse flow is likely but the magnitude appears censored:

```text
y[t] ≈ max(U[t] - S[t], 0)
```

Typical evidence:

```text
- Solar is high.
- Raw net load is flat or near zero for a long midday period.
- No clear positive reflected bump.
- Simple sign flip does not recover a plausible magnitude.
```

Operational action:

```text
Flag as RPF-censored.
Do not treat as ordinary sign flip unless reconstruction is explicitly defined.
```

### 9.3 telemetry_anomaly

Use this when the day is dominated by data quality issues:

```text
- sudden spike/drop;
- flatline;
- missing block;
- impossible night-solar effect;
- discontinuity unrelated to solar;
- shape inconsistent with either reflection or clipping.
```

Operational action:

```text
Do not auto-correct.
Send to manual review or data-quality pipeline.
```

### 9.4 normal_low_load

Use this when the day has low net load but no convincing sign-error evidence:

```text
- Smooth positive midday valley.
- No reflected bump.
- No clear near-zero clipping plateau.
- Correction does not improve underlying-load plausibility.
```

Operational action:

```text
No correction.
```

### 9.5 unclear

Use this when the reviewer or model cannot confidently distinguish mechanisms:

```text
- Weak boundaries.
- Corrected and uncorrected curves both plausible.
- Small midday bump only.
- Conflict between features.
```

Operational action:

```text
Set confidence = unsure.
Use conservative handling in primary evaluation.
```

---

## 10. Confidence flag

The confidence flag should indicate annotation certainty, not model probability alone.

Suggested manual rule:

```text
confidence = sure
    if the reviewer can clearly assign the day to one mechanism
    and the correction/no-correction decision is visually defensible.

confidence = unsure
    if the day is borderline, mechanism is unclear,
    or physical and telemetry evidence conflict.
```

For model outputs, a score-margin rule can support confidence:

```text
margin = abs(Q - tau)

high-confidence model prediction if margin >= delta_conf
low-confidence model prediction  if margin <  delta_conf
```

For labelled Beta review, store manual confidence independently from model confidence.

---

## 11. Training objective options

### 11.1 Day-level binary training

For each day, use the best candidate feature vector:

```text
phi_star[d] = phi(W_star[d])
```

Train a day classifier:

```text
P(z[d] = 1 | phi_star[d]) = sigmoid(beta_0 + beta^T * phi_star[d])
```

This is simple but depends on how `W_star[d]` is chosen during training.

### 11.2 Candidate-level ranking training

For each day, assign a relevance score to every candidate based on overlap with the reference RPF interval.

If the true window is `W_true`, candidate IoU is:

```text
IoU(W, W_true) = length(intersection(W, W_true)) / length(union(W, W_true))
```

Candidate relevance for RPF days:

```text
rel(W) = 4 if IoU(W, W_true) >= 0.85
rel(W) = 3 if 0.70 <= IoU(W, W_true) < 0.85
rel(W) = 2 if 0.50 <= IoU(W, W_true) < 0.70
rel(W) = 1 if 0.25 <= IoU(W, W_true) < 0.50
rel(W) = 0 if IoU(W, W_true) < 0.25
```

For non-RPF days:

```text
rel(NULL) = 4
rel(W)    = 0 for all non-null W
```

This avoids treating near-correct candidate windows as completely negative.

Recommended objective:

```text
Use candidate ranking if interval labels are trusted.
Use day-level classifier if only day labels are trusted.
```

### 11.3 Minimal-parameter recommendation

Paper-friendly version:

```text
1. Generate dense candidates.
2. Compute compact physical feature vector.
3. Train regularised logistic regression or shallow LightGBM.
4. Select one global threshold using validation F1.
5. Decode one window or no window.
```

---

## 12. Threshold selection

Select `tau` on validation data only.

Primary objective:

```text
tau_star = threshold that maximises F1 on validation data
```

If false corrections are operationally costly, use a precision constraint:

```text
tau_star = threshold that maximises recall on validation data
           subject to precision >= P_min
```

or:

```text
tau_star = threshold that maximises F1 on validation data
           subject to precision >= P_min
```

Recommended:

```text
Use one global threshold for the main paper result.
Do not tune thresholds separately for each Beta site.
```

---

## 13. Evaluation design

### 13.1 Alpha spatiotemporal validation

Use held-out sites and held-out later time periods.

Example:

```text
Outer folds:
    hold out complete substations for testing
    test period = later year

Training:
    different substations
    earlier period

Validation:
    different substations and/or earlier validation period
```

This tests generalisation across:

```text
1. unseen substations; and
2. future dates.
```

### 13.2 Beta validation

For actual manually reviewed Beta sites, report multiple metrics:

```text
Metric A: conservative-all
    Treat unsure as non-correction.

Metric B: sure-only
    Exclude unsure days from evaluation.

Metric C: abstention-aware
    Allow the model to flag uncertain/manual-review cases.
```

This matters because Beta labels are manually reviewed and may contain visually ambiguous cases.

### 13.3 Metrics

Day-level precision:

```text
Precision_day = TP_day / (TP_day + FP_day)
```

Day-level recall:

```text
Recall_day = TP_day / (TP_day + FN_day)
```

Day-level F1:

```text
F1_day = 2 * Precision_day * Recall_day / (Precision_day + Recall_day)
```

Interval-level precision:

```text
Precision_int = TP_int / (TP_int + FP_int)
```

Interval-level recall:

```text
Recall_int = TP_int / (TP_int + FN_int)
```

Interval-level F1:

```text
F1_int = 2 * Precision_int * Recall_int / (Precision_int + Recall_int)
```

Boundary error:

```text
MAE_start = mean(abs(s_hat[d] - s_true[d]) for positive days)
MAE_end   = mean(abs(e_hat[d] - e_true[d]) for positive days)
```

Window IoU:

```text
IoU_day = length(intersection(W_hat[d], W_true[d])) / length(union(W_hat[d], W_true[d]))
```

---

## 14. Recommended compact feature set

The full method can compute many features, but the first paper-friendly version should use a compact set.

Recommended MVP features:

```text
1. bridge_improvement
2. total_variation_improvement
3. curvature_improvement
4. boundary_level_improvement
5. boundary_reflection_min
6. solar_netload_correlation_inside_window
7. solar_netload_derivative_correlation_inside_window
8. pseudo_load_negative_fraction
9. raw_bump_prominence_site_normalised
10. solar_peak_distance
11. missing_fraction_inside_window
12. site_rank_of_best_counterfactual_score
13. rolling_mean_best_score_15d
```

If the goal is maximum interpretability, start with only the first eight.

If the goal is better Beta transfer, add site rank and rolling temporal context.

---

## 15. Pseudocode

```python
for site in sites:
    for day in days:
        y = observed_net_load[site, day]
        S = estimated_solar[site, day]

        # Step 1: interpolate short gaps for feature calculation only
        y_feat, missing_mask = interpolate_short_gaps(y)

        # Step 2: generate dense daytime candidate windows
        candidates = generate_windows(
            start_time="06:00",
            end_time="18:00",
            min_duration_hours=0.5,
            max_duration_hours=8.0,
        )
        candidates.append(NULL_WINDOW)

        best_score = score_null
        best_window = NULL_WINDOW
        best_features = None

        # Step 3: score every candidate window
        for W in candidates:
            if W is NULL_WINDOW:
                continue

            U_forward = S + y_feat
            U_corrected = S + y_feat
            U_corrected[W] = S[W] - y_feat[W]

            features = compute_physical_features(
                y=y_feat,
                S=S,
                U_forward=U_forward,
                U_corrected=U_corrected,
                W=W,
                missing_mask=missing_mask,
            )

            score = model_score(features)

            if score > best_score:
                best_score = score
                best_window = W
                best_features = features

        # Step 4: day-level decision
        if best_score >= threshold:
            rpf_day_label = 1
            rpf_interval_label = intervals_inside(best_window)
        else:
            rpf_day_label = 0
            rpf_interval_label = all_zero

        # Step 5: mechanism classification
        rpf_mechanism = classify_mechanism(
            best_features,
            y=y_feat,
            S=S,
            W=best_window,
            rpf_day_label=rpf_day_label,
        )

        # Step 6: confidence flag
        confidence = classify_confidence(
            score=best_score,
            threshold=threshold,
            mechanism=rpf_mechanism,
            feature_conflicts=check_feature_conflicts(best_features),
        )
```

---

## 16. Main differences from previous m7/m8 style

| Previous approach | New method |
|---|---|
| Candidate generation depends on local maxima/minima | Dense physically bounded daytime windows |
| Interval-level independent prediction | Structured one-window decoding |
| Learns shape mostly from simulated Alpha morphology | Uses explicit counterfactual physical reconstruction |
| Candidate labels are hard positive/negative | Near-correct windows can be ranked by IoU |
| Missing value can remove a whole day | Missingness is handled and represented as a feature |
| Many Beta errors caused by candidate ceiling | Candidate recall becomes high; scoring becomes the main task |
| All RPF treated as one mechanism | Separates reflection, clipping/deadband, telemetry anomaly, normal low load, unclear |

---

## 17. Suggested method name

Possible names:

```text
Structured Counterfactual RPF Correction (SCRC)
Physics-Guided Counterfactual Window Correction (PCWC)
Structured Counterfactual Window Inference (SCWI)
Reverse-Flow Counterfactual Window Model (RFCW)
```

Recommended name:

```text
Physics-Guided Counterfactual Window Correction (PCWC)
```

---

## 18. Suggested paper wording

Draft method description:

> We formulate RPF sign correction as a structured counterfactual window-inference problem. For each site-day, the method enumerates physically plausible daytime candidate windows and compares two reconstructions of underlying demand: a no-correction interpretation, `U_F[t] = S[t] + y[t]`, and a sign-corrected interpretation, `U_R[t] = S[t] - y[t]` inside the candidate window. Candidate windows are scored by whether the correction improves bridge consistency, smoothness, boundary reflection, and solar-net-load co-movement. The final decoder selects either no correction or one continuous correction window. This formulation avoids independent interval predictions and directly encodes the physical expectation that RPF sign errors appear as contiguous solar-centred events.

---

## 19. Open design choices for later iteration

```text
1. Use fixed 06:00–18:00 candidate bounds or adaptive solar-active bounds?
2. Use linear logistic model or shallow LightGBM for candidate score?
3. Train using day labels only or candidate IoU ranking?
4. Treat clipped_deadband as positive RPF day, separate censored class, or excluded sensitivity case?
5. Report primary Beta metric as conservative-all or sure-only?
6. Use site-normalised rank features in the main model or only as an operational-data calibration layer?
7. Should temporal context be included in the primary model or only as post-processing sensitivity?
```

---

## 20. Recommended v1 implementation plan

### v1.0 clean paper model

```text
Candidate generation:
    dense windows, 06:00–18:00, 0.5–8 hours

Features:
    bridge improvement
    total-variation improvement
    curvature improvement
    boundary-level improvement
    boundary-reflection minimum
    solar-net-load correlation
    derivative correlation
    pseudo-load negative fraction
    missing fraction
    solar peak distance

Model:
    regularised logistic regression or shallow LightGBM

Decoder:
    one window or no window

Threshold:
    one validation-calibrated global threshold
```

### v1.1 operational Beta calibration

Add:

```text
site-normalised feature ranks
15-day rolling physical score
mechanism classification
confidence flag
```

### v1.2 optional extension

Add a separate clipped/deadband detector:

```text
near-zero plateau duration under high solar
solar strength relative to historical underlying demand
flat valley score
```

Report this separately from pure sign-flip correction.

---

## 21. Minimal version if reviewer asks for fewer parameters

If the method still feels too parameter-heavy, use this minimal version:

```text
Candidate windows:
    06:00–18:00, duration 0.5–8 hours

Features:
    bridge_improvement
    total_variation_improvement
    boundary_reflection_min
    solar_netload_correlation
    pseudo_load_negative_fraction
    missing_fraction

Model:
    regularised logistic regression

Decision:
    one global validation threshold

Decoder:
    best one-window correction or no correction
```

This gives a clean physical story with very few moving parts.

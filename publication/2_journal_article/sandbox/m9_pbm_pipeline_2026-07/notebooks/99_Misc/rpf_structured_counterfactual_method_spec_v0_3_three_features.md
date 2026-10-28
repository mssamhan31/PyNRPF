# Physics-Guided Structured Window Method for RPF Sign Correction

**Version:** v0.3 three-feature main method, VS Code-safe Markdown  
**Purpose:** Specify a simple, physically interpretable reverse-power-flow (RPF) sign-correction method using only three core features.  
**Design goal:** Avoid many hand-tuned rules and thresholds while preserving the key physical logic of the correction task.

This file intentionally uses plain-text equations inside code blocks so it can be read in VS Code without a LaTeX extension.

---

## 1. Core idea

For each site-day, do not classify each 15-minute interval independently. Instead, treat the day as one structured decision:

```text
Either:
    no RPF sign-correction window

Or:
    one contiguous RPF sign-correction window W = [s, e]
```

For each physically plausible candidate window, reconstruct the underlying demand in two ways:

```text
No-correction interpretation:
    U_no[t] = S[t] + y[t]

Correction interpretation for candidate window W:
    U_corr[t; W] = S[t] - y[t]    if t is inside W
                   S[t] + y[t]    otherwise
```

where:

```text
y[t] = measured net load, currently recorded as positive
S[t] = estimated behind-the-meter solar generation
U[t] = reconstructed underlying demand
```

Then ask:

```text
Does U_corr[t; W] look more physically plausible than U_no[t]?
```

The main method uses only three physical checks:

```text
1. Bridge improvement
2. Roughness improvement
3. Slope-continuity improvement
```

Everything else is treated as backlog or future extension.

---

## 2. Inputs and outputs

### 2.1 Inputs

For site `i`, day `d`, and interval `t`:

```text
y[i,d,t] = observed net load
S[i,d,t] = estimated solar generation
```

Typical resolution:

```text
15 minutes
```

but the method can be adapted to other resolutions.

### 2.2 Outputs

Day-level output:

```text
rpf_day_label = 1 if the day requires RPF sign correction
rpf_day_label = 0 otherwise
```

Window output:

```text
rpf_start_time = start of selected correction window, if rpf_day_label = 1
rpf_end_time   = end of selected correction window, if rpf_day_label = 1
```

Interval output:

```text
rpf_interval_label[t] = 1 if t is inside selected correction window
rpf_interval_label[t] = 0 otherwise
```

Optional reviewer output:

```text
rpf_confidence in {sure, unsure}
rpf_mechanism  in {reflection_wrong_sign, clipped_deadband,
                   telemetry_anomaly, normal_low_load, unclear}
```

---

## 3. Candidate-window generation

### 3.1 Candidate set

For each site-day, generate candidate windows:

```text
W = [s, e]
```

where:

```text
s = candidate start interval
e = candidate end interval
```

Use broad physical constraints only:

```text
1. s and e must be inside broad daylight period, e.g. 06:00 to 18:00.
2. Window duration must be between 0.5 and 8 hours.
3. Window must be contiguous.
```

For 15-minute data:

```text
minimum duration = 0.5 hours = 2 intervals
maximum duration = 8.0 hours = 32 intervals
```

So:

```text
2 <= e - s + 1 <= 32
```

The candidate set is:

```text
CandidateSet = {NULL} union {all plausible daytime windows W = [s, e]}
```

where:

```text
NULL = no correction
```

### 3.2 Why dense candidate windows?

The aim is high recall. We do not want to miss true RPF days because a local-peak or local-minimum rule failed.

The method therefore checks many plausible windows and lets the physical score decide which one is best.

---

## 4. Counterfactual reconstruction

For each non-null candidate window `W = [s, e]`, define two reconstructed underlying-demand curves.

### 4.1 No-correction reconstruction

```text
U_no[t] = S[t] + y[t]
```

This assumes the measured net load is correctly signed for the whole day.

### 4.2 Candidate-correction reconstruction

```text
U_corr[t; W] = S[t] - y[t]    if s <= t <= e
               S[t] + y[t]    otherwise
```

This assumes that only the candidate window has wrong-sign RPF.

---

## 5. Three core physical features

For every candidate window `W`, compute three improvement features.

Each feature compares:

```text
U_no
```

against:

```text
U_corr[.; W]
```

The feature should be positive when correction makes the reconstructed underlying-demand curve more physically plausible.

---

## 5.1 Feature 1: Bridge improvement

### 5.1.1 Intuition

Underlying demand should usually connect smoothly from the period before RPF to the period after RPF.

If the candidate correction is correct, the corrected curve inside the window should look like a plausible bridge between the pre-window and post-window demand levels.

### 5.1.2 Context windows

Use a small shoulder before and after the candidate window.

Let:

```text
K = number of context intervals
```

Recommended default for 15-minute data:

```text
K = 4 intervals = 1 hour
```

Define:

```text
LeftContext(W)  = {s-K, ..., s-1}
RightContext(W) = {e+1, ..., e+K}
```

If context intervals go outside the day, either:

```text
1. skip that candidate, or
2. use available context only if at least half the context exists.
```

Recommended simple choice:

```text
Skip candidates without enough left and right context.
```

### 5.1.3 Anchor levels

Use the median underlying-demand level before and after the candidate window.

Because the context is outside the candidate, use the no-correction reconstruction:

```text
left_anchor(W)  = median(U_no[t] for t in LeftContext(W))
right_anchor(W) = median(U_no[t] for t in RightContext(W))
```

Median is preferred over mean because it is robust to spikes.

### 5.1.4 Linear bridge

For each interval `t` inside `W`, construct a straight-line bridge between the two anchors.

Let:

```text
L = e - s + 1
j = t - s + 1
```

where:

```text
j = 1, 2, ..., L
```

Define:

```text
bridge[t; W] = left_anchor(W)
               + (j / (L + 1)) * (right_anchor(W) - left_anchor(W))
```

This avoids forcing the bridge to equal the anchors exactly at the first and last interval inside the window.

### 5.1.5 Bridge error

No-correction bridge error:

```text
bridge_error_no(W)
= median(abs(U_no[t] - bridge[t; W]) for t inside W)
```

Correction bridge error:

```text
bridge_error_corr(W)
= median(abs(U_corr[t; W] - bridge[t; W]) for t inside W)
```

### 5.1.6 Bridge improvement

Use a scale-free improvement score:

```text
bridge_improvement(W)
= (bridge_error_no(W) - bridge_error_corr(W))
  / (bridge_error_no(W) + bridge_error_corr(W) + eps)
```

where:

```text
eps = small positive number to avoid division by zero
```

Interpretation:

```text
positive value  -> correction improves bridge plausibility
zero            -> correction and no correction are similar
negative value  -> correction makes bridge plausibility worse
```

---

## 5.2 Feature 2: Roughness improvement

### 5.2.1 Intuition

A wrong-sign reflection often creates an artificial bump or cusp in the reconstructed underlying demand.

If the candidate correction is correct, the corrected underlying-demand curve should become smoother around the candidate window.

### 5.2.2 Expanded local region

Compute roughness over the candidate plus its shoulders:

```text
LocalRegion(W) = LeftContext(W) union W union RightContext(W)
```

This is better than using only `W`, because boundary jumps matter.

### 5.2.3 Roughness definition

Use mean absolute first difference.

For a series `x[t]` and interval set `A`:

```text
roughness(x; A)
= mean(abs(x[t] - x[t-1]) for consecutive intervals t-1 and t inside A)
```

Mean is preferred over sum so longer windows are not automatically penalised.

### 5.2.4 Roughness improvement

No-correction roughness:

```text
roughness_no(W) = roughness(U_no; LocalRegion(W))
```

Correction roughness:

```text
roughness_corr(W) = roughness(U_corr[.; W]; LocalRegion(W))
```

Improvement:

```text
roughness_improvement(W)
= (roughness_no(W) - roughness_corr(W))
  / (roughness_no(W) + roughness_corr(W) + eps)
```

Interpretation:

```text
positive value  -> correction makes the curve smoother
zero            -> correction does not change roughness materially
negative value  -> correction makes the curve rougher
```

---

## 5.3 Feature 3: Slope-continuity improvement

### 5.3.1 Intuition

At a true sign-error boundary, the raw curve often has a kink caused by sign reflection.

After applying sign correction inside the window, the slope before and after each boundary should become more continuous.

This feature checks both boundaries:

```text
left boundary  = transition into W at s
right boundary = transition out of W at e
```

### 5.3.2 Robust slope estimates

Instead of using only one interval-to-interval difference, estimate slopes using medians over short windows.

Use the same context length:

```text
K = 4 intervals = 1 hour for 15-minute data
```

For any series `x[t]`, define a median first-difference slope over a set of consecutive intervals `A`:

```text
slope(x; A)
= median(x[t] - x[t-1] for consecutive intervals t-1 and t inside A)
```

### 5.3.3 Left-boundary discontinuity

For a given reconstructed curve `U`, define:

```text
left_slope_before(U, W) = slope(U; LeftContext(W))
left_slope_after(U, W)  = slope(U; first K intervals inside W)
```

Then:

```text
left_discontinuity(U, W)
= abs(left_slope_before(U, W) - left_slope_after(U, W))
```

### 5.3.4 Right-boundary discontinuity

Define:

```text
right_slope_before(U, W) = slope(U; last K intervals inside W)
right_slope_after(U, W)  = slope(U; RightContext(W))
```

Then:

```text
right_discontinuity(U, W)
= abs(right_slope_before(U, W) - right_slope_after(U, W))
```

### 5.3.5 Total slope discontinuity

For no correction:

```text
slope_discontinuity_no(W)
= left_discontinuity(U_no, W) + right_discontinuity(U_no, W)
```

For candidate correction:

```text
slope_discontinuity_corr(W)
= left_discontinuity(U_corr[.; W], W) + right_discontinuity(U_corr[.; W], W)
```

### 5.3.6 Slope-continuity improvement

```text
slope_continuity_improvement(W)
= (slope_discontinuity_no(W) - slope_discontinuity_corr(W))
  / (slope_discontinuity_no(W) + slope_discontinuity_corr(W) + eps)
```

Interpretation:

```text
positive value  -> correction makes boundary slopes more continuous
zero            -> correction does not change boundary continuity materially
negative value  -> correction makes boundary slopes less continuous
```

---

## 6. Candidate score

### 6.1 Three-feature vector

For each candidate window `W`, define:

```text
f1(W) = bridge_improvement(W)
f2(W) = roughness_improvement(W)
f3(W) = slope_continuity_improvement(W)
```

### 6.2 Simple equal-weight score

The lowest-parameter version is:

```text
Score(W) = f1(W) + f2(W) + f3(W)
```

Because each feature is already scale-free and usually bounded around `[-1, 1]`, equal weighting is a reasonable first version.

### 6.3 Optional robust z-score version

If the three features have noticeably different distributions, use robust normalisation based on training data only:

```text
z_k(W) = (f_k(W) - median_train(f_k)) / (IQR_train(f_k) + eps)
```

Then:

```text
Score(W) = z_1(W) + z_2(W) + z_3(W)
```

Recommended initial paper version:

```text
Use raw scale-free features first.
Use robust z-score version only as an ablation if needed.
```

### 6.4 Learned-weight version

If equal weights are too weak, use a very small logistic regression:

```text
Score(W) = beta_0 + beta_1*f1(W) + beta_2*f2(W) + beta_3*f3(W)
```

To avoid overfitting:

```text
1. Use strong regularisation.
2. Learn beta only from training/validation data.
3. Keep one global day-level threshold.
4. Avoid site-specific thresholds in the main method.
```

Recommended reporting:

```text
Main method: equal-weight or logistic 3-feature score.
Ablation: compare equal-weight versus learned-weight score.
```

---

## 7. Day-level decoding

For each day, compute `Score(W)` for every non-null candidate window.

Select the best window:

```text
W_star = argmax Score(W) over all non-null W
```

Define day score:

```text
Q = Score(W_star)
```

Classify the day using one threshold:

```text
rpf_day_label = 1 if Q >= tau
rpf_day_label = 0 if Q < tau
```

where:

```text
tau = validation-calibrated threshold
```

If `rpf_day_label = 1`:

```text
rpf_start_time = start of W_star
rpf_end_time   = end of W_star
rpf_interval_label[t] = 1 if t is inside W_star, else 0
```

If `rpf_day_label = 0`:

```text
rpf_start_time = blank
rpf_end_time   = blank
rpf_interval_label[t] = 0 for all t
```

---

## 8. Corrected net-load output

For clean reflection-type sign error:

```text
if rpf_interval_label[t] = 1:
    y_corrected[t] = -y[t]
else:
    y_corrected[t] = y[t]
```

This correction is appropriate for:

```text
rpf_mechanism = reflection_wrong_sign
```

For likely clipped/deadband behaviour:

```text
rpf_mechanism = clipped_deadband
```

do not automatically use simple sign flip as the final corrected value, because the export magnitude may be censored or lost.

---

## 9. Confidence and mechanism flags

The model can output a binary correction decision, but manual review should also record confidence and mechanism.

### 9.1 Confidence

```text
rpf_confidence = sure
```

Use when the visual and physical evidence is clear.

```text
rpf_confidence = unsure
```

Use when corrected and uncorrected interpretations are both plausible.

### 9.2 Mechanism

Recommended mechanisms:

```text
reflection_wrong_sign
clipped_deadband
telemetry_anomaly
normal_low_load
unclear
```

Simple definitions:

```text
reflection_wrong_sign:
    Clear positive reflected bump. Sign flip produces smooth corrected curve.

clipped_deadband:
    Long near-zero plateau under high solar. Reverse flow likely, but magnitude may be lost.

telemetry_anomaly:
    Spike, flatline, missing block, or discontinuity not physically aligned with solar.

normal_low_load:
    Smooth positive midday valley caused by low load and high solar; no clear sign error.

unclear:
    Ambiguous or borderline case.
```

---

## 10. Validation design

### 10.1 Alpha validation

Use spatiotemporal holdout:

```text
1. Hold out complete substations.
2. Hold out a later test period.
3. Train only on earlier periods and different substations.
```

This checks whether the method generalises across both:

```text
site
```

and:

```text
time
```

### 10.2 Beta validation

Use Beta as actual manually reviewed operational data.

Recommended reporting:

```text
1. Conservative-all evaluation:
   unsure days treated as non-correction.

2. Sure-only evaluation:
   exclude unsure days.

3. Optional abstention-aware evaluation:
   model can output correction / no correction / manual review needed.
```

### 10.3 Threshold selection

Choose `tau` using validation data only.

Recommended objective:

```text
maximise day-level F1
```

or, if operational false positives are costly:

```text
maximise recall subject to precision >= P_min
```

Use one global threshold in the main method.

---

## 11. Metrics

### 11.1 Day-level metrics

A day is positive if it contains any RPF correction window.

Report:

```text
precision_day = TP_day / (TP_day + FP_day)
recall_day    = TP_day / (TP_day + FN_day)
F1_day        = 2 * precision_day * recall_day / (precision_day + recall_day)
```

### 11.2 Interval-level metrics

Evaluate all intervals against reviewed interval labels:

```text
precision_interval
recall_interval
F1_interval
```

### 11.3 Boundary metrics

For true-positive days, report:

```text
start_error_minutes = abs(predicted_start - reviewed_start)
end_error_minutes   = abs(predicted_end - reviewed_end)
```

Also report:

```text
IoU = length(predicted_window intersection reviewed_window)
      / length(predicted_window union reviewed_window)
```

---

## 12. Pseudocode

```text
for each site-day:

    read y[t] and S[t]

    compute U_no[t] = S[t] + y[t]

    CandidateSet = {all daytime windows W satisfying duration bounds}

    for each W in CandidateSet:

        compute U_corr[t; W]:
            if t inside W:
                U_corr[t; W] = S[t] - y[t]
            else:
                U_corr[t; W] = S[t] + y[t]

        compute bridge_improvement(W)
        compute roughness_improvement(W)
        compute slope_continuity_improvement(W)

        Score(W) = bridge_improvement(W)
                 + roughness_improvement(W)
                 + slope_continuity_improvement(W)

    W_star = window with highest Score(W)
    Q = Score(W_star)

    if Q >= tau:
        rpf_day_label = 1
        rpf_interval_label[t] = 1 if t inside W_star else 0
        y_corrected[t] = -y[t] if t inside W_star else y[t]
    else:
        rpf_day_label = 0
        rpf_interval_label[t] = 0 for all t
        y_corrected[t] = y[t]
```

---

## 13. Main method summary

The proposed main method is:

```text
1. Generate all physically plausible daytime windows.
2. For each window, reconstruct underlying demand with and without sign correction.
3. Score each window using only:
       a. bridge improvement,
       b. roughness improvement,
       c. slope-continuity improvement.
4. Select the best window.
5. Compare its score with one validation-calibrated threshold.
6. If positive, correct that one contiguous window by sign flip.
```

This is intentionally simpler than the exploratory 20-rule model. It is easier to defend in a paper because each feature corresponds to a direct physical expectation of underlying demand.

---

## 14. Backlog / future extensions

The following ideas are useful but should not be part of the main method yet. They can be tested later as ablations or future work.

### 14.1 Solar co-movement

Inside reflected RPF windows, measured positive net load may move with solar generation.

Possible features:

```text
corr(S[t], y[t]) inside W
corr(diff(S[t]), diff(y[t])) inside W
same-sign derivative fraction between S and y inside W
```

Reason for backlog:

```text
Useful for clean reflected-bump cases, but can fail on cloudy days or clipped/deadband cases.
```

### 14.2 Curvature improvement

Possible feature:

```text
curvature_improvement = reduction in mean absolute second difference after correction
```

Reason for backlog:

```text
Related to roughness improvement. May add complexity without enough independent value.
```

### 14.3 Boundary level discontinuity

Possible feature:

```text
level_continuity_improvement = reduction in level jumps at start and end boundaries
```

Reason for backlog:

```text
Partly overlaps with slope-continuity improvement.
```

### 14.4 Pseudo-load stability

Inside a candidate RPF window:

```text
pseudo_load[t] = S[t] - y[t]
```

Possible checks:

```text
mean(pseudo_load)
standard deviation(pseudo_load)
fraction(pseudo_load < 0)
```

Reason for backlog:

```text
Useful as a plausibility check, but sensitive to solar estimation error.
```

### 14.5 Temporal context

Possible features:

```text
rolling mean of day scores over neighbouring 7, 15, or 21 days
fill isolated one-day gaps in long RPF runs
veto isolated anomalous positives
```

Reason for backlog:

```text
Very useful empirically, but can look like a hand-tuned post-processing rule. Better to add only after the three-feature physical model is established.
```

### 14.6 Mechanism-specific model

Possible classes:

```text
reflection_wrong_sign
clipped_deadband
telemetry_anomaly
normal_low_load
unclear
```

Reason for backlog:

```text
Likely important for Beta, but it expands the task beyond simple sign correction into mechanism classification and censored-load reconstruction.
```

### 14.7 Learned candidate ranker

Possible models:

```text
logistic regression
XGBoost ranker
LightGBM ranker
small monotonic gradient-boosted tree
```

Reason for backlog:

```text
Can improve performance, but the main paper method should first establish that the physical three-feature score works.
```

---

## 15. Recommended experiments for this version

Run the following ablations:

```text
A. Bridge only
B. Bridge + roughness
C. Bridge + roughness + slope continuity
D. Equal-weight 3-feature score
E. Logistic-regression 3-feature score
```

Report:

```text
Alpha spatiotemporal F1
Beta conservative-all F1
Beta sure-only F1
interval F1
boundary error
```

The preferred final method is the simplest variant whose Beta performance is acceptable while maintaining strong Alpha spatiotemporal performance.


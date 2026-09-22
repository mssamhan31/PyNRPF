# Leakage audit

What each outer fold was permitted to read. Every number below is derived from committed Phase 3 outputs; the scorer was not re-run and no dataset was opened.

Global, before any fold: the scorer outputs (r, window, runner-up, margin) for every site-day, computed in Phase 3 with a cohort-level label-free floor φ (a pre-existing property of the frozen scores, recorded in the Phase 0 decision). The per-day features z, L, z_per_slot, logL and m are label-free functions of those outputs.

Per outer fold:

- Fold alpha_alpha_A: outer-training stations 17 (9 Alpha, 8 Beta), 11,644 eligible days with labels. Held-out alpha_A: 1,039 days, scorer outputs and label-free features only; labels used only to score the result.
- Fold alpha_alpha_B: outer-training stations 17 (9 Alpha, 8 Beta), 11,630 eligible days with labels. Held-out alpha_B: 1,055 days, scorer outputs and label-free features only; labels used only to score the result.
- Fold alpha_alpha_C: outer-training stations 17 (9 Alpha, 8 Beta), 11,630 eligible days with labels. Held-out alpha_C: 1,055 days, scorer outputs and label-free features only; labels used only to score the result.
- Fold alpha_alpha_D: outer-training stations 17 (9 Alpha, 8 Beta), 11,649 eligible days with labels. Held-out alpha_D: 1,035 days, scorer outputs and label-free features only; labels used only to score the result.
- Fold alpha_alpha_E: outer-training stations 17 (9 Alpha, 8 Beta), 11,650 eligible days with labels. Held-out alpha_E: 1,035 days, scorer outputs and label-free features only; labels used only to score the result.
- Fold alpha_alpha_F: outer-training stations 17 (9 Alpha, 8 Beta), 11,652 eligible days with labels. Held-out alpha_F: 1,033 days, scorer outputs and label-free features only; labels used only to score the result.
- Fold alpha_alpha_G: outer-training stations 17 (9 Alpha, 8 Beta), 11,648 eligible days with labels. Held-out alpha_G: 1,037 days, scorer outputs and label-free features only; labels used only to score the result.
- Fold alpha_alpha_H: outer-training stations 17 (9 Alpha, 8 Beta), 11,644 eligible days with labels. Held-out alpha_H: 1,039 days, scorer outputs and label-free features only; labels used only to score the result.
- Fold alpha_alpha_I: outer-training stations 17 (9 Alpha, 8 Beta), 11,641 eligible days with labels. Held-out alpha_I: 1,042 days, scorer outputs and label-free features only; labels used only to score the result.
- Fold alpha_alpha_J: outer-training stations 17 (9 Alpha, 8 Beta), 11,629 eligible days with labels. Held-out alpha_J: 1,055 days, scorer outputs and label-free features only; labels used only to score the result.
- Fold beta_beta_A: outer-training stations 17 (10 Alpha, 7 Beta), 12,347 eligible days with labels. Held-out beta_A: 366 days, scorer outputs and label-free features only; labels used only to score the result.
- Fold beta_beta_B: outer-training stations 17 (10 Alpha, 7 Beta), 12,451 eligible days with labels. Held-out beta_B: 366 days, scorer outputs and label-free features only; labels used only to score the result.
- Fold beta_beta_C: outer-training stations 17 (10 Alpha, 7 Beta), 12,344 eligible days with labels. Held-out beta_C: 366 days, scorer outputs and label-free features only; labels used only to score the result.
- Fold beta_beta_D: outer-training stations 17 (10 Alpha, 7 Beta), 12,498 eligible days with labels. Held-out beta_D: 366 days, scorer outputs and label-free features only; labels used only to score the result.
- Fold beta_beta_E: outer-training stations 17 (10 Alpha, 7 Beta), 12,376 eligible days with labels. Held-out beta_E: 364 days, scorer outputs and label-free features only; labels used only to score the result.
- Fold beta_beta_F: outer-training stations 17 (10 Alpha, 7 Beta), 12,397 eligible days with labels. Held-out beta_F: 364 days, scorer outputs and label-free features only; labels used only to score the result.
- Fold beta_beta_G: outer-training stations 17 (10 Alpha, 7 Beta), 12,399 eligible days with labels. Held-out beta_G: 366 days, scorer outputs and label-free features only; labels used only to score the result.
- Fold beta_beta_H: outer-training stations 17 (10 Alpha, 7 Beta), 12,348 eligible days with labels. Held-out beta_H: 364 days, scorer outputs and label-free features only; labels used only to score the result.

Inner loop: within the 17 outer-training stations only; each inner held-out station's labels are used only to score the inner prediction; C4's null distribution and every logistic are fitted on the 16 inner-training stations.

Selection: by station-macro inner log loss under the inner Beta energy-precision gate, fixed in PLAN.md before any outer result was computed.

D1 (diagnostic only): standardises each station's z by that station's own null-day median and MAD, including the held-out station's unlabelled days; transductive by design and excluded from selection and from the success result.

B3 (diagnostic only): receives the cohort indicator; excluded from selection and from the success result.

R0 (reference): fitted on the same-cohort outer-training stations only, as in Phase 3.

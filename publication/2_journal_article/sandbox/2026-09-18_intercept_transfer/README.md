# Sandbox: intercept transfer for M9 (18 September 2026)

Question: with the M9 scorer frozen and the calibration slope frozen from the other population, how many reviewed days of a new population are needed to set the intercept, and is the untouched foreign calibration safe before any review? See `PLAN.md` for the design and `RESULTS.md` for the answer.

Reads committed Phase 3 outputs only (`sandbox/2026-09-16_phase3_release/outputs/05_m9`, `06_site_days`). Nothing outside this folder is modified; nothing is committed.

## Run

From the repository root, with the project environment active:

```powershell
.\.venv\Scripts\Activate.ps1
python publication\2_journal_article\sandbox\2026-09-18_intercept_transfer\run.py
python publication\2_journal_article\sandbox\2026-09-18_intercept_transfer\plot.py
```

About six minutes (2 directions × 2 levels × 2 designs × 5 sample sizes × 20 draws × 8 to 10 stations). Seed 9.

## Files

| file | what it is |
|---|---|
| `PLAN.md` | design, benchmarks, sampling designs, the success reading fixed in advance |
| `run.py` | the experiment |
| `plot.py` | the learning-curve figure from `curve_summary.csv` |
| `curve.csv` | pooled metrics per direction, level, design, sample size and draw |
| `curve_summary.csv` | median and 5–95 percentiles over draws per cell |
| `station_k27.csv` | per-station results at 27 review-design days |
| `intercepts.csv` | fitted intercepts per station and draw, and the benchmark pairs |
| `fig_learning_curve.png` | Beta energy precision and Energy IoU against reviewed days |
| `RESULTS.md` | observations, interpretations, options |

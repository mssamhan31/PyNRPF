# Sandbox: cross-cohort calibration for M9 (18 September 2026)

Question: can one shared calibration, fitted without cohort or station identifiers, replace the frozen within-cohort calibration while the M9 scorer and its evidence stay unchanged? See `PLAN.md` for the design and `RESULTS.md` for the answer.

Everything here reads committed Phase 3 outputs only (`sandbox/2026-09-16_phase3_release/outputs/05_m9`, `06_site_days`, `01_folds`). Nothing outside this folder is modified; nothing is committed.

## Run

From the repository root, with the project environment active:

```powershell
.\.venv\Scripts\Activate.ps1
python publication\2_journal_article\sandbox\2026-09-18_cross_cohort_calibration\run.py
```

About seven minutes (18 outer folds × 17 inner folds × 5 candidates, plus a 1,000-draw station bootstrap per candidate and group). Seed 9.

## Files

| file | what it is |
|---|---|
| `PLAN.md` | the approved experimental plan, candidates, validation design, risks and the decisions taken on 18 September |
| `run.py` | the whole experiment: data, candidates, nested validation, metrics, outputs |
| `RESULTS.md` | observations, interpretations and the options for the methodological decision |
| `LEAKAGE_AUDIT.md` | what each outer fold was permitted to read |
| `predictions_outer.parquet` | held-out predictions per site-day and candidate (p, outcome, energies) |
| `fits.csv`, `selection.csv` | fitted coefficients per fold; the inner selection table per outer fold |
| `metrics_pooled.csv`, `metrics_station.csv`, `reliability.csv`, `bootstrap.csv` | the metrics |
| `results_summary.json` | selection counts and the folds where the inner gate fell back |

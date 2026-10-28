# 2026-09-22 — bridge anchors at the window edge

A sandbox experiment on the locked M9 scorer for wrong reverse power flow (RPF) sign
detection. It asks whether drawing the straight bridge from a candidate window's own
edge slots, rather than from the nearest finite readings outside it, helps the beta_D
days whose labelled span abuts a missing reading, and what it costs elsewhere. It
changes no default: `--anchors nearest` remains the frozen method, reproduced here bit
for bit. `PLAN.md` states the question and the variants; `RESULTS.md` the numbers and
their reading.

## Inputs

| Input | Where |
|---|---|
| Scorer, harness, comparison script | `../../m9_dev/m9_scorer.py`, `m9_eval.py`, `compare_runs.py` (this experiment added `--anchors` and `--runs-dir`) |
| Frozen settings | `../../m9_dev/decisions/frozen_method_rev2.md`: `--variant sq --p 0 --sigma overnight --c 0.7 --stat llr --missing mask_windows --edges inwardx` |
| Frozen reference run | `../../m9_dev/runs/phase5_final_rev2/` (nearest anchors, scorer sha256 `f05c148d…`) |
| Data | `../../dataset/final/dataset_alpha.parquet`, `dataset_beta.parquet` (hashes recorded in every `runs/*/config.json`) |

## Outputs

| Output | Content |
|---|---|
| `runs/nearest/`, `runs/edge/`, `runs/gap_edge/` | One run per anchor rule: `config.json` (settings, scorer and data hashes), `predictions_<cohort>.csv` (held-out per site-day), `summary_pooled.csv`, `summary_station.csv`, `calibration_fits.csv` |
| `logs/<rule>.log` | The harness's printed report for each run |
| `tables/compare_nearest.csv`, `compare_edge.csv`, `compare_gap_edge.csv` | `compare_runs.py` per-station deltas: `nearest` against the frozen reference, then `edge` and `gap_edge` against `nearest` |
| `tables/reproduction_check.md` | Row-by-row identity of `runs/nearest` with the frozen reference |
| `tables/pooled.csv` and `.md` | Pooled Beta `sure`, Alpha and Beta `unsure` metrics per rule |
| `tables/per_station.csv` and `.md` | Per-station metrics per rule, both cohorts |
| `tables/beta_D_gap_days.csv` and `.md` | beta_D labelled RPF days whose span abuts a gap: window, r, p, decision, exact match and IoU per rule |
| `tables/outcome_changes.csv` and `.md` | Per station, days whose outcome or window changed against `nearest` |

`runs/*/scores_<cohort>.parquet`, the harness's scoring cache, is removed after each
run: it duplicates the prediction columns and is rebuilt by the commands below.

## Reproduce

From the repository root in PowerShell, with the project environment installed:

```powershell
$py = ".\.venv\Scripts\python.exe"
$m9 = "publication/2_journal_article/m9_dev"
$sb = "publication/2_journal_article/sandbox/2026-09-22_bridge_anchors"

& $py -m pytest $m9/tests -q                       # 39 tests, 16 of them on the anchor rules

foreach ($rule in "nearest", "edge", "gap_edge") {
  & $py $m9/m9_eval.py --run $rule --variant sq --p 0 --sigma overnight --c 0.7 `
        --stat llr --missing mask_windows --edges inwardx --anchors $rule --runs-dir $sb/runs `
        | Tee-Object -FilePath $sb/logs/$rule.log
  Remove-Item $sb/runs/$rule/scores_*.parquet
}

& $py $m9/compare_runs.py --incumbent phase5_final_rev2 --candidate $sb/runs/nearest  --out-dir $sb/tables
& $py $m9/compare_runs.py --incumbent $sb/runs/nearest --candidate $sb/runs/edge     --out-dir $sb/tables
& $py $m9/compare_runs.py --incumbent $sb/runs/nearest --candidate $sb/runs/gap_edge --out-dir $sb/tables
& $py $sb/make_tables.py
```

Each run scores about 13,000 site-days against 1,176 windows and takes under a minute.
`compare_runs.py` accepts a run as a name under `m9_dev/runs` or as a path to a run
folder. `make_tables.py` reads the three runs, the frozen reference and the Beta
dataset, and writes every table under `tables/`.

# m9_dev — M9 development workspace

Development of the compact counterfactual M9 method for reverse power flow (RPF) sign
errors, following `M9_development_plan.md` (2026-09-16). This folder is
self-contained: it reads the frozen datasets in `../dataset/final/` and writes only
under `runs/`, `weak_sites/` and `notes/`. Nothing else in `2_journal_article/` is
modified by anything here.

## What the method is

For each site-day, two reconstructions of underlying demand are compared: keep the
recorded net-load sign (U0 = S + y), or negate it inside one contiguous window
(S − y inside, S + y outside). Every window in 06:00–18:00 is scored by how much
better a straight line through its two outside anchors fits the corrected
reconstruction than the uncorrected one, divided by the day's overnight noise
scale and the window length. `NO_CORRECTION` scores zero and is ranked jointly.
The winning score is mapped to a probability by a two-coefficient logistic on a
signed-log transform, and one public control *c* gives `AUTO_CORRECT`,
`AUTO_KEEP` or `UNCERTAIN`. One component, no fitted scoring parameters.

## Files

| File | Purpose |
|---|---|
| `m9_scorer.py` | Reference implementation: reconstruction, all-window scoring, ranking, calibration, decision |
| `m9_metrics.py` | Energy IoU, energy precision, site-day metrics, sure-day recall, calibration reliability |
| `m9_eval.py` | Leave-one-station-out harness; writes a run folder |
| `tests/` | Synthetic fixtures only — `pytest publication/2_journal_article/m9_dev/tests -q` |
| `decisions/` | Phase 0 record and the frozen-method record, written before the results they govern |
| `runs/<name>/` | `config.json`, cached scores, held-out predictions, pooled and per-station summaries |
| `weak_sites/` | Failure-class diagnosis figures and tables |
| `notes/` | One entry per iteration round: where, why, remedy, result, decision |

## Running

From the repository root, with the project environment active:

```powershell
pytest publication/2_journal_article/m9_dev/tests -q
python publication/2_journal_article/m9_dev/m9_eval.py --run phase2_baseline                 # original recommendation
python publication/2_journal_article/m9_dev/m9_eval.py --run final --stat llr --p 0 --missing mask_windows                  # revision 1
python publication/2_journal_article/m9_dev/m9_eval.py --run final --stat llr --p 0 --missing mask_windows --edges inwardx  # revision 2 (current)
python publication/2_journal_article/m9_dev/m9_eval.py --run <name> --variant abs      # Candidate B
python publication/2_journal_article/m9_dev/m9_eval.py --run <name> --variant tv       # Candidate C
python publication/2_journal_article/m9_dev/m9_eval.py --run <name> --sigma station    # station-level scale
python publication/2_journal_article/m9_dev/m9_eval.py --run <name> --anchors gap_edge --runs-dir <folder>   # bridge anchored on the window edge beside a gap; run lands in <folder>/<name>
```

`--anchors` (`nearest`, the frozen default; `edge`; `gap_edge`) chooses where the bridge is
anchored; `--runs-dir` places the run outside `runs/`, for sandbox experiments such as
`../sandbox/2026-09-22_bridge_anchors/`. `compare_runs.py` takes the same `--runs-dir`
and accepts a path to a run folder in place of a name.

A full run scores 13,000 site-days against 1,176 windows each and takes a few
minutes. Per-station tables are the ones to read; pooled numbers hide station
failures.

## Status

| Phase | State |
|---|---|
| 0 — decisions | `decisions/2026-09-16_phase0.md` |
| 1 — reference implementation and fixtures | done: 18 fixtures, ruff clean |
| 2 — baseline | `runs/phase2_baseline/` |
| 3 — failure-driven rounds | six: rounds 2–4 adopted (2 confirmed by Samhan), 1 and 5 rejected, 6 probed and not run — `notes/round_*.md` |
| 4 — screen A / B / C | A retained — `runs/p4b_abs_nearest`, `runs/p4_tv_llr_mask` |
| 5 — freeze and final run | rev 1: `decisions/frozen_method.md`, `runs/phase5_final/`; **rev 2 (edge rule, round 7): `decisions/frozen_method_rev2.md`, `runs/phase5_final_rev2/`** |

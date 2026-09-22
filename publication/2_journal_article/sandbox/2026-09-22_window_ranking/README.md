# Sandbox: within-day window ranking of M9 (22 September 2026)

Question: when M9 scores every candidate window of a labelled reverse power flow (RPF) day with the frozen revision-2 settings, where does the true window sit in the ranking, how good is the rank-1 window, and how much would a reviewer choosing among the top three or five gain? See `RESULTS.md` for the answer.

Everything here is computed from the frozen datasets, the frozen scorer (`m9_dev/m9_scorer.py`, imported, never copied) and the committed Phase 3 outputs (`sandbox/2026-09-16_phase3_release/outputs/05_m9`, `06_site_days`). The scoring is label-free; labels enter only when the ranked windows are measured against the truth. Nothing outside this folder is modified; nothing is committed. The method is not changed: no parameter is fitted and no rule is altered.

## Run

From the repository root, with the project environment active:

```powershell
.\.venv\Scripts\Activate.ps1
python publication\2_journal_article\sandbox\2026-09-22_window_ranking\run.py
```

About 20–40 seconds depending on machine load (10–16 s to load and hash-check the datasets, the rest to score 4,010 days). Deterministic; no random numbers: two runs on 22 September produced byte-identical tables. The run stops with an error if the configuration's `m9` block is not the frozen revision 2 or if the recomputed best windows differ from the committed ones.

## Inputs (read-only)

| input | used for |
|---|---|
| `config/final_evaluation.yaml` and the datasets it names (hash-checked by `final_eval.config.load`) | the Phase 3 population: complete site-days, cohort sigma floor, labels |
| `m9_dev/m9_scorer.py` | admissibility, bridge residuals and the likelihood-ratio evidence; `best_window` for the tie-break check |
| `final_eval/data.py`, `final_eval/m9.py` | the same loader and `sigma_floor` the Phase 3 engine used |
| `.../outputs/05_m9/scores_{alpha,beta}.parquet` | committed best window, score, admissible count per site-day (self-check) |
| `.../outputs/06_site_days/site_days.parquet` (m9 rows) | committed candidate energies per site-day (energy check) |

## Populations

Every complete site-day with at least one labelled slot: Alpha 3,381 days (all `controlled`), Beta `sure` 470 days, Beta `unsure` 159 days (reported separately, never pooled with `sure`). All 4,010 have at least one admissible window. Per-station tables cover the two headline populations (Alpha, Beta sure).

## Definitions

- **Window**: an inclusive slot pair [start, end] inside slots 24–71 (06:00–18:00); 1,176 windows. Scored by the frozen evidence r_W = (L/2) log((RSS_u + λ)/(RSS_c + λ)), λ = L·floor²; the floor is the cohort's smallest non-zero overnight step (Alpha 1.40e-7 MW, Beta 8.03e-9 MW, identical to the Phase 3 manifest).
- **Admissible ranking**: the windows that pass the frozen admissibility (interior finite, an anchor on each side, edges at local minima of net load with the inward one-slot tolerance and the gap exemption), ordered by score descending, then shorter, then earlier — the scorer's tie-break. Rank 1 is the window M9 proposes (or, when the null wins, the window it keeps for inspection).
- **All-windows ranking**: the same order over every window with a finite score, admissibility ignored. A window holding a missing reading has no score and is "not scoreable".
- **Truth intervals**: the labelled slots. **Truth span**: first to last labelled slot. On Alpha 1,430 days (42%) have non-contiguous labels, so the span holds unlabelled slots; the exact-match rank uses the span, every IoU, precision and recall uses the labelled slots.
- **Rank of the exact span**: the truth span's position in the ranking; "not admissible" when the span fails the admissibility rules (or lies outside slots 24–71), "not scoreable" in the all-windows ranking when it holds a missing reading.
- **Rank of IoU ≥ 0.8**: the position of the first window in the ranking whose interval IoU with the labelled slots reaches 0.8; "none ≥ 0.8" when no ranked window does.
- **Bins**: 1, 2, 3, 4–10, >10, plus the exclusion bins above.
- **Energies**: a slot's correction energy is 2·|y|·0.25 MWh (missing readings carry none); candidate = sum over the window, correct = sum over window ∩ labelled slots, required = sum over the labelled slots; energy IoU = correct / (candidate + required − correct); energy precision = correct / candidate. Pooled by summing MWh before dividing. Interval precision, recall and F1 pool the slot counts.
- **Oracles**: among the top k admissible windows, the one with the highest interval IoU (reported with interval metrics) and the one with the highest energy IoU (reported with energy metrics). These use the truth to choose, so they are upper bounds on what a reviewer could gain, not a reviewer model.
- **Geometry** of the rank-1 window [a, b] against the truth span [at, bt]: exact; too long (start / end / both sides extended, never shortened); too short (inside the span); shifted earlier / later (overlapping, neither contains the other); disjoint.

## Outputs

`tables/` (every table as `.csv` and `.md`):

| file | what it is |
|---|---|
| `self_check` | per cohort: days checked, sigma floor, admissible-count, best-window, null-decision and tie-break mismatches, maximum score difference, energy differences against Phase 3 |
| `per_day` | one row per labelled RPF day: truth span, contiguity, admissibility of the truth and the reason if not, all four ranks and bins, rank-1 window, its score, IoU, slot counts, energies, geometry, edge offsets, top-3 and top-5 oracle picks, the scorer's own best window |
| `rank_distribution_population`, `_station`, `_alpha_contiguity` | counts and shares per bin for both rankings and both targets; `*_admissible.md` and `*_all_windows.md` are the readable "count (share)" forms |
| `rank1_quality_population`, `_station`, `_alpha_contiguity` | exact share, IoU summary, candidate-set ceiling (mean best admissible IoU), pooled interval and energy metrics of the rank-1 window |
| `oracle_top_k` | best-of-top-1/3/5 per population |
| `failure_geometry` | geometry categories with counts, shares, mean edge offsets (rank-1 minus truth, slots), share of days with edges within one slot, mean extension of too-long windows |
| `truth_admissibility` | why truth spans are excluded from the admissible ranking |

`figures/` (PNG, 200 dpi, Arial, journal palette): `fig01_rank_distribution` (bins per population, admissible ranking, exact and IoU ≥ 0.8), `fig02_rank1_iou_cdf` (cumulative distribution of the rank-1 window's interval IoU), `fig03_station_exact_rank1` (share of exact rank-1 per station, headline populations).

## Files

| file | what it is |
|---|---|
| `run.py` | the whole study: loading, label-free scoring, self-check, ranking, metrics, tables, figures |
| `RESULTS.md` | observations, interpretation, what the study supports and what it does not |
| `README.md` | this file |

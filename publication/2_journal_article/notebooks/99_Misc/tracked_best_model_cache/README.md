# Tracked Best Model Cache

This folder stores small, high-value artifacts for model-development continuity across computers.

The goal is to make the current best interpretable model easy to inspect after `git pull`, without tracking bulky generated HTML, full candidate-window tables, or ignored experiment output folders.

## Current Cache

| Folder | Model | Purpose |
|---|---|---|
| `physical_score_c14_best/` | `m9_pbm_physical_score`, C14 best pre-XGB model | Portable decision/debug cache for the best physical-score model before the overnight XGB/RF/MLP experiment. |

## Tracking Policy

Tracked here:

- compact model configuration;
- final-label daily feature cache;
- per-site thresholds;
- daily prediction audit;
- metric summaries;
- FP/FN review index;
- source/hash manifest.

Not tracked here:

- generated FP/FN HTML pages;
- full dense candidate-window tables;
- ignored experiment output folders;
- notebook execution outputs.

Regenerate bulky review outputs locally from the tracked cache when needed.

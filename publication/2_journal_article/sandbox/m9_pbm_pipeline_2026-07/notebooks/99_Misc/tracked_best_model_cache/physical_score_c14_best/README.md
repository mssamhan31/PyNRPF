# Physical Score C14 Best Cache

This is a portable cache for the best pre-overnight interpretable physical-score model:

`R2_beta_loso_plus_alpha / G_b1p0_r1p5_sc0p5`

This is the model that achieved approximately:

- Beta all-days day F1: `0.8096`
- Beta sure-only day F1: `0.8994`

The cache is designed for fast continuation on another computer after `git pull`.

## Files

| File | Contents |
|---|---|
| `best_model_config.json` | Model name, variant, weights, threshold policy, and headline metric. |
| `daily_feature_cache.csv` | Alpha/Beta daily physical features with current final labels and confidence. |
| `best_model_thresholds.csv` | Per-heldout-site thresholds from the C14 `R2_beta_loso_plus_alpha` run. |
| `best_model_daily_predictions.csv` | Current-label Beta day predictions, scores, thresholds, selected windows, true windows, and weighted feature components. |
| `best_model_metrics_current_labels.csv` | Pooled, macro-site, and per-site day metrics for Beta all-days and Beta sure-only. |
| `fp_fn_review_index.csv` | Sure-only FP/FN rows to review or regenerate HTML from. |
| `fp_fn_counts_by_site.csv` | Count summary of sure-only FP/FN days by Beta site. |
| `review_examples/` | Top-confidence FP and FN HTML galleries plus C16 review indexes for quick manual inspection. |
| `source_c14_*.csv` | Compact source summaries copied from the original C14 experiment outputs. |
| `artifact_manifest.csv` | SHA256 hashes for tracked cache files and important source files. |

## Important Notes

- This cache uses the current `dataset/final/dataset_beta.parquet` confidence values.
- Current final Beta confidence count in this cache is `2330` sure days and `598` unsure days.
- The cache tracks two compact review HTML galleries for quick inspection: `review_examples/fp_top_confidence_12days.html` and `review_examples/fn_top_confidence_12days.html`.
- The review examples use the local `review_examples/plotly-3.6.0.min.js` runtime, so they do not depend on the Plotly CDN.
- Larger regenerated HTML batches should still be written under `99_Misc/outputs/`, not inside this tracked cache.
- The tracked daily cache avoids rescanning dense candidate windows when continuing model development.

## Quick Checks

After pulling on another computer, the cache is usable if:

- `artifact_manifest.csv` exists;
- `daily_feature_cache.csv` has both `alpha` and `beta` rows;
- `best_model_daily_predictions.csv` has `2928` Beta site-days;
- `fp_fn_review_index.csv` lists sure-only FP/FN cases for manual inspection.
- `review_examples/fp_top_confidence_12days.html` and `review_examples/fn_top_confidence_12days.html` open locally for quick visual checks.

## Suggested Local Workflow

1. Use `best_model_metrics_current_labels.csv` to confirm the baseline.
2. Use `fp_fn_review_index.csv` to select examples for manual review.
3. Open the two HTML files in `review_examples/` to inspect the highest-confidence FP and FN examples.
4. Use `daily_feature_cache.csv` to try new lightweight weight/threshold changes without rebuilding candidate windows.
5. Write regenerated HTML or new experiment outputs under `99_Misc/outputs/`, not inside this tracked cache.

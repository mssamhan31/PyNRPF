# Changelog

## 0.4.0 (unreleased)

The package now ships one method, M9, and nothing else.

- `pynrpf.run(frame)` scores every site-day of a pandas frame and returns a site-day table
  and an interval table; `pynrpf.spark.run_per_site` does the same on a Spark frame per
  site; `pynrpf run` is the command line for a CSV.
- The method lives in `pynrpf.m9`, one module per step: stories, windows, edges, bridge,
  misfit, evidence, winner, calibration, decision.
- Release calibration fitted on eight Ausgrid substations with reviewed real errors, and a
  frozen evidence floor, both recorded with their provenance; `fit_calibration` refits the
  intercept for a new population from reviewed days with the slope kept.
- Dependencies reduced to numpy and pandas; scikit-learn, pyspark and the paper toolchain
  are optional extras.
- Removed: the M7 and M8 inference plugins, M8 training, artefact bundles, the YAML
  configuration system, the model scaffold and drift monitoring. M7 and M8 remain as the
  paper's baselines under `publication/2_journal_article/paper/baselines/`.
- The journal paper's evaluation is a local package `publication/2_journal_article/paper/`
  called by seven notebooks; the reference run is committed under `results/` with manifests.

## 0.3.0 (24 March 2026)

Conference-paper release: M7 rule and M8 two-stage XGBoost behind one inference API, with
training, artefact bundles and a Databricks-oriented configuration. Archived on Zenodo.

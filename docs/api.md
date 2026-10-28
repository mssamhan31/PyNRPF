# API

Everything public is importable from `pynrpf`. The names a pipeline needs:

| name | what it is |
|---|---|
| `run(frame, *, columns=None, c=0.7, calibration=RELEASE_CALIBRATION, phi=RELEASE_PHI) -> Result` | the whole method on a pandas frame; see [Quick start](quickstart.md) |
| `Result` | `site_days` and `intervals` frames, `provenance`, and `summary()` |
| `Provenance` | `version`, `method`, `calibration_intercept`, `calibration_slope`, `calibration_note`, `phi`, `c`; `as_dict()` for JSON or a table row |
| `provenance(c=0.7, calibration=RELEASE_CALIBRATION, phi=RELEASE_PHI) -> Provenance` | the record `run` attaches to its result; a Spark job builds it with the arguments it gave `run_per_site` |
| `pynrpf.spark.run_per_site(sdf, ...)` | the same run on a Spark frame; see [Spark](spark.md) |

/// details | Show details
    type: more

## Every other public name

| name | what it is |
|---|---|
| `score_siteday(y, s, phi=RELEASE_PHI) -> Score` | steps 1 to 7 on one day: 96-slot arrays of net load and solar (MW) |
| `Score` | `input_ok`, `n_admissible`, `winner`, `runner_up` (each a `Candidate` with `start`, `end`, `evidence`), `evidence` (r*) |
| `Calibration(intercept, slope, provenance="")` | step 8; `probability(evidence)`, `evidence_at(p)` |
| `RELEASE_CALIBRATION` | the Beta-8 pair, intercept −4.590, slope 2.162 |
| `RELEASE_PHI` | the evidence floor, 8.03e-9 MW |
| `fit_calibration(evidence, labels, slope=None, provenance="")` | fit the intercept with the slope fixed, or both numbers (needs scikit-learn) |
| `pynrpf.schemas.SITE_DAYS`, `INTERVALS` | the output tables as `(name, dtype, spark type, meaning)` tuples |

## Step modules and errors

The step modules under `pynrpf.m9` (`stories`, `windows`, `edges`, `bridge`, `misfit`,
`evidence`, `winner`, `calibration`, `decision`) are importable for research and are
documented in [Method](method.md); `pynrpf.impact` holds the energy and minimum-demand
numbers. Errors: `KeyError` for a missing column, `ValueError` for timestamps off the grid,
duplicated readings, or an invalid `c`.

///

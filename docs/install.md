# Install

Python 3.10 or later. The package depends on numpy and pandas only. Pin the version in
anything that has to be reproducible:

```powershell
python -m pip install pynrpf==0.4.0
```

## Databricks

Pin the version. Two routes give the same files, built from the same tagged commit by the
release workflow:

1. **PyPI**, when the workspace can reach it: `%pip install pynrpf==0.4.0` in a notebook, or
   `pynrpf==0.4.0` in the cluster or job library list.
2. **A wheel in a Unity Catalog Volume**, when it cannot, or when a pipeline has to give the
   same result every time it runs: the steps under
   [A pinned wheel for a pipeline](#a-pinned-wheel-for-a-pipeline).

The runtime needs Python 3.10 or later. pyspark is already on the cluster, so the Spark adapter
needs nothing beyond the package.
[`examples/databricks_smoke.py`](https://github.com/mssamhan31/PyNRPF/blob/main/examples/databricks_smoke.py)
is a notebook that installs the pin and runs the method on synthetic data, in pandas and
through Spark; run it once on a new cluster before any real readings.

## A pinned wheel for a pipeline

A scheduled pipeline should hold its own copy of the release, so that a run never depends on
reaching PyPI and the installed files cannot change. The wheel is downloaded once.

1. **Download and check.** Take `pynrpf-0.4.0-py3-none-any.whl` and `SHA256SUMS` from the
   [GitHub Release](https://github.com/mssamhan31/PyNRPF/releases). The hash of the downloaded
   wheel must equal the wheel's line in `SHA256SUMS`; upper and lower case do not matter:

    ```powershell
    Get-FileHash .\pynrpf-0.4.0-py3-none-any.whl -Algorithm SHA256   # the file you downloaded
    Get-Content .\SHA256SUMS                                          # what the release published
    ```

2. **Store it under its version.** Copy the wheel to a folder named for the release, such as
   `/Volumes/<catalog>/<schema>/<volume>/pynrpf/0.4.0/`. Never overwrite a stored wheel: a later
   release goes in its own folder, so an earlier run can always be repeated.
3. **Install it by path.** For a scheduled job, add the wheel as a dependent library of the job
   task (library source *Volumes*) or of the cluster. In a notebook:

    ```python
    %pip install /Volumes/<catalog>/<schema>/<volume>/pynrpf/0.4.0/pynrpf-0.4.0-py3-none-any.whl
    ```

4. **Fix the Databricks Runtime version** of the job cluster. The wheel pins PyNRPF only; Python,
   numpy, pandas and Spark come from the runtime.
5. **Store the provenance record** with every output table, as shown under [Spark](spark.md).

/// details | Show details
    type: more

## From a clone of the repository

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Extras

| extra | adds | when you need it |
|---|---|---|
| `fit` | scikit-learn | fitting both calibration numbers; the intercept-only refit needs nothing extra |
| `spark` | pyspark | the Spark adapter, outside Databricks |
| `paper` | scikit-learn, xgboost, holidays, pyyaml, pyarrow, matplotlib, jupyterlab, tabulate | reproducing the journal paper |
| `dev` | pytest, ruff, build, mkdocs | working on the repository |

```powershell
python -m pip install -e .[dev,paper]
```

## Reproducibility

The method has no randomness: the same readings, the same version and the same settings give
the same tables. Pin the version, and keep the provenance record with every output:
`result.provenance` in Python, `provenance.json` from the command line, and
`pynrpf.provenance(c, calibration, phi)` for a Spark job. The record holds the package
version, the two calibration numbers and their note, the evidence floor and the control. It
does not record the runtime, so note the Databricks Runtime version beside it.

## What the pinned wheel does and does not fix

The wheel fixes PyNRPF itself. Its two dependencies come from the runtime: the package needs
pandas 2.1 or later and numpy 1.26 or later. On a runtime that ships older ones, pip fetches
newer versions at install time, which needs PyPI or an internal mirror and brings in versions
nobody chose. Pick a runtime that already meets both, and keep it fixed.

The identity that runs the job needs read access to the Volume. Databricks describes the
notebook, cluster and job-task forms under
[Install libraries from a volume](https://docs.databricks.com/aws/en/libraries/volume-libraries).

`pynrpf==0.4.0` from PyPI is also a pin and installs the same files, but every cluster start
then depends on reaching PyPI.

///

# Install

Python 3.10 or later. The package depends on numpy and pandas only. Pin the version in
anything that has to be reproducible:

```powershell
python -m pip install pynrpf==0.4.0
```

From a clone of the repository:

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

## Databricks

Pin the version. Two routes give the same files, built from the same tagged commit by the
release workflow:

1. **PyPI**, when the workspace can reach it: `%pip install pynrpf==0.4.0` in a notebook, or
   `pynrpf==0.4.0` in the cluster or job library list.
2. **A wheel in a Unity Catalog Volume**, when it cannot: download
   `pynrpf-0.4.0-py3-none-any.whl` and `SHA256SUMS` from the
   [GitHub Release](https://github.com/mssamhan31/PyNRPF/releases), check the sum, copy the
   wheel to a Volume and install it by path:

    ```python
    %pip install /Volumes/<catalog>/<schema>/<volume>/pynrpf-0.4.0-py3-none-any.whl
    ```

The runtime needs Python 3.10 or later. pyspark is already on the cluster, so the Spark adapter
needs nothing beyond the package.
[`examples/databricks_smoke.py`](https://github.com/mssamhan31/PyNRPF/blob/main/examples/databricks_smoke.py)
is a notebook that installs the pin and runs the method on synthetic data, in pandas and
through Spark; run it once on a new cluster before any real readings.

## Reproducibility

The method has no randomness: the same readings, the same version and the same settings give
the same tables. Pin the version, and keep the provenance record with every output:
`result.provenance` in Python, `provenance.json` from the command line, and
`pynrpf.provenance(c, calibration, phi)` for a Spark job. The record holds the package
version, the two calibration numbers and their note, the evidence floor and the control.

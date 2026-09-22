# Install

Python 3.10 or later. The package depends on numpy and pandas only.

```powershell
python -m pip install pynrpf
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

Either install from PyPI on the cluster or job (`%pip install pynrpf==0.4.0`), or build the
wheel once (`python -m build --wheel`), copy it to a Unity Catalog Volume and install it from
there. The Spark adapter needs nothing beyond the package: pyspark is already on the cluster.

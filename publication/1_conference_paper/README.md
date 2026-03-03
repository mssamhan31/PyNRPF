# Conference Paper Archive (`1_conference_paper`)

This folder is a frozen, self-contained archive of the conference publication
artifacts. It is isolated from ongoing package and model development.

## Structure

```text
publication/1_conference_paper/
  config/run.yaml
  dataset/raw/
    rpf_dataset.parquet
    sha256.txt
  notebooks/
    01_reproduce_key_numbers.ipynb
    01_reproduce_key_numbers.executed.ipynb
    02_publication_figures.ipynb
    02_publication_figures.executed.ipynb
    03_publication_tables.ipynb
    03_publication_tables.executed.ipynb
  models/
    xgb1_day.pkl
    xgb2_timestamp.pkl
  outputs/
    metrics__local_dev.json
    metrics__local_dev.yaml
    publication_figures/
    publication_tables/
  src/
  requirements.txt
```

## Run (standalone inside archive)

From the repository root:

```powershell
cd publication/1_conference_paper
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install jupyter nbconvert
```

Execute notebooks in order:

```powershell
.\.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/01_reproduce_key_numbers.ipynb --output 01_reproduce_key_numbers.executed.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=7200
.\.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/02_publication_figures.ipynb --output 02_publication_figures.executed.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=7200
.\.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/03_publication_tables.ipynb --output 03_publication_tables.executed.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=7200
```

Outputs are written under `publication/1_conference_paper/outputs`.

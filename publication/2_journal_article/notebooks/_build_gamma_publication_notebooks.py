"""Build readable Gamma-forecast and publication-inventory notebooks."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

KERNEL_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def write_notebook(path: Path, cells: list) -> Path:
    notebook = nbf.v4.new_notebook(cells=cells, metadata=KERNEL_METADATA)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)
    return path


def build_03(notebook_dir: Path) -> Path:
    cells = [
        markdown(
            """
# 03 Gamma Forecast Impact

**Research question.** How much do RPF sign errors, and their correction by
`m9_pbm`, change direct seven-day-ahead net-load point forecasts?

Gamma is **Beta substation B**. It was selected because it has the largest
raw-versus-manual data-error RMSE and a practically material correction effect;
it is not described as the substation with the most RPF days. The notebook
creates three real data conditions, trains three forecast models, evaluates
September 2024, and writes publication-ready outputs. No synthetic, smoke-only,
or placeholder metric rows are permitted.

**Inputs:** final Gamma data, the cached unlabeled Beta-B candidate features,
and the Beta-B outer-fold `m9_pbm` artifact from Notebook 02e.  
**Outputs:** corrected Gamma series, forecast audits, two metric tables, four
figures, and one reproducibility manifest.  
**Expected runtime:** about two to five minutes after Notebook 02b exists.
"""
        ),
        markdown(
            """
## 1. Imports, Paths, And Visible Configuration

The path search works from JupyterLab, VS Code, or the repository root. The
displayed horizon, lookback, test month, and XGBoost settings are the complete
forecast contract used below.
"""
        ),
        code(
            """
from pathlib import Path
import sys
import time

import pandas as pd
from IPython.display import Image, display


def find_notebook_directory() -> Path:
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if candidate.name == "notebooks" and (candidate / "_gamma_forecast.py").exists():
            return candidate
        nested = candidate / "publication" / "2_journal_article" / "notebooks"
        if (nested / "_gamma_forecast.py").exists():
            return nested
    raise FileNotFoundError("Could not locate the journal notebook directory.")


NOTEBOOK_DIR = find_notebook_directory()
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from _gamma_forecast import (  # noqa: E402
    CONDITION_COLUMNS,
    apply_m9_pbm_correction,
    build_forecast_examples,
    fit_direct_forecasts,
    forecast_impact_table,
    forecast_metric_rows,
    gamma_data_error_metrics,
    load_beta_b_model,
)
from _m9_pbm_data import (  # noqa: E402
    artifact_inventory,
    load_dataset,
    load_experiment_config,
    manifest_payload,
    output_dirs,
    resolve_paths,
    validate_input_hashes,
    write_csv,
    write_manifest,
    write_parquet,
)
from _m9_pbm_plotting import (  # noqa: E402
    plot_gamma_data_error,
    plot_gamma_example_week,
    plot_gamma_forecast_residuals,
    plot_gamma_forecast_rmse,
)

STARTED_AT = time.time()
ARTICLE_ROOT = NOTEBOOK_DIR.parent
CONFIG = load_experiment_config(ARTICLE_ROOT)
PATHS = resolve_paths(ARTICLE_ROOT, CONFIG)
SLUG = "03_gamma_forecast_impact"
OUTPUT_DIRS = output_dirs(PATHS, SLUG)

display(pd.Series(CONFIG["forecast"], name="forecast_configuration"))
display(pd.Series(CONFIG["windows"], name="experiment_windows"))
"""
        ),
        markdown(
            """
## 2. Inputs And Leakage Audit

The correction artifact was produced when Beta B was the **outer held-out
substation**. It was trained and tuned using sure-labelled days from Beta
substations A, C, D, E, F, G, and H. Alpha and Beta-B labels were both excluded.
The candidate cache contains physical features only; labels are joined later
solely to construct the manual reference and evaluate the result.
"""
        ),
        code(
            """
GAMMA_PATH = PATHS.final_data / "dataset_gamma.parquet"
CANDIDATE_PATH = (
    PATHS.intermediate
    / "02b_m9_pbm_candidate_features"
    / "_partitions"
    / "beta_beta_B_candidates.parquet"
)
MODEL_PATH = PATHS.manifests / "02e_m9_pbm_beta_B_outer_fold_model.json"

hash_audit = validate_input_hashes(PATHS, CONFIG)
gamma = load_dataset("gamma", article_root=ARTICLE_ROOT, config=CONFIG)
model = load_beta_b_model(MODEL_PATH)
candidates = pd.read_parquet(CANDIDATE_PATH)

assert gamma["substation_id"].unique().tolist() == ["beta_B"]
assert gamma.groupby("date").size().eq(96).all()
assert candidates["date"].nunique() == gamma["date"].nunique() == 366
assert not {"label_day", "label_interval", "confidence"}.intersection(candidates.columns)

leakage_audit = pd.Series(
    {
        "heldout_substation": model["heldout_substation"],
        "training_substations": ", ".join(model["training_substations"]),
        "heldout_labels_used": model["heldout_labels_used"],
        "alpha_used": model["alpha_used"],
        "candidate_days": candidates["date"].nunique(),
        "gamma_intervals": len(gamma),
    },
    name="leakage_audit",
)
display(hash_audit)
display(leakage_audit)
"""
        ),
        markdown(
            r"""
## 3. Three Gamma Data Conditions

For each quarter-hour $t$, the observed raw net load is $y(t)$. The physical
model selects one candidate window per day and predicts a day only when its
weighted score reaches the fixed Beta-B threshold:

\[
s(W)=w_1F_1(W)+w_3F_3(W)+w_4F_4(W),
\qquad
\widehat d=\mathbf 1\{\max_W s(W)\ge\tau\}.
\]

The model-corrected and manually corrected series are

\[
x_{m9}(t)=
\begin{cases}
-y(t), & \widehat d=1 \text{ and }t\in\widehat W,\\
y(t), & \text{otherwise},
\end{cases}
\qquad
x_{manual}(t)=
\begin{cases}
-y(t), & z(t)=1,\\
y(t), & z(t)=0.
\end{cases}
\]

**Notation.** $W$ is a candidate window; $F_1,F_3,F_4$ are bridge,
slope-continuity, and duration scores; $w_1,w_3,w_4$ are fixed fitted weights;
$\tau$ is the fixed threshold; $\widehat W$ is the highest-scoring window;
$\widehat d$ is the model's day decision; and $z(t)$ is the final manual
interval label. Crucially, $z(t)$ does not enter the `m9_pbm` score or decision.
"""
        ),
        code(
            """
SERIES_PATH = OUTPUT_DIRS["intermediate"] / "01_gamma_series.parquet"
DATA_ERROR_PATH = OUTPUT_DIRS["metrics"] / "01_gamma_data_error_metrics.csv"
DATA_ERROR_TABLE = OUTPUT_DIRS["tables"] / "table01_gamma_data_error_summary.csv"

gamma_series, m9_day_predictions = apply_m9_pbm_correction(
    gamma,
    candidates,
    model,
)
data_error_metrics = gamma_data_error_metrics(
    gamma_series,
    test_start=CONFIG["windows"]["gamma_forecast_test_start"],
    test_end=CONFIG["windows"]["gamma_forecast_test_end"],
)
write_parquet(gamma_series, SERIES_PATH)
write_csv(data_error_metrics, DATA_ERROR_PATH)
write_csv(data_error_metrics, DATA_ERROR_TABLE)

display(pd.Series(model["weights"], name="fixed_Beta_B_weights"))
print(f"Fixed threshold: {model['threshold']:.6f}")
display(data_error_metrics)
"""
        ),
        markdown(
            r"""
## 4. Direct Seven-Day-Ahead Forecast Design

Each row is one point forecast, not one element of a seven-day trajectory from
a common origin. For every September target $t$, the origin is exactly
$o=t-7\text{ days}$, and the predictor is

\[
\widehat x(t\mid o)=f_\theta\left(\mathcal H_{14}(o),c(t)\right).
\]

**Notation.** $t$ is one 15-minute target timestamp; $o$ is its forecast
origin; $\mathcal H_{14}(o)$ is the 14-day history ending at $o$; $c(t)$
contains only calendar values known for the target; $f_\theta$ is seasonal
naive, linear regression, or XGBoost; and $\theta$ denotes parameters fitted
once using target examples ending before September. No feature observes a value after $o$, and
the learned models are not refitted during September.

The 14-day history is represented by the same-quarter-hour values at the origin
and preceding 13 days, robust whole-window summaries, last-day summaries, and
the count of finite readings. Calendar features use cyclic time-of-day,
day-of-week, and month terms plus a weekend indicator.
"""
        ),
        code(
            """
TEST_START = CONFIG["windows"]["gamma_forecast_test_start"]
TEST_END = CONFIG["windows"]["gamma_forecast_test_end"]
TRAIN_START = gamma_series["timestamp"].min().strftime("%Y-%m-%d")
TRAIN_END = (pd.Timestamp(TEST_START) - pd.Timedelta(minutes=15)).strftime("%Y-%m-%d")
HORIZON_DAYS = int(CONFIG["forecast"]["horizon_days"])
LOOKBACK_DAYS = int(CONFIG["forecast"]["lookback_days"])

train_examples = {}
test_examples = {}
matrix_paths = []
for condition, column in CONDITION_COLUMNS.items():
    train = build_forecast_examples(
        gamma_series,
        column,
        target_start=TRAIN_START,
        target_end=TRAIN_END,
        horizon_days=HORIZON_DAYS,
        lookback_days=LOOKBACK_DAYS,
    )
    test = build_forecast_examples(
        gamma_series,
        column,
        target_start=TEST_START,
        target_end=TEST_END,
        horizon_days=HORIZON_DAYS,
        lookback_days=LOOKBACK_DAYS,
    )
    assert train["target_timestamp"].max() < pd.Timestamp(TEST_START, tz="UTC")
    assert test["target_timestamp"].sub(test["origin_timestamp"]).eq(
        pd.Timedelta(days=HORIZON_DAYS)
    ).all()
    train_examples[condition] = train
    test_examples[condition] = test
    train_path = OUTPUT_DIRS["intermediate"] / f"02_train_{condition}.parquet"
    test_path = OUTPUT_DIRS["intermediate"] / f"03_test_{condition}.parquet"
    write_parquet(train, train_path)
    write_parquet(test, test_path)
    matrix_paths.extend([train_path, test_path])

design_audit = pd.DataFrame(
    [
        {
            "data_condition": condition,
            "training_examples": len(train_examples[condition]),
            "training_target_end": train_examples[condition]["target_timestamp"].max(),
            "test_examples": len(test_examples[condition]),
            "test_origin_end": test_examples[condition]["origin_timestamp"].max(),
        }
        for condition in CONDITION_COLUMNS
    ]
)
display(design_audit)
"""
        ),
        markdown(
            """
## 5. Fit Once, Then Forecast September

Seasonal naive copies the value at the exact seven-day origin. Linear
regression standardises median-imputed physical-history features. XGBoost uses
the fixed configuration displayed in Section 1. Each learned model is fitted
once under each data condition. All predictions are evaluated against the
manually corrected target, so changing the training-data condition is the only
intended comparison.
"""
        ),
        code(
            """
prediction_frames = []
fit_audits = []
for condition in CONDITION_COLUMNS:
    condition_predictions, condition_audit = fit_direct_forecasts(
        train_examples[condition],
        test_examples[condition],
        data_condition=condition,
        config=CONFIG,
    )
    prediction_frames.append(condition_predictions)
    fit_audits.append(condition_audit)

forecast_predictions = pd.concat(prediction_frames, ignore_index=True)
fit_audit = pd.concat(fit_audits, ignore_index=True)
PREDICTION_PATH = OUTPUT_DIRS["intermediate"] / "04_gamma_forecast_predictions.parquet"
FIT_AUDIT_PATH = OUTPUT_DIRS["intermediate"] / "05_forecast_fit_audit.csv"
write_parquet(forecast_predictions, PREDICTION_PATH)
write_csv(fit_audit, FIT_AUDIT_PATH)

assert forecast_predictions["status"].eq("complete").all()
assert not forecast_predictions["is_placeholder"].any()
assert fit_audit["fit_count_per_learned_model"].eq(1).all()
display(fit_audit)
"""
        ),
        markdown(
            """
## 6. Forecast Metrics And Impact

RMSE and MAE use every target for which both the prediction and manually
corrected reference are finite. The valid-target count is displayed rather
than silently treating missing measurements as zero. The impact table reports
the RMSE change from raw to `m9_pbm`-corrected training data for each model and
shows the remaining distance to the ideal manually corrected condition.
"""
        ),
        code(
            """
FORECAST_METRICS_PATH = OUTPUT_DIRS["metrics"] / "02_gamma_forecast_metrics.csv"
FORECAST_TABLE = OUTPUT_DIRS["tables"] / "table02_gamma_forecast_impact.csv"

forecast_metrics = forecast_metric_rows(forecast_predictions)
impact = forecast_impact_table(forecast_metrics)
write_csv(forecast_metrics, FORECAST_METRICS_PATH)
write_csv(impact, FORECAST_TABLE)
display(forecast_metrics)
display(impact)
"""
        ),
        markdown(
            """
## 7. Publication Figures

The weekly curve exposes where the correction acts. The remaining figures
separate data error from forecast error and show the residual distributions,
so an average RMSE difference is not the only evidence available.
"""
        ),
        code(
            """
FIGURE_WEEK = OUTPUT_DIRS["figures"] / "fig01_gamma_raw_m9_manual_example_week.png"
FIGURE_DATA_ERROR = OUTPUT_DIRS["figures"] / "fig02_gamma_data_error_rmse.png"
FIGURE_FORECAST = OUTPUT_DIRS["figures"] / "fig03_gamma_forecast_rmse.png"
FIGURE_RESIDUAL = OUTPUT_DIRS["figures"] / "fig04_gamma_forecast_residuals.png"

plot_gamma_example_week(gamma_series, FIGURE_WEEK)
plot_gamma_data_error(data_error_metrics, FIGURE_DATA_ERROR)
plot_gamma_forecast_rmse(forecast_metrics, FIGURE_FORECAST)
plot_gamma_forecast_residuals(forecast_predictions, FIGURE_RESIDUAL)
for figure_path in [FIGURE_WEEK, FIGURE_DATA_ERROR, FIGURE_FORECAST, FIGURE_RESIDUAL]:
    display(Image(filename=figure_path))
"""
        ),
        markdown(
            """
## 8. Interpretation And Limitations

The code below prints the measured effects without assuming that correction
must improve every forecast model. This is a single-substation case study, and
the manually corrected series remains label-dependent. The Beta-B correction
is leakage-isolated with respect to labels and Alpha, but feature-family design
was informed by the wider Beta development process. Missing target readings
are excluded from metric denominators and are counted explicitly.
"""
        ),
        code(
            """
full_data = data_error_metrics.loc[data_error_metrics["scope"].eq("full_gamma")]
raw_data = full_data.loc[full_data["data_condition"].eq("raw_uncorrected")].iloc[0]
m9_data = full_data.loc[full_data["data_condition"].eq("m9_pbm_corrected")].iloc[0]
print(f"Full-year raw data-error RMSE: {raw_data['rmse_MW']:.3f} MW")
print(f"Full-year m9 data-error RMSE:  {m9_data['rmse_MW']:.3f} MW")
print(
    "Full-year RMSE reduction: "
    f"{m9_data['rmse_reduction_vs_raw_MW']:.3f} MW "
    f"({m9_data['rmse_reduction_vs_raw_pct']:.1f}%)"
)
display(impact[["model_label", "m9_rmse_reduction_vs_raw_MW", "m9_rmse_reduction_vs_raw_pct"]])
"""
        ),
        markdown(
            """
## 9. Reproducibility Manifest And Output Inventory

The manifest records all input hashes and output paths. The final assertions
fail if any declared artifact is missing, empty, placeholder, or inconsistent
with the fixed forecast design.
"""
        ),
        code(
            """
FINAL_OUTPUTS = [
    SERIES_PATH,
    DATA_ERROR_PATH,
    FORECAST_METRICS_PATH,
    DATA_ERROR_TABLE,
    FORECAST_TABLE,
    FIGURE_WEEK,
    FIGURE_DATA_ERROR,
    FIGURE_FORECAST,
    FIGURE_RESIDUAL,
]
manifest = manifest_payload(
    paths=PATHS,
    config=CONFIG,
    started_at=STARTED_AT,
    inputs=[PATHS.config, GAMMA_PATH, CANDIDATE_PATH, MODEL_PATH],
    outputs=FINAL_OUTPUTS,
    row_counts={
        "gamma_intervals": len(gamma_series),
        "gamma_days": gamma_series["date"].nunique(),
        "forecast_predictions": len(forecast_predictions),
        "valid_reference_targets": int(forecast_predictions["y_reference"].notna().sum()),
    },
)
manifest.update(
    {
        "status": "publication_ready",
        "gamma_substation": "beta_B",
        "contains_placeholder_forecasts": False,
        "correction_artifact": MODEL_PATH.name,
        "heldout_labels_used": False,
        "alpha_used": False,
        "forecast_models_fit_once_before_test_month": True,
        "local_intermediates": [
            str(path.relative_to(PATHS.article))
            for path in [*matrix_paths, PREDICTION_PATH, FIT_AUDIT_PATH]
        ],
    }
)
MANIFEST_PATH = write_manifest(PATHS, f"{SLUG}.json", manifest)
inventory = artifact_inventory(
    {path.name: path for path in [*FINAL_OUTPUTS, MANIFEST_PATH]},
    relative_to=PATHS.article,
)
display(inventory)
assert inventory["exists"].all() and inventory["bytes"].gt(0).all()
"""
        ),
    ]
    return write_notebook(notebook_dir / "03_gamma_forecast_impact.ipynb", cells)


def build_04(notebook_dir: Path) -> Path:
    cells = [
        markdown(
            """
# 04 Publication Tables And Figures

**Purpose.** Consolidate the compact, paper-facing outputs from Notebooks
02a-02g and 03 into one explicit publication inventory and a small final table
set. This notebook does not rerun model development and does not read
exploratory caches.

Every final table is exported as CSV, Markdown, and LaTeX. Missing upstream
artifacts are displayed and treated as execution errors rather than silently
omitted.

**Inputs:** compact tables, figures, and manifests from 02a-02g and 03.  
**Outputs:** seven final table families in three formats, an upstream inventory,
and a reproducibility manifest.  
**Expected runtime:** under one minute after the upstream notebooks have run.
"""
        ),
        markdown(
            """
## 1. Imports And Output Contract

Only compact tables, figures, and manifests from the repeatable notebook
workflow are declared below. Paths are relative to the article folder and are
portable across laptops.
"""
        ),
        code(
            """
from pathlib import Path
import sys
import time

import pandas as pd
from IPython.display import display


def find_notebook_directory() -> Path:
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if candidate.name == "notebooks" and (candidate / "_m9_pbm_data.py").exists():
            return candidate
        nested = candidate / "publication" / "2_journal_article" / "notebooks"
        if (nested / "_m9_pbm_data.py").exists():
            return nested
    raise FileNotFoundError("Could not locate the journal notebook directory.")


NOTEBOOK_DIR = find_notebook_directory()
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from _m9_pbm_data import (  # noqa: E402
    artifact_inventory,
    load_experiment_config,
    manifest_payload,
    output_dirs,
    resolve_paths,
    write_manifest,
    write_table_formats,
)

STARTED_AT = time.time()
ARTICLE_ROOT = NOTEBOOK_DIR.parent
CONFIG = load_experiment_config(ARTICLE_ROOT)
PATHS = resolve_paths(ARTICLE_ROOT, CONFIG)
SLUG = "04_publication_tables_figures"
OUTPUT_DIRS = output_dirs(PATHS, SLUG)
"""
        ),
        markdown(
            """
## 2. Upstream Artifact Inventory

The table registry selects headline regime, ablation, weight, ML, final
evaluation, localisation, and Gamma forecast results. The figure registry
checks every paper-facing figure generated by 02a-02g and 03. Notebook 02b has
no required plot, so its compact feature-quality table is included in the
inventory as its hand-off artifact.
"""
        ),
        code(
            """
SOURCE_TABLES = {
    "table01_m9_training_regimes": PATHS.tables
    / "02c_m9_pbm_training_regimes"
    / "table01_regime_headline_metrics.csv",
    "table02_m9_ablation_by_feature_count": PATHS.tables
    / "02d_m9_pbm_feature_ablation"
    / "table01_best_by_feature_count.csv",
    "table03_m9_weight_optimisation": PATHS.tables
    / "02e_m9_pbm_weight_optimisation"
    / "table01_equal_vs_grid_vs_random.csv",
    "table04_m9_physical_vs_ml": PATHS.tables
    / "02f_m9_pbm_ml_comparison"
    / "table01_physical_vs_ml.csv",
    "table05_m9_final_evaluation": PATHS.tables
    / "02g_m9_pbm_final_evaluation"
    / "table01_final_headline_metrics.csv",
    "table06_m9_localisation_and_energy": PATHS.tables
    / "02g_m9_pbm_final_evaluation"
    / "table03_localisation_and_energy.csv",
    "table07_gamma_forecast_impact": PATHS.tables
    / "03_gamma_forecast_impact"
    / "table02_gamma_forecast_impact.csv",
}

UPSTREAM_FIGURES = {
    "02a_method_example": PATHS.figures
    / "02a_m9_pbm_method_example"
    / "fig01_m9_pbm_alpha_F_2024-02-17.png",
    "02c_regime_metrics": PATHS.figures
    / "02c_m9_pbm_training_regimes"
    / "fig01_regime_precision_recall_f1.png",
    "02c_regime_thresholds": PATHS.figures
    / "02c_m9_pbm_training_regimes"
    / "fig02_thresholds_by_heldout_substation.png",
    "02d_ablation_feature_count": PATHS.figures
    / "02d_m9_pbm_feature_ablation"
    / "fig01_f1_by_feature_count.png",
    "02d_ablation_top_subsets": PATHS.figures
    / "02d_m9_pbm_feature_ablation"
    / "fig02_top_subset_performance.png",
    "02d_ablation_feature_evidence": PATHS.figures
    / "02d_m9_pbm_feature_ablation"
    / "fig03_feature_frequency_and_marginal_effect.png",
    "02e_weight_simplex": PATHS.figures
    / "02e_m9_pbm_weight_optimisation"
    / "fig01_weight_simplex_performance.png",
    "02e_weight_stability": PATHS.figures
    / "02e_m9_pbm_weight_optimisation"
    / "fig02_selected_weights_by_fold.png",
    "02f_physical_vs_ml": PATHS.figures
    / "02f_m9_pbm_ml_comparison"
    / "fig01_physical_vs_ml_precision_recall_f1.png",
    "02g_confusion": PATHS.figures
    / "02g_m9_pbm_final_evaluation"
    / "fig01_final_confusion_matrices.png",
    "02g_window_iou": PATHS.figures
    / "02g_m9_pbm_final_evaluation"
    / "fig02_window_iou_distribution.png",
    "02g_energy": PATHS.figures
    / "02g_m9_pbm_final_evaluation"
    / "fig03_energy_metric_summary.png",
    "02g_review_burden": PATHS.figures
    / "02g_m9_pbm_final_evaluation"
    / "fig04_auto_accept_manual_review_and_errors.png",
    "02g_coverage_scores": PATHS.figures
    / "02g_m9_pbm_final_evaluation"
    / "fig05_auto_accept_precision_recall_f1.png",
    "03_gamma_week": PATHS.figures
    / "03_gamma_forecast_impact"
    / "fig01_gamma_raw_m9_manual_example_week.png",
    "03_gamma_data_error": PATHS.figures
    / "03_gamma_forecast_impact"
    / "fig02_gamma_data_error_rmse.png",
    "03_gamma_forecast_rmse": PATHS.figures
    / "03_gamma_forecast_impact"
    / "fig03_gamma_forecast_rmse.png",
    "03_gamma_residuals": PATHS.figures
    / "03_gamma_forecast_impact"
    / "fig04_gamma_forecast_residuals.png",
}

SUPPORTING_OUTPUTS = {
    "02b_feature_quality": PATHS.tables
    / "02b_m9_pbm_candidate_features"
    / "table02_feature_quality_summary.csv",
    **{
        f"manifest_{name}": PATHS.manifests / f"{name}.json"
        for name in [
            "02a_m9_pbm_method_example",
            "02b_m9_pbm_candidate_features",
            "02c_m9_pbm_training_regimes",
            "02d_m9_pbm_feature_ablation",
            "02e_m9_pbm_weight_optimisation",
            "02f_m9_pbm_ml_comparison",
            "02g_m9_pbm_final_evaluation",
            "03_gamma_forecast_impact",
        ]
    },
}

upstream = {**SOURCE_TABLES, **UPSTREAM_FIGURES, **SUPPORTING_OUTPUTS}
upstream_inventory = artifact_inventory(upstream, relative_to=PATHS.article)
display(upstream_inventory)
assert upstream_inventory["exists"].all(), "Required upstream outputs are missing."
assert upstream_inventory["bytes"].gt(0).all(), "An upstream output is empty."
"""
        ),
        markdown(
            """
## 3. Export Final Paper Tables

The exported values are copied from compact upstream tables without
re-estimating or manually transcribing metrics. CSV supports programmatic use,
Markdown supports rapid review, and LaTeX supports manuscript drafting.
"""
        ),
        code(
            """
exported_paths = []
table_audit_rows = []
for final_name, source_path in SOURCE_TABLES.items():
    table = pd.read_csv(source_path)
    paths = write_table_formats(table, OUTPUT_DIRS["tables"] / final_name)
    exported_paths.extend(paths)
    table_audit_rows.append(
        {
            "final_table": final_name,
            "source": str(source_path.relative_to(PATHS.article)).replace("\\\\", "/"),
            "rows": len(table),
            "columns": len(table.columns),
        }
    )

table_audit = pd.DataFrame(table_audit_rows)
inventory_paths = write_table_formats(
    upstream_inventory,
    OUTPUT_DIRS["tables"] / "table00_upstream_artifact_inventory",
)
exported_paths.extend(inventory_paths)
display(table_audit)
"""
        ),
        markdown(
            """
## 4. Final Checks And Manifest

This final audit verifies all three formats for every table and records the
upstream dependency graph. Figures remain in their originating notebook
folders so the inventory has one authoritative path for each image.
"""
        ),
        code(
            """
final_inventory = artifact_inventory(
    {path.name + "_" + path.parent.name: path for path in exported_paths},
    relative_to=PATHS.article,
)
assert final_inventory["exists"].all() and final_inventory["bytes"].gt(0).all()

manifest = manifest_payload(
    paths=PATHS,
    config=CONFIG,
    started_at=STARTED_AT,
    inputs=list(upstream.values()),
    outputs=exported_paths,
    row_counts={
        "upstream_artifacts": len(upstream_inventory),
        "final_table_families": len(SOURCE_TABLES),
        "final_table_files": len(exported_paths),
        "upstream_figures": len(UPSTREAM_FIGURES),
    },
)
manifest.update(
    {
        "status": "publication_ready",
        "source_notebooks": ["02a", "02b", "02c", "02d", "02e", "02f", "02g", "03"],
        "missing_upstream_outputs": 0,
        "figure_paths": {
            name: str(path.relative_to(PATHS.article)).replace("\\\\", "/")
            for name, path in UPSTREAM_FIGURES.items()
        },
    }
)
MANIFEST_PATH = write_manifest(PATHS, f"{SLUG}.json", manifest)
display(final_inventory)
print(
    f"Wrote {len(SOURCE_TABLES)} final table families and checked "
    f"{len(UPSTREAM_FIGURES)} figures."
)
print(f"Manifest: {MANIFEST_PATH.relative_to(PATHS.article)}")
"""
        ),
    ]
    return write_notebook(notebook_dir / "04_publication_tables_figures.ipynb", cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", choices=["03", "04", "all"], default="all")
    args = parser.parse_args()
    notebook_dir = Path(__file__).resolve().parent
    builders = {"03": build_03, "04": build_04}
    selected = list(builders) if args.notebook == "all" else [args.notebook]
    for name in selected:
        print(builders[name](notebook_dir))


if __name__ == "__main__":
    main()

"""Fast figure-only rendering from persisted journal experiment results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import _experiment_helpers as characterisation
import pandas as pd
from _figure_sources import figure_source_path, load_figure_source, write_figure_source
from _m9_pbm_data import load_experiment_config, resolve_paths
from _m9_pbm_plotting import (
    plot_ablation_by_feature_count,
    plot_ablation_feature_evidence,
    plot_auto_accept_burden,
    plot_coverage_scores,
    plot_energy_summary,
    plot_final_confusion_matrices,
    plot_gamma_data_error,
    plot_gamma_example_week,
    plot_gamma_forecast_residuals,
    plot_gamma_forecast_rmse,
    plot_method_example,
    plot_physical_vs_ml,
    plot_regime_metrics,
    plot_regime_thresholds,
    plot_selected_weights,
    plot_top_ablation_subsets,
    plot_weight_simplex,
    plot_window_iou_distribution,
)

SUPPORTED_SLUGS = {
    "01_characterisation",
    "02a_m9_pbm_method_example",
    "02c_m9_pbm_training_regimes",
    "02d_m9_pbm_feature_ablation",
    "02e_m9_pbm_weight_optimisation",
    "02f_m9_pbm_ml_comparison",
    "02g_m9_pbm_final_evaluation",
    "03_gamma_forecast_impact",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required persisted result is missing: {path}")
    return pd.read_csv(path)


def _source(
    source_root: Path,
    slug: str,
    name: str,
    factory: Callable[[], pd.DataFrame],
    *,
    required_columns: tuple[str, ...],
    refresh: bool,
) -> pd.DataFrame:
    path = figure_source_path(source_root, slug, name)
    if refresh or not path.exists():
        write_figure_source(factory(), path, required_columns=required_columns)
    return load_figure_source(path, required_columns=required_columns)


def _render_characterisation(paths, refresh: bool) -> list[Path]:
    slug = "01_characterisation"
    intermediate = paths.intermediate / slug
    daytype_path = intermediate / "05_rpf_month_daytype_summary.csv"

    def daytype_factory() -> pd.DataFrame:
        if daytype_path.exists():
            return _read_csv(daytype_path)
        cfg = characterisation.load_config(paths.article)
        return pd.concat(
            [
                characterisation.rpf_daytype_summary(
                    characterisation.load_dataset(paths.article, cfg, "alpha"), "Alpha"
                ),
                characterisation.rpf_daytype_summary(
                    characterisation.load_dataset(paths.article, cfg, "beta"), "Beta"
                ),
            ],
            ignore_index=True,
        )

    occurrence = _source(
        paths.figure_sources,
        slug,
        "site_occurrence",
        lambda: _read_csv(intermediate / "02_rpf_occurrence_by_site.csv"),
        required_columns=("dataset", "substation_id", "rpf_day_pct"),
        refresh=refresh,
    )
    temporal = _source(
        paths.figure_sources,
        slug,
        "temporal_summary",
        lambda: _read_csv(intermediate / "03_rpf_temporal_summary.csv"),
        required_columns=("dataset", "level", "rpf_interval_pct"),
        refresh=refresh,
    )
    events = _source(
        paths.figure_sources,
        slug,
        "event_durations",
        lambda: _read_csv(intermediate / "04_rpf_event_summary.csv"),
        required_columns=("dataset", "duration_hours"),
        refresh=refresh,
    )
    daytype = _source(
        paths.figure_sources,
        slug,
        "month_daytype",
        daytype_factory,
        required_columns=("dataset", "month", "daytype", "rpf_site_day_pct"),
        refresh=refresh,
    )
    event_counts = _source(
        paths.figure_sources,
        slug,
        "event_count_distribution",
        lambda: _read_csv(intermediate / "06_rpf_event_count_by_day_distribution.csv"),
        required_columns=("dataset", "plot_category", "n_rpf_site_days"),
        refresh=refresh,
    )
    return characterisation.write_characterisation_figures(
        occurrence,
        temporal,
        events,
        daytype,
        event_counts,
        paths.figures / slug,
    )


def _render_02a(paths, refresh: bool) -> list[Path]:
    slug = "02a_m9_pbm_method_example"
    source = _source(
        paths.figure_sources,
        slug,
        "method_example",
        lambda: _read_csv(
            paths.tables / slug / "table01_alpha_F_2024-02-17_plot_data.csv"
        ),
        required_columns=("timestamp", "candidate_window", "bridge_anchor"),
        refresh=refresh,
    )
    output = paths.figures / slug / "fig01_m9_pbm_alpha_F_2024-02-17.png"
    return [plot_method_example(source, output)]


def _render_02c(paths, refresh: bool) -> list[Path]:
    slug = "02c_m9_pbm_training_regimes"
    metric_path = paths.metrics / slug / "01_day_metrics.csv"
    pooled = _source(
        paths.figure_sources,
        slug,
        "pooled_beta_sure_metrics",
        lambda: _read_csv(metric_path).loc[
            lambda data: data["confidence_scope"].eq("beta_sure")
            & data["aggregation"].eq("pooled")
        ],
        required_columns=("regime", "precision", "recall", "f1"),
        refresh=refresh,
    )
    thresholds = _source(
        paths.figure_sources,
        slug,
        "fold_thresholds",
        lambda: _read_csv(paths.metrics / slug / "02_thresholds_by_fold.csv"),
        required_columns=("regime", "heldout_substation", "threshold"),
        refresh=refresh,
    )
    figure_dir = paths.figures / slug
    return [
        plot_regime_metrics(pooled, figure_dir / "fig01_regime_precision_recall_f1.png"),
        plot_regime_thresholds(
            thresholds,
            figure_dir / "fig02_thresholds_by_heldout_substation.png",
        ),
    ]


def _render_02d(paths, refresh: bool) -> list[Path]:
    slug = "02d_m9_pbm_feature_ablation"
    metrics = _source(
        paths.figure_sources,
        slug,
        "all_subset_metrics",
        lambda: _read_csv(paths.metrics / slug / "01_all_511_subset_metrics.csv"),
        required_columns=("feature_count", "feature_set_short", "beta_sure_f1"),
        refresh=refresh,
    )
    best = _source(
        paths.figure_sources,
        slug,
        "best_by_feature_count",
        lambda: _read_csv(paths.tables / slug / "table01_best_by_feature_count.csv"),
        required_columns=("feature_count", "beta_sure_f1"),
        refresh=refresh,
    )
    evidence = _source(
        paths.figure_sources,
        slug,
        "feature_evidence",
        lambda: _read_csv(paths.tables / slug / "table02_feature_frequency.csv"),
        required_columns=(
            "feature_number",
            "top_25_frequency_pct",
            "mean_paired_delta_f1",
        ),
        refresh=refresh,
    )
    figure_dir = paths.figures / slug
    return [
        plot_ablation_by_feature_count(metrics, best, figure_dir / "fig01_f1_by_feature_count.png"),
        plot_top_ablation_subsets(metrics, figure_dir / "fig02_top_subset_performance.png"),
        plot_ablation_feature_evidence(
            evidence,
            figure_dir / "fig03_feature_frequency_and_marginal_effect.png",
        ),
    ]


def _render_02e(paths, refresh: bool) -> list[Path]:
    slug = "02e_m9_pbm_weight_optimisation"

    def search_factory() -> pd.DataFrame:
        return pd.concat(
            [
                _read_csv(paths.metrics / slug / "01_grid_search_results.csv"),
                _read_csv(paths.metrics / slug / "02_random_search_results.csv"),
            ],
            ignore_index=True,
        )

    search = _source(
        paths.figure_sources,
        slug,
        "weight_search",
        search_factory,
        required_columns=("weight_F1", "weight_F3", "weight_F4", "inner_macro_f1"),
        refresh=refresh,
    )
    selected = _source(
        paths.figure_sources,
        slug,
        "selected_weights",
        lambda: _read_csv(paths.metrics / slug / "04_selected_weights_and_thresholds.csv"),
        required_columns=("strategy", "heldout_substation", "weight_F1", "weight_F3", "weight_F4"),
        refresh=refresh,
    )
    figure_dir = paths.figures / slug
    return [
        plot_weight_simplex(search, figure_dir / "fig01_weight_simplex_performance.png"),
        plot_selected_weights(selected, figure_dir / "fig02_selected_weights_by_fold.png"),
    ]


def _render_02f(paths, refresh: bool) -> list[Path]:
    slug = "02f_m9_pbm_ml_comparison"

    def comparison_factory() -> pd.DataFrame:
        data = _read_csv(paths.tables / slug / "table01_physical_vs_ml.csv")
        keep = data["aggregation"].eq("pooled") & (
            data["model"].eq("m9_pbm_optimised_physical")
            | data["regime"].eq("beta_only")
        )
        data = data.loc[keep].copy()
        labels = {
            "m9_pbm_optimised_physical": "Optimised physical",
            "dnn": "DNN / Beta only",
            "random_forest": "RF / Beta only",
            "xgboost": "XGBoost / Beta only",
        }
        data["display_label"] = data["model"].map(labels)
        order = list(labels)
        data["model"] = pd.Categorical(data["model"], categories=order, ordered=True)
        return data.sort_values("model")

    comparison = _source(
        paths.figure_sources,
        slug,
        "beta_only_model_comparison",
        comparison_factory,
        required_columns=("display_label", "precision", "recall", "f1"),
        refresh=refresh,
    )
    output = paths.figures / slug / "fig01_physical_vs_ml_precision_recall_f1.png"
    return [plot_physical_vs_ml(comparison, output)]


def _render_02g(paths, refresh: bool) -> list[Path]:
    slug = "02g_m9_pbm_final_evaluation"

    def window_audit_factory() -> pd.DataFrame:
        interval_audit = pd.read_parquet(
            paths.intermediate / slug / "heldout_prediction_audit.parquet"
        )
        rows = []
        for (substation, date), group in interval_audit.groupby(
            ["substation_id", "date"], sort=True
        ):
            true_slots = set(group.loc[group["label_interval"], "slot"].astype(int))
            predicted_slots = set(
                group.loc[group["predicted_interval"], "slot"].astype(int)
            )
            union = true_slots | predicted_slots
            rows.append(
                {
                    "substation_id": substation,
                    "date": date,
                    "confidence": group["confidence"].iloc[0],
                    "true_day": bool(group["label_day"].max()),
                    "predicted_day": bool(group["predicted_day"].iloc[0]),
                    "window_iou": len(true_slots & predicted_slots) / len(union)
                    if union
                    else 1.0,
                }
            )
        return pd.DataFrame(rows)

    day_metrics = _source(
        paths.figure_sources,
        slug,
        "day_metrics",
        lambda: _read_csv(paths.metrics / slug / "01_day_metrics.csv"),
        required_columns=("confidence_scope", "aggregation", "tp", "fp", "fn", "tn"),
        refresh=refresh,
    )
    window_audit = _source(
        paths.figure_sources,
        slug,
        "window_audit",
        window_audit_factory,
        required_columns=("confidence", "true_day", "predicted_day", "window_iou"),
        refresh=refresh,
    )
    energy = _source(
        paths.figure_sources,
        slug,
        "energy_metrics",
        lambda: _read_csv(paths.metrics / slug / "04_energy_metrics.csv"),
        required_columns=("confidence_scope", "aggregation", "interval_scope", "energy_f1"),
        refresh=refresh,
    )
    coverage = _source(
        paths.figure_sources,
        slug,
        "coverage_metrics",
        lambda: _read_csv(paths.metrics / slug / "05_confidence_coverage_metrics.csv"),
        required_columns=("coverage_pct", "manual_review_days", "fp", "fn", "f1"),
        refresh=refresh,
    )
    figure_dir = paths.figures / slug
    return [
        plot_final_confusion_matrices(
            day_metrics,
            figure_dir / "fig01_final_confusion_matrices.png",
        ),
        plot_window_iou_distribution(
            window_audit,
            figure_dir / "fig02_window_iou_distribution.png",
        ),
        plot_energy_summary(energy, figure_dir / "fig03_energy_metric_summary.png"),
        plot_auto_accept_burden(
            coverage,
            figure_dir / "fig04_auto_accept_manual_review_and_errors.png",
        ),
        plot_coverage_scores(
            coverage,
            figure_dir / "fig05_auto_accept_precision_recall_f1.png",
        ),
    ]


def _render_03(paths, refresh: bool) -> list[Path]:
    slug = "03_gamma_forecast_impact"
    series = _source(
        paths.figure_sources,
        slug,
        "gamma_series",
        lambda: pd.read_parquet(paths.intermediate / slug / "01_gamma_series.parquet"),
        required_columns=(
            "timestamp",
            "raw_uncorrected_MW",
            "m9_pbm_corrected_MW",
            "manually_corrected_MW",
        ),
        refresh=refresh,
    )
    data_error = _source(
        paths.figure_sources,
        slug,
        "data_error_metrics",
        lambda: _read_csv(paths.metrics / slug / "01_gamma_data_error_metrics.csv"),
        required_columns=("scope", "data_condition", "rmse_MW"),
        refresh=refresh,
    )
    forecast = _source(
        paths.figure_sources,
        slug,
        "forecast_metrics",
        lambda: _read_csv(paths.metrics / slug / "02_gamma_forecast_metrics.csv"),
        required_columns=("model", "data_condition", "rmse_MW"),
        refresh=refresh,
    )
    predictions = _source(
        paths.figure_sources,
        slug,
        "forecast_predictions",
        lambda: pd.read_parquet(
            paths.intermediate / slug / "04_gamma_forecast_predictions.parquet"
        ),
        required_columns=("model", "data_condition", "y_pred", "y_reference"),
        refresh=refresh,
    )
    figure_dir = paths.figures / slug
    return [
        plot_gamma_example_week(series, figure_dir / "fig01_gamma_raw_m9_manual_example_week.png"),
        plot_gamma_data_error(data_error, figure_dir / "fig02_gamma_data_error_rmse.png"),
        plot_gamma_forecast_rmse(forecast, figure_dir / "fig03_gamma_forecast_rmse.png"),
        plot_gamma_forecast_residuals(
            predictions,
            figure_dir / "fig04_gamma_forecast_residuals.png",
        ),
    ]


RENDERERS = {
    "01_characterisation": _render_characterisation,
    "02a_m9_pbm_method_example": _render_02a,
    "02c_m9_pbm_training_regimes": _render_02c,
    "02d_m9_pbm_feature_ablation": _render_02d,
    "02e_m9_pbm_weight_optimisation": _render_02e,
    "02f_m9_pbm_ml_comparison": _render_02f,
    "02g_m9_pbm_final_evaluation": _render_02g,
    "03_gamma_forecast_impact": _render_03,
}


def render_notebook_figures(
    article_root: Path,
    slug: str,
    *,
    refresh_sources: bool = False,
) -> list[Path]:
    """Render one notebook's figures without running its scientific pipeline."""

    if slug not in SUPPORTED_SLUGS:
        raise ValueError(f"Unsupported figure notebook: {slug}")
    config = load_experiment_config(article_root)
    paths = resolve_paths(article_root, config)
    outputs = RENDERERS[slug](paths, refresh_sources)
    receipt = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "figure_only",
        "notebook_slug": slug,
        "refresh_sources": refresh_sources,
        "figures": [str(path.relative_to(paths.article)) for path in outputs],
        "figure_sources": [
            str(path.relative_to(paths.article))
            for path in sorted((paths.figure_sources / slug).glob("*.parquet"))
        ],
    }
    paths.manifests.mkdir(parents=True, exist_ok=True)
    receipt_path = paths.manifests / f"{slug}_render.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return outputs
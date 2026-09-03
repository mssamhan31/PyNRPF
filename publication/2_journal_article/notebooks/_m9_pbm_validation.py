"""Leakage-safe split, threshold, and metric helpers for m9_pbm experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

METRIC_COLUMNS = [
    "support",
    "positive_support",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "f1",
]


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    metrics: dict[str, float | int]
    sweep: pd.DataFrame


def binary_metrics(
    truth: Iterable[bool],
    prediction: Iterable[bool],
) -> dict[str, float | int]:
    """Compute stable binary counts and P/R/F1 with zero-division set to zero."""

    true = np.asarray(list(truth), dtype=bool)
    pred = np.asarray(list(prediction), dtype=bool)
    if true.shape != pred.shape:
        raise ValueError("Truth and prediction must have the same shape.")
    tp = int((true & pred).sum())
    fp = int((~true & pred).sum())
    fn = int((true & ~pred).sum())
    tn = int((~true & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "support": int(len(true)),
        "positive_support": int(true.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def metric_rows(
    frame: pd.DataFrame,
    *,
    truth_column: str = "true_day",
    prediction_column: str = "predicted_day",
    substation_column: str = "substation_id",
) -> pd.DataFrame:
    """Return pooled, per-substation, and macro-substation metric rows."""

    pooled = {
        "aggregation": "pooled",
        "substation_id": "",
        **binary_metrics(frame[truth_column], frame[prediction_column]),
    }
    substations = []
    for substation, group in frame.groupby(substation_column, sort=True):
        substations.append(
            {
                "aggregation": "substation",
                "substation_id": substation,
                **binary_metrics(group[truth_column], group[prediction_column]),
            }
        )
    substation_frame = pd.DataFrame(substations)
    macro = {
        "aggregation": "macro_substation",
        "substation_id": "",
        "support": int(substation_frame["support"].sum()),
        "positive_support": int(substation_frame["positive_support"].sum()),
        "tp": int(substation_frame["tp"].sum()),
        "fp": int(substation_frame["fp"].sum()),
        "fn": int(substation_frame["fn"].sum()),
        "tn": int(substation_frame["tn"].sum()),
        "precision": float(substation_frame["precision"].mean()),
        "recall": float(substation_frame["recall"].mean()),
        "f1": float(substation_frame["f1"].mean()),
    }
    return pd.concat(
        [pd.DataFrame([pooled]), substation_frame, pd.DataFrame([macro])],
        ignore_index=True,
    )


def beta_loso_folds(frame: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return one train/test index pair for each Beta held-out substation."""

    if "substation_id" not in frame:
        raise ValueError("LOSO input requires substation_id.")
    substations = sorted(frame["substation_id"].astype(str).unique())
    if len(substations) < 2:
        raise ValueError("LOSO requires at least two substations.")
    folds = []
    for heldout in substations:
        test_mask = frame["substation_id"].eq(heldout).to_numpy()
        train_index = frame.index[~test_mask].to_numpy()
        test_index = frame.index[test_mask].to_numpy()
        folds.append((heldout, train_index, test_index))
    return folds


def assert_heldout_absent(training: pd.DataFrame, heldout_substation: str) -> None:
    """Fail loudly if an outer or inner held-out substation reaches tuning data."""

    present = training["substation_id"].astype(str).eq(heldout_substation).any()
    if present:
        raise AssertionError(f"Held-out substation {heldout_substation} is present in training.")


def _group_weights(data: pd.DataFrame, dataset_balanced: bool) -> pd.Series:
    groups = data[["dataset", "substation_id"]].drop_duplicates().copy()
    if dataset_balanced:
        dataset_counts = groups["dataset"].value_counts()
        groups["weight"] = groups["dataset"].map(
            lambda value: 1 / len(dataset_counts) / dataset_counts[value]
        )
    else:
        groups["weight"] = 1 / len(groups)
    return groups.set_index(["dataset", "substation_id"])["weight"]


def _threshold_metrics(
    data: pd.DataFrame,
    *,
    threshold: float,
    score_column: str,
    dataset_balanced: bool,
) -> dict[str, float | int]:
    group_rows = []
    for (dataset, substation), group in data.groupby(["dataset", "substation_id"], sort=True):
        metrics = binary_metrics(group["true_day"], group[score_column].ge(threshold))
        group_rows.append({"dataset": dataset, "substation_id": substation, **metrics})
    grouped = pd.DataFrame(group_rows)
    weights = _group_weights(data, dataset_balanced)
    grouped["weight"] = [
        weights.loc[(row.dataset, row.substation_id)] for row in grouped.itertuples()
    ]
    pooled = binary_metrics(data["true_day"], data[score_column].ge(threshold))
    return {
        "macro_precision": float((grouped["precision"] * grouped["weight"]).sum()),
        "macro_recall": float((grouped["recall"] * grouped["weight"]).sum()),
        "macro_f1": float((grouped["f1"] * grouped["weight"]).sum()),
        "pooled_precision": pooled["precision"],
        "pooled_recall": pooled["recall"],
        "pooled_f1": pooled["f1"],
        "tp": pooled["tp"],
        "fp": pooled["fp"],
        "fn": pooled["fn"],
        "tn": pooled["tn"],
    }


def select_threshold(
    training: pd.DataFrame,
    *,
    score_column: str = "score",
    dataset_balanced: bool = False,
) -> ThresholdSelection:
    """Select threshold by macro-substation F1 using the PRD tie-break order."""

    required = {"dataset", "substation_id", "true_day", score_column}
    missing = required - set(training.columns)
    if missing:
        raise ValueError(f"Threshold input is missing columns: {sorted(missing)}")
    data = training[[*required]].dropna().copy()
    if data.empty:
        raise ValueError("Cannot select a threshold from an empty training frame.")
    data["tuning_group"] = (
        data["dataset"].astype(str) + "|" + data["substation_id"].astype(str)
    )
    groups = sorted(data["tuning_group"].unique())
    group_index = {group: index for index, group in enumerate(groups)}
    group_codes = data["tuning_group"].map(group_index).to_numpy(dtype=int)
    truth = data["true_day"].astype(bool).to_numpy()

    scores_ascending = np.sort(data[score_column].astype(float).unique())
    score_codes_ascending = np.searchsorted(
        scores_ascending, data[score_column].to_numpy(dtype=float)
    )
    score_codes_descending = len(scores_ascending) - 1 - score_codes_ascending
    tp_increments = np.zeros((len(scores_ascending), len(groups)), dtype=float)
    fp_increments = np.zeros_like(tp_increments)
    np.add.at(
        tp_increments,
        (score_codes_descending[truth], group_codes[truth]),
        1,
    )
    np.add.at(
        fp_increments,
        (score_codes_descending[~truth], group_codes[~truth]),
        1,
    )
    tp = np.vstack([np.zeros((1, len(groups))), np.cumsum(tp_increments, axis=0)])
    fp = np.vstack([np.zeros((1, len(groups))), np.cumsum(fp_increments, axis=0)])

    group_support = data.groupby("tuning_group").size().reindex(groups).to_numpy(dtype=float)
    group_positive = (
        data.groupby("tuning_group")["true_day"].sum().reindex(groups).to_numpy(dtype=float)
    )
    group_negative = group_support - group_positive
    fn = group_positive[None, :] - tp
    tn = group_negative[None, :] - fp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(
        tp,
        group_positive[None, :],
        out=np.zeros_like(tp),
        where=group_positive[None, :] > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )

    group_lookup = (
        data[["tuning_group", "dataset", "substation_id"]]
        .drop_duplicates()
        .set_index("tuning_group")
        .loc[groups]
    )
    if dataset_balanced:
        dataset_counts = group_lookup["dataset"].value_counts()
        weights = np.array(
            [
                1 / len(dataset_counts) / dataset_counts[row.dataset]
                for row in group_lookup.itertuples()
            ]
        )
    else:
        weights = np.full(len(groups), 1 / len(groups))

    pooled_tp = tp.sum(axis=1)
    pooled_fp = fp.sum(axis=1)
    pooled_fn = fn.sum(axis=1)
    pooled_tn = tn.sum(axis=1)
    pooled_precision = np.divide(
        pooled_tp,
        pooled_tp + pooled_fp,
        out=np.zeros_like(pooled_tp),
        where=(pooled_tp + pooled_fp) > 0,
    )
    pooled_recall = np.divide(
        pooled_tp,
        pooled_tp + pooled_fn,
        out=np.zeros_like(pooled_tp),
        where=(pooled_tp + pooled_fn) > 0,
    )
    pooled_f1 = np.divide(
        2 * pooled_precision * pooled_recall,
        pooled_precision + pooled_recall,
        out=np.zeros_like(pooled_precision),
        where=(pooled_precision + pooled_recall) > 0,
    )
    thresholds = np.r_[scores_ascending[-1] + 1e-12, scores_ascending[::-1]]
    sweep = pd.DataFrame(
        {
            "threshold": thresholds,
            "macro_precision": precision @ weights,
            "macro_recall": recall @ weights,
            "macro_f1": f1 @ weights,
            "pooled_precision": pooled_precision,
            "pooled_recall": pooled_recall,
            "pooled_f1": pooled_f1,
            "tp": pooled_tp.astype(int),
            "fp": pooled_fp.astype(int),
            "fn": pooled_fn.astype(int),
            "tn": pooled_tn.astype(int),
        }
    )
    best = sweep.sort_values(
        ["macro_f1", "macro_precision", "macro_recall", "threshold"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).iloc[0]
    metrics = {key: best[key] for key in best.index if key != "threshold"}
    return ThresholdSelection(float(best["threshold"]), metrics, sweep)


def equal_weights(features: Sequence[str]) -> dict[str, float]:
    if not features:
        raise ValueError("At least one feature is required.")
    return {feature: 1 / len(features) for feature in features}


def all_nonempty_feature_subsets(features: Sequence[str]) -> pd.DataFrame:
    """Enumerate every nonempty subset with stable integer masks and equal weights."""

    if not features:
        raise ValueError("At least one feature is required.")
    rows = []
    for subset_mask in range(1, 2 ** len(features)):
        active = [
            feature for index, feature in enumerate(features) if subset_mask & (1 << index)
        ]
        row: dict[str, object] = {
            "subset_mask": subset_mask,
            "feature_count": len(active),
            "feature_set": " + ".join(active),
        }
        row.update({f"includes_{feature}": feature in active for feature in features})
        rows.append(row)
    return pd.DataFrame(rows)


def subset_equal_weight_matrix(
    definitions: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:
    """Convert subset definitions to one unit-sum equal-weight row per subset."""

    matrix = np.column_stack(
        [definitions[f"includes_{feature}"].to_numpy(dtype=float) for feature in features]
    )
    counts = matrix.sum(axis=1)
    if (counts == 0).any():
        raise ValueError("Subset definitions include an empty feature set.")
    return matrix / counts[:, None]


def simplex_grid(step: float = 0.05, minimum: float = 0.05) -> pd.DataFrame:
    """Enumerate three-feature weights that meet the positive unit-simplex rules."""

    if step <= 0 or minimum <= 0 or minimum * 3 > 1:
        raise ValueError("Invalid simplex grid step or minimum.")
    units = round(1 / step)
    minimum_units = round(minimum / step)
    rows = []
    for first in range(minimum_units, units + 1):
        for second in range(minimum_units, units + 1):
            third = units - first - second
            if third < minimum_units:
                continue
            rows.append(
                {
                    "weight_F1": first * step,
                    "weight_F3": second * step,
                    "weight_F4": third * step,
                }
            )
    result = pd.DataFrame(rows)
    if not np.allclose(result.sum(axis=1), 1.0):
        raise AssertionError("Generated grid weights do not sum to one.")
    return result


def random_simplex_weights(
    count: int,
    *,
    minimum: float = 0.05,
    seed: int = 9,
) -> pd.DataFrame:
    """Generate seeded positive three-feature weights with the required minimum."""

    if count < 1 or minimum <= 0 or minimum * 3 >= 1:
        raise ValueError("Invalid random simplex count or minimum.")
    generator = np.random.default_rng(seed)
    base = generator.dirichlet(np.ones(3), size=count)
    weights = minimum + (1 - 3 * minimum) * base
    return pd.DataFrame(weights, columns=["weight_F1", "weight_F3", "weight_F4"])


def cross_validated_weight_results(
    daily_scores: pd.DataFrame,
    definitions: pd.DataFrame,
    *,
    excluded_substation: str | None = None,
) -> pd.DataFrame:
    """Evaluate candidate weights by inner LOSO using sure Beta days only."""

    required_daily = {"dataset", "substation_id", "true_day", "confidence"}
    missing_daily = required_daily - set(daily_scores.columns)
    if missing_daily:
        raise ValueError(f"Daily scores are missing columns: {sorted(missing_daily)}")
    required_definitions = {
        "weight_id",
        "search_origin",
        "weight_F1",
        "weight_F3",
        "weight_F4",
        "score_column",
    }
    missing_definitions = required_definitions - set(definitions.columns)
    if missing_definitions:
        raise ValueError(
            f"Weight definitions are missing columns: {sorted(missing_definitions)}"
        )

    eligible = daily_scores.loc[
        daily_scores["dataset"].eq("beta") & daily_scores["confidence"].eq("sure")
    ].copy()
    if excluded_substation is not None:
        eligible = eligible.loc[~eligible["substation_id"].eq(excluded_substation)]
        assert_heldout_absent(eligible, excluded_substation)
    inner_substations = sorted(eligible["substation_id"].unique())
    if len(inner_substations) < 2:
        raise ValueError("Nested weight selection requires at least two Beta substations.")

    rows = []
    base_columns = ["dataset", "substation_id", "true_day"]
    for definition in definitions.itertuples(index=False):
        fold_metrics = []
        thresholds = []
        for inner_heldout in inner_substations:
            training_mask = ~eligible["substation_id"].eq(inner_heldout)
            training = eligible.loc[training_mask, base_columns].copy()
            training["score"] = eligible.loc[training_mask, definition.score_column].to_numpy()
            assert_heldout_absent(training, inner_heldout)
            selection = select_threshold(training, score_column="score")
            evaluation = eligible.loc[~training_mask]
            prediction = evaluation[definition.score_column].ge(selection.threshold)
            fold_metrics.append(binary_metrics(evaluation["true_day"], prediction))
            thresholds.append(selection.threshold)
        fold_frame = pd.DataFrame(fold_metrics)
        rows.append(
            {
                "weight_id": definition.weight_id,
                "search_origin": definition.search_origin,
                "weight_F1": definition.weight_F1,
                "weight_F3": definition.weight_F3,
                "weight_F4": definition.weight_F4,
                "excluded_outer_substation": excluded_substation or "none_full_beta_loso",
                "inner_substations": len(inner_substations),
                "inner_macro_precision": float(fold_frame["precision"].mean()),
                "inner_macro_recall": float(fold_frame["recall"].mean()),
                "inner_macro_f1": float(fold_frame["f1"].mean()),
                "inner_pooled_tp": int(fold_frame["tp"].sum()),
                "inner_pooled_fp": int(fold_frame["fp"].sum()),
                "inner_pooled_fn": int(fold_frame["fn"].sum()),
                "mean_inner_threshold": float(np.mean(thresholds)),
                "minimum_inner_threshold": float(np.min(thresholds)),
                "maximum_inner_threshold": float(np.max(thresholds)),
            }
        )
    return pd.DataFrame(rows)


def select_best_weight_result(results: pd.DataFrame) -> pd.Series:
    """Select one weight vector with the PRD macro-F1 and precision tie-breaks."""

    if results.empty:
        raise ValueError("Cannot select a weight from empty results.")
    return results.sort_values(
        [
            "inner_macro_f1",
            "inner_macro_precision",
            "inner_macro_recall",
            "weight_id",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).iloc[0]


def ml_hyperparameter_definitions(config: dict) -> pd.DataFrame:
    """Expand the small declared DNN, RF, and XGBoost parameter grids."""

    ml = config["m9_pbm"]["ml_comparison"]
    rows = []
    model_grids = {
        "dnn": [
            {
                "hidden_layer_sizes": tuple(hidden),
                "alpha": alpha,
                "max_iter": ml["dnn"]["max_iter"],
            }
            for hidden, alpha in product(
                ml["dnn"]["hidden_layer_sizes"], ml["dnn"]["alpha"]
            )
        ],
        "random_forest": [
            {
                "n_estimators": estimators,
                "max_depth": depth,
                "min_samples_leaf": leaf,
            }
            for estimators, depth, leaf in product(
                ml["random_forest"]["n_estimators"],
                ml["random_forest"]["max_depth"],
                ml["random_forest"]["min_samples_leaf"],
            )
        ],
        "xgboost": [
            {
                "n_estimators": estimators,
                "max_depth": depth,
                "learning_rate": rate,
                "subsample": ml["xgboost"]["subsample"],
                "colsample_bytree": ml["xgboost"]["colsample_bytree"],
            }
            for estimators, depth, rate in product(
                ml["xgboost"]["n_estimators"],
                ml["xgboost"]["max_depth"],
                ml["xgboost"]["learning_rate"],
            )
        ],
    }
    for model, parameter_sets in model_grids.items():
        for index, parameters in enumerate(parameter_sets):
            serialisable = {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in parameters.items()
            }
            rows.append(
                {
                    "model": model,
                    "hyperparameter_id": f"{model}_{index:02d}",
                    "parameters": parameters,
                    "parameters_json": json.dumps(serialisable, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def balanced_training_weights(
    training: pd.DataFrame,
    *,
    dataset_balanced: bool,
) -> np.ndarray:
    """Give datasets, classes, and substations controlled total influence."""

    required = {"dataset", "substation_id", "true_day"}
    missing = required - set(training.columns)
    if missing:
        raise ValueError(f"Training frame is missing weight columns: {sorted(missing)}")
    data = training.reset_index(drop=True)
    weights = np.zeros(len(data), dtype=float)
    datasets = sorted(data["dataset"].unique())
    dataset_targets = {
        dataset: 1 / len(datasets) if dataset_balanced else 1.0 for dataset in datasets
    }
    if not dataset_balanced and len(datasets) > 1:
        dataset_sizes = data["dataset"].value_counts(normalize=True)
        dataset_targets = dataset_sizes.to_dict()

    for dataset in datasets:
        dataset_mask = data["dataset"].eq(dataset).to_numpy()
        dataset_frame = data.loc[dataset_mask]
        classes = sorted(dataset_frame["true_day"].astype(bool).unique())
        for true_day in classes:
            class_mask = dataset_mask & data["true_day"].astype(bool).eq(true_day).to_numpy()
            class_frame = data.loc[class_mask]
            substations = sorted(class_frame["substation_id"].unique())
            target = dataset_targets[dataset] / len(classes) / len(substations)
            for substation in substations:
                group_mask = class_mask & data["substation_id"].eq(substation).to_numpy()
                weights[group_mask] = target / group_mask.sum()
    if (weights <= 0).any():
        raise AssertionError("Every ML training row must receive positive weight.")
    return weights / weights.mean()


def fit_ml_classifier(
    model: str,
    parameters: dict,
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    seed: int,
):
    """Fit one deterministic classifier from the declared compact grids."""

    if model == "dnn":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                **parameters,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
            ),
        )
        estimator.fit(x, y, mlpclassifier__sample_weight=sample_weight)
        return estimator
    if model == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier(
            **parameters,
            random_state=seed,
            n_jobs=1,
        )
        estimator.fit(x, y, sample_weight=sample_weight)
        return estimator
    if model == "xgboost":
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            **parameters,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
        )
        estimator.fit(x, y, sample_weight=sample_weight)
        return estimator
    raise ValueError(f"Unknown ML model: {model}")


def ml_probability(estimator, features: np.ndarray) -> np.ndarray:
    """Return the positive-class probability for any supported estimator."""

    return estimator.predict_proba(features)[:, 1]


def evaluate_ml_hyperparameters(
    pool: pd.DataFrame,
    definitions: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    fold_substations: Sequence[str],
    dataset_balanced: bool,
    seed: int,
) -> pd.DataFrame:
    """Evaluate declared hyperparameters by training-only-threshold LOSO."""

    rows = []
    for definition in definitions.itertuples(index=False):
        fold_rows = []
        thresholds = []
        for heldout in fold_substations:
            evaluation_mask = pool["substation_id"].eq(heldout)
            training = pool.loc[~evaluation_mask].copy()
            evaluation = pool.loc[evaluation_mask].copy()
            assert_heldout_absent(training, heldout)
            sample_weight = balanced_training_weights(
                training, dataset_balanced=dataset_balanced
            )
            estimator = fit_ml_classifier(
                definition.model,
                definition.parameters,
                training[list(feature_columns)].to_numpy(dtype=float),
                training["true_day"].astype(bool).to_numpy(),
                sample_weight,
                seed=seed,
            )
            training["probability"] = ml_probability(
                estimator, training[list(feature_columns)].to_numpy(dtype=float)
            )
            threshold = select_threshold(
                training,
                score_column="probability",
                dataset_balanced=dataset_balanced,
            ).threshold
            prediction = ml_probability(
                estimator, evaluation[list(feature_columns)].to_numpy(dtype=float)
            ) >= threshold
            fold_rows.append(binary_metrics(evaluation["true_day"], prediction))
            thresholds.append(threshold)
        fold_metrics = pd.DataFrame(fold_rows)
        rows.append(
            {
                "model": definition.model,
                "hyperparameter_id": definition.hyperparameter_id,
                "parameters_json": definition.parameters_json,
                "inner_macro_precision": float(fold_metrics["precision"].mean()),
                "inner_macro_recall": float(fold_metrics["recall"].mean()),
                "inner_macro_f1": float(fold_metrics["f1"].mean()),
                "mean_inner_threshold": float(np.mean(thresholds)),
                "minimum_inner_threshold": float(np.min(thresholds)),
                "maximum_inner_threshold": float(np.max(thresholds)),
            }
        )
    return pd.DataFrame(rows)


def select_best_ml_hyperparameter(results: pd.DataFrame, model: str) -> pd.Series:
    """Select one model-specific configuration by macro F1 and PR tie-breaks."""

    candidates = results.loc[results["model"].eq(model)]
    if candidates.empty:
        raise ValueError(f"No hyperparameter results for {model}.")
    return candidates.sort_values(
        [
            "inner_macro_f1",
            "inner_macro_precision",
            "inner_macro_recall",
            "hyperparameter_id",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).iloc[0]


def run_nested_ml_outer_experiment(
    training_pool: pd.DataFrame,
    outer_evaluation: pd.DataFrame,
    definitions: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    inner_fold_substations: Sequence[str],
    dataset_balanced: bool,
    seed: int,
    regime: str,
    outer_identifier: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select ML hyperparameters inside training data, then predict one outer set."""

    inner_results = evaluate_ml_hyperparameters(
        training_pool,
        definitions,
        feature_columns=feature_columns,
        fold_substations=inner_fold_substations,
        dataset_balanced=dataset_balanced,
        seed=seed,
    )
    definition_lookup = definitions.set_index("hyperparameter_id")
    selected_rows = []
    decision_parts = []
    for model in sorted(definitions["model"].unique()):
        selected = select_best_ml_hyperparameter(inner_results, model)
        definition = definition_lookup.loc[selected["hyperparameter_id"]]
        sample_weight = balanced_training_weights(
            training_pool, dataset_balanced=dataset_balanced
        )
        estimator = fit_ml_classifier(
            model,
            definition["parameters"],
            training_pool[list(feature_columns)].to_numpy(dtype=float),
            training_pool["true_day"].astype(bool).to_numpy(),
            sample_weight,
            seed=seed,
        )
        threshold_training = training_pool[
            ["dataset", "substation_id", "true_day"]
        ].copy()
        threshold_training["probability"] = ml_probability(
            estimator,
            training_pool[list(feature_columns)].to_numpy(dtype=float),
        )
        threshold = select_threshold(
            threshold_training,
            score_column="probability",
            dataset_balanced=dataset_balanced,
        ).threshold
        decisions = outer_evaluation[
            ["dataset", "substation_id", "date", "true_day", "confidence"]
        ].copy()
        decisions["probability"] = ml_probability(
            estimator,
            outer_evaluation[list(feature_columns)].to_numpy(dtype=float),
        )
        decisions["predicted_day"] = decisions["probability"].ge(threshold)
        decisions["model"] = model
        decisions["regime"] = regime
        decisions["outer_identifier"] = outer_identifier
        decisions["threshold"] = threshold
        decisions["hyperparameter_id"] = selected["hyperparameter_id"]
        decision_parts.append(decisions)
        selected_rows.append(
            {
                "regime": regime,
                "outer_identifier": outer_identifier,
                "model": model,
                "hyperparameter_id": selected["hyperparameter_id"],
                "parameters_json": selected["parameters_json"],
                "inner_macro_precision": selected["inner_macro_precision"],
                "inner_macro_recall": selected["inner_macro_recall"],
                "inner_macro_f1": selected["inner_macro_f1"],
                "outer_training_rows": len(training_pool),
                "outer_threshold": threshold,
            }
        )
    inner_results.insert(0, "outer_identifier", outer_identifier)
    inner_results.insert(0, "regime", regime)
    return inner_results, pd.DataFrame(selected_rows), pd.concat(decision_parts, ignore_index=True)


def confidence_coverage_metrics(
    predictions: pd.DataFrame,
    coverage_levels_pct: Sequence[int],
    *,
    truth_column: str = "true_day",
    prediction_column: str = "predicted_day",
    margin_column: str = "confidence_margin",
) -> pd.DataFrame:
    """Evaluate high-margin auto-accept decisions at fixed coverage levels."""

    required = {truth_column, prediction_column, margin_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Coverage input is missing columns: {sorted(missing)}")
    ordered = predictions.sort_values(
        [margin_column, "substation_id", "date"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    rows = []
    for coverage in coverage_levels_pct:
        auto_count = int(np.ceil(len(ordered) * coverage / 100))
        accepted = ordered.iloc[:auto_count]
        metrics = binary_metrics(accepted[truth_column], accepted[prediction_column])
        rows.append(
            {
                "coverage_pct": coverage,
                "auto_accepted_days": auto_count,
                "manual_review_days": len(ordered) - auto_count,
                "manual_review_pct": 100 * (len(ordered) - auto_count) / len(ordered),
                "true_rpf_days_remaining_for_review": int(
                    ordered.iloc[auto_count:][truth_column].sum()
                ),
                "auto_errors": metrics["fp"] + metrics["fn"],
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def recommended_coverage(
    coverage: pd.DataFrame,
    *,
    minimum_precision: float,
    minimum_f1: float,
) -> pd.Series | None:
    eligible = coverage.loc[
        coverage["precision"].ge(minimum_precision) & coverage["f1"].ge(minimum_f1)
    ]
    if eligible.empty:
        return None
    return eligible.sort_values("coverage_pct", ascending=False).iloc[0]

from __future__ import annotations

import io
from contextlib import redirect_stdout
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from src.evaluate import evaluate_day_level, evaluate_interval_level
from src.features import build_xgb1_features, build_xgb2_features
from src.io import req
from src.m7_threshold import run_m7


def _make_clf(params: Dict[str, Any]) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective=params.get("objective", "binary:logistic"),
        eval_metric=params.get("eval_metric", "aucpr"),
        tree_method=params.get("tree_method", "hist"),
        learning_rate=params.get("eta", 0.1),
        n_estimators=int(params.get("n_estimators", 500)),
        max_depth=int(params.get("max_depth", 6)),
        min_child_weight=int(params.get("min_child_weight", 3)),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        scale_pos_weight=params.get("scale_pos_weight", 5),
        random_state=int(params.get("seed", 9)),
        missing=np.nan,
    )


def _split_cfg(cfg: Dict[str, Any]) -> Dict[str, str]:
    split = req(cfg, "split")
    return {
        "train_end": str(req(split, "train_end")),
        "test_start": str(req(split, "test_start")),
    }


def _interval_hours_to_minutes(hours: float) -> int:
    return int(round(float(hours) * 60.0))


def _collect_test_metrics(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    col_site: str,
    col_ts: str,
    col_net: str,
    col_gt: str,
    pred_day_col: str,
    pred_flag_col: str,
) -> Dict[str, float]:
    split_cfg = _split_cfg(cfg)
    with redirect_stdout(io.StringIO()):
        day = evaluate_day_level(
            df,
            col_site,
            col_ts,
            col_net,
            col_gt,
            pred_day_col,
            split_cfg,
            rounding=3,
        )
        interval_tp = evaluate_interval_level(
            df,
            col_site,
            col_ts,
            col_net,
            col_gt,
            pred_day_col,
            pred_flag_col,
            split_cfg,
            tp_days_only=True,
            rounding=3,
        )
    return {
        "day_precision": float(day["test"]["precision"]),
        "day_recall": float(day["test"]["recall"]),
        "day_f1": float(day["test"]["f1"]),
        "interval_tp_precision": float(interval_tp["test"]["precision"]),
        "interval_tp_recall": float(interval_tp["test"]["recall"]),
        "interval_tp_f1": float(interval_tp["test"]["f1"]),
    }


def run_m7_one_at_a_time_sweep(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    col_site: str,
    col_ts: str,
    col_net: str,
    col_solar: str,
    col_gt: str,
) -> pd.DataFrame:
    base_m7 = deepcopy(req(cfg, "m7_threshold"))
    sweep_specs: Sequence[Tuple[str, Iterable[float]]] = (
        ("solar_peak_window_hours", (1.5, 2.5, 3.5)),
        ("min_threshold", (0.01, 0.025, 0.04)),
        ("min_threshold_both", (0.15, 0.25, 0.35)),
    )
    rows: List[Dict[str, Any]] = []

    for sweep_name, values in sweep_specs:
        for value in values:
            trial_cfg = deepcopy(cfg)
            trial_m7 = trial_cfg.setdefault("m7_threshold", {})
            trial_m7.update(base_m7)
            if sweep_name == "solar_peak_window_hours":
                trial_m7["peak_window_minutes"] = _interval_hours_to_minutes(float(value))
            else:
                trial_m7[sweep_name] = float(value)

            result = run_m7(df, trial_cfg, col_site, col_ts, col_net, col_solar)
            metrics = _collect_test_metrics(
                result,
                trial_cfg,
                col_site,
                col_ts,
                col_net,
                col_gt,
                "m7_rpf_day",
                "m7_rpf_flag",
            )
            rows.append(
                {
                    "sweep_name": sweep_name,
                    "solar_peak_window_hours": round(
                        float(trial_m7["peak_window_minutes"]) / 60.0, 3
                    ),
                    "min_threshold": float(trial_m7["min_threshold"]),
                    "min_threshold_both": float(trial_m7["min_threshold_both"]),
                    **metrics,
                }
            )

    columns = [
        "sweep_name",
        "solar_peak_window_hours",
        "min_threshold",
        "min_threshold_both",
        "day_precision",
        "day_recall",
        "day_f1",
        "interval_tp_precision",
        "interval_tp_recall",
        "interval_tp_f1",
    ]
    return pd.DataFrame(rows, columns=columns)


def _build_m8_result_frame(
    df: pd.DataFrame,
    day_df: pd.DataFrame,
    col_site: str,
    col_ts: str,
    pred_day: np.ndarray,
    prob_day: np.ndarray,
    ts_results: pd.DataFrame | None,
) -> pd.DataFrame:
    result = df.copy()
    if "date" not in result.columns:
        result["date"] = result[col_ts].dt.date

    mapped_day = day_df[[col_site, "date"]].copy()
    mapped_day["m8_rpf_day"] = pred_day.astype(bool)
    mapped_day["m8_prob_day"] = prob_day.astype(float)
    day_map = mapped_day.set_index([col_site, "date"])[["m8_rpf_day", "m8_prob_day"]]
    idx = result.set_index([col_site, "date"]).index

    result["m8_rpf_day"] = idx.map(day_map["m8_rpf_day"]).values
    result["m8_prob_day"] = idx.map(day_map["m8_prob_day"]).values
    result["m8_rpf_day"] = result["m8_rpf_day"].fillna(False).astype(bool)

    if ts_results is not None and not ts_results.empty:
        result = result.merge(ts_results, on=[col_site, col_ts], how="left")
        result["m8_rpf_flag"] = (
            result["m8_rpf_flag"].fillna(False).infer_objects(copy=False).astype(bool)
        )
    else:
        result["m8_rpf_flag"] = False
        result["m8_prob_ts"] = np.nan

    return result


def _sample_random_triplets(
    seed: int,
    n_random: int,
    default_triplet: Tuple[float, int, float],
) -> List[Tuple[float, int, float]]:
    rng = np.random.default_rng(seed)
    seen = {default_triplet}
    triplets: List[Tuple[float, int, float]] = []
    while len(triplets) < n_random:
        eta = round(float(np.exp(rng.uniform(np.log(0.03), np.log(0.30)))), 4)
        max_depth = int(rng.integers(4, 11))
        scale_pos_weight = round(float(np.exp(rng.uniform(np.log(2.0), np.log(10.0)))), 3)
        triplet = (eta, max_depth, scale_pos_weight)
        if triplet in seen:
            continue
        seen.add(triplet)
        triplets.append(triplet)
    return triplets


def run_m8_random_search(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    col_site: str,
    col_ts: str,
    col_net: str,
    col_solar: str,
    col_gt: str,
    seed: int = 123,
    trials: int = 5,
) -> pd.DataFrame:
    if trials < 1:
        raise ValueError("trials must be at least 1")

    m8_cfg = req(cfg, "m8_xgb")
    xgb1_base = deepcopy(req(m8_cfg, "xgb1_day"))
    xgb2_base = deepcopy(req(m8_cfg, "xgb2_timestamp"))
    thr1 = float(xgb1_base.get("threshold", 0.5))
    thr2 = float(xgb2_base.get("threshold", 0.5))

    default_triplet = (
        round(float(xgb1_base["eta"]), 4),
        int(xgb1_base["max_depth"]),
        round(float(xgb1_base["scale_pos_weight"]), 3),
    )
    random_triplets = _sample_random_triplets(seed, trials - 1, default_triplet)
    triplets = [default_triplet] + random_triplets

    day_df, feat_cols1, label1 = build_xgb1_features(
        df, cfg, col_site, col_ts, col_net, col_solar, col_gt
    )
    day_df = day_df.copy()
    day_df["_date_ts"] = pd.to_datetime(day_df["date"])

    split = req(cfg, "split")
    test_start = pd.Timestamp(req(split, "test_start"))
    test_end = pd.Timestamp(req(split, "test_end"))
    is_test1 = (day_df["_date_ts"] >= test_start) & (day_df["_date_ts"] <= test_end)
    train1 = day_df.loc[~is_test1]

    X_train1 = train1[feat_cols1].values.astype(np.float32)
    y_train1 = train1[label1].values.astype(np.uint8)
    X_all1 = day_df[feat_cols1].values.astype(np.float32)

    rows: List[Dict[str, Any]] = []
    for eta, max_depth, scale_pos_weight in triplets:
        xgb1_cfg = deepcopy(xgb1_base)
        xgb2_cfg = deepcopy(xgb2_base)
        for params in (xgb1_cfg, xgb2_cfg):
            params["eta"] = eta
            params["max_depth"] = int(max_depth)
            params["scale_pos_weight"] = scale_pos_weight

        clf1 = _make_clf(xgb1_cfg)
        clf1.fit(X_train1, y_train1)
        prob_day = clf1.predict_proba(X_all1)[:, 1]
        pred_day = (prob_day >= thr1).astype(np.uint8)

        pos_keys = day_df.loc[pred_day.astype(bool), [col_site, "date"]].copy()
        ts_results: pd.DataFrame | None = None
        if not pos_keys.empty:
            ts_df, feat_cols2, label2 = build_xgb2_features(
                df,
                cfg,
                day_df.drop(columns=["_date_ts"]),
                pos_keys,
                col_site,
                col_ts,
                col_net,
                col_solar,
                col_gt,
            )
            if not ts_df.empty:
                ts_df = ts_df.copy()
                ts_df["_date_ts"] = pd.to_datetime(ts_df["date"])
                is_test2 = (ts_df["_date_ts"] >= test_start) & (ts_df["_date_ts"] <= test_end)
                train2 = ts_df.loc[~is_test2]
                if not train2.empty and train2[label2].nunique() > 1:
                    clf2 = _make_clf(xgb2_cfg)
                    X_train2 = train2[feat_cols2].values.astype(np.float32)
                    y_train2 = train2[label2].values.astype(np.uint8)
                    clf2.fit(X_train2, y_train2)

                    X_all2 = ts_df[feat_cols2].values.astype(np.float32)
                    prob_ts = clf2.predict_proba(X_all2)[:, 1]
                    pred_ts = (prob_ts >= thr2).astype(bool)
                    ts_results = ts_df[[col_site, col_ts]].copy()
                    ts_results["m8_rpf_flag"] = pred_ts
                    ts_results["m8_prob_ts"] = prob_ts

        result = _build_m8_result_frame(
            df,
            day_df.drop(columns=["_date_ts"]),
            col_site,
            col_ts,
            pred_day,
            prob_day,
            ts_results,
        )
        metrics = _collect_test_metrics(
            result,
            cfg,
            col_site,
            col_ts,
            col_net,
            col_gt,
            "m8_rpf_day",
            "m8_rpf_flag",
        )
        rows.append(
            {
                "eta": eta,
                "max_depth": int(max_depth),
                "scale_pos_weight": scale_pos_weight,
                **metrics,
            }
        )

    columns = [
        "eta",
        "max_depth",
        "scale_pos_weight",
        "day_precision",
        "day_recall",
        "day_f1",
        "interval_tp_precision",
        "interval_tp_recall",
        "interval_tp_f1",
    ]
    return pd.DataFrame(rows, columns=columns)


__all__ = [
    "run_m7_one_at_a_time_sweep",
    "run_m8_random_search",
]

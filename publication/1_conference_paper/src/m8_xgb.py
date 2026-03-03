"""m8_xgb -- Two-stage XGBoost method for RPF detection.

Stage 1 (XGB1): Day-level classifier — which (site, date) pairs have RPF?
Stage 2 (XGB2): Interval-level classifier — which 15-min intervals within
                XGB1-positive days are affected?

Both models use fixed hyperparameters and thresholds from ``config/run.yaml``.
Trained models are saved to ``outputs/`` as pickle files.
"""
from __future__ import annotations

import pickle
import time as _time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features import build_xgb1_features, build_xgb2_features
from src.io import req


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clf(params: Dict[str, Any]) -> xgb.XGBClassifier:
    """Instantiate an ``XGBClassifier`` from a config dict."""
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


def _print_cm(name: str, tp: int, fp: int, fn: int, tn: int) -> None:
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    print(f"  [{name}] TP={tp}  FP={fp}  FN={fn}  TN={tn}  "
          f"P={P:.3f}  R={R:.3f}  F1={F1:.3f}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_m8(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    col_site: str,
    col_ts: str,
    col_net: str,
    col_solar: str,
    col_gt: str,
) -> pd.DataFrame:
    """Run the two-stage XGBoost RPF detection pipeline.

    Adds columns:

    - ``m8_rpf_day``   : bool — day-level RPF flag from XGB1.
    - ``m8_rpf_flag``  : bool — interval-level RPF flag from XGB2.
    - ``net_load_MW_m8``: float — sign-corrected net load.
    - ``m8_prob_day``  : float — XGB1 probability.
    - ``m8_prob_ts``   : float — XGB2 probability (NaN where XGB2 didn't run).

    Returns a copy of *df* with these columns appended.
    """
    t0 = _time.perf_counter()

    m8_cfg = req(cfg, "m8_xgb")
    split = req(cfg, "split")
    repo_root = Path(__file__).resolve().parents[1]
    output_dir_cfg = Path(cfg.get("paths", {}).get("output_dir", "outputs"))
    output_dir = output_dir_cfg if output_dir_cfg.is_absolute() else (repo_root / output_dir_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    xgb1_cfg = req(m8_cfg, "xgb1_day")
    xgb2_cfg = req(m8_cfg, "xgb2_timestamp")
    thr1 = float(xgb1_cfg.get("threshold", 0.5))
    thr2 = float(xgb2_cfg.get("threshold", 0.5))

    test_start = pd.Timestamp(split["test_start"])
    test_end = pd.Timestamp(split["test_end"])

    # ================================================================
    # STAGE 1: XGB1 — Day-level classification
    # ================================================================
    print("=" * 60)
    print("STAGE 1: XGB1 day-level classification")
    print("=" * 60)

    day_df, feat_cols1, label1 = build_xgb1_features(
        df, cfg, col_site, col_ts, col_net, col_solar, col_gt
    )

    # Split
    day_df["_date_ts"] = pd.to_datetime(day_df["date"])
    is_test1 = (day_df["_date_ts"] >= test_start) & (day_df["_date_ts"] <= test_end)
    train1 = day_df.loc[~is_test1]
    test1 = day_df.loc[is_test1]

    X_train1 = train1[feat_cols1].values.astype(np.float32)
    y_train1 = train1[label1].values.astype(np.uint8)
    X_test1 = test1[feat_cols1].values.astype(np.float32)
    y_test1 = test1[label1].values.astype(np.uint8)
    print(f"  Train: {len(y_train1):,} rows ({y_train1.sum():,} positive)")
    print(f"  Test:  {len(y_test1):,} rows ({y_test1.sum():,} positive)")

    # Train
    clf1 = _make_clf(xgb1_cfg)
    clf1.fit(X_train1, y_train1)

    # Predict on ALL data
    X_all1 = day_df[feat_cols1].values.astype(np.float32)
    p1_all = clf1.predict_proba(X_all1)[:, 1]
    pred1_all = (p1_all >= thr1).astype(int)

    day_df["m8_prob_day"] = p1_all
    day_df["m8_rpf_day"] = pred1_all.astype(bool)

    # Evaluate on test set
    p1_te = p1_all[is_test1.values]
    pred1_te = pred1_all[is_test1.values]
    tp1 = int(((pred1_te == 1) & (y_test1 == 1)).sum())
    fp1 = int(((pred1_te == 1) & (y_test1 == 0)).sum())
    fn1 = int(((pred1_te == 0) & (y_test1 == 1)).sum())
    tn1 = int(((pred1_te == 0) & (y_test1 == 0)).sum())
    _print_cm("XGB1 TEST", tp1, fp1, fn1, tn1)

    # Save XGB1 model
    xgb1_path = output_dir / "xgb1_day.pkl"
    with open(xgb1_path, "wb") as f:
        pickle.dump({"model": clf1, "feat_cols": feat_cols1, "threshold": thr1}, f)
    print(f"  Model saved: {xgb1_path}")

    # ================================================================
    # STAGE 2: XGB2 — Interval-level classification
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: XGB2 interval-level classification")
    print("=" * 60)

    # Positive keys from XGB1 (both train and test)
    pos_keys = day_df.loc[day_df["m8_rpf_day"], [col_site, "date"]].copy()
    print(f"  XGB1-positive site-days: {len(pos_keys):,}")

    ts_df, feat_cols2, label2 = build_xgb2_features(
        df, cfg, day_df, pos_keys, col_site, col_ts, col_net, col_solar, col_gt
    )

    # Split
    ts_df["_date_ts"] = pd.to_datetime(ts_df["date"])
    is_test2 = (ts_df["_date_ts"] >= test_start) & (ts_df["_date_ts"] <= test_end)
    train2 = ts_df.loc[~is_test2]
    test2 = ts_df.loc[is_test2]

    X_train2 = train2[feat_cols2].values.astype(np.float32)
    y_train2 = train2[label2].values.astype(np.uint8)
    X_test2 = test2[feat_cols2].values.astype(np.float32)
    y_test2 = test2[label2].values.astype(np.uint8)
    print(f"  Train: {len(y_train2):,} rows ({y_train2.sum():,} positive)")
    print(f"  Test:  {len(y_test2):,} rows ({y_test2.sum():,} positive)")

    # Train
    clf2 = _make_clf(xgb2_cfg)
    clf2.fit(X_train2, y_train2)

    # Predict on ALL XGB2 data
    X_all2 = ts_df[feat_cols2].values.astype(np.float32)
    p2_all = clf2.predict_proba(X_all2)[:, 1]
    pred2_all = (p2_all >= thr2).astype(int)

    ts_df["m8_prob_ts"] = p2_all
    ts_df["m8_rpf_flag"] = pred2_all.astype(bool)

    # Evaluate on test set
    p2_te = p2_all[is_test2.values]
    pred2_te = pred2_all[is_test2.values]
    tp2 = int(((pred2_te == 1) & (y_test2 == 1)).sum())
    fp2 = int(((pred2_te == 1) & (y_test2 == 0)).sum())
    fn2 = int(((pred2_te == 0) & (y_test2 == 1)).sum())
    tn2 = int(((pred2_te == 0) & (y_test2 == 0)).sum())
    _print_cm("XGB2 TEST", tp2, fp2, fn2, tn2)

    # Save XGB2 model
    xgb2_path = output_dir / "xgb2_timestamp.pkl"
    with open(xgb2_path, "wb") as f:
        pickle.dump({"model": clf2, "feat_cols": feat_cols2, "threshold": thr2}, f)
    print(f"  Model saved: {xgb2_path}")

    # ================================================================
    # MAP RESULTS BACK TO ORIGINAL DataFrame
    # ================================================================
    print("\n" + "=" * 60)
    print("Mapping results to original DataFrame")
    print("=" * 60)

    result = df.copy()
    if "date" not in result.columns:
        result["date"] = result[col_ts].dt.date

    # --- Day-level: m8_rpf_day, m8_prob_day ---
    day_map = day_df.set_index([col_site, "date"])[["m8_rpf_day", "m8_prob_day"]]
    idx = result.set_index([col_site, "date"]).index
    result["m8_rpf_day"] = idx.map(day_map["m8_rpf_day"]).values
    result["m8_prob_day"] = idx.map(day_map["m8_prob_day"]).values
    result["m8_rpf_day"] = result["m8_rpf_day"].fillna(False).astype(bool)

    # --- Interval-level: m8_rpf_flag, m8_prob_ts ---
    # Build a lookup from XGB2 results
    ts_results = ts_df[[col_site, col_ts, "m8_rpf_flag", "m8_prob_ts"]].copy()
    result = result.merge(
        ts_results, on=[col_site, col_ts], how="left", suffixes=("", "_xgb2")
    )
    result["m8_rpf_flag"] = result["m8_rpf_flag"].fillna(False).infer_objects(copy=False).astype(bool)

    # --- Corrected MW ---
    result["net_load_MW_m8"] = np.where(
        result["m8_rpf_flag"],
        -result[col_net].values,
        result[col_net].values,
    )
    if result[col_net].isna().any():
        result.loc[result[col_net].isna(), "net_load_MW_m8"] = np.nan

    # --- Day-level confusion (test set, using XGB1 results) ---
    elapsed = _time.perf_counter() - t0
    n_day_flagged = int(result.groupby([col_site, "date"])["m8_rpf_day"].first().sum())
    n_interval_flagged = int(result["m8_rpf_flag"].sum())

    print(f"\nm8_xgb complete ({elapsed:.1f}s):")
    print(f"  Days flagged (XGB1):       {n_day_flagged:,}")
    print(f"  Intervals flagged (XGB2):  {n_interval_flagged:,}")
    print(f"  Thresholds: xgb1={thr1}, xgb2={thr2}")

    return result

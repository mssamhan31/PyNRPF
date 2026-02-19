"""Evaluation helpers for RPF detection methods.

Computes day-level and interval-level precision / recall / F1 for both
m7 (deterministic) and m8 (XGBoost), on train and test splits.

Ground truth is derived from ``net_load_ground_truth`` — this column is
**never** used in the detection logic of m7 or m8; it is used here *only*
for scoring their outputs.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rounding: int = 3,
) -> Dict[str, Any]:
    """Compute classification metrics from boolean/int arrays.

    Returns
    -------
    dict with keys: tp, fp, fn, tn, precision, recall, f1, support
    """
    y_t = np.asarray(y_true, dtype=bool)
    y_p = np.asarray(y_pred, dtype=bool)
    tp = int((y_p & y_t).sum())
    fp = int((y_p & ~y_t).sum())
    fn = int((~y_p & y_t).sum())
    tn = int((~y_p & ~y_t).sum())
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(P, rounding),
        "recall": round(R, rounding),
        "f1": round(F1, rounding),
        "support": int(y_t.sum()),
    }


def _print_metrics(label: str, m: Dict[str, Any]) -> None:
    """Print a single metrics dict in a compact format."""
    print(f"  [{label}]  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}  "
          f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")


# ---------------------------------------------------------------------------
# Day-level evaluation
# ---------------------------------------------------------------------------

def evaluate_day_level(
    df: pd.DataFrame,
    col_site: str,
    col_ts: str,
    col_net: str,
    col_gt: str,
    pred_day_col: str,
    split_cfg: Dict[str, str],
    rounding: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """Day-level evaluation for a single method.

    Excludes days where ``col_net`` (the *input* column) has any negative
    value — these are already-obvious RPF and not the algorithm's target.

    Parameters
    ----------
    pred_day_col : str
        Column with the method's day-level boolean prediction (e.g.
        ``"m7_rpf_day"`` or ``"m8_rpf_day"``).
    split_cfg : dict
        Must contain ``"train_end"`` and ``"test_start"`` date strings.
    """
    # Aggregate to (site, date)
    day_agg = df.groupby([col_site, "date"]).agg(
        y_pred=(pred_day_col, "first"),
        gt_min=(col_gt, "min"),
        net_min=(col_net, "min"),
    ).reset_index()
    day_agg["y_true"] = day_agg["gt_min"] < 0

    # Exclude days where raw net_load_MW is already negative
    day_agg["has_negative_mw"] = day_agg["net_min"] < 0
    n_excluded = int(day_agg["has_negative_mw"].sum())
    day_agg = day_agg.loc[~day_agg["has_negative_mw"]].copy()

    # Split
    dates = pd.to_datetime(day_agg["date"])
    train_mask = dates <= pd.Timestamp(split_cfg["train_end"])
    test_mask = dates >= pd.Timestamp(split_cfg["test_start"])

    train_m = compute_metrics(
        day_agg.loc[train_mask, "y_true"],
        day_agg.loc[train_mask, "y_pred"],
        rounding,
    )
    test_m = compute_metrics(
        day_agg.loc[test_mask, "y_true"],
        day_agg.loc[test_mask, "y_pred"],
        rounding,
    )

    print(f"\n  Day-level ({pred_day_col})  "
          f"[excluded {n_excluded} days with negative MW]")
    _print_metrics("TRAIN", train_m)
    _print_metrics("TEST ", test_m)

    return {"train": train_m, "test": test_m, "excluded_days": n_excluded}


# ---------------------------------------------------------------------------
# Interval-level evaluation
# ---------------------------------------------------------------------------

def evaluate_interval_level(
    df: pd.DataFrame,
    col_site: str,
    col_ts: str,
    col_net: str,
    col_gt: str,
    pred_flag_col: str,
    split_cfg: Dict[str, str],
    daytime_start_hour: int = 6,
    daytime_end_hour: int = 18,
    rpf_days_only: bool = False,
    rounding: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """Interval-level evaluation for a single method.

    Filters to daytime intervals (``daytime_start_hour`` <= hour <
    ``daytime_end_hour``).

    If ``rpf_days_only=True``, further restricts to (site, date) groups
    where the ground truth has at least one negative interval — this
    answers "on true RPF days, how well does the algorithm classify
    individual intervals?"
    """
    work = df[[col_site, "date", col_ts, col_net, col_gt, pred_flag_col]].copy()

    # Daytime filter
    hour = work[col_ts].dt.hour
    work = work.loc[(hour >= daytime_start_hour) & (hour < daytime_end_hour)].copy()

    # Drop rows where ground truth is null
    work = work.dropna(subset=[col_gt])

    # Exclude days where raw net_load_MW is already negative
    neg_days = (
        df.groupby([col_site, "date"])[col_net]
        .min()
        .reset_index(name="_net_min")
    )
    neg_days["_has_neg"] = neg_days["_net_min"] < 0
    work = work.merge(
        neg_days[[col_site, "date", "_has_neg"]],
        on=[col_site, "date"], how="left",
    )
    n_excl = int(work["_has_neg"].fillna(False).sum())
    work = work.loc[~work["_has_neg"].fillna(False)].copy()

    # RPF-days-only filter
    if rpf_days_only:
        day_gt = (
            work.groupby([col_site, "date"])[col_gt]
            .apply(lambda s: (s < 0).any())
            .reset_index(name="_is_rpf_day")
        )
        work = work.merge(day_gt, on=[col_site, "date"], how="left")
        work = work.loc[work["_is_rpf_day"].fillna(False)].copy()

    # Ground truth at interval level
    y_true = (work[col_gt] < 0).values
    y_pred = work[pred_flag_col].values.astype(bool)

    # Split
    dates = pd.to_datetime(work["date"])
    train_mask = dates <= pd.Timestamp(split_cfg["train_end"])
    test_mask = dates >= pd.Timestamp(split_cfg["test_start"])

    train_m = compute_metrics(y_true[train_mask], y_pred[train_mask], rounding)
    test_m = compute_metrics(y_true[test_mask], y_pred[test_mask], rounding)

    suffix = " (RPF days only)" if rpf_days_only else " (all days)"
    print(f"\n  Interval-level ({pred_flag_col}){suffix}")
    _print_metrics("TRAIN", train_m)
    _print_metrics("TEST ", test_m)

    return {"train": train_m, "test": test_m}


# ---------------------------------------------------------------------------
# Full evaluation for one method
# ---------------------------------------------------------------------------

def evaluate_method(
    df: pd.DataFrame,
    col_site: str,
    col_ts: str,
    col_net: str,
    col_gt: str,
    pred_day_col: str,
    pred_flag_col: str,
    split_cfg: Dict[str, str],
    method_name: str,
    rounding: int = 3,
) -> Dict[str, Any]:
    """Run all evaluations for a single method and return a nested dict."""
    print(f"\n{'=' * 60}")
    print(f"  Evaluation: {method_name}")
    print(f"{'=' * 60}")

    day = evaluate_day_level(
        df, col_site, col_ts, col_net, col_gt,
        pred_day_col, split_cfg, rounding,
    )
    interval_all = evaluate_interval_level(
        df, col_site, col_ts, col_net, col_gt,
        pred_flag_col, split_cfg, rpf_days_only=False, rounding=rounding,
    )
    interval_rpf = evaluate_interval_level(
        df, col_site, col_ts, col_net, col_gt,
        pred_flag_col, split_cfg, rpf_days_only=True, rounding=rounding,
    )

    return {
        "day": day,
        "interval_all": interval_all,
        "interval_rpf_days_only": interval_rpf,
    }


# ---------------------------------------------------------------------------
# Confusion matrix plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "",
    ax: Any = None,
    cmap: str = "viridis",
) -> Any:
    """Plot a 2x2 confusion matrix heatmap with counts annotated.

    Returns the matplotlib figure (or *ax.figure* if ax was provided).
    """
    import matplotlib.pyplot as plt

    y_t = np.asarray(y_true, dtype=bool)
    y_p = np.asarray(y_pred, dtype=bool)
    tp = int((y_p & y_t).sum())
    fp = int((y_p & ~y_t).sum())
    fn = int((~y_p & y_t).sum())
    tn = int((~y_p & ~y_t).sum())
    # Rows = actual class, Cols = predicted class
    # Row 0 = Positive (True), Row 1 = Negative (False)
    # Col 0 = Pred Positive, Col 1 = Pred Negative
    cm = np.array([[tp, fn], [fp, tn]])

    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.5, 4))
    else:
        fig = ax.figure

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, label="Count", shrink=0.8)

    # Annotate cells
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            colour = "white" if val < cm.max() * 0.5 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center",
                    fontsize=16, fontweight="bold", color=colour)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Pos", "Pred Neg"])
    ax.set_yticklabels(["Actual Pos", "Actual Neg"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    sub = f"P={P:.3f}  R={R:.3f}  F1={F1:.3f}"
    ax.set_title(f"{title}\n{sub}" if title else sub, fontsize=11)

    if own_fig:
        fig.tight_layout()
    return fig


def plot_all_confusion_matrices(
    df: pd.DataFrame,
    col_site: str,
    col_ts: str,
    col_net: str,
    col_gt: str,
    split_cfg: Dict[str, str],
    output_path: Optional[str] = None,
) -> Any:
    """Plot a 2x2 grid of confusion matrices (test set only).

    Grid layout:
        m7 day-level      |  m7 interval-level
        m8 day-level      |  m8 interval-level

    Interval panels use daytime (6-18h) only.  Days where ``col_net``
    has negative values are excluded.
    """
    import matplotlib.pyplot as plt

    # ── Test set filter ───────────────────────────────────────────────
    dates = pd.to_datetime(df["date"])
    test = df.loc[dates >= pd.Timestamp(split_cfg["test_start"])].copy()

    # ── Exclude negative-MW days ──────────────────────────────────────
    neg_day_keys = (
        test.groupby([col_site, "date"])[col_net]
        .min()
        .reset_index(name="_nm")
    )
    neg_day_keys["_excl"] = neg_day_keys["_nm"] < 0
    keep_keys = neg_day_keys.loc[~neg_day_keys["_excl"], [col_site, "date"]]

    test = test.merge(keep_keys, on=[col_site, "date"], how="inner")

    # ── Day-level ground truth ────────────────────────────────────────
    day_agg = test.groupby([col_site, "date"]).agg(
        gt_min=(col_gt, "min"),
    ).reset_index()
    day_agg["y_true"] = day_agg["gt_min"] < 0

    # ── Daytime intervals for interval-level ──────────────────────────
    hour = test[col_ts].dt.hour
    test_day = test.loc[(hour >= 6) & (hour < 18)].copy()
    test_day = test_day.dropna(subset=[col_gt])

    # ── Build figure ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Confusion Matrices — Test Set",
                 fontsize=14, fontweight="bold", y=0.98)

    methods = [
        ("m7_rpf_day", "m7_rpf_flag", "m7 (DTR)"),
        ("m8_rpf_day", "m8_rpf_flag", "m8 (XGBoost)"),
    ]

    for row_idx, (day_col, flag_col, label) in enumerate(methods):
        # Day-level panel
        if day_col in test.columns:
            day_pred = (
                test.groupby([col_site, "date"])[day_col]
                .first().reset_index()
            )
            d = day_agg.merge(day_pred, on=[col_site, "date"], how="inner")
            plot_confusion_matrix(
                d["y_true"].values, d[day_col].values,
                title=f"{label} — Day-Level", ax=axes[row_idx, 0],
            )
        else:
            axes[row_idx, 0].text(
                0.5, 0.5, f"{label} not available",
                ha="center", va="center",
                transform=axes[row_idx, 0].transAxes,
            )

        # Interval-level panel
        if flag_col in test_day.columns:
            y_true_int = (test_day[col_gt] < 0).values
            y_pred_int = test_day[flag_col].values.astype(bool)
            plot_confusion_matrix(
                y_true_int, y_pred_int,
                title=f"{label} — Interval (daytime)",
                ax=axes[row_idx, 1],
            )
        else:
            axes[row_idx, 1].text(
                0.5, 0.5, f"{label} not available",
                ha="center", va="center",
                transform=axes[row_idx, 1].transAxes,
            )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        print(f"\nConfusion matrix plot saved: {output_path}")

    return fig

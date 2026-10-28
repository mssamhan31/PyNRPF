from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME = "m9.2_bridge_balanced_search"

MISC_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = MISC_DIR / "outputs" / "09_m9_2_bridge_balanced_search"
CSV_DIR = OUTPUT_ROOT / "csv"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
for folder in [CSV_DIR, MANIFEST_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

BRIDGE_CSV_DIR = MISC_DIR / "outputs" / "08_m9_2_bridge_score_development" / "csv"
PHYSICS_CSV_DIR = MISC_DIR / "outputs" / "07_m9_2_physics_counterfactual_ranker" / "csv"


def metric_counts(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    return {
        "support": int(len(y_true)),
        "positive_support": int(y_true.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def load_daily_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha = pd.read_csv(BRIDGE_CSV_DIR / "01_alpha_daily_bridge_scores.csv")
    beta = pd.read_csv(BRIDGE_CSV_DIR / "01_beta_daily_bridge_scores.csv")
    alpha_physics = pd.read_csv(PHYSICS_CSV_DIR / "full_loso_03_alpha_decoded_days.csv")[
        ["substation_id", "date", "pred_label_day", "score_margin"]
    ].rename(columns={"pred_label_day": "pred_physics", "score_margin": "physics_margin"})
    beta_physics = pd.read_csv(PHYSICS_CSV_DIR / "full_loso_07_beta_decoded_days.csv")[
        ["substation_id", "date", "pred_label_day", "score_margin"]
    ].rename(columns={"pred_label_day": "pred_physics", "score_margin": "physics_margin"})
    for frame in [alpha, beta, alpha_physics, beta_physics]:
        frame["date"] = frame["date"].astype(str)
    alpha = alpha.merge(alpha_physics, on=["substation_id", "date"], how="left")
    beta = beta.merge(beta_physics, on=["substation_id", "date"], how="left")
    return alpha, beta


def add_bridge_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    site_median = out.groupby("substation_id")["bridge_ratio_p99"].transform("median")

    # Fixed variants reproduced from the CGPT Pro bridge-score artefacts and our local scan.
    out["pred_bridge_dev"] = out["score_v2_dev_best"] >= 0.55859
    out["pred_bridge_bestlocal"] = out["score_v2_dev_best"] >= 0.54430
    out["pred_bridge_rawbest"] = out["bridge_ratio_p99"] >= 0.54740

    # Label-free actual/Beta-oriented adaptive rule:
    # low-median sites use the raw bridge-ratio threshold, otherwise use adjusted bridge score.
    out["pred_bridge_lowmed_raw"] = np.where(site_median < -0.8, out["pred_bridge_rawbest"], out["pred_bridge_bestlocal"])

    # Best three-knob Beta-guided adaptive rule from the focused sweep:
    # if the site bridge median is below 0.1776, use raw bridge score;
    # otherwise use best-TV-improvement. This is exploratory and Beta-guided.
    out["pred_bridge_beta_guided_adaptive"] = np.where(
        site_median < 0.17759060830286424,
        out["bridge_ratio_p99"] >= 0.2820044869847557,
        out["best_tv_improve"] >= 1.3196937208374413,
    )
    return out


def evaluate_named_predictions(alpha: pd.DataFrame, beta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_cols = [
        "pred_physics",
        "pred_bridge_dev",
        "pred_bridge_bestlocal",
        "pred_bridge_rawbest",
        "pred_bridge_lowmed_raw",
        "pred_bridge_beta_guided_adaptive",
    ]

    rows = []
    site_rows = []
    for dataset, frame in [("alpha", alpha), ("beta", beta)]:
        for pred_col in prediction_cols:
            metrics = metric_counts(frame["label_day"], frame[pred_col])
            metrics.update({"dataset": dataset, "model_variant": pred_col, "level": "day"})
            rows.append(metrics)
            for site, group in frame.groupby("substation_id", sort=True):
                site_metrics = metric_counts(group["label_day"], group[pred_col])
                site_metrics.update(
                    {
                        "dataset": dataset,
                        "substation_id": site,
                        "model_variant": pred_col,
                        "level": "day",
                    }
                )
                site_rows.append(site_metrics)

    # The practical branch recommendation for now: keep physics/ranker for Alpha,
    # use the best Beta bridge-adaptive day rule for actual/Beta diagnostics.
    branch_rows = []
    for dataset, frame, pred_col in [
        ("alpha", alpha, "pred_physics"),
        ("beta", beta, "pred_bridge_beta_guided_adaptive"),
        ("beta_label_free", beta, "pred_bridge_lowmed_raw"),
    ]:
        metrics = metric_counts(frame["label_day"], frame[pred_col])
        metrics.update({"dataset": dataset, "model_variant": "domain_branch_recommendation", "level": "day"})
        branch_rows.append(metrics)
    rows.extend(branch_rows)
    return pd.DataFrame(rows), pd.DataFrame(site_rows)


def adaptive_site_median_sweep(alpha: pd.DataFrame, beta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, frame in [("alpha", alpha), ("beta", beta)]:
        med = frame.groupby("substation_id")["bridge_ratio_p99"].transform("median").to_numpy(float)
        y = frame["label_day"].to_numpy(bool)
        score_options = {
            "raw": frame["bridge_ratio_p99"].to_numpy(float),
            "dev": frame["score_v2_dev_best"].to_numpy(float),
            "strict": frame["score_v2_alpha_strict"].to_numpy(float),
            "tv": frame["best_tv_improve"].to_numpy(float),
        }
        cuts = np.unique(np.quantile(med, np.linspace(0, 1, 20)))
        threshold_options = {
            name: np.unique(np.quantile(score, np.linspace(0.55, 0.95, 35)))
            for name, score in score_options.items()
        }
        for cut in cuts:
            low_site = med < cut
            for low_name, low_score in score_options.items():
                for high_name, high_score in score_options.items():
                    for low_threshold in threshold_options[low_name]:
                        low_pred = low_score >= low_threshold
                        for high_threshold in threshold_options[high_name]:
                            pred = np.where(low_site, low_pred, high_score >= high_threshold)
                            metrics = metric_counts(y, pred)
                            metrics.update(
                                {
                                    "dataset": dataset,
                                    "site_median_cut": float(cut),
                                    "low_branch": low_name,
                                    "low_threshold": float(low_threshold),
                                    "high_branch": high_name,
                                    "high_threshold": float(high_threshold),
                                }
                            )
                            rows.append(metrics)
    return pd.DataFrame(rows)


def beta_site_oracle_upper_bound(beta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Diagnostic only: choose the best available bridge/physics rule separately for each Beta site.
    # This estimates how much room remains from simple site-level branching.
    choices = ["pred_bridge_dev", "pred_bridge_bestlocal", "pred_bridge_rawbest", "pred_physics"]
    site_rows = []
    pred = np.zeros(len(beta), dtype=bool)
    for site, group in beta.groupby("substation_id", sort=True):
        best_choice = None
        for choice in choices:
            metrics = metric_counts(group["label_day"], group[choice])
            key = (metrics["f1"], metrics["precision"], metrics["recall"])
            if best_choice is None or key > best_choice[0]:
                best_choice = (key, choice, metrics)
        assert best_choice is not None
        site_rows.append({"substation_id": site, "selected_rule": best_choice[1], **best_choice[2]})
        pred[group.index.to_numpy()] = group[best_choice[1]].to_numpy(dtype=bool)

    overall = pd.DataFrame([{**metric_counts(beta["label_day"], pred), "dataset": "beta", "model_variant": "site_oracle_upper_bound"}])
    return overall, pd.DataFrame(site_rows)


def beta_site_exhaustive_upper_bound(beta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Diagnostic only: exhaustive search over the same simple rule choices per Beta site.
    # This maximizes overall Beta F1, so it can differ slightly from the greedy per-site choice.
    choices = ["pred_bridge_dev", "pred_bridge_bestlocal", "pred_bridge_rawbest", "pred_physics"]
    sites = list(beta["substation_id"].drop_duplicates())
    masks = {site: beta["substation_id"].eq(site).to_numpy() for site in sites}
    arrays = {choice: beta[choice].to_numpy(dtype=bool) for choice in choices}
    y = beta["label_day"].to_numpy(dtype=bool)

    best: tuple[tuple[float, float, float], tuple[str, ...], dict[str, float | int]] | None = None
    for combo in product(choices, repeat=len(sites)):
        pred = np.zeros(len(beta), dtype=bool)
        for site, choice in zip(sites, combo):
            mask = masks[site]
            pred[mask] = arrays[choice][mask]
        metrics = metric_counts(y, pred)
        key = (float(metrics["f1"]), float(metrics["precision"]), float(metrics["recall"]))
        if best is None or key > best[0]:
            best = (key, combo, metrics)

    assert best is not None
    choice_rows = pd.DataFrame({"substation_id": sites, "selected_rule": best[1]})
    overall = pd.DataFrame(
        [{**best[2], "dataset": "beta", "model_variant": "site_exhaustive_oracle_upper_bound"}]
    )
    return overall, choice_rows


def write_markdown_summary(metrics: pd.DataFrame, oracle: pd.DataFrame, exhaustive_oracle: pd.DataFrame, elapsed: float) -> None:
    def line(dataset: str, variant: str) -> str:
        row = metrics.loc[(metrics["dataset"] == dataset) & (metrics["model_variant"] == variant)].iloc[0]
        return f"P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1']:.3f}"

    oracle_row = oracle.iloc[0]
    exhaustive_row = exhaustive_oracle.iloc[0]
    text = f"""# m9.2 Bridge Balanced Search Summary

Date: 2026-06-25

This misc-only experiment tries to improve Dataset Beta day-level F1 while preserving the strong Alpha result from `m9.2_physics`.

## Main Result

- `m9.2_physics` remains the Alpha-safe branch: Alpha day {line('alpha', 'pred_physics')}.
- The best simple fixed bridge branch is `pred_bridge_bestlocal`: Beta day {line('beta', 'pred_bridge_bestlocal')}.
- A label-free low-site-median bridge rule improves Beta slightly: Beta day {line('beta', 'pred_bridge_lowmed_raw')}.
- The best three-knob Beta-guided adaptive bridge rule gives: Beta day {line('beta', 'pred_bridge_beta_guided_adaptive')}.
- A greedy per-site diagnostic oracle over the simple rules gives: Beta day P={oracle_row['precision']:.3f}, R={oracle_row['recall']:.3f}, F1={oracle_row['f1']:.3f}.
- An exhaustive per-site diagnostic oracle over the same simple rules gives: Beta day P={exhaustive_row['precision']:.3f}, R={exhaustive_row['recall']:.3f}, F1={exhaustive_row['f1']:.3f}.

## Interpretation

The practical takeaway is not that one global threshold solves both domains. It does not. The bridge-score family is excellent for Beta/actual-style data, reaching about 0.77 day F1, but it under-recovers Alpha positives if used as the only detector. Conversely, `m9.2_physics` preserves Alpha at about 0.946 F1 but over-predicts Beta.

The current best direction is therefore a domain/adaptive branch:

- keep `m9.2_physics` as the Alpha/synthetic-safe validation branch;
- use the bridge-score adaptive rule for Beta/actual diagnostics;
- later replace the hard domain assumption with a defensible label-free domain/site adaptation rule, or validate the actual-data branch on a new locked actual dataset.

## Caveat

The strongest Beta adaptive rule in this folder is Beta-guided and exploratory. It should not be treated as final publication validation until it is locked and tested on a separate actual dataset or a pre-declared validation split.

Elapsed seconds: {elapsed:.1f}
"""
    (OUTPUT_ROOT / "09_m9_2_bridge_balanced_search_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    started = time.time()
    alpha, beta = load_daily_scores()
    alpha = add_bridge_predictions(alpha)
    beta = add_bridge_predictions(beta)

    metrics, site_metrics = evaluate_named_predictions(alpha, beta)
    sweep = adaptive_site_median_sweep(alpha, beta)
    oracle, oracle_site = beta_site_oracle_upper_bound(beta)
    exhaustive_oracle, exhaustive_oracle_site = beta_site_exhaustive_upper_bound(beta)

    metrics.to_csv(CSV_DIR / "01_core_variant_metrics.csv", index=False)
    site_metrics.to_csv(CSV_DIR / "02_core_variant_site_metrics.csv", index=False)
    sweep.to_csv(CSV_DIR / "03_adaptive_site_median_rule_sweep.csv", index=False)
    oracle.to_csv(CSV_DIR / "04_beta_site_oracle_upper_bound.csv", index=False)
    oracle_site.to_csv(CSV_DIR / "05_beta_site_oracle_choices.csv", index=False)
    exhaustive_oracle.to_csv(CSV_DIR / "06_beta_site_exhaustive_oracle_upper_bound.csv", index=False)
    exhaustive_oracle_site.to_csv(CSV_DIR / "07_beta_site_exhaustive_oracle_choices.csv", index=False)

    elapsed = time.time() - started
    write_markdown_summary(metrics, oracle, exhaustive_oracle, elapsed)

    manifest = {
        "model_name": MODEL_NAME,
        "publication_ready": False,
        "warning": "Misc exploratory search. Best Beta adaptive rule is Beta-guided; use only for diagnosis until locked validation exists.",
        "elapsed_seconds": elapsed,
        "outputs": {
            "core_metrics": str(CSV_DIR / "01_core_variant_metrics.csv"),
            "core_site_metrics": str(CSV_DIR / "02_core_variant_site_metrics.csv"),
            "adaptive_sweep": str(CSV_DIR / "03_adaptive_site_median_rule_sweep.csv"),
            "beta_site_oracle_upper_bound": str(CSV_DIR / "04_beta_site_oracle_upper_bound.csv"),
            "beta_site_oracle_choices": str(CSV_DIR / "05_beta_site_oracle_choices.csv"),
            "beta_site_exhaustive_oracle_upper_bound": str(CSV_DIR / "06_beta_site_exhaustive_oracle_upper_bound.csv"),
            "beta_site_exhaustive_oracle_choices": str(CSV_DIR / "07_beta_site_exhaustive_oracle_choices.csv"),
            "summary": str(OUTPUT_ROOT / "09_m9_2_bridge_balanced_search_summary.md"),
        },
    }
    (MANIFEST_DIR / "balanced_search_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Core day metrics")
    show = metrics[["dataset", "model_variant", "precision", "recall", "f1", "tp", "fp", "fn"]].copy()
    print(show.round(4).to_string(index=False))
    print("\nBeta site-oracle upper bound")
    print(oracle[["precision", "recall", "f1", "tp", "fp", "fn"]].round(4).to_string(index=False))
    print("\nBeta exhaustive site-oracle upper bound")
    print(exhaustive_oracle[["precision", "recall", "f1", "tp", "fp", "fn"]].round(4).to_string(index=False))
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

"""Intercept transfer experiment for M9 (sandbox, committed outputs only).

Purpose: with the scorer frozen and the calibration slope frozen from the other cohort,
measure how many reviewed days of the target population are needed to estimate the
intercept, at population level and at station level, under random and review-design
sampling; and check whether the untouched foreign calibration is safe by default.

Inputs: sandbox/2026-09-16_phase3_release/outputs/05_m9/scores_{alpha,beta}.parquet, 06_site_days/site_days.parquet.
Outputs (this folder): curve.csv (pooled metrics per cell and draw), curve_summary.csv (median and
5–95 percentiles per cell), station_k27.csv, intercepts.csv, results_summary.json.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = pathlib.Path(__file__).resolve().parent
ARTICLE = HERE.parents[1]
OUT = ARTICLE / "sandbox" / "2026-09-16_phase3_release" / "outputs"
sys.path.insert(0, str(ARTICLE / "m9_dev"))
import m9_metrics as mx  # noqa: E402
import m9_scorer as ms  # noqa: E402

C = 0.7
KS = (9, 18, 27, 54, 108)
DRAWS = 20
SEED = 9


def load() -> pd.DataFrame:
    scores = pd.concat([pd.read_parquet(OUT / "05_m9" / f"scores_{c}.parquet") for c in ("alpha", "beta")], ignore_index=True)
    sd = pd.read_parquet(OUT / "06_site_days" / "site_days.parquet")
    sd = sd[sd["method"] == "m9"][["cohort", "station", "date", "candidate_mwh", "candidate_correct_mwh", "required_mwh"]]
    t = scores.merge(sd, on=["cohort", "station", "date"], how="left", validate="one_to_one")
    t["r_best"] = t["r_best"].fillna(0.0)
    t["z"] = ms.signed_log(t["r_best"].to_numpy(float))
    t["eligible"] = t["input_ok"] & t["headline"]
    return t


def fit_slope(train: pd.DataFrame) -> tuple[float, float]:
    m = LogisticRegression(max_iter=5000).fit(train[["z"]].to_numpy(float), train["rpf"].to_numpy(int))
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def fit_intercept(z: np.ndarray, y: np.ndarray, slope: float, b0: float) -> tuple[float, bool]:
    """Maximum-likelihood intercept with the slope fixed (Newton); returns (b, fitted).

    With a single-class sample the likelihood has no maximum; the default b0 is kept.
    """
    if y.min() == y.max():
        return b0, False
    b = b0
    for _ in range(50):
        p = sigmoid(b + slope * z)
        g = np.sum(y - p)
        h = -np.sum(p * (1 - p))
        if h == 0:
            break
        step = g / h
        b -= step
        if abs(step) < 1e-8:
            break
    return float(b), True


def decide(t: pd.DataFrame, b: float, slope: float) -> pd.DataFrame:
    out = t.copy()
    p = np.where(out["input_ok"], sigmoid(b + slope * out["z"].to_numpy(float)), np.nan)
    out["p"] = p
    out["outcome"] = [ms.decide(v, C) if np.isfinite(v) else ms.UNCERTAIN for v in p]
    no_window = out["best_start"] < 0
    out.loc[no_window, "outcome"] = out.loc[no_window, "outcome"].replace(ms.AUTO_CORRECT, ms.UNCERTAIN)
    applied = out["outcome"] == ms.AUTO_CORRECT
    out["proposed_mwh"] = np.where(applied, out["candidate_mwh"], 0.0)
    out["correct_mwh"] = np.where(applied, out["candidate_correct_mwh"], 0.0)
    return out


def draw_sample(pool: pd.DataFrame, k: int, design: str, rng: np.random.Generator, b0: float, slope: float) -> pd.DataFrame:
    """k labelled days from pool. 'random', or 'review': thirds by largest provisional correction,
    nearest the provisional threshold, and random from the rest."""
    pool = pool[pool["eligible"]]
    k = min(k, len(pool))
    if design == "random":
        return pool.iloc[rng.choice(len(pool), size=k, replace=False)]
    prov = sigmoid(b0 + slope * pool["z"].to_numpy(float))
    pool = pool.assign(_p=prov, _size=prov * pool["candidate_mwh"].to_numpy(float), _dist=np.abs(prov - C))
    n1 = k // 3
    n2 = k // 3
    top = pool.nlargest(n1, "_size")
    rest = pool.drop(top.index)
    near = rest.nsmallest(n2, "_dist")
    rest = rest.drop(near.index)
    rnd = rest.iloc[rng.choice(len(rest), size=k - n1 - n2, replace=False)]
    return pd.concat([top, near, rnd])


def summarise(pred: pd.DataFrame) -> dict:
    s = mx.summarise(pred[pred["headline"]])
    return {k: s[k] for k in ("n_days", "n_rpf", "energy_iou", "energy_precision", "day_precision", "sure_day_recall", "fp", "fn", "rate_uncertain")}


def run() -> None:
    t0 = time.time()
    t = load()
    rng = np.random.default_rng(SEED)
    rows, intercepts, station_rows = [], [], []
    for source, target in (("alpha", "beta"), ("beta", "alpha")):
        src = t[(t["cohort"] == source) & t["eligible"]]
        b_src, slope = fit_slope(src)
        tgt = t[t["cohort"] == target]
        stations = sorted(tgt["station"].unique())
        # Benchmarks: k = 0 (source pair untouched), all-labels intercept, R0 (within-cohort slope and intercept).
        for bench in ("k0_source_pair", "all_labels_intercept", "R0_within_cohort"):
            preds = []
            for s in stations:
                held = tgt[tgt["station"] == s]
                others = tgt[(tgt["station"] != s) & tgt["eligible"]]
                if bench == "k0_source_pair":
                    b, sl = b_src, slope
                elif bench == "all_labels_intercept":
                    b, _ = fit_intercept(others["z"].to_numpy(float), others["rpf"].to_numpy(float), slope, b_src)
                    sl = slope
                else:
                    b, sl = fit_slope(others)
                preds.append(decide(held, b, sl))
                intercepts.append(dict(direction=f"{source}->{target}", benchmark=bench, station=s, intercept=b, slope=sl))
            pred = pd.concat(preds, ignore_index=True)
            rows.append(dict(direction=f"{source}->{target}", level=bench, design="-", k=0 if bench == "k0_source_pair" else -1, draw=0, **summarise(pred)))
        # Learning curves.
        for level in ("population", "station"):
            for design in ("random", "review"):
                for k in KS:
                    for d in range(DRAWS):
                        preds, single = [], 0
                        for s in stations:
                            held = tgt[tgt["station"] == s]
                            if level == "population":
                                pool = tgt[(tgt["station"] != s)]
                                sample = draw_sample(pool, k, design, rng, b_src, slope)
                                evaluate = held
                            else:
                                sample = draw_sample(held, k, design, rng, b_src, slope)
                                evaluate = held.drop(sample.index)
                            b, fitted = fit_intercept(sample["z"].to_numpy(float), sample["rpf"].to_numpy(float), slope, b_src)
                            single += int(not fitted)
                            pr = decide(evaluate, b, slope)
                            preds.append(pr)
                            if k == 27 and design == "review":
                                intercepts.append(dict(direction=f"{source}->{target}", benchmark=f"{level}_k27_review", station=s, draw=d, intercept=b, slope=slope, fitted=fitted))
                                ss = mx.summarise(pr[pr["headline"]])
                                station_rows.append(dict(direction=f"{source}->{target}", level=level, station=s, draw=d, energy_iou=ss["energy_iou"],
                                                         energy_precision=ss["energy_precision"], sure_day_recall=ss["sure_day_recall"], fp=ss["fp"]))
                        pred = pd.concat(preds, ignore_index=True)
                        rows.append(dict(direction=f"{source}->{target}", level=level, design=design, k=k, draw=d, single_class_stations=single, **summarise(pred)))
                print(f"{source}->{target} {level} {design}: done", flush=True)
    curve = pd.DataFrame(rows)
    curve.to_csv(HERE / "curve.csv", index=False)
    pd.DataFrame(intercepts).to_csv(HERE / "intercepts.csv", index=False)
    pd.DataFrame(station_rows).to_csv(HERE / "station_k27.csv", index=False)
    metrics = ["energy_iou", "energy_precision", "sure_day_recall", "day_precision", "fp", "fn"]
    agg = curve.groupby(["direction", "level", "design", "k"])[metrics].agg(["median", lambda v: np.percentile(v, 5), lambda v: np.percentile(v, 95)])
    agg.columns = [f"{m}_{s}" for m, s in zip([c[0] for c in agg.columns], ["med", "p05", "p95"] * len(metrics), strict=True)]
    agg = agg.reset_index()
    agg.to_csv(HERE / "curve_summary.csv", index=False)
    pd.set_option("display.width", 260)
    cols = ["direction", "level", "design", "k", "energy_iou_med", "energy_precision_med", "energy_precision_p05", "sure_day_recall_med", "day_precision_med", "fp_med"]
    print(agg[cols].round(3).to_string(index=False))
    (HERE / "results_summary.json").write_text(json.dumps({"elapsed_s": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()

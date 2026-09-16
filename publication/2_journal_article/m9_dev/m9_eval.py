"""Leave-one-station-out development harness for M9.

Purpose: score every site-day of Alpha and Beta with the reference scorer (label-free,
so scoring is done once and cached), then for each station fit the two calibration
coefficients on the other stations of the same cohort, apply the fixed public
control c, and pool held-out decisions. Only the calibration is fitted per fold.

Inputs:  dataset/final/dataset_alpha.parquet and dataset_beta.parquet (read-only).
Outputs: runs/<run_name>/config.json, scores_<cohort>.parquet (cached scoring),
         predictions_<cohort>.csv (held-out per site-day), summary_pooled.csv,
         summary_station.csv, and a printed report.
Key steps: load 96-slot site-days; compute the sigma floor per cohort; score; LOSO
         calibrate on 'sure' (Beta) or all (Alpha) training days; decide; summarise.

This is development evidence, not the final evaluation: the design choices were made
with all stations visible. Beta 'unsure' days are scored and reported separately and
never enter a calibration fit.

Usage (from the repository root):
    python publication/2_journal_article/m9_dev/m9_eval.py --run phase2_baseline
    python publication/2_journal_article/m9_dev/m9_eval.py --run r1_abs --variant abs
    python publication/2_journal_article/m9_dev/m9_eval.py --run r1_fullday --sigma fullday
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m9_metrics as mx  # noqa: E402
import m9_scorer as ms  # noqa: E402

ARTICLE = HERE.parent
DATA = ARTICLE / "dataset" / "final"
RUNS = HERE / "runs"
COHORTS = ("alpha", "beta")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sitedays(cohort: str) -> list[dict]:
    """Complete 96-slot site-days as dicts of arrays; incomplete days are dropped and counted."""
    df = pd.read_parquet(DATA / f"dataset_{cohort}.parquet")
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["d"] = df["ts"].dt.date.astype(str)
    df = df.sort_values(["substation_id", "ts"])
    days = []
    for (sid, d), g in df.groupby(["substation_id", "d"], sort=False):
        if len(g) != ms.SLOTS:
            continue
        days.append(dict(cohort=cohort, station=sid, date=d,
                         confidence=g["confidence"].iloc[0] if "confidence" in g else "controlled",
                         y=g["net_load_MW"].to_numpy(float), s=g["solar_MW"].to_numpy(float),
                         truth=g["label_interval"].to_numpy(bool)))
    return days


def sigma_floor(days: list[dict]) -> float:
    """Smallest non-zero absolute overnight step across the cohort: a label-free resolution proxy."""
    steps = np.concatenate([np.abs(np.diff((d["y"] + d["s"])[ms.OVERNIGHT])) for d in days])
    steps = steps[np.isfinite(steps) & (steps > 0)]
    return float(steps.min()) if steps.size else 1e-3


def station_scale(days: list[dict], floor: float, day_scale: str) -> dict[str, float]:
    """Per-station robust scale: median over the station's days of the chosen day-level scale.

    Label-free but uses every day of the station, including days later held out; the
    Phase 0 record accepts this mild leakage for a scale that carries no label information.
    """
    fn = ms.overnight_scale if day_scale == "overnight" else ms.fullday_scale
    out: dict[str, list[float]] = {}
    for d in days:
        out.setdefault(d["station"], []).append(fn(d["y"] + d["s"], floor))
    return {k: float(np.median(v)) for k, v in out.items()}


def score_cohort(days: list[dict], floor: float, variant: str, p_exp: float, sigma_mode: str, stat: str = "gain", missing: str = "abstain_day") -> pd.DataFrame:
    """Score every site-day; returns one row per day with windows, evidence and energies."""
    per_station = sigma_mode.startswith("station_")
    day_scale = sigma_mode.split("_")[-1]  # overnight | fullday
    scales = station_scale(days, floor, day_scale) if per_station else {}
    rows = []
    for d in days:
        sig = scales.get(d["station"]) if per_station else None
        r = ms.score_siteday(d["y"], d["s"], floor, variant=variant, p_exp=p_exp, sigma=sig, scale=day_scale, stat=stat, missing=missing)
        p_mwh, req, c_mwh = mx.energy_terms(d["y"], d["truth"], r["best_start"], r["best_end"])
        idx = np.flatnonzero(d["truth"])
        rows.append(dict(cohort=d["cohort"], station=d["station"], date=d["date"], confidence=d["confidence"],
                         rpf=int(d["truth"].any()),
                         true_start=int(idx[0]) if idx.size else -1, true_end=int(idx[-1]) if idx.size else -1,
                         true_slots=int(idx.size), required_mwh=req, proposed_mwh=p_mwh, correct_mwh=c_mwh,
                         **r))
    return pd.DataFrame(rows)


def loso(scores: pd.DataFrame, c: float) -> tuple[pd.DataFrame, list[dict]]:
    """Held-out probabilities and outcomes; calibration fitted on other stations' eligible days."""
    out, fits = [], []
    eligible = scores["input_ok"] & scores["confidence"].isin(["sure", "controlled"])
    for sid in sorted(scores["station"].unique()):
        train = scores[(scores["station"] != sid) & eligible]
        test = scores[scores["station"] == sid].copy()
        cal = ms.Calibrator().fit(train["r_best"].to_numpy(), train["rpf"].to_numpy())
        p = np.where(test["input_ok"], cal.probability(test["r_best"].fillna(0.0).to_numpy()), np.nan)
        test["p"] = p
        test["outcome"] = [ms.decide(v, c) if np.isfinite(v) else ms.UNCERTAIN for v in p]
        # A day the model would correct but whose best candidate is the null cannot
        # happen (p > c implies a positive window); guard anyway.
        test.loc[test["best_start"] < 0, "outcome"] = test.loc[test["best_start"] < 0, "outcome"].replace(ms.AUTO_CORRECT, ms.UNCERTAIN)
        out.append(test)
        fits.append(dict(held_out=sid, alpha=cal.alpha, beta=cal.beta, raw_threshold_correct=cal.raw_threshold(c),
                         raw_threshold_keep=cal.raw_threshold(1 - c), n_train=len(train)))
    return pd.concat(out, ignore_index=True), fits


def run(run_name: str, variant: str, p_exp: float, sigma_mode: str, c: float, stat: str = "gain", missing: str = "abstain_day") -> None:
    out_dir = RUNS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    config = dict(run=run_name, variant=variant, p_exp=p_exp, sigma_mode=sigma_mode, c=c, stat=stat, missing=missing,
                  scan=[ms.SCAN_START, ms.SCAN_END], code_sha256=sha256_file(HERE / "m9_scorer.py"),
                  data_sha256={f"dataset_{k}.parquet": sha256_file(DATA / f"dataset_{k}.parquet") for k in COHORTS})
    pooled, stations, fits_all = [], [], []
    for cohort in COHORTS:
        days = load_sitedays(cohort)
        floor = sigma_floor(days)
        config[f"sigma_floor_{cohort}"] = floor
        cache = out_dir / f"scores_{cohort}.parquet"
        scores = score_cohort(days, floor, variant, p_exp, sigma_mode, stat, missing)
        scores.to_parquet(cache, index=False)
        pred, fits = loso(scores, c)
        pred.to_csv(out_dir / f"predictions_{cohort}.csv", index=False)
        fits_all += [dict(cohort=cohort, **f) for f in fits]
        head = pred[pred["confidence"].isin(["sure", "controlled"])]
        rel = mx.calibration_reliability(head.loc[head["input_ok"], "p"], head.loc[head["input_ok"], "rpf"])
        pooled.append(dict(cohort=cohort, group="headline", **mx.summarise(head), **rel))
        if cohort == "beta":
            uns = pred[pred["confidence"] == "unsure"]
            pooled.append(dict(cohort=cohort, group="unsure_sensitivity", **mx.summarise(uns)))
        st = mx.station_table(head)
        st.insert(0, "cohort", cohort)
        stations.append(st)
    pd.DataFrame(pooled).to_csv(out_dir / "summary_pooled.csv", index=False)
    pd.concat(stations).to_csv(out_dir / "summary_station.csv", index=False)
    pd.DataFrame(fits_all).to_csv(out_dir / "calibration_fits.csv", index=False)
    config["elapsed_s"] = round(time.time() - t0, 1)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report(out_dir)


def report(out_dir: pathlib.Path) -> None:
    pd.set_option("display.width", 220)
    pooled = pd.read_csv(out_dir / "summary_pooled.csv")
    st = pd.read_csv(out_dir / "summary_station.csv")
    cols = ["cohort", "group", "n_days", "n_rpf", "energy_iou", "energy_precision", "day_f1", "day_precision",
            "day_recall", "sure_day_recall", "sure_day_uncertain_rate", "rate_auto_correct", "rate_uncertain", "ece"]
    print(f"\n=== {out_dir.name}: pooled ===")
    print(pooled[[c for c in cols if c in pooled]].round(4).to_string(index=False))
    scols = ["cohort", "station", "n_days", "n_rpf", "required_mwh", "energy_iou", "energy_precision",
             "sure_day_recall", "sure_day_uncertain_rate", "day_precision", "day_f1", "rate_auto_correct"]
    print(f"\n=== {out_dir.name}: per station (headline days only) ===")
    print(st[scols].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name; outputs go to runs/<run>/")
    ap.add_argument("--variant", default="sq", choices=["sq", "abs", "tv"])
    ap.add_argument("--p", type=float, default=1.0, help="duration exponent; 1 = per-slot mean")
    ap.add_argument("--sigma", default="overnight", choices=["overnight", "fullday", "station_overnight", "station_fullday"])
    ap.add_argument("--c", type=float, default=0.7, help="public confidence control")
    ap.add_argument("--stat", default="gain", choices=["gain", "llr"], help="evidence statistic")
    ap.add_argument("--missing", default="abstain_day", choices=["abstain_day", "mask_windows"])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.report_only:
        report(RUNS / a.run)
    else:
        run(a.run, a.variant, a.p, a.sigma, a.c, a.stat, a.missing)
